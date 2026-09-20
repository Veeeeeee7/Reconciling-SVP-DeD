import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def log(message, file='log.txt'):
    with open(file, 'a') as f:
        f.write(message + '\n')
        print(message)

def make_policy(df_data):
    """
    Create behavior policy π_b from the dataset.
    """
    # count occurrences of each state-action pair
    SA_count = df_data.groupby(['s:state', 'a:action']).size() \
        .unstack().reindex(index=range(nS), columns=range(nA)).fillna(0)

    # behavior policy
    π_b = SA_count.div(SA_count.sum(axis=1), axis=0)

    # only allow actions frequently used by clinicians
    SA_mask = (SA_count >= 6)

    # for states without any "available" actions, allow the most frequent action
    for s in range(nS): # changed from nS-1
        if SA_mask.loc[s].sum() == 0:
            SA_mask.loc[s, SA_count.loc[s].argmax()] = True

    return π_b, SA_mask, SA_count

def make_transition_matrix(df_data, nA, nS_total, S_survival, S_death):
    """
    Create the empirical transition matrix from the dataset.
    """
    # count occurrences of each transition
    SAS_count = df_data.groupby(['s:state', 'a:action', 's:next_state']).size().reset_index(name='count')

    # Create the transition matrix
    P = np.full((nS_total, nA, nS_total), np.nan)
    for i, row in SAS_count.iterrows():
        P[row['s:state'], row['a:action'], row['s:next_state']] = row['count']

    # Normalize the transition matrix
    P = P / np.nansum(P, axis=2, keepdims=True)

    # Set the transition probabilities for terminal states
    P[S_survival, :, :] = 0
    P[S_survival, :, S_survival] = 1
    P[S_death, :, :] = 0
    P[S_death, :, S_death] = 1

    return P

def make_gymP(P, R, nS, nA, nS_total, S_survival, S_death):
    """
    Convert the transition and reward matrices to the gym format.
    """
    gymP = defaultdict(lambda: defaultdict(list))
    for s in range(nS):
        for a in range(nA):
            for next_s in range(nS_total):
                if not np.isnan(P[s, a, next_s]):
                    prob = P[s, a, next_s]
                    reward = R[s, a, next_s]
                    done = int(next_s in [S_survival, S_death])
                    gymP[s][a].append((prob, next_s, reward, done))
    return gymP


def value_iteration_masked(gymP, nS, nA, SA_mask, gamma, theta=1e-10):
    V = np.zeros(nS)
    for _ in tqdm(itertools.count()):
        V_new = V.copy()
        for s in range(nS):
            ## V[s] = max {a} sum {s', r} P[s', r | s, a] * (r + gamma * V[s'])
            Q_s = np.zeros((nA))
            for a in range(nA):
                Q_s[a] = sum(p * (r + (0 if done else gamma * V[s_])) for p, s_, r, done in gymP[s][a])

            Q_s[~SA_mask[s]] = np.nan
            new_v = np.nanmax(Q_s)
            V_new[s] = new_v
        if np.isclose(np.linalg.norm(V_new - V), theta):
            break
        V = V_new


    pi = np.zeros((nS, nA))
    for s in range(nS):
        Q_s = np.zeros(nA)
        for a in range(nA):
            if a in gymP[s]:
                Q_s[a] = sum(
                    p * (r + (0 if done else gamma * V[s_]))
                    for p, s_, r, done in gymP[s][a]
                )

        Q_s[~SA_mask[s]] = np.nan

        if np.all(np.isnan(Q_s)):
            continue

        best_a = np.nanargmax(Q_s)
        pi[s, :] = 0.0
        pi[s, best_a] = 1.0

    return V, pi


def V2Q(P, V, nA, nS, SA_mask, gamma, mode='svp'):
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in P[s]:
            if not SA_mask[s, a]:
                Q[s, a] = np.nan
                continue
            for p, s_, r, done in P[s][a]:
                # Q[s, a] = np.sum(p * (r + gamma * V[s_] * (1 - done)) for p, s_, r, done in P[s][a])
                if mode == 'svp':
                    if s_ == 750:  # survival state
                        Q[s, a] += p * (r + gamma * 1 * (1 - done))
                    elif s_ == 751:  # death state
                        Q[s, a] += p * (r + gamma * -1 * (1 - done))
                    else:
                        Q[s, a] += p * (r + gamma * V[s_] * (1 - done))
                elif mode == 'ded':
                    if s_ == 750:  # survival state
                        Q[s, a] += p * (r + gamma * 0 * (1 - done))
                    elif s_ == 751:  # death state
                        Q[s, a] += p * (r + gamma * -1 * (1 - done))
                    else:
                        Q[s, a] += p * (r + gamma * V[s_] * (1 - done))
    return Q

def svp_masked(P, V_star, nS, nA, SA_mask, gamma, zeta, theta=1e-10, max_iter=1000,
               min_iter=0, return_status=False, optimal_policies=None):
    # min_iter=0 reproduces the solver used for all reported results; min_iter=10
    # reproduces the experiments.py variant (skips the checks for the first
    # min_iter sweeps and does not store those policies as cycle candidates).
    is_max_iter = False
    V = V_star.copy().astype(float)
    policies = []
    svp_policy = None
    n_iter = 0

    # diagnostics for the cycle-handling description in the paper (Sec. 3).
    # purely observational: nothing below changes the policy that is returned.
    status = {'terminated_by': None, 'n_sweeps': 0, 'cycle_start': None,
              'cycle_len': None, 'n_empty_before_fallback': 0, 'zeta': zeta,
              'min_iter': min_iter}

    Q = V2Q(P, V, nA, nS, SA_mask, gamma, mode='svp')
    pi = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        Q_s = Q[s].copy()

        # set invalid actions to nan
        Q_s[~SA_mask[s]] = np.nan
        pi[s] = (Q_s >= (1 - zeta) * V_star[s]) & SA_mask[s]

    policies.append(pi)

    while True:
        delta = 0.0
        for s in range(nS):
            old_v = V[s]
            Q_s = np.zeros(nA)

            # exploratory policy (compute one-step lookahead state-value function Q)
            for a in P[s]:
                # Q_s[a] = np.sum(p * (r + gamma * V[s_] * (1 - done)) for p, s_, r, done in P[s][a])
                for p, s_, r, done in P[s][a]:
                    if s_ == 750:  # survival state
                        Q_s[a] += p * (r + gamma * 1 * (1 - done))
                    elif s_ == 751:  # death state
                        Q_s[a] += p * (r + gamma * -1 * (1 - done))
                    else:
                        Q_s[a] += p * (r + gamma * V[s_] * (1 - done))
            # set invalid actions to nan
            Q_s[~SA_mask[s]] = np.nan

            # determine cutoff for both near greedy beneficial and harmful actions
            if V_star[s] >= 0:
                Q_cutoff = (1 - zeta) * V_star[s]
            else:
                Q_cutoff = V_star[s] - zeta * abs(V_star[s])

            # find indices of actions that meet the cutoff and are valid
            Pi_S = np.argwhere((Q_s >= Q_cutoff) & SA_mask[s])

            # update state-value function V using the best action from the selected set or worst action if none selected
            if len(Pi_S) > 0:
                new_v = Q_s[Pi_S].min()
            else:
                new_v = Q_s.max()

            V[s] = new_v
            delta = max(delta, np.abs(new_v - old_v))

        # update policy
        Q = V2Q(P, V, nA, nS, SA_mask, gamma, mode='svp')
        pi = np.zeros((nS, nA))
        for s in range(nS):
            if V_star[s] >= 0:
                threshold = (1 - zeta) * V_star[s]
            else:
                threshold = V_star[s] - zeta * abs(V_star[s])
            pi[s] = (Q[s] >= threshold) & SA_mask[s]

        n_iter += 1

        if n_iter < min_iter:
            continue

        if ((policies[-1] == pi).all() and delta < theta):
            svp_policy = pi
            iter = n_iter
            status['terminated_by'] = 'converged'
            print("Converged")
            break

        is_cycle = False
        for i, past_pi in enumerate(policies):
            if (past_pi == pi).all():
                is_cycle = True
                cycle_start = i
                cycle_policies = policies[cycle_start:]
                core_bool = (cycle_policies[0] > 0)
                for pol in cycle_policies[1:]:
                    core_bool &= (pol > 0)
                svp_policy = core_bool.astype(float)
        if is_cycle:
            iter = n_iter
            status['terminated_by'] = 'cycle'
            status['cycle_start'] = cycle_start
            status['cycle_len'] = len(policies) - cycle_start
            print("Cycle detected")
            break

        if n_iter >= max_iter:
            svp_policy = pi
            is_max_iter = True
            iter = n_iter
            status['terminated_by'] = 'max_iter'
            print("Max iterations reached")
            break

        policies.append(pi)

    # uses optimal actions for states without recommendations
    # optimal_policies is only used for the greedy fallback below; passing a
    # cached copy avoids recomputing the same value iteration on every call.
    if optimal_policies is None:
        _, optimal_policies = value_iteration_masked(P, nS, nA, SA_mask, gamma, theta=theta)
    for s in range(nS):
        recommendation_exist = False
        for a in range(nA):
            if svp_policy[s][a] != 0 & SA_mask[s][a]: recommendation_exist = True
        if not recommendation_exist:
            status['n_empty_before_fallback'] += 1
            for a in range(nA):
                if SA_mask[s][a]:
                    svp_policy[s][a] = optimal_policies[s][a]

    status['n_sweeps'] = n_iter
    if return_status:
        return V, svp_policy, is_max_iter, iter, status
    return V, svp_policy, is_max_iter, iter

def ded_deadend(Q_d, nS, nA, SA_mask, threshold):
    pi_ded_deadend = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            if Q_d[s, a] <= -threshold and SA_mask[s, a]:
                pi_ded_deadend[s, a] = 1
    return pi_ded_deadend

def compute_conflict_fraction(pi1, pi2, SA_mask, nS, nA):
    conflicts = np.zeros(nS)
    for s in range(nS):
        if any(pi1[s, a] and pi2[s, a] for a in range(nA)):
            conflicts[s] += 1
        elif all(SA_mask[s, a] == False for a in range(nA)):
            conflicts[s] = np.nan
    return np.nanmean(conflicts)

def compute_iou(pi1, pi2, SA_mask, nS, use_0_for_empty=True):
    ious = np.zeros(nS)
    for s in range(nS):
        if all(SA_mask[s, a] == False for a in range(nA)):
            ious[s] = np.nan
            continue
        actions1 = set(np.where(pi1[s] > 0)[0])
        actions2 = set(np.where(pi2[s] > 0)[0])
        intersection = actions1.intersection(actions2)
        union = actions1.union(actions2)
        if len(union) > 0:
            ious[s] = len(intersection) / len(union)
        else:
            if use_0_for_empty:
                ious[s] = 0.0
            else:
                ious[s] = 1.0

    return np.nanmean(ious)

def visualize_svp_and_bad_sizes(zeta_vals, dt_vals, svp_map, bad_map,
                                save_path="figures/svp_vs_bad_sizes_fixed.pdf"):
    """
    One plot, two y-axes:
      - Green (left y): SVP size vs zeta (mean over death_threshold)
      - Red   (right y): DeD eliminated size vs death_threshold (mean over zeta)

    Note: both curves share the same x-axis range (usually [0, 1]),
    but the x-values correspond to different hyperparameters.
    """
    svp_map = np.asarray(svp_map)
    bad_map = np.asarray(bad_map)

    if svp_map.shape != bad_map.shape:
        raise ValueError(f"svp_map shape {svp_map.shape} must match bad_map shape {bad_map.shape}")
    if svp_map.shape != (len(dt_vals), len(zeta_vals)):
        raise ValueError(
            f"Expected maps with shape (len(dt_vals), len(zeta_vals)) = "
            f"({len(dt_vals)}, {len(zeta_vals)}), got {svp_map.shape}"
        )

    # SVP depends on zeta -> average over dt for each zeta
    svp_curve = svp_map.mean(axis=0)   # (len(zeta_vals),)

    # DeD depends on death_threshold -> average over zeta for each dt
    bad_curve = bad_map.mean(axis=1)   # (len(dt_vals),)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Left axis: SVP (green)
    ax1.plot(zeta_vals, svp_curve, color="green", linewidth=2)
    ax1.set_xlabel("Hyperparameter value (ζ for SVP, θ_D for DeD)")
    ax1.set_ylabel("Avg SVP recommended actions per state", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    # Right axis: DeD eliminated (red)
    ax2 = ax1.twinx()
    ax2.plot(dt_vals, bad_curve, color="red", linewidth=2)
    ax2.set_ylabel("Avg DeD eliminated actions per state", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Add one additional ticker at the bottom of each y-axis
    ax1_yticks = list(ax1.get_yticks())
    # ax1_yticks.append(ax1.get_ylim()[0])
    ax1.set_yticks(ax1_yticks)

    ax2_yticks = list(ax2.get_yticks())
    # ax2_yticks.append(ax2.get_ylim()[0])
    ax2.set_yticks(ax2_yticks)

    # ax1.set_title("SVP size vs ζ (green) and DeD size vs θ_D (red)")

    # Optional: make the shared x-limits cover both ranges
    x_min = min(float(np.min(zeta_vals)), float(np.min(dt_vals)))
    x_max = max(float(np.max(zeta_vals)), float(np.max(dt_vals)))
    ax1.set_xlim(x_min, x_max)

    # Legend (green/red)
    legend_handles = [
        Line2D([0], [0], color="green", lw=2, label="SVP size vs ζ"),
        Line2D([0], [0], color="red",   lw=2, label="DeD size vs θ_D"),
    ]
    ax1.legend(handles=legend_handles, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # Clear previous log file
    if os.path.exists('log.txt'):
        os.remove('log.txt')
    with open('log.txt', 'w') as f:
        f.write('')

    # Define constants
    nS = 750
    nA = 25
    nS_term = 2
    S_survival = 750
    S_death = 751
    nS_total = nS + nS_term

    # Load the data
    data_dir = 'mimic_sepsis_data_2025/'
    df = pd.read_csv(data_dir + 'traj_shifted_train.csv', dtype={"a:action": "Int64", 'a:next_action': "Int64"})


    # Create behavior policy and transition matrix
    π_b, SA_mask, SA_count = make_policy(df)
    P = make_transition_matrix(df, nA, nS_total, S_survival, S_death)

    # SVP
    R_svp = np.zeros((nS_total, nA, nS_total))
    R_svp[:, :, S_survival] = 1
    R_svp[:, :, S_death] = -1
    P_svp = make_gymP(P, R_svp, nS, nA, nS_total, S_survival, S_death)
    V_star, pi_star = value_iteration_masked(P_svp, nS, nA, SA_mask.values, gamma=1.0, theta=1e-10)

    R_ded_deadend = np.zeros((nS_total, nA, nS_total))
    R_ded_deadend[:, :, S_death] = -1
    P_deadend = make_gymP(P, R_ded_deadend, nS, nA, nS_total, S_survival, S_death)
    V_ded_deadend, _ = value_iteration_masked(P_deadend, nS, nA, SA_mask.values, gamma=1.0, theta=1e-10)
    Q_ded_deadend = V2Q(P_deadend, V_ded_deadend, nA, nS, SA_mask.values, gamma=1.0, mode='ded')
    # pi_ded_deadend = ded_deadend(Q_ded_deadend, nS, nA, SA_mask.values, threshold=0.5)
    # avg_size_ded_deadend = np.mean(np.sum(pi_ded_deadend, axis=1))
    # print(f"DeD-Deadend average policy size: {avg_size_ded_deadend}")


    zeta_values = np.arange(0.01, 1.00, 0.01)
    death_thresholds = np.arange(0.01, 1.00, 0.01)

    svp_sizes = np.zeros(len(zeta_values))
    ded_sizes = np.zeros(len(death_thresholds))
    pi_svp_list = np.zeros((len(zeta_values), nS, nA))
    pi_ded_list = np.zeros((len(death_thresholds), nS, nA))

    conflict_fractions = np.zeros((len(zeta_values), len(death_thresholds)))
    ious = np.zeros((len(zeta_values), len(death_thresholds)))

    for i, zeta in enumerate(zeta_values):
        V_svp, pi_svp, is_max_iter, iter = svp_masked(P_svp, V_star, nS, nA, SA_mask.values, gamma=1.0, zeta=zeta, theta=1e-10)
        avg_size_svp = np.mean(np.sum(pi_svp, axis=1))
        print(f'Zeta: {zeta}, SVP iters: {iter}, Max iter reached: {is_max_iter}, SVP size: {avg_size_svp}')
        svp_sizes[i] = avg_size_svp
        pi_svp_list[i] = pi_svp

    for j, death_threshold in enumerate(death_thresholds):
        pi_ded_deadend = ded_deadend(Q_ded_deadend, nS, nA, SA_mask.values, threshold=death_threshold)
        avg_size_ded_deadend = np.mean(np.sum(pi_ded_deadend, axis=1))
        ded_sizes[j] = avg_size_ded_deadend
        pi_ded_list[j] = pi_ded_deadend

    np.save('results/svp_policies.npy', pi_svp_list)
    np.save('results/ded_policies.npy', pi_ded_list)
    np.save('results/svp_sizes.npy', svp_sizes)
    np.save('results/ded_sizes.npy', ded_sizes)

    for i, zeta in enumerate(zeta_values):
        for j, death_threshold in enumerate(death_thresholds):
            avg_size_svp = svp_sizes[i]
            avg_size_ded_deadend = ded_sizes[j]
            pi_svp = pi_svp_list[i]
            pi_ded_deadend = pi_ded_list[j]

            conflict_fraction = compute_conflict_fraction(pi_svp, pi_ded_deadend, SA_mask.values, nS, nA)
            iou = compute_iou(pi_svp, pi_ded_deadend, SA_mask.values, nS, use_0_for_empty=True)
            conflict_fractions[i, j] = conflict_fraction
            ious[i, j] = iou

            # print(f"zeta: {zeta:.2f}, death_threshold: {death_threshold:.2f}, SVP size: {avg_size_svp:.2f}, DeD-Deadend size: {avg_size_ded_deadend:.2f}, Conflict fraction: {conflict_fraction:.4f}, Avg IoU: {iou:.4f}")

    np.save('results/conflict_fractions.npy', conflict_fractions)
    np.save('results/ious.npy', ious)


    visualize_svp_and_bad_sizes(
        zeta_vals=zeta_values,
        dt_vals=death_thresholds,
        svp_map=svp_sizes.reshape(1, -1).repeat(len(death_thresholds), axis=0),
        bad_map=ded_sizes.reshape(-1, 1).repeat(len(zeta_values), axis=1),
        save_path="visualizations/svp_vs_ded_sizes.pdf"
    )
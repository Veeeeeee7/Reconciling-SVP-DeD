import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def svp_masked(P, V_star, nS, nA, SA_mask, gamma, zeta, theta=1e-10, max_iter=1000, min_iter=10):
    is_max_iter = False
    V = V_star.copy().astype(float)
    policies = []
    svp_policy = None
    n_iter = 0

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
            break

        if n_iter >= max_iter:
            svp_policy = pi
            is_max_iter = True
            iter = n_iter
            break

        policies.append(pi)


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

def compute_iou(pi1, pi2, SA_mask, nS, use_0_for_empty=False):
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

def compute_top_k_conflict_states(pi1, pi2, nS, nA, k):
    conflicts = np.zeros(nS)
    for s in range(nS):
        for a in range(nA):
            if pi1[s, a] and pi2[s, a]:
                conflicts[s] += 1
    top_k_states = np.argsort(conflicts)[-k:][::-1]
    return top_k_states

def compute_conflict_set_for_state(pi1, pi2, state, nA):
    conflict_actions = []
    for a in range(nA):
        if pi1[state, a] and pi2[state, a]:
            conflict_actions.append(a)
    return conflict_actions

if __name__ == "__main__":
    nS = 750
    nA = 25
    S_survival = 750
    S_death = 751
    nS_total = nS + 2

    # Load the data
    data_dir = 'mimic_sepsis_data_2025/'
    train_df = pd.read_csv(data_dir + 'traj_shifted_train.csv', dtype={"a:action": "Int64", 'a:next_action': "Int64"})
    val_df = pd.read_csv(data_dir + 'traj_shifted_val.csv', dtype={"a:action": "Int64", 'a:next_action': "Int64"})
    test_df = pd.read_csv(data_dir + 'traj_shifted_test.csv', dtype={"a:action": "Int64", 'a:next_action': "Int64"})

    # Create behavior policy and transition matrix
    train_π_b, train_SA_mask, train_SA_count = make_policy(train_df)
    train_P = make_transition_matrix(train_df, nA, nS_total, S_survival, S_death)
    val_π_b, val_SA_mask, val_SA_count = make_policy(val_df)
    val_P = make_transition_matrix(val_df, nA, nS_total, S_survival, S_death)

    trajectories = train_df['traj'].unique()
    patient_total = len(trajectories)
    death_total = 0
    state_action_deaths = set()
    state_action_total = nS * nA - np.sum(train_SA_mask.values == False)
    for traj_id in trajectories:
        traj_data = train_df[train_df['traj'] == traj_id]
        last_row = traj_data.iloc[-1]
        last_state = last_row['s:state']
        last_action = last_row['a:action']
        end_state = last_row['s:next_state']
        if end_state == S_death:
            state_action_deaths.add((last_state, last_action))
            death_total += 1
    state_action_deaths = len(state_action_deaths)
    death_rate_per_patient = death_total / patient_total

    print(f'Death rate per state in training data: {death_total}/{patient_total} = {death_rate_per_patient:.4f}')
    # print(f'Death rate per state, action in training data: {state_action_deaths}/{state_action_total} = {state_action_deaths/state_action_total:.4f}')


    # SVP
    zeta = 0.2
    print(f"Choose zeta={zeta} for SVP and death rate threshold={death_rate_per_patient:.4f} for DeD")
    train_R_svp = np.zeros((nS_total, nA, nS_total))
    train_R_svp[:, :, S_survival] = 1
    train_R_svp[:, :, S_death] = -1
    train_P_svp = make_gymP(train_P, train_R_svp, nS, nA, nS_total, S_survival, S_death)
    train_V_star, train_π_star = value_iteration_masked(train_P_svp, nS, nA, train_SA_mask.values, gamma=1.0, theta=1e-10)
    train_V_star_svp, train_π_svp, is_max_iter, iter = svp_masked(train_P_svp, train_V_star, nS, nA, train_SA_mask.values, gamma=1.0, zeta=zeta, theta=1e-10)
    train_size_svp = np.mean(np.sum(train_π_svp, axis=1))
    print(f"SVP average policy size: {train_size_svp}")

    zeta2 = 0.8
    print(f'Second zeta={zeta2}')
    train_V_star_svp2, train_π_svp2, is_max_iter2, iter2 = svp_masked(train_P_svp, train_V_star, nS, nA, train_SA_mask.values, gamma=1.0, zeta=zeta2, theta=1e-10)
    train_size_svp2 = np.mean(np.sum(train_π_svp2, axis=1))
    print(f"SVP (zeta={zeta2}) average policy size: {train_size_svp2}")

    # DeD
    train_R_ded = np.zeros((nS_total, nA, nS_total))
    train_R_ded[:, :, S_death] = -1
    train_P_ded = make_gymP(train_P, train_R_ded, nS, nA, nS_total, S_survival, S_death)
    train_V_star_ded, train_π_star_ded = value_iteration_masked(train_P_ded, nS, nA, train_SA_mask.values, gamma=1.0, theta=1e-10)
    train_Q_ded = V2Q(train_P_ded, train_V_star_ded, nA, nS, train_SA_mask.values, 1.0, mode='ded')
    train_π_ded = ded_deadend(train_Q_ded, nS, nA, train_SA_mask.values, threshold=death_rate_per_patient)
    train_size_ded = np.mean(np.sum(train_π_ded, axis=1))
    print(f"DeD average policy size: {train_size_ded}")

    # Find indices in train_π_svp that conflict with train_π_ded
    conflict_states_svp_ded = []
    for s in range(nS):
        for a in range(nA):
            if train_π_svp[s, a] and train_π_ded[s, a]:
                conflict_states_svp_ded.append([s,a])
                break


    print(f"\nConflicting states between SVP (zeta={zeta}) and DeD: {len(conflict_states_svp_ded)} states")
    print(f"Conflict state indices: {conflict_states_svp_ded}")

    # Find indices in train_π_svp2 that conflict with train_π_ded
    conflict_states_svp2_ded = []
    for s in range(nS):
        for a in range(nA):
            if train_π_svp2[s, a] and train_π_ded[s, a]:
                conflict_states_svp2_ded.append([s,a])
                break


    print(f"\nConflicting states between SVP (zeta={zeta2}) and DeD: {len(conflict_states_svp2_ded)} states")
    print(f"Conflict state indices: {conflict_states_svp2_ded}")

    # sorting by DeD V-values for visualization
    sort_indices = np.argsort(train_V_star_ded)
    train_Q_ded_ordered = train_Q_ded[sort_indices]
    train_π_svp_ordered = train_π_svp[sort_indices]
    train_π_svp2_ordered = train_π_svp2[sort_indices]

    # Find ordered indices for conflict states in train_π_svp2_ordered
    conflict_indices_svp2 = []
    for s, a in conflict_states_svp2_ded:
        ordered_idx = np.where(sort_indices == s)[0]
        if len(ordered_idx) > 0:
            conflict_indices_svp2.append(ordered_idx[0])

    print(f"\nOrdered indices for SVP2 conflict states: {conflict_indices_svp2}")

    hist_list = [i in  conflict_indices_svp2 for i in range(nS)]

    def plot_true_hist_by_bucket(bool_list, bucket_size=50, title=None):
        b = np.asarray(bool_list, dtype=bool)
        n = b.size
        if n == 0:
            raise ValueError("bool_list is empty")

        # number of buckets (ceil)
        nb = (n + bucket_size - 1) // bucket_size

        # count Trues per bucket
        counts = np.array([b[i*bucket_size : (i+1)*bucket_size].sum() for i in range(nb)])

        # x positions = bucket index (0,1,2,...)
        x = np.arange(nb)

        plt.figure()
        plt.bar(x, counts)
        plt.xlabel(f"Bucket index (size={bucket_size})")
        plt.ylabel("# True values in bucket")
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()
    plot_true_hist_by_bucket(hist_list, bucket_size=50)

    conflict_fraction = compute_conflict_fraction(train_π_svp, train_π_ded, train_SA_mask.values, nS, nA)
    iou = compute_iou(train_π_svp, train_π_ded, train_SA_mask.values, nS, use_0_for_empty=True)
    print(f"Conflict fraction between SVP and DeD: {conflict_fraction:.14f}")
    print(f"Mean IOU between SVP and DeD: {iou:.14f}")

    conflict_fraction_2 = compute_conflict_fraction(train_π_svp2, train_π_ded, train_SA_mask.values, nS, nA)
    iou_2 = compute_iou(train_π_svp2, train_π_ded, train_SA_mask.values, nS, use_0_for_empty=True)
    print(f"Conflict fraction between SVP (zeta={zeta2}) and DeD: {conflict_fraction_2:.14f}")
    print(f"Mean IOU between SVP (zeta={zeta2}) and DeD: {iou_2:.14f}")


    additional_indices = np.where(sort_indices == 546)[0].tolist() + np.where(sort_indices == 47)[0].tolist()
    points = [
        (211, 23), (323, 19), (444, 24), (490, 14), (531, 5), (540, 14), (728, 24), (177, 13), (52, 19)
    ]
    red_circle_points = []
    for point in points:
        orig_state, action = point
        ordered_idx = np.where(sort_indices == orig_state)[0]
        if len(ordered_idx) > 0:
            red_circle_points.append((ordered_idx[0], action))

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # --- settings ---
    N_MAIN_STATES = 16
    THRESH = 0.0973


    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    seen = set()
    additional_indices = [i for i in additional_indices if not (i in seen or seen.add(i))]

    # --- build combined row index list: main (top) + separator row + additional (bottom) ---
    total_states = train_Q_ded_ordered.shape[0]

    main_rows = list(range(min(N_MAIN_STATES, total_states)))
    add_rows = [i for i in additional_indices if 0 <= i < total_states and i not in main_rows]

    SEP = None  # sentinel for the "..." row
    row_sel = main_rows + ([SEP] if len(add_rows) > 0 else []) + add_rows

    # map from original row index -> plotted y coordinate
    orig_to_y = {}
    y = 0
    for r in row_sel:
        if r is SEP:
            y += 1
            continue
        orig_to_y[r] = y
        y += 1

    # --- construct plotting arrays with a NaN separator row (so it draws as blank) ---
    nA = train_Q_ded_ordered.shape[1]
    Q_rows, Pi1_rows, Pi2_rows, yticklabels = [], [], [], []

    for r in row_sel:
        if r is SEP:
            Q_rows.append(np.full((nA,), np.nan))
            Pi1_rows.append(np.zeros((nA,), dtype=int))
            Pi2_rows.append(np.zeros((nA,), dtype=int))
            yticklabels.append("...")
        else:
            Q_rows.append(train_Q_ded_ordered[r, :])
            Pi1_rows.append(train_π_svp_ordered[r, :])
            Pi2_rows.append(train_π_svp2_ordered[r, :])
            yticklabels.append(f"State {sort_indices[r]}")

    Q   = np.vstack(Q_rows)
    Pi1 = np.vstack(Pi1_rows)
    Pi2 = np.vstack(Pi2_rows)

    if not (Q.shape == Pi1.shape == Pi2.shape):
        raise ValueError(f"Shape mismatch: Q {Q.shape}, Pi1 {Pi1.shape}, Pi2 {Pi2.shape}. Expected all equal.")

    nS, nA = Q.shape

    # --- figure/axes with room for colorbar on the right ---
    fig, ax = plt.subplots(figsize=(14, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.25)

    # --- heatmap of DeD Q-values ---
    cmap = plt.get_cmap("Reds_r").copy()
    cmap.set_bad(color="white")  # NaN separator row becomes white
    im = ax.imshow(Q, aspect="auto", interpolation="nearest", cmap=cmap)

    # --- axes ticks/labels (all tick labels fontsize 20) ---
    ax.set_xticks(np.arange(nA))
    ax.set_xticklabels([f"Action {j}" for j in range(nA)], rotation=45, ha="right", fontsize=20)
    ax.set_yticks(np.arange(nS))
    ax.set_yticklabels(yticklabels, fontsize=20)

    # --- colorbar ---
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"DeD $Q$-values", fontsize=32)
    cbar.ax.tick_params(labelsize=20)

    # --- 1) DeD crosses: red where Q < -THRESH ---
    ys_ded, xs_ded = np.where(Q < -THRESH)
    ax.scatter(xs_ded, ys_ded, marker="x", s=175, linewidths=2, c="red")

    # --- 2) SVP1: dark blue circles where Pi1 == 1 ---
    ys1, xs1 = np.where(Pi1 == 1)
    ax.scatter(xs1, ys1, marker="o", s=120, facecolors="none", edgecolors="blue", linewidths=2)

    # --- 3) SVP2: light blue bigger circles where Pi2 == 1 ---
    light_blue = "#6FA8FF"
    ys2, xs2 = np.where(Pi2 == 1)
    ax.scatter(xs2, ys2, marker="o", s=400, facecolors="none", edgecolors=light_blue, linewidths=2)

    # --- 4) Custom black circles at specific (original_state, action) points ---
    rx, ry = [], []
    for (orig_state, action) in red_circle_points:
        if orig_state in orig_to_y and 0 <= action < nA:
            ry.append(orig_to_y[orig_state])
            rx.append(action)

    if len(rx) > 0:
        ax.scatter(
            rx, ry,
            marker="o",
            s=75,
            facecolors="black",
            edgecolors="black",
            linewidths=0
        )

    # --- optional: draw a horizontal divider line through the "..." row ---
    if SEP in row_sel:
        sep_y = row_sel.index(SEP)
        ax.hlines(sep_y, -0.5, nA - 0.5, colors="black", linewidth=1.0, alpha=0.35)

    # --- keep grid cells aligned ---
    ax.set_xlim(-0.5, nA - 0.5)
    ax.set_ylim(nS - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig("mimic_state_level_plot.pdf", dpi=800, bbox_inches="tight")
    plt.show()
    plt.close(fig)
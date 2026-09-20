import numpy as np
import pickle
from lifegate import LifeGate
from svp import value_iter, value_iter_near_greedy, V2Q
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def MDP_lifegate(env, types='regular', deadend_threshold=0.7):
    P = {}
    width, height = env.scr_w, env.scr_h
    for y in range(height):
        for x in range(width):
            s = y * width + x
            P[s] = {}
            if [x, y] in env.barriers:
                continue
            for a in env.legal_actions:
                P[s][a] = []
                new_x, new_y = x, y
                reward, done = 0.0, False

                # dead-end special drag
                if [x, y] in env.dead_ends:
                    if [x + 1, y] in env.deaths:
                        done = True
                        if types in ('death','regular'): reward = -1.0
                    P[s][a].append((deadend_threshold, s + 1, reward, done))
                    P[s][a].append((1 - deadend_threshold, s, 0.0, False))
                    continue

                # terminal states
                if [new_x, new_y] in env.deaths:
                    if types in ('death','regular'): reward = -1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue
                if [new_x, new_y] in env.recoveries:
                    if types in ('recovery','regular'): reward = +1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue

                # normal move
                if a==1:   new_y -=1
                elif a==2: new_y +=1
                elif a==3: new_x -=1
                elif a==4: new_x +=1

                # bounce off walls/barriers
                if (new_x<0 or new_y<0 or new_x>=width or new_y>=height or [new_x,new_y] in env.barriers):
                    new_x, new_y = x, y

                s_next = new_y*width + new_x
                s_drag = s + 1
                reward_drag, done_drag = 0.0, False

                if [new_x,new_y] in env.deaths:
                    done = True
                    if types in ('death','regular'): reward = -1.0
                elif [new_x,new_y] in env.recoveries:
                    done = True
                    if types in ('recovery','regular'): reward = +1.0

                if [x+1,y] in env.deaths:
                    done_drag = True
                    if types in ('death','regular'): reward_drag = -1.0

                P[s][a].append((1-env.death_drag, s_next, reward, done))
                P[s][a].append((  env.death_drag,   s_drag, reward_drag, done_drag))
    return P

def bad_policies(Q_d, env_death, env, threshold):
    env_death.P = MDP_lifegate(env, types='death', deadend_threshold=threshold)
    env_death.nS = env_death.scr_w * env_death.scr_h
    env_death.nA = env_death.nb_actions
    pi_bad = np.zeros((env_death.nS, env_death.nA), dtype=int)
    for s in range(env_death.nS):
        for a in range(env_death.nA):
            if Q_d[s,a] <= -threshold:
                pi_bad[s,a] = 1
    return pi_bad

def fraction_conflict(pi_svp, pi_bad, valid_states):
    conflicts = 0
    for s in range(pi_bad.shape[0]):
        if any(pi_svp[s,a] and pi_bad[s,a] for a in range(pi_bad.shape[1])):
            conflicts += 1
    return conflicts / valid_states

def cleaned(pi, lifegate_states, dead_states, dead_ends):
    for s in range(pi.shape[0]):
        if s in lifegate_states or s in dead_states or s in dead_ends:
            pi[s,:] = 0
    return pi

def train_search_pair(
    env, env_death, gamma,
    zeta_vals, dt_vals,
    barrier_states, lifegate_states, dead_states, dead_ends,
    theta=1e-10, max_iter=1000
):
    n_dt, n_z = len(dt_vals), len(zeta_vals)
    iou_map = np.zeros(100)
    total_states  = env.scr_w * env.scr_h
    valid_states  = total_states - len(barrier_states)

    # initialize regular MDP + optimal value function for SVP
    env.P = MDP_lifegate(env, types='regular', deadend_threshold=0.7)
    env.nS, env.nA = env.scr_w * env.scr_h, env.nb_actions
    V_star, _ = value_iter(env, gamma, theta=theta)
    lst_pi_svp = []

    # compute SVP policies for each zeta
    for zeta in zeta_vals:
        V_svp, pi_svp, _, _ = value_iter_near_greedy(
            env, gamma, zeta, V_star,
            theta=theta, max_iter=max_iter)
        pi_svp = cleaned(pi_svp, lifegate_states, dead_states, dead_ends)
        lst_pi_svp.append(pi_svp)

    # initialize death MDP + optimal value function for DeD
    env_death.P = MDP_lifegate(env, types='death', deadend_threshold=0.7)
    env_death.nS, env_death.nA = env_death.scr_w * env_death.scr_h, env_death.nb_actions
    V_d, _ = value_iter(env_death, gamma, theta=theta)
    Q_d = V2Q(env_death, V_d, gamma)
    lst_pi_bad = []

    # compute DeD policies for each deadend threshold
    for dt in dt_vals:
        pi_bad = bad_policies(Q_d, env_death, env, dt)
        lst_pi_bad.append(pi_bad)


    # loop over all pairs of (dt, zeta) and compute inconsistency metrics
    for s in range(env.nS):
        for i, dt in enumerate(dt_vals):
            for j, zeta in enumerate(zeta_vals):
                # fill maps
                pi_bad = lst_pi_bad[i]
                pi_svp = lst_pi_svp[j]

                svp_actions = set([a for a in range(env.nA) if pi_svp[s,a]])
                bad_actions = set([a for a in range(env.nA) if pi_bad[s,a]])

                intersection = svp_actions.intersection(bad_actions)
                union = svp_actions.union(bad_actions)

                if len(union) > 0:
                    iou_map[s] += len(intersection) / len(union)

    iou_map /= len(dt_vals) * len(zeta_vals)
    return iou_map.reshape(10, 10)


def plot_avg_iou_lifegate(
    avg_iou_state,
    barrier_states, lifegate_states, dead_ends, dead_states,
    grid_size=10,
    title="average IOU per state (SVP vs. DeD)",
    cmap="viridis",
    save_path="figures/avg_iou_state_heatmap.pdf",
    alpha=0.75,
):
    # Times-like serif fonts + serif math
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    avg_iou_state = np.array(avg_iou_state, dtype=float, copy=True)

    # Mask out special states
    for i in range(avg_iou_state.shape[0]):
        for j in range(avg_iou_state.shape[1]):
            state = i * grid_size + j
            if state in barrier_states or state in lifegate_states or state in dead_ends or state in dead_states:
                avg_iou_state[i, j] = np.nan

    # Print the min and max of the average IOU state
    min_iou = np.nanmin(avg_iou_state)
    max_iou = np.nanmax(avg_iou_state)
    print(f"Min average IOU: {min_iou}, Max average IOU: {max_iou}")

    plt.figure(figsize=(12, 12))
    plt.imshow(avg_iou_state, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0, alpha=alpha)

    # Label each cell with its value (skip NaNs)
    for i in range(avg_iou_state.shape[0]):
        for j in range(avg_iou_state.shape[1]):
            val = avg_iou_state[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=24, color="black")

    cbar = plt.colorbar()
    cbar.set_label("Avg IOU", fontsize=48)
    cbar.ax.tick_params(labelsize=24)

    # plt.title(title, fontsize=32)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_state_frequency_heatmap(results, states_to_ignore, save_path="figures/state_conflict_frequency_heatmap.pdf"):
    """Per-state normalized conflict frequency.

    NF(s) = (1/|H|) * sum_h 1[ pi_SVP^(h)(s) INTERSECT pi_DeD^(h)(s) != empty ]

    with |H| = |zeta_vals| * |dt_vals|.  Dividing by |H| (rather than by the
    largest per-state count) makes each cell a literal probability over the
    hyperparameter grid, matching the definition of NF(s) in the paper.
    """
    # Times-like serif fonts + serif math
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    incons_tensor = np.array(results["incons_tensor"])
    state_conflict_counts = np.sum(incons_tensor, axis=(1, 2), dtype=np.float64)

    # Mask out states to ignore
    for s in states_to_ignore:
        state_conflict_counts[s] = np.nan

    # Normalize by the size of the hyperparameter grid H, NOT by the max count.
    H = len(results["dt_vals"]) * len(results["zeta_vals"])
    min_count = np.nanmin(state_conflict_counts)
    max_count = np.nanmax(state_conflict_counts)
    print(f"Min conflict count (ignoring NaNs): {min_count}, Max conflict count (ignoring NaNs): {max_count}")
    print(f"|H| = {len(results['dt_vals'])} x {len(results['zeta_vals'])} = {H}")
    print(f"NF range: min={min_count / H:.4f}, max={max_count / H:.4f}")
    normalized_conflicts = state_conflict_counts / H

    # Reshape to 10x10 grid
    grid_conflicts = normalized_conflicts.reshape((10, 10))

    plt.figure(figsize=(12, 12))
    plt.imshow(grid_conflicts, cmap="hot_r", origin="upper", vmin=0.0)

    # Label each cell with its value
    for i in range(grid_conflicts.shape[0]):
        for j in range(grid_conflicts.shape[1]):
            val = grid_conflicts[i, j]
            if np.isfinite(val):
                color = "white" if val > 0.5 else "black"
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=24, color=color)

    cbar = plt.colorbar()
    cbar.set_label("Normalized Conflict Frequency", fontsize=48)
    cbar.ax.tick_params(labelsize=24)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    rs = np.random.RandomState(1234)
    env       = LifeGate(state_mode='tabular', rng=rs, death_drag=0.4, fixed_life=True)
    env_death = LifeGate(state_mode='tabular', rng=rs, death_drag=0.4, fixed_life=True)

    barrier_states  = [0,1,2,3,4,51,52,53,54]
    lifegate_states = [5,6,7]
    dead_states     = [8,9,19,29,39,49,59,69,79,89,99]
    dead_ends       = [45,46,47,48,55,56,57,58,65,66,67,68,
                       75,76,77,78,85,86,87,88,95,96,97,98]

    # train on a fine grid, ζ, dt ∈ [0,1] step=0.01
    zeta_vals = np.arange(0.01, 1.00, 0.01)
    dt_vals   = np.arange(0.01, 1.00, 0.01)


    iou_map = np.load('results/iou_map.npy')


    plot_avg_iou_lifegate(
        iou_map,
        barrier_states, lifegate_states, dead_ends, dead_states,
        grid_size=10,
        cmap="Reds",
        save_path="figures/state_level_iou_heatmap.pdf",
        alpha=1,
    )

    # NOTE: results/trained_policies.pkl is byte-identical to
    # results/trained_policies06.pkl (drag = 0.6).  The drag-0.4 sweep -- the
    # configuration this script's env is built with, and the one behind the main
    # policy-level heatmap -- lives in results/trained_policies04.pkl.
    with open("results/trained_policies04.pkl", "rb") as f:
        results = pickle.load(f)

    plot_state_frequency_heatmap(results, barrier_states + lifegate_states + dead_states + dead_ends, save_path='figures/state_conflict_frequency_heatmap.pdf')
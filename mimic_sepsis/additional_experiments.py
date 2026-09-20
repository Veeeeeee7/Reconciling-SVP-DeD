"""
mortality_evaluation.py

Evaluates two conflict resolution strategies across a full (zeta, theta) grid:
    - SVP-priority:    at conflicting states, always keep SVP's recommended
                       action set (ignore DeD's warning).
    - Random-priority: at conflicting states, flip a fair coin — heads follows
                       SVP (keep conflicting actions), tails follows DeD (remove
                       them). Each conflicting state resolves independently.

At non-conflicting states both strategies use the same action set and produce
identical mortality estimates — the difference comes entirely from how conflicts
are resolved.

For each starting state the policy takes one step, lands in a next state, and
reads off the empirical mortality rate of that state (fraction of real training
trajectories that visited the state and eventually died).  These are averaged
across all starting states.

Usage
-----
    python mortality_evaluation.py
    python mortality_evaluation.py --use-saved-data

Outputs (all written to results/)
-------
    mortality_svp_priority.npy     shape (n_zeta, n_theta)
    mortality_random_priority.npy  shape (n_zeta, n_theta)
    mortality_results.csv
    figures/mortality_heatmap_svp.pdf
    figures/mortality_heatmap_random.pdf
    figures/mortality_diff_heatmap.pdf  (random - svp; positive means SVP is better)
    figures/mortality_vs_zeta.pdf
    figures/mortality_vs_theta.pdf
"""

import os
import itertools
import multiprocessing as mp
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants (must match consistency.py / experiments.py)
# ---------------------------------------------------------------------------
nS       = 750
nA       = 25
nS_term  = 2
S_survival = 750
S_death    = 751
nS_total   = nS + nS_term

DATA_DIR    = "mimic_sepsis_data_2025/"
RESULTS_DIR = "results/"
FIGURES_DIR = "figures/"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_FILE = "log.txt"

def log(message=""):
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")
    print(message)

# ---------------------------------------------------------------------------
# Shared helper functions (mirrors consistency.py exactly so the policies
# produced here are bit-for-bit identical)
# ---------------------------------------------------------------------------

def make_policy(df_data):
    SA_count = (
        df_data.groupby(["s:state", "a:action"])
        .size()
        .unstack()
        .reindex(index=range(nS), columns=range(nA))
        .fillna(0)
    )
    pi_b    = SA_count.div(SA_count.sum(axis=1), axis=0)
    SA_mask = SA_count >= 6
    for s in range(nS):
        if SA_mask.loc[s].sum() == 0:
            SA_mask.loc[s, SA_count.loc[s].argmax()] = True
    return pi_b, SA_mask, SA_count


def make_transition_matrix(df_data):
    SAS_count = (
        df_data.groupby(["s:state", "a:action", "s:next_state"])
        .size()
        .reset_index(name="count")
    )
    P = np.full((nS_total, nA, nS_total), np.nan)
    for _, row in SAS_count.iterrows():
        P[int(row["s:state"]), int(row["a:action"]), int(row["s:next_state"])] = row["count"]
    P = P / np.nansum(P, axis=2, keepdims=True)
    P[S_survival, :, :] = 0;  P[S_survival, :, S_survival] = 1
    P[S_death,    :, :] = 0;  P[S_death,    :, S_death]    = 1
    return P


def make_gymP(P, R):
    gymP = defaultdict(lambda: defaultdict(list))
    for s in range(nS):
        for a in range(nA):
            for ns in range(nS_total):
                if not np.isnan(P[s, a, ns]):
                    done = int(ns in (S_survival, S_death))
                    gymP[s][a].append((P[s, a, ns], ns, R[s, a, ns], done))
    return gymP


def value_iteration_masked(gymP, SA_mask, gamma=1.0, theta=1e-10):
    V = np.zeros(nS)
    for _ in tqdm(itertools.count(), desc="  Value iteration", leave=False):
        V_new = V.copy()
        for s in range(nS):
            Q_s = np.array([
                sum(p * (r + (0 if done else gamma * V[ns]))
                    for p, ns, r, done in gymP[s][a])
                for a in range(nA)
            ], dtype=float)
            Q_s[~SA_mask[s]] = np.nan
            V_new[s] = np.nanmax(Q_s)
        if np.isclose(np.linalg.norm(V_new - V), theta):
            break
        V = V_new

    # greedy policy from converged V
    pi = np.zeros((nS, nA))
    for s in range(nS):
        Q_s = np.array([
            sum(p * (r + (0 if done else gamma * V[ns]))
                for p, ns, r, done in gymP[s][a])
            for a in range(nA)
        ], dtype=float)
        Q_s[~SA_mask[s]] = np.nan
        if not np.all(np.isnan(Q_s)):
            pi[s, np.nanargmax(Q_s)] = 1.0
    return V, pi


def V2Q(gymP, V, SA_mask, gamma=1.0, mode="svp"):
    """Compute Q from a value function for the given mode."""
    Q = np.full((nS, nA), np.nan)
    for s in range(nS):
        for a in range(nA):
            if not SA_mask[s, a]:
                continue
            val = 0.0
            for p, ns, r, done in gymP[s][a]:
                if mode == "svp":
                    bootstrap = 1.0 if ns == S_survival else (-1.0 if ns == S_death else V[ns])
                else:  # ded
                    bootstrap = 0.0 if ns == S_survival else (-1.0 if ns == S_death else V[ns])
                val += p * (r + (0.0 if done else gamma * bootstrap))
            Q[s, a] = val
    return Q


def svp_masked(gymP_svp, V_star, SA_mask, zeta, gamma=1.0, theta=1e-10, max_iter=1000, min_iter=10):
    """Reproduce the SVP algorithm from consistency.py."""
    V        = V_star.copy().astype(float)
    policies = []
    n_iter   = 0

    def _cutoff(s):
        return (1 - zeta) * V_star[s] if V_star[s] >= 0 else V_star[s] - zeta * abs(V_star[s])

    def _build_pi(V_cur):
        Q  = V2Q(gymP_svp, V_cur, SA_mask, gamma=gamma, mode="svp")
        pi = np.zeros((nS, nA))
        for s in range(nS):
            pi[s] = (Q[s] >= _cutoff(s)) & SA_mask[s]
        return pi, Q

    pi, _ = _build_pi(V)
    policies.append(pi.copy())

    while True:
        delta = 0.0
        for s in range(nS):
            old_v = V[s]
            Q_s   = np.array([
                sum(p * (r + (0 if done else gamma * (
                    1.0 if ns == S_survival else (-1.0 if ns == S_death else V[ns])
                ))) for p, ns, r, done in gymP_svp[s][a])
                for a in range(nA)
            ], dtype=float)
            Q_s[~SA_mask[s]] = np.nan
            Pi_s = np.argwhere((Q_s >= _cutoff(s)) & SA_mask[s])
            V[s] = Q_s[Pi_s].min() if len(Pi_s) > 0 else np.nanmax(Q_s)
            delta = max(delta, abs(V[s] - old_v))

        pi, _ = _build_pi(V)
        n_iter += 1

        if n_iter < min_iter:
            policies.append(pi.copy())
            continue

        if (policies[-1] == pi).all() and delta < theta:
            svp_policy = pi
            break

        # cycle detection
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
                break
        if is_cycle:
            break

        if n_iter >= max_iter:
            svp_policy = pi
            break

        policies.append(pi.copy())

    # greedy fallback: ensure every state has at least one recommended action
    _, opt_pi = value_iteration_masked(gymP_svp, SA_mask, gamma=gamma)
    for s in range(nS):
        if not any(svp_policy[s, a] and SA_mask[s, a] for a in range(nA)):
            svp_policy[s] = opt_pi[s]

    return V, svp_policy


def ded_policy(Q_d, SA_mask, threshold):
    """Returns a boolean mask of ELIMINATED (risky) actions."""
    pi = np.zeros((nS, nA), dtype=bool)
    for s in range(nS):
        for a in range(nA):
            if SA_mask[s, a] and Q_d[s, a] <= -threshold:
                pi[s, a] = True
    return pi


# ---------------------------------------------------------------------------
# Combined policy builders
# ---------------------------------------------------------------------------

def build_combined_policy_prioritize_svp(pi_svp, pi_ded_bad, SA_mask, opt_pi):
    """
    Strategy: trust SVP.  The final recommendation set for state s is
        pi_svp(s)          (SVP keeps conflicting actions)
    If pi_svp is empty for some state, fall back to the greedy optimal action.
    """
    combined = pi_svp.copy().astype(float)
    # Greedy fallback (should already be handled inside svp_masked, but be safe)
    for s in range(nS):
        if not any(combined[s, a] and SA_mask[s, a] for a in range(nA)):
            combined[s] = opt_pi[s]
    return combined


def build_combined_policy_prioritize_ded(pi_svp, pi_ded_bad, SA_mask, opt_pi):
    """
    Strategy: trust DeD.  Remove any action that DeD flags as risky, even if
    SVP recommends it.
        final(s) = pi_svp(s) \ pi_ded_bad(s)
    If the result is empty, fall back to the greedy optimal action (we cannot
    leave the clinician with zero choices).
    """
    combined = pi_svp.copy().astype(float)
    combined[pi_ded_bad] = 0.0  # remove DeD-flagged actions

    for s in range(nS):
        if not any(combined[s, a] and SA_mask[s, a] for a in range(nA)):
            combined[s] = opt_pi[s]
    return combined


# ---------------------------------------------------------------------------
# Mortality simulation (one-step rollout)
# Note: the actual evaluation is done inside _worker using empirical_mortality.
# This function is kept for reference.
# ---------------------------------------------------------------------------

def simulate_mortality_one_step(combined_policy, SA_mask, start_states,
                                 P_global_arr, V_ded_arr, rng):
    """
    For each starting state:
      - At non-conflicting states: the combined policy already reflects the
        resolution (SVP-priority or DeD-priority), so sample uniformly from
        its recommended action set.
      - At conflicting states: same — the combined policy has already resolved
        the conflict, so we sample from the resulting action set.

    After sampling an action, draw the next state from the empirical transition
    matrix and read off its mortality rate from -V_ded[next_state].

    Non-conflicting states where SVP and DeD agree contribute equally to both
    strategies.  Only conflicting states differ between the two policies, which
    is where the mortality difference comes from.

    Skips (s, a) pairs with no observed transitions in the training data.

    Returns
    -------
    mean_mortality : float
    """
    policy_actions = []
    for s in range(nS):
        acts = [a for a in range(nA)
                if combined_policy[s, a] > 0 and SA_mask[s, a]]
        if not acts:
            acts = [int(np.argmax(SA_mask[s]))]
        policy_actions.append(acts)

    mortality_rates = []
    for s in start_states:
        action = rng.choice(policy_actions[s])

        row   = P_global_arr[s, action]
        total = np.nansum(row)
        if total <= 0:
            continue   # unseen (s, a) — skip

        probs      = np.nan_to_num(row, nan=0.0) / total
        next_state = int(rng.choice(nS_total, p=probs))

        if next_state == S_death:
            mortality_rates.append(1.0)
        elif next_state == S_survival:
            mortality_rates.append(0.0)
        else:
            mortality_rates.append(float(-V_ded_arr[next_state]))

    if not mortality_rates:
        return np.nan
    return float(np.mean(mortality_rates))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_heatmap(matrix, zeta_vals, theta_vals, title, save_path, cmap="RdYlGn_r"):
    """
    matrix has shape (n_zeta, n_theta): axis 0 = zeta (x), axis 1 = theta (y).
    Transpose so imshow gets (n_theta, n_zeta): rows = theta, cols = zeta.
    origin='lower' puts (zeta=min, theta=min) at bottom-left.
    Lower mortality is better — RdYlGn_r makes low values green, high values red.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(
        matrix.T,                   # (n_theta, n_zeta): rows=theta on y, cols=zeta on x
        extent=[zeta_vals[0], zeta_vals[-1], theta_vals[0], theta_vals[-1]],
        origin="lower",             # (zeta=min, theta=min) at bottom-left
        aspect="auto",
        cmap=cmap,
        vmin=np.nanmin(matrix),
        vmax=np.nanmax(matrix),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mortality Rate (lower is better)", fontsize=32)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel(r"$\zeta$", fontsize=32)
    ax.set_ylabel(r"$\theta_D$", fontsize=32)
    ax.tick_params(labelsize=20)
    ax.axvline(x=0.2,    color="blue", linestyle="--", linewidth=1.5)
    ax.axhline(y=0.0973, color="blue", linestyle="--", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {save_path}")


def plot_diff_heatmap(diff, zeta_vals, theta_vals, save_path):
    """
    diff = mortality_ded_priority - mortality_svp_priority.
    Positive = DeD-priority has higher mortality = SVP-priority is safer.
    Negative = SVP-priority has higher mortality = DeD-priority is safer.

    matrix has shape (n_zeta, n_theta): transpose once inside for correct orientation.
    Colormap RdBu_r: red = positive (SVP safer), blue = negative (DeD safer).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    vabs = np.nanmax(np.abs(diff))
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(
        diff.T,                     # (n_theta, n_zeta): rows=theta on y, cols=zeta on x
        extent=[zeta_vals[0], zeta_vals[-1], theta_vals[0], theta_vals[-1]],
        origin="lower",             # (zeta=min, theta=min) at bottom-left
        aspect="auto",
        cmap="RdBu_r",             # red = SVP safer, blue = DeD safer
        vmin=-vabs,
        vmax=vabs,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        r"Mortality(Random Priority) $-$ Mortality(SVP Priority)"
        "\nRed = SVP safer  |  Blue = Random safer",
        fontsize=24,
    )
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel(r"$\zeta$", fontsize=32)
    ax.set_ylabel(r"$\theta_D$", fontsize=32)
    ax.tick_params(labelsize=20)
    ax.axvline(x=0.2,    color="black", linestyle="--", linewidth=1.5)
    ax.axhline(y=0.0973, color="black", linestyle="--", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _worker(args):
    """
    Evaluate one (i, j) hyperparameter pair using a one-step rollout.

    At every starting state:
      - If no conflict between SVP and DeD: both strategies use the same
        action set (SVP recommendations minus DeD exclusions) and produce
        identical mortality estimates.
      - If conflict:
          - SVP-priority:    always keep SVP's recommended set (sample from {2, 5}).
          - Random-priority: flip a fair coin — heads keeps SVP's set (sample from
                             {2, 5}), tails removes DeD-flagged actions (sample from
                             {2}). Each conflicting state resolves independently.

    After sampling an action, draw the next state from the empirical transition
    matrix and read off empirical_mortality[next_state].  Average across all
    starting states.

    Parameters
    ----------
    args : tuple
        (i, j, pi_svp, pi_ded_bad, SA_mask_arr, opt_pi, start_states,
         P_global_arr, empirical_mortality_arr, seed)

    Returns
    -------
    (i, j, mort_svp, mort_random)
    """
    (i, j, pi_svp, pi_ded_bad, SA_mask_arr, opt_pi,
     start_states, P_global_arr, empirical_mortality_arr, seed) = args

    rng = np.random.default_rng(seed)

    def _resolve_actions(s, use_svp_priority, rng_local):
        svp_acts = [a for a in range(nA)
                    if pi_svp[s, a] > 0 and SA_mask_arr[s, a]]
        ded_bad  = [a for a in range(nA)
                    if pi_ded_bad[s, a] and SA_mask_arr[s, a]]

        conflict = any(a in ded_bad for a in svp_acts)

        if not conflict:
            # Both strategies agree — use SVP set (DeD has nothing to exclude here)
            return svp_acts if svp_acts else [int(np.argmax(SA_mask_arr[s]))]

        if use_svp_priority:
            # Always keep SVP's set (includes conflicting actions)
            return svp_acts if svp_acts else [int(np.argmax(SA_mask_arr[s]))]
        else:
            # Flip a coin: heads = prioritize SVP (keep conflicting actions),
            #              tails = prioritize DeD (remove conflicting actions)
            if rng_local.random() < 0.5:
                # Coin landed SVP: sample from full SVP set
                return svp_acts if svp_acts else [int(np.argmax(SA_mask_arr[s]))]
            else:
                # Coin landed DeD: remove DeD-flagged actions from SVP set
                safe = [a for a in svp_acts if a not in ded_bad]
                return safe if safe else [int(np.argmax(SA_mask_arr[s]))]

    def _simulate(use_svp_priority):
        mortality_rates = []
        for s in start_states:
            acts   = _resolve_actions(s, use_svp_priority, rng)
            action = rng.choice(acts)

            row   = P_global_arr[s, action]
            total = np.nansum(row)
            if total <= 0:
                continue   # unseen (s, a) — skip

            probs      = np.nan_to_num(row, nan=0.0) / total
            next_state = int(rng.choice(nS_total, p=probs))

            if next_state == S_death:
                mortality_rates.append(1.0)
            elif next_state == S_survival:
                mortality_rates.append(0.0)
            else:
                mort = empirical_mortality_arr[next_state]
                if not np.isnan(mort):
                    mortality_rates.append(float(mort))

        if not mortality_rates:
            return np.nan
        return float(np.mean(mortality_rates))

    mort_svp    = _simulate(use_svp_priority=True)
    mort_random = _simulate(use_svp_priority=False)

    return i, j, mort_svp, mort_random


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-saved-data",
        action="store_true",
        help="Skip all computation and load mortality arrays from results/ to regenerate figures.",
    )
    args = parser.parse_args()

    # Clear log file
    with open(LOG_FILE, "w") as f:
        f.write("")

    zeta_values      = np.round(np.arange(0.01, 1.00, 0.01), 4)
    death_thresholds = np.round(np.arange(0.01, 1.00, 0.01), 4)
    n_zeta           = len(zeta_values)
    n_theta          = len(death_thresholds)

    if args.use_saved_data:
        # ------------------------------------------------------------------
        # Fast path: load saved results and skip all computation
        # ------------------------------------------------------------------
        log("--use-saved-data flag set: loading results from disk …")

        svp_path    = RESULTS_DIR + "mortality_svp_priority.npy"
        random_path = RESULTS_DIR + "mortality_random_priority.npy"

        if not os.path.exists(svp_path) or not os.path.exists(random_path):
            raise FileNotFoundError(
                f"Could not find saved results at {svp_path} and {random_path}. "
                "Run without --use-saved-data first."
            )

        mortality_svp_priority    = np.load(svp_path)
        mortality_random_priority = np.load(random_path)
        log(f"  Loaded mortality_svp_priority.npy    shape={mortality_svp_priority.shape}")
        log(f"  Loaded mortality_random_priority.npy shape={mortality_random_priority.shape}")

    else:
        # ------------------------------------------------------------------
        # Full computation path
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # 1.  Load data
        # ------------------------------------------------------------------
        log("Loading data …")
        df_train = pd.read_csv(
            DATA_DIR + "traj_shifted_train.csv",
            dtype={"a:action": "Int64", "a:next_action": "Int64"},
        )
        log(f"  Train: {len(df_train)} rows")

        _, SA_mask, _ = make_policy(df_train)
        SA_mask_arr   = SA_mask.values

        P_raw    = make_transition_matrix(df_train)
        P_global = P_raw
        log("  Transition matrix built from training data.")

        # ------------------------------------------------------------------
        # 1b. Empirical mortality rate per state
        # ------------------------------------------------------------------
        log("\nComputing empirical mortality rate per state …")

        if "icustayid" in df_train.columns:
            traj_col = "icustayid"
        elif "bloc" in df_train.columns:
            df_train["_traj_id"] = (df_train["bloc"] == 1).cumsum()
            traj_col = "_traj_id"
        else:
            df_train["_traj_id"] = df_train.index
            traj_col = "_traj_id"

        traj_died = (
            df_train.groupby(traj_col)["s:next_state"]
            .apply(lambda x: int((x == S_death).any()))
        )
        df_train["_died"] = df_train[traj_col].map(traj_died)

        # For each state s, find trajectories whose FIRST state is s,
        # then compute the fraction of those that eventually died.
        # This gives the mortality rate of starting a trajectory from state s.
        first_rows = df_train.sort_values("bloc").groupby(traj_col).first().reset_index() \
            if "bloc" in df_train.columns \
            else df_train.groupby(traj_col).first().reset_index()

        empirical_mortality = np.full(nS, np.nan)
        for s in range(nS):
            starting_here = first_rows[first_rows["s:state"] == s]
            if len(starting_here) == 0:
                continue
            empirical_mortality[s] = starting_here["_died"].mean()

        n_covered = int(np.sum(~np.isnan(empirical_mortality)))
        log(f"  Empirical mortality computed for {n_covered}/{nS} states "
            f"(based on trajectories starting at each state).")

        # ------------------------------------------------------------------
        # 2.  Value iteration for SVP and DeD
        # ------------------------------------------------------------------
        log("\nBuilding reward matrices and gymP …")

        R_svp                   = np.zeros((nS_total, nA, nS_total))
        R_svp[:, :, S_survival] =  1.0
        R_svp[:, :, S_death]    = -1.0
        gymP_svp                = make_gymP(P_raw, R_svp)

        R_ded                   = np.zeros((nS_total, nA, nS_total))
        R_ded[:, :, S_death]    = -1.0
        gymP_ded                = make_gymP(P_raw, R_ded)

        log("Running value iteration for SVP (V*) …")
        V_star, opt_pi = value_iteration_masked(gymP_svp, SA_mask_arr)

        log("Running value iteration for DeD …")
        V_ded, _  = value_iteration_masked(gymP_ded, SA_mask_arr)
        Q_ded     = V2Q(gymP_ded, V_ded, SA_mask_arr, gamma=1.0, mode="ded")

        # ------------------------------------------------------------------
        # 3.  Cache SVP and DeD policies across the grid
        # ------------------------------------------------------------------
        SVP_CACHE_PATH = RESULTS_DIR + "pi_svp_grid.npy"
        DED_CACHE_PATH = RESULTS_DIR + "pi_ded_grid.npy"

        if os.path.exists(SVP_CACHE_PATH):
            log(f"\nLoading cached SVP policies from {SVP_CACHE_PATH} …")
            pi_svp_arr = np.load(SVP_CACHE_PATH)
        else:
            log("\nCaching SVP policies across zeta grid …")
            pi_svp_arr = np.zeros((n_zeta, nS, nA))
            for i, zeta in enumerate(tqdm(zeta_values, desc="SVP grid")):
                _, pi_svp = svp_masked(gymP_svp, V_star, SA_mask_arr, zeta=zeta)
                pi_svp_arr[i] = pi_svp
            np.save(SVP_CACHE_PATH, pi_svp_arr)
            log(f"  Saved SVP grid to {SVP_CACHE_PATH}")

        if os.path.exists(DED_CACHE_PATH):
            log(f"Loading cached DeD policies from {DED_CACHE_PATH} …")
            pi_ded_arr = np.load(DED_CACHE_PATH)
        else:
            log("\nCaching DeD policies across theta grid …")
            pi_ded_arr = np.zeros((n_theta, nS, nA), dtype=bool)
            for j, theta in enumerate(death_thresholds):
                pi_ded_arr[j] = ded_policy(Q_ded, SA_mask_arr, threshold=float(theta))
            np.save(DED_CACHE_PATH, pi_ded_arr)
            log(f"  Saved DeD grid to {DED_CACHE_PATH}")

        # ------------------------------------------------------------------
        # 4.  Collect starting states
        # ------------------------------------------------------------------
        grouped = df_train.groupby(traj_col)
        start_states = []
        for pid, traj in grouped:
            traj = traj.sort_values("bloc") if "bloc" in df_train.columns else traj
            s0 = int(traj.iloc[0]["s:state"])
            if s0 < nS:
                start_states.append(s0)
        start_states = list(start_states)
        log(f"\n  Using {len(start_states)} starting states (full training set) …")

        # ------------------------------------------------------------------
        # 5.  Evaluate mortality across (zeta, theta) grid  (parallelised)
        # ------------------------------------------------------------------
        log("\nEvaluating mortality across hyperparameter grid …")

        mortality_svp_priority    = np.full((n_zeta, n_theta), np.nan)
        mortality_random_priority = np.full((n_zeta, n_theta), np.nan)

        tasks = [
            (i, j,
             pi_svp_arr[i],
             pi_ded_arr[j],
             SA_mask_arr,
             opt_pi,
             start_states,
             P_global,
             empirical_mortality,
             i * n_theta + j)
            for i in range(n_zeta)
            for j in range(n_theta)
        ]

        n_cpus = max(1, mp.cpu_count() - 1)
        log(f"  Using {n_cpus} worker processes for {len(tasks)} tasks …")

        with mp.Pool(processes=n_cpus) as pool:
            for i, j, mort_svp, mort_random in tqdm(
                pool.imap_unordered(_worker, tasks),
                total=len(tasks),
                desc="Grid search",
            ):
                mortality_svp_priority[i, j]    = mort_svp
                mortality_random_priority[i, j] = mort_random

        # ------------------------------------------------------------------
        # 6.  Save raw arrays
        # ------------------------------------------------------------------
        np.save(RESULTS_DIR + "mortality_svp_priority.npy",    mortality_svp_priority)
        np.save(RESULTS_DIR + "mortality_random_priority.npy", mortality_random_priority)
        log(f"\nSaved mortality arrays to {RESULTS_DIR}")

        # ------------------------------------------------------------------
        # 7.  Save CSV summary
        # ------------------------------------------------------------------
        rows = []
        for i, zeta in enumerate(zeta_values):
            for j, theta in enumerate(death_thresholds):
                rows.append({
                    "zeta":                    zeta,
                    "theta_D":                 theta,
                    "mortality_svp_priority":    mortality_svp_priority[i, j],
                    "mortality_random_priority": mortality_random_priority[i, j],
                    "diff_random_minus_svp":     mortality_random_priority[i, j] - mortality_svp_priority[i, j],
                })
        df_results = pd.DataFrame(rows)
        df_results.to_csv(RESULTS_DIR + "mortality_results.csv", index=False)
        log(f"Saved CSV to {RESULTS_DIR}mortality_results.csv")

    # ------------------------------------------------------------------
    # 8.  Summary statistics  (always runs)
    # ------------------------------------------------------------------
    diff = mortality_random_priority - mortality_svp_priority  # positive => SVP is better

    log("\n=== Summary ===")
    log(f"  Mortality range (SVP-priority):    {np.nanmin(mortality_svp_priority):.4f} – {np.nanmax(mortality_svp_priority):.4f}")
    log(f"  Mortality range (Random-priority): {np.nanmin(mortality_random_priority):.4f} – {np.nanmax(mortality_random_priority):.4f}")
    log(f"  Mean mortality (SVP-priority):     {np.nanmean(mortality_svp_priority):.4f}")
    log(f"  Mean mortality (Random-priority):  {np.nanmean(mortality_random_priority):.4f}")
    log(f"  Fraction of (ζ,θ) pairs where SVP-priority has lower mortality: {np.nanmean(diff > 0):.3f}")
    log(f"  Fraction of (ζ,θ) pairs where Random-priority has lower mortality: {np.nanmean(diff < 0):.3f}")
    log(f"  Fraction where they are equal: {np.nanmean(diff == 0):.3f}")

    focal_zeta  = 0.20
    focal_theta = 0.10
    i_focal = int(np.argmin(np.abs(zeta_values  - focal_zeta)))
    j_focal = int(np.argmin(np.abs(death_thresholds - focal_theta)))
    log(f"\n  Focal point (ζ={zeta_values[i_focal]:.2f}, θ_D={death_thresholds[j_focal]:.2f}):")
    log(f"    Mortality (SVP-priority):    {mortality_svp_priority[i_focal, j_focal]:.4f}")
    log(f"    Mortality (Random-priority): {mortality_random_priority[i_focal, j_focal]:.4f}")

    # ------------------------------------------------------------------
    # 9.  Plots  (always runs)
    # ------------------------------------------------------------------
    log("\nGenerating figures …")

    plot_heatmap(
        mortality_svp_priority,
        zeta_values, death_thresholds,
        title="Mortality Rate — SVP Priority at Conflicts",
        save_path=FIGURES_DIR + "mortality_heatmap_svp.pdf",
    )
    plot_heatmap(
        mortality_random_priority,
        zeta_values, death_thresholds,
        title="Mortality Rate — Random Priority at Conflicts",
        save_path=FIGURES_DIR + "mortality_heatmap_random.pdf",
    )
    plot_diff_heatmap(
        diff,
        zeta_values, death_thresholds,
        save_path=FIGURES_DIR + "mortality_diff_heatmap.pdf",
    )

    # Line plot: mortality vs zeta, averaged over theta
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(zeta_values, np.nanmean(mortality_svp_priority,    axis=1),
            color="steelblue", lw=2, label="SVP Priority at Conflicts")
    ax.plot(zeta_values, np.nanmean(mortality_random_priority, axis=1),
            color="tomato", lw=2, linestyle="--", label="Random Priority at Conflicts")
    ax.fill_between(zeta_values,
                    np.nanmin(mortality_svp_priority, axis=1),
                    np.nanmax(mortality_svp_priority, axis=1),
                    alpha=0.15, color="steelblue")
    ax.fill_between(zeta_values,
                    np.nanmin(mortality_random_priority, axis=1),
                    np.nanmax(mortality_random_priority, axis=1),
                    alpha=0.15, color="tomato")
    ax.axvline(x=0.2, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xlabel(r"$\zeta$ (SVP near-optimality margin)", fontsize=32)
    ax.set_ylabel("Mortality Rate (lower is better)", fontsize=32)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "mortality_vs_zeta.pdf", dpi=500, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIGURES_DIR}mortality_vs_zeta.pdf")

    # Line plot: mortality vs theta, averaged over zeta
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(death_thresholds, np.nanmean(mortality_svp_priority,    axis=0),
            color="steelblue", lw=2, label="SVP Priority at Conflicts")
    ax.plot(death_thresholds, np.nanmean(mortality_random_priority, axis=0),
            color="tomato", lw=2, linestyle="--", label="Random Priority at Conflicts")
    ax.fill_between(death_thresholds,
                    np.nanmin(mortality_svp_priority, axis=0),
                    np.nanmax(mortality_svp_priority, axis=0),
                    alpha=0.15, color="steelblue")
    ax.fill_between(death_thresholds,
                    np.nanmin(mortality_random_priority, axis=0),
                    np.nanmax(mortality_random_priority, axis=0),
                    alpha=0.15, color="tomato")
    ax.axvline(x=0.0973, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xlabel(r"$\theta_D$ (DeD death threshold)", fontsize=32)
    ax.set_ylabel("Mortality Rate (lower is better)", fontsize=32)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "mortality_vs_theta.pdf", dpi=500, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIGURES_DIR}mortality_vs_theta.pdf")

    log("\n=== Average Mortality Across All Hyperparameters ===")
    log(f"  Average mortality (SVP-priority):    {np.nanmean(mortality_svp_priority):.4f}")
    log(f"  Average mortality (Random-priority): {np.nanmean(mortality_random_priority):.4f}")

    log("\nDone.")
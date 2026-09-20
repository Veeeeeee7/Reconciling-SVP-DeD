"""
Bootstrap resampling for MIMIC-III figures 7, 8 (and conflict-fraction curve used in Fig 5).

Strategy:
  - Resample patient trajectories (by icustayid) with replacement.
  - For each bootstrap replicate, rebuild the transition matrix, run value
    iteration for SVP and DeD, then compute:
      * svp_sizes[zeta]            -> Fig 7 (left y-axis)
      * ded_sizes[theta]           -> Fig 7 (right y-axis)
      * conflict_fractions[z, th]  -> Fig 8 heatmap
      * per-state conflict freq    -> Fig 5 (state-level)
  - After N_BOOT replicates, take the 2.5th and 97.5th percentiles as the 95% CI
    and the median as the central estimate used in the figures.

  --use-saved-boots flag: skip rerunning replicates and load existing
    bootstrap arrays from results/bootstrap/, recomputing only the CI
    at the new 95% level.

Usage:
    python bootstrap_mimic.py
    python bootstrap_mimic.py --use-saved-boots
"""

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.lines import Line2D
from joblib import Parallel, delayed

# ── constants ────────────────────────────────────────────────────────────────
nS       = 750
nA       = 25
nS_term  = 2
S_surv   = 750
S_death  = 751
nS_total = nS + nS_term
GAMMA    = 1.0
THETA    = 1e-10
MAX_ITER = 1000
N_BOOT   = 100        # number of bootstrap replicates
CI_LO    = 2.5        # lower percentile  -> 95% CI
CI_HI    = 97.5       # upper percentile
SEED     = 42

DATA_DIR    = 'mimic_sepsis_data_2025/'
RESULTS_DIR = 'results/bootstrap/'
FIGURES_DIR = 'figures/bootstrap/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

zeta_values      = np.arange(0.01, 1.00, 0.01)
death_thresholds = np.arange(0.01, 1.00, 0.01)
n_zeta  = len(zeta_values)
n_theta = len(death_thresholds)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ── helper functions (mirror consistency.py) ──────────────────────────────────

def make_policy_from_df(df):
    SA_count = (df.groupby(['s:state', 'a:action']).size()
                  .unstack()
                  .reindex(index=range(nS), columns=range(nA))
                  .fillna(0))
    SA_mask = (SA_count >= 6).values  # bool ndarray (nS, nA)
    for s in range(nS):
        if not SA_mask[s].any():
            SA_mask[s, int(SA_count.iloc[s].argmax())] = True
    return SA_mask


def make_transition_matrix_from_df(df):
    SAS = (df.groupby(['s:state', 'a:action', 's:next_state'])
             .size().reset_index(name='count'))
    P = np.full((nS_total, nA, nS_total), np.nan)
    for _, row in SAS.iterrows():
        P[int(row['s:state']), int(row['a:action']), int(row['s:next_state'])] = row['count']
    P = P / np.nansum(P, axis=2, keepdims=True)
    for st in (S_surv, S_death):
        P[st, :, :] = 0.0
        P[st, :, st] = 1.0
    return P


def make_gymP(P, R):
    gymP = defaultdict(lambda: defaultdict(list))
    for s in range(nS):
        for a in range(nA):
            for ns in range(nS_total):
                if not np.isnan(P[s, a, ns]):
                    done = int(ns in (S_surv, S_death))
                    gymP[s][a].append((P[s, a, ns], ns, R[s, a, ns], done))
    return gymP


def value_iteration_masked(gymP, SA_mask):
    V = np.zeros(nS)
    while True:
        V_new = V.copy()
        for s in range(nS):
            Q_s = np.full(nA, np.nan)
            for a in range(nA):
                if SA_mask[s, a] and gymP[s][a]:
                    Q_s[a] = sum(p * (r + (0 if done else GAMMA * V[ns]))
                                 for p, ns, r, done in gymP[s][a])
            v = np.nanmax(Q_s)
            V_new[s] = v if np.isfinite(v) else 0.0
        if np.linalg.norm(V_new - V) < THETA:
            break
        V = V_new
    return V


def V2Q(gymP, V, SA_mask, mode='svp'):
    Q = np.full((nS, nA), np.nan)
    for s in range(nS):
        for a in range(nA):
            if not SA_mask[s, a]:
                continue
            q = 0.0
            for p, ns, r, done in gymP[s][a]:
                if mode == 'svp':
                    v_ns = 1.0 if ns == S_surv else (-1.0 if ns == S_death else V[ns])
                else:  # ded
                    v_ns = 0.0 if ns == S_surv else (-1.0 if ns == S_death else V[ns])
                q += p * (r + GAMMA * v_ns * (1 - done))
            Q[s, a] = q
    return Q


def svp_masked(gymP_svp, V_star, SA_mask, zeta):
    V = V_star.copy().astype(float)
    policies = []
    n_iter = 0

    def _make_pi(Q, Vs):
        pi = np.zeros((nS, nA), dtype=float)
        for s in range(nS):
            thr = (1 - zeta) * Vs[s] if Vs[s] >= 0 else Vs[s] - zeta * abs(Vs[s])
            pi[s] = (Q[s] >= thr) & SA_mask[s]
        return pi

    Q = V2Q(gymP_svp, V, SA_mask, mode='svp')
    policies.append(_make_pi(Q, V_star))

    while True:
        for s in range(nS):
            Q_s = np.full(nA, np.nan)
            for a in range(nA):
                if SA_mask[s, a] and gymP_svp[s][a]:
                    v_ns_sum = 0.0
                    for p, ns, r, done in gymP_svp[s][a]:
                        v_ns = 1.0 if ns == S_surv else (-1.0 if ns == S_death else V[ns])
                        v_ns_sum += p * (r + GAMMA * v_ns * (1 - done))
                    Q_s[a] = v_ns_sum
            Q_s[~SA_mask[s]] = np.nan
            thr = ((1 - zeta) * V_star[s] if V_star[s] >= 0
                   else V_star[s] - zeta * abs(V_star[s]))
            valid = np.where((Q_s >= thr) & SA_mask[s])[0]
            V[s] = Q_s[valid].min() if len(valid) > 0 else np.nanmax(Q_s)

        Q = V2Q(gymP_svp, V, SA_mask, mode='svp')
        pi = _make_pi(Q, V_star)
        n_iter += 1

        if (policies[-1] == pi).all():
            svp_policy = pi
            break

        for i, past in enumerate(policies):
            if (past == pi).all():
                cycle = policies[i:]
                core = cycle[0] > 0
                for p2 in cycle[1:]:
                    core &= (p2 > 0)
                svp_policy = core.astype(float)
                return svp_policy

        if n_iter >= MAX_ITER:
            svp_policy = pi
            break
        policies.append(pi)

    # greedy fallback for empty states
    _, pi_greedy = _greedy_policy(gymP_svp, V_star, SA_mask)
    for s in range(nS):
        if not svp_policy[s].any():
            svp_policy[s] = pi_greedy[s]
    return svp_policy


def _greedy_policy(gymP, V, SA_mask):
    pi = np.zeros((nS, nA))
    for s in range(nS):
        Q_s = np.full(nA, np.nan)
        for a in range(nA):
            if SA_mask[s, a] and gymP[s][a]:
                Q_s[a] = sum(p * (r + (0 if done else GAMMA * V[ns]))
                             for p, ns, r, done in gymP[s][a])
        if np.any(np.isfinite(Q_s)):
            pi[s, np.nanargmax(Q_s)] = 1.0
    return V, pi


def ded_policy(Q_d, SA_mask, threshold):
    pi = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            if SA_mask[s, a] and Q_d[s, a] <= -threshold:
                pi[s, a] = 1.0
    return pi


def compute_conflict_fraction(pi_svp, pi_ded, SA_mask):
    conflicts = []
    for s in range(nS):
        if not SA_mask[s].any():
            continue
        conflicts.append(float(np.any((pi_svp[s] > 0) & (pi_ded[s] > 0))))
    return np.mean(conflicts)


def compute_state_conflict(pi_svp, pi_ded, SA_mask):
    """Returns binary conflict indicator per state."""
    out = np.zeros(nS)
    for s in range(nS):
        if SA_mask[s].any():
            out[s] = float(np.any((pi_svp[s] > 0) & (pi_ded[s] > 0)))
    return out


# ── one bootstrap replicate ───────────────────────────────────────────────────

def run_one_replicate(df_full, patient_ids, rng_seed):
    rng = np.random.RandomState(rng_seed)
    sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
    df_boot = df_full[df_full['traj'].isin(sampled_ids)].copy()

    SA_mask = make_policy_from_df(df_boot)
    P       = make_transition_matrix_from_df(df_boot)

    R_svp = np.zeros((nS_total, nA, nS_total))
    R_svp[:, :, S_surv]  =  1.0
    R_svp[:, :, S_death] = -1.0
    gymP_svp = make_gymP(P, R_svp)

    R_ded = np.zeros((nS_total, nA, nS_total))
    R_ded[:, :, S_death] = -1.0
    gymP_ded = make_gymP(P, R_ded)

    V_star   = value_iteration_masked(gymP_svp, SA_mask)
    V_ded    = value_iteration_masked(gymP_ded, SA_mask)
    Q_ded    = V2Q(gymP_ded, V_ded, SA_mask, mode='ded')

    svp_sizes_rep = np.zeros(n_zeta)
    ded_sizes_rep = np.zeros(n_theta)
    cf_rep        = np.zeros((n_zeta, n_theta))
    sc_rep        = np.zeros((n_zeta, n_theta, nS))

    pi_svp_list = []
    for i, zeta in enumerate(zeta_values):
        pi_svp = svp_masked(gymP_svp, V_star, SA_mask, zeta)
        svp_sizes_rep[i] = pi_svp.sum(axis=1).mean()
        pi_svp_list.append(pi_svp)

    pi_ded_list = []
    for j, thr in enumerate(death_thresholds):
        pi_ded = ded_policy(Q_ded, SA_mask, thr)
        ded_sizes_rep[j] = pi_ded.sum(axis=1).mean()
        pi_ded_list.append(pi_ded)

    for i in range(n_zeta):
        for j in range(n_theta):
            cf_rep[i, j] = compute_conflict_fraction(pi_svp_list[i], pi_ded_list[j], SA_mask)
            sc_rep[i, j] = compute_state_conflict(pi_svp_list[i], pi_ded_list[j], SA_mask)

    return svp_sizes_rep, ded_sizes_rep, cf_rep, sc_rep


# ── CI computation and figures ────────────────────────────────────────────────

def compute_ci_and_plot(svp_boots, ded_boots, cf_boots, sc_boots):
    def ci(arr, axis=0):
        med = np.median(arr, axis=axis)
        lo  = np.percentile(arr, CI_LO,  axis=axis)
        hi  = np.percentile(arr, CI_HI, axis=axis)
        return med, lo, hi

    svp_med, svp_lo, svp_hi = ci(svp_boots)
    ded_med, ded_lo, ded_hi = ci(ded_boots)
    cf_med,  cf_lo,  cf_hi  = ci(cf_boots)

    sc_mean_boots = sc_boots.mean(axis=(1, 2))   # (N_BOOT, nS)
    sc_med, sc_lo, sc_hi = ci(sc_mean_boots)

    np.savez(RESULTS_DIR + 'summaries_95ci.npz',
             svp_med=svp_med, svp_lo=svp_lo, svp_hi=svp_hi,
             ded_med=ded_med, ded_lo=ded_lo, ded_hi=ded_hi,
             cf_med=cf_med,   cf_lo=cf_lo,   cf_hi=cf_hi,
             sc_med=sc_med,   sc_lo=sc_lo,   sc_hi=sc_hi)
    print("Saved 95% CI summaries.")

    ci_label = '95% CI'

    # ── Figure 7: SVP and DeD sizes with CI bands ─────────────────────────────
    fig, ax1 = plt.subplots(figsize=(16, 12))
    ax2 = ax1.twinx()

    ax1.plot(zeta_values, svp_med, color='green', linewidth=2)
    ax1.fill_between(zeta_values, svp_lo, svp_hi, color='green', alpha=0.2)
    ax1.set_xlabel(r'Hyperparameter Value ($\zeta$ for SVP, $\theta_D$ for DeD)',
                   fontsize=32)
    ax1.set_ylabel('Avg SVP Recommended Actions per State', color='green', fontsize=32)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelcolor='green', labelsize=20)

    ax2.plot(death_thresholds, ded_med, color='red', linewidth=2)
    ax2.fill_between(death_thresholds, ded_lo, ded_hi, color='red', alpha=0.2)
    ax2.set_ylabel('Avg DeD Eliminated Actions per State', color='red', fontsize=32)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=20)

    handles = [
        Line2D([0], [0], color='green', lw=2, label=r'SVP size vs. $\zeta$'),
        Line2D([0], [0], color='red',   lw=2, label=r'DeD size vs. $\theta_D$'),
        matplotlib.patches.Patch(color='gray', alpha=0.3, label=ci_label),
    ]
    ax1.legend(handles=handles, loc='best', fontsize=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + 'fig7_sizes_95ci.pdf', dpi=800, bbox_inches='tight')
    plt.close()
    print("Saved Figure 7 (95% CI).")

    # ── Figure 8: conflict fraction heatmap (median) ──────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(cf_med.T, extent=[0, 1, 0, 1], origin='lower',
                   aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=cf_boots.max())
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Conflict Fraction', fontsize=32)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel(r'$\zeta$', fontsize=32)
    ax.set_ylabel(r'$\theta_D$', fontsize=32)
    ax.tick_params(labelsize=20)
    ax.axvline(x=0.2,    color='blue', linestyle='--')
    ax.axhline(y=0.0973, color='blue', linestyle='--')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + 'fig8_conflict_heatmap_median_95ci.pdf', dpi=800, bbox_inches='tight')
    plt.close()

    # CI width heatmap
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow((cf_hi - cf_lo).T, extent=[0, 1, 0, 1], origin='lower',
                   aspect='auto', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'{ci_label} Width (Conflict Fraction)', fontsize=32)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel(r'$\zeta$', fontsize=32)
    ax.set_ylabel(r'$\theta_D$', fontsize=32)
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + 'fig8_conflict_heatmap_ci_width_95ci.pdf', dpi=800, bbox_inches='tight')
    plt.close()
    print("Saved Figure 8 (95% CI).")

    # ── Figure 5 (state-level): bar plot with CI bounds ───────────────────────
    sort_idx = np.argsort(sc_med)[::-1][:30]
    x = np.arange(len(sort_idx))

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(x, sc_med[sort_idx], color='salmon', label='Median conflict freq')
    ax.errorbar(x,
                sc_med[sort_idx],
                yerr=[sc_med[sort_idx] - sc_lo[sort_idx],
                      sc_hi[sort_idx] - sc_med[sort_idx]],
                fmt='none', color='black', capsize=4, linewidth=1.5,
                label=ci_label)
    ax.set_xticks(x)
    ax.set_xticklabels([f'State {sort_idx[i]}' for i in range(len(sort_idx))],
                       rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Avg Normalized Conflict Frequency', fontsize=20)
    ax.set_xlabel('State', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + 'fig5_state_conflict_95ci.pdf', dpi=800, bbox_inches='tight')
    plt.close()
    print("Saved Figure 5 (state-level, 95% CI).")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use-saved-boots',
        action='store_true',
        help='Load existing bootstrap arrays from results/bootstrap/ and '
             'recompute CI at 95%% level without rerunning replicates.',
    )
    args = parser.parse_args()

    if args.use_saved_boots:
        print('--use-saved-boots flag set: loading existing bootstrap arrays …')
        svp_boots = np.load(RESULTS_DIR + 'svp_sizes_boots.npy')
        ded_boots = np.load(RESULTS_DIR + 'ded_sizes_boots.npy')
        cf_boots  = np.load(RESULTS_DIR + 'conflict_fractions_boots.npy')
        sc_boots  = np.load(RESULTS_DIR + 'state_conflict_freq_boots.npy')
        print(f'  Loaded {svp_boots.shape[0]} replicates from disk.')
    else:
        print("Loading data …")
        df = pd.read_csv(DATA_DIR + 'traj_shifted_train.csv',
                         dtype={"a:action": "Int64", 'a:next_action': "Int64"})
        patient_ids = df['traj'].unique()
        print(f"  {len(df)} rows, {len(patient_ids)} patients")

        seeds = np.random.RandomState(SEED).randint(0, 2**31, size=N_BOOT)

        print(f"Running {N_BOOT} bootstrap replicates …")
        results = Parallel(n_jobs=-1, verbose=5)(
            delayed(run_one_replicate)(df, patient_ids, int(s)) for s in seeds
        )

        svp_boots = np.stack([r[0] for r in results])
        ded_boots = np.stack([r[1] for r in results])
        cf_boots  = np.stack([r[2] for r in results])
        sc_boots  = np.stack([r[3] for r in results])

        np.save(RESULTS_DIR + 'svp_sizes_boots.npy',           svp_boots)
        np.save(RESULTS_DIR + 'ded_sizes_boots.npy',           ded_boots)
        np.save(RESULTS_DIR + 'conflict_fractions_boots.npy',  cf_boots)
        np.save(RESULTS_DIR + 'state_conflict_freq_boots.npy', sc_boots)
        print("Saved bootstrap arrays.")

    compute_ci_and_plot(svp_boots, ded_boots, cf_boots, sc_boots)
    print("\nDone. All outputs in", RESULTS_DIR, "and", FIGURES_DIR)


if __name__ == '__main__':
    main()
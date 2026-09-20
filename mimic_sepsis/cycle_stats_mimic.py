"""
Cycle statistics for SVP's near-greedy value iteration on MIMIC-III.

How often does SVP's near-greedy value iteration cycle instead of converging on
the MIMIC-III sepsis MDP?

For every zeta in {0.01, ..., 0.99} we run consistency.svp_masked and record how
the iteration terminated ('converged', 'cycle', 'max_iter'). The cycle
resolution itself (intersection of the policies in the cycle, then greedy
fallback for states left with an empty set) is unchanged -- this script only
reads the termination status that consistency.py now reports.

Two solver variants are swept, because the repo contains both:
  min_iter=0   -- consistency.py, the solver behind every reported result
  min_iter=10  -- experiments.py (Fig. 5), which skips the convergence/cycle
                  checks for 10 sweeps and does not store those policies

Writes results/cycle_stats_mimic.csv and appends to results/cycle_stats_mimic.log
"""
import os
import sys
import time
import contextlib
import multiprocessing as mp
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import consistency as C

nS, nA = 750, 25
S_SURVIVAL, S_DEATH = 750, 751
nS_TOTAL = nS + 2
DATA = os.path.join(HERE, 'mimic_sepsis_data_2025', 'traj_shifted_train.csv')
RESULTS_DIR = os.path.join(HERE, 'results')

# consistency.make_policy reads these as module-level globals
C.nS, C.nA = nS, nA

_G = {}   # inherited by forked workers; avoids pickling the gym transition dict


def setup():
    df = pd.read_csv(DATA, dtype={"a:action": "Int64", 'a:next_action': "Int64"})
    pi_b, SA_mask, SA_count = C.make_policy(df)
    P = C.make_transition_matrix(df, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)
    R_svp = np.zeros((nS_TOTAL, nA, nS_TOTAL))
    R_svp[:, :, S_SURVIVAL] = 1
    R_svp[:, :, S_DEATH] = -1
    gymP = C.make_gymP(P, R_svp, nS, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn), \
            contextlib.redirect_stderr(dn):
        V_star, pi_star = C.value_iteration_masked(gymP, nS, nA, SA_mask.values,
                                                   gamma=1.0, theta=1e-10)
    return gymP, SA_mask.values, V_star, pi_star, len(df)


def _one(args):
    zeta, min_iter = args
    t0 = time.time()
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn), \
            contextlib.redirect_stderr(dn):
        _, pi_svp, _, _, st = C.svp_masked(
            _G['gymP'], _G['V_star'], nS, nA, _G['SA_mask'], gamma=1.0,
            zeta=zeta, theta=1e-10, max_iter=1000, min_iter=min_iter,
            return_status=True, optimal_policies=_G['pi_star'])
    st['mean_svp_size'] = float(pi_svp.sum(axis=1).mean())
    st['zeta'] = round(float(zeta), 2)
    st['seconds'] = round(time.time() - t0, 2)
    return st


COLS = ['min_iter', 'zeta', 'terminated_by', 'n_sweeps', 'cycle_start',
        'cycle_len', 'n_empty_before_fallback', 'mean_svp_size', 'seconds']
CSV_PATH = os.path.join(RESULTS_DIR, 'cycle_stats_mimic.csv')


def _load_done():
    if not os.path.exists(CSV_PATH):
        return []
    out = []
    with open(CSV_PATH) as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if line:
                out.append(dict(zip(COLS, line.split(','))))
    return out


def main():
    # Resumable: each call solves as many (min_iter, zeta) pairs as fit in
    # --budget seconds, appends them to results/cycle_stats_mimic.csv, and
    # exits. Re-run until it reports 0 remaining.
    t0 = time.time()
    variants = [int(x) for x in (os.environ.get('VARIANTS', '0,10')).split(',')]
    budget = float(os.environ.get('BUDGET', '1e9'))

    zeta_vals = np.arange(0.01, 1.00, 0.01)
    done = {(int(r['min_iter']), r['zeta']) for r in _load_done()}
    jobs = [(z, mi) for mi in variants for z in zeta_vals
            if (mi, str(round(float(z), 2))) not in done]
    if not jobs:
        print("0 remaining")
    else:
        gymP, SA_mask, V_star, pi_star, n_rows = setup()
        _G.update(gymP=gymP, SA_mask=SA_mask, V_star=V_star, pi_star=pi_star)
        print(f"setup done ({n_rows} training rows) in {time.time() - t0:.1f}s; "
              f"{len(jobs)} pairs left")

        n_proc = max(1, min(os.cpu_count() or 1, 8))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        new_file = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, 'a') as f, mp.Pool(n_proc) as pool:
            if new_file:
                f.write(','.join(COLS) + '\n')
            n_new = 0
            for r in pool.imap_unordered(_one, jobs):
                f.write(','.join(str(r[c]) for c in COLS) + '\n')
                f.flush()
                n_new += 1
                if time.time() - t0 > budget:
                    pool.terminate()
                    break
        print(f"solved {n_new} this run; "
              f"{len(jobs) - n_new} remaining ({time.time() - t0:.0f}s)")

    rows = [dict(r, min_iter=int(r['min_iter']), n_sweeps=int(r['n_sweeps']),
                 cycle_len=(None if r['cycle_len'] == 'None' else int(r['cycle_len'])))
            for r in _load_done()]
    summary = []
    for mi in sorted({r['min_iter'] for r in rows}):
        rr = [r for r in rows if r['min_iter'] == mi]
        n = len(rr)
        if n == 0:
            continue
        n_cycle = sum(r['terminated_by'] == 'cycle' for r in rr)
        n_conv = sum(r['terminated_by'] == 'converged' for r in rr)
        n_cap = sum(r['terminated_by'] == 'max_iter' for r in rr)
        summary.append(
            f"MIMIC-III (min_iter={mi}): {n_cycle}/{n} zeta values cycled "
            f"({100.0 * n_cycle / n:.1f}%), {n_conv} converged, {n_cap} hit the "
            f"1000-sweep cap; max sweeps used = {max(r['n_sweeps'] for r in rr)}; "
            f"cycle lengths {sorted({r['cycle_len'] for r in rr if r['cycle_len']})}")

    if not summary:
        return
    log_path = os.path.join(RESULTS_DIR, 'cycle_stats_mimic.log')
    with open(log_path, 'a') as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(zeta = 0.01..0.99, gamma=1, max_iter=1000, SA_mask >= 6) ===\n")
        for line in summary:
            f.write(line + '\n')
    print('\n'.join(summary))
    print(f"\nwrote {CSV_PATH}\nappended {log_path}\ntotal {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()

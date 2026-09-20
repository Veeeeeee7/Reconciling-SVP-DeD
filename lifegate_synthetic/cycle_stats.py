"""
Cycle statistics for SVP's near-greedy value iteration in LifeGate.

How often does SVP's near-greedy value iteration cycle instead of converging?

For every zeta in {0.01, ..., 0.99} we run svp.value_iter_near_greedy on the
LifeGate MDP and record how the iteration terminated ('converged', 'cycle', or
'max_iter'). The cycle resolution itself (intersection of the policies in the
cycle, then greedy fallback for empty states) is unchanged -- this script only
reads the termination status that svp.py now reports.

Usage:
    python3 cycle_stats.py                 # drag 0.4 (paper default)
    python3 cycle_stats.py 0.0 0.2 0.4 0.6 # all four drags used in Fig. 4

Writes results/cycle_stats_lifegate.csv and appends a summary to
results/cycle_stats_lifegate.log
"""
import os
import sys
import time
import numpy as np

from lifegate import LifeGate
from svp import value_iter, value_iter_near_greedy
from train_policies import MDP_lifegate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# same geometry as train_policies.py
BARRIER_STATES = [0, 1, 2, 3, 4, 51, 52, 53, 54]
LIFEGATE_STATES = [5, 6, 7]
DEAD_STATES = [8, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
DEAD_ENDS = [45, 46, 47, 48, 55, 56, 57, 58, 65, 66, 67, 68,
             75, 76, 77, 78, 85, 86, 87, 88, 95, 96, 97, 98]


def run_drag(drag, zeta_vals, gamma=1, theta=1e-10, max_iter=1000):
    rs = np.random.RandomState(1234)
    env = LifeGate(state_mode='tabular', rng=rs, death_drag=drag, fixed_life=True)
    env.P = MDP_lifegate(env, types='regular', deadend_threshold=0.7)
    env.nS, env.nA = env.scr_w * env.scr_h, env.nb_actions

    V_star, _ = value_iter(env, gamma, theta=theta)

    rows = []
    for zeta in zeta_vals:
        t0 = time.time()
        _, pi_svp, _, n_iter, status = value_iter_near_greedy(
            env, gamma, zeta, V_star, theta=theta, max_iter=max_iter,
            return_status=True)
        # mean set size over the 53 neutral states actually analyzed in the paper
        excluded = set(BARRIER_STATES) | set(LIFEGATE_STATES) | set(DEAD_STATES) | set(DEAD_ENDS)
        neutral = [s for s in range(env.nS) if s not in excluded]
        rows.append(dict(
            drag=drag, zeta=round(float(zeta), 2),
            terminated_by=status['terminated_by'], n_sweeps=status['n_sweeps'],
            cycle_start=status['cycle_start'], cycle_len=status['cycle_len'],
            n_empty_before_fallback=status['n_empty_before_fallback'],
            mean_svp_size_neutral=float(pi_svp[neutral].sum(axis=1).mean()),
            seconds=round(time.time() - t0, 2)))
    return rows


def main():
    drags = [float(x) for x in sys.argv[1:]] or [0.4]
    zeta_vals = np.arange(0.01, 1.00, 0.01)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    summary = []
    for drag in drags:
        rows = run_drag(drag, zeta_vals)
        all_rows += rows
        n = len(rows)
        n_cycle = sum(r['terminated_by'] == 'cycle' for r in rows)
        n_conv = sum(r['terminated_by'] == 'converged' for r in rows)
        n_cap = sum(r['terminated_by'] == 'max_iter' for r in rows)
        max_sweeps = max(r['n_sweeps'] for r in rows)
        cyc_zetas = [r['zeta'] for r in rows if r['terminated_by'] == 'cycle']
        summary.append(
            f"LifeGate drag={drag}: {n_cycle}/{n} zeta values cycled "
            f"({100.0 * n_cycle / n:.1f}%), {n_conv} converged, {n_cap} hit the "
            f"{1000}-sweep cap; max sweeps used = {max_sweeps}; "
            f"cycling zeta range = "
            f"{(min(cyc_zetas), max(cyc_zetas)) if cyc_zetas else 'none'}")

    header = ('drag,zeta,terminated_by,n_sweeps,cycle_start,cycle_len,'
              'n_empty_before_fallback,mean_svp_size_neutral,seconds')
    csv_path = os.path.join(RESULTS_DIR, 'cycle_stats_lifegate.csv')
    with open(csv_path, 'w') as f:
        f.write(header + '\n')
        for r in all_rows:
            f.write(','.join(str(r[k]) for k in header.split(',')) + '\n')

    log_path = os.path.join(RESULTS_DIR, 'cycle_stats_lifegate.log')
    with open(log_path, 'a') as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(zeta = 0.01..0.99, gamma=1, max_iter=1000) ===\n")
        for line in summary:
            f.write(line + '\n')
    print('\n'.join(summary))
    print(f"\nwrote {csv_path}\nappended {log_path}")


if __name__ == '__main__':
    main()

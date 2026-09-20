"""
One-step mortality of the four tiers in the reconciliation ordering.

    T1  SVP \ DeD    recommended by SVP, not flagged by DeD
    T2  SVP n DeD    the conflicting actions
    T3  neither      available, neither recommended nor flagged
    T4  DeD \ SVP    flagged by DeD, not recommended by SVP

m(s,a) = sum_{s'} P_hat(s'|s,a) * mort(s'), scoring 1 for death, 0 for discharge and
otherwise the fraction of training visits to s' whose trajectory ended in death.

The tiers are compared ONLY on a common population of states, because conflicts arise
almost exclusively in the most severe states: averaging T2 over conflicting states and
T1 over all 750 states compares patient severity, not action quality. Two populations
are reported:

    all-conflict     every state holding at least one conflicting action
    removable        conflicting states that also retain an unflagged SVP action
                     (the population where the ordering can change a recommendation)

Each state contributes once: tier means are averaged within a state across the
(zeta, theta_D) settings at which it is conflicting, then across states. 95% CIs are
cluster bootstraps over states. Pairwise tests are paired over states offering both tiers.

Usage: python3 tier_mortality.py [n_bootstrap]
Writes results/reconciliation/tier_mortality_*.csv and appends to R5_log.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conflict_action_eval as E

OUT = os.path.join(HERE, 'results', 'reconciliation')
TIERS = ['T1_svp_only', 'T2_conflict', 'T3_neither', 'T4_ded_only']
LABEL = {'T1_svp_only': 'T1  SVP \\ DeD (unflagged SVP)',
         'T2_conflict': 'T2  SVP n DeD (conflicting)',
         'T3_neither':  'T3  neither (unflagged, not recommended)',
         'T4_ded_only': 'T4  DeD \\ SVP (flagged, not recommended)'}


def tier_means(M, valid, S, D, states):
    """Mean m over each tier for the given states. Returns (n_states, 4) with nans."""
    T = [S & ~D & valid, S & D & valid, ~S & ~D & valid, ~S & D & valid]
    out = np.full((len(states), 4), np.nan)
    Mz = np.nan_to_num(M, nan=0.0)
    for k, t in enumerate(T):
        tt = t[states]
        c = tt.sum(1)
        sm = (Mz[states] * tt).sum(1)
        nz = c > 0
        out[nz, k] = sm[nz] / c[nz]
    return out


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    t0 = time.time()
    df, mask, counts, mort, support = E.load()
    keys = [(s, a) for s in range(750) for a in range(25) if mask[s, a]]
    m, _ = E.point_and_boot(support, mort, keys, 1)
    M = np.full((750, 25), np.nan)
    for (s, a), v in m.items():
        M[s, a] = v
    valid = mask & ~np.isnan(M)

    svp = np.load(os.path.join(HERE, 'results', 'svp_policies.npy')) > 0
    ded = np.load(os.path.join(HERE, 'results', 'ded_policies.npy')) > 0
    print(f"setup {time.time()-t0:.0f}s")

    acc = {'all_conflict': {}, 'removable': {}}     # state -> list of (4,) rows
    for i in range(99):
        S = svp[i] & mask
        for j in range(99):
            D = ded[j] & mask
            inter = S & D
            ni, sz = inter.sum(1), S.sum(1)
            conf = np.nonzero(ni > 0)[0]
            if conf.size == 0:
                continue
            rows = tier_means(M, valid, S, D, conf)
            rem = ni[conf] < sz[conf]
            for k, s in enumerate(conf):
                acc['all_conflict'].setdefault(int(s), []).append(rows[k])
                if rem[k]:
                    acc['removable'].setdefault(int(s), []).append(rows[k])
    print(f"grid swept {time.time()-t0:.0f}s")

    rng = np.random.default_rng(0)
    out_lines = []
    for pop, d in acc.items():
        states = sorted(d)
        X = np.array([np.nanmean(np.array(v), axis=0) for v in d.values()])  # (n_states, 4)
        pd.DataFrame(X, index=states, columns=TIERS).to_csv(
            os.path.join(OUT, f'tier_mortality_{pop}.csv'), index_label='state')
        out_lines.append(f"\npopulation: {pop}  ({len(states)} distinct states)")
        out_lines.append(f"  {'tier':44s} {'n states':>9} {'mean':>8}  {'95% CI':>18}")
        for k, t in enumerate(TIERS):
            col = X[:, k]
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                out_lines.append(f"  {LABEL[t]:44s} {0:>9}      empty")
                continue
            v = col[ok]
            bs = np.array([v[rng.integers(len(v), size=len(v))].mean() for _ in range(B)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out_lines.append(f"  {LABEL[t]:44s} {ok.sum():>9} {v.mean():>8.4f}  "
                             f"[{lo:.4f}, {hi:.4f}]")
        out_lines.append("  paired within-state comparisons:")
        for a, b in [('T1_svp_only', 'T2_conflict'), ('T2_conflict', 'T3_neither'),
                     ('T3_neither', 'T4_ded_only'), ('T1_svp_only', 'T3_neither'),
                     ('T2_conflict', 'T4_ded_only')]:
            ia, ib = TIERS.index(a), TIERS.index(b)
            ok = ~(np.isnan(X[:, ia]) | np.isnan(X[:, ib]))
            n = int(ok.sum())
            if n < 3:
                out_lines.append(f"    {a} vs {b}: n={n} (too few)")
                continue
            xa, xb = X[ok, ia], X[ok, ib]
            p = stats.wilcoxon(xa, xb).pvalue
            arrow = '<' if xa.mean() < xb.mean() else '>'
            out_lines.append(f"    {a:12s} {xa.mean():.4f} {arrow} {b:12s} {xb.mean():.4f}"
                             f"   n={n:4d}  diff {xa.mean()-xb.mean():+.4f}  "
                             f"Wilcoxon p={p:.2g}  ({100*(xa<xb).mean():.0f}% of states)")
    txt = '\n'.join(out_lines)
    print(txt)
    with open(os.path.join(OUT, 'R5_log.md'), 'a') as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} — four-tier one-step mortality "
                f"(cluster bootstrap over states, B={B}) ===\n{txt}\n")
    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()

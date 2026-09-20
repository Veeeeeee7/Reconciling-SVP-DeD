"""
Action-level comparison of conflicting actions against their SVP alternatives.

Question: when DeD flags an action that SVP recommends, and SVP still has another
recommendation that DeD does NOT flag, is the flagged action actually worse?
If the two are comparable, DeD's elimination is not supported by outcomes and our
ordering (keep the conflicting action, rank it above neutral actions) is justified.

Unit of analysis: a conflicting (state, action) pair, i.e. a state-action pair that
is in the SVP set and flagged by DeD, in a state where the SVP set retains at least
one unflagged action. These are collected over the full 99x99 (zeta, theta_D) grid
from the cached policy arrays, then deduplicated -- the same pair recurs at many
hyperparameter settings and those instances are not independent.

For each such pair we compare, within the same state:
    m(s, a_conflict)     the DeD-flagged action SVP recommends
    m(s, a_alt)          the SVP actions DeD does not flag (mean, and the best one)
where m(s,a) = sum_{s'} P_hat(s'|s,a) * mort(s') is the one-step model-based
mortality estimate: 1 if s' is death, 0 if discharge, else the fraction of training
visits to s' whose trajectory ended in death.

Because each (s,a) has only 6-203 observed transitions, confidence intervals come
from a nonparametric bootstrap that resamples each (s,a)'s observed transitions
(mort(s') is held fixed as a plug-in estimate).

"Comparable" is tested with TOST equivalence, not by a non-significant difference:
we report the smallest margin delta for which the two are equivalent at alpha=0.05.

Usage: python3 conflict_action_eval.py [n_bootstrap]
Writes results/reconciliation/conflict_actions_*.csv and appends to R5_log.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import consistency as C

nS, nA = 750, 25
S_SURVIVAL, S_DEATH = 750, 751
nS_TOTAL = nS + 2
DATA = os.path.join(HERE, 'mimic_sepsis_data_2025', 'traj_shifted_train.csv')
OUT = os.path.join(HERE, 'results', 'reconciliation')
C.nS, C.nA = nS, nA


def load():
    df = pd.read_csv(DATA, dtype={"a:action": "Int64", 'a:next_action': "Int64"})
    _, SA, SAc = C.make_policy(df)
    mask, counts = SA.values, SAc.values

    died = df.groupby('traj')['s:next_state'].apply(lambda x: int((x == S_DEATH).any()))
    df['_died'] = df['traj'].map(died)
    mort = np.full(nS_TOTAL, np.nan)
    for s, v in df.groupby('s:state')['_died'].mean().items():
        if 0 <= s < nS:
            mort[s] = v
    mort[S_DEATH] = 1.0
    mort[S_SURVIVAL] = 0.0

    # raw transition counts, kept unnormalized so the bootstrap can resample them
    sas = df.groupby(['s:state', 'a:action', 's:next_state']).size()
    support = {}
    for (s, a, s2), n in sas.items():
        if 0 <= s < nS and np.isfinite(mort[s2]):
            support.setdefault((int(s), int(a)), []).append((int(s2), int(n)))
    return df, mask, counts, mort, support


def point_and_boot(support, mort, keys, B, seed=0):
    """Exact m(s,a) plus B bootstrap replicates, resampling each pair's transitions."""
    rng = np.random.default_rng(seed)
    m = {}
    boot = {}
    for k in keys:
        sup = support.get(k)
        if not sup:
            m[k] = np.nan
            boot[k] = np.full(B, np.nan)
            continue
        s2 = np.array([x[0] for x in sup])
        n = np.array([x[1] for x in sup], float)
        p = n / n.sum()
        v = mort[s2]
        m[k] = float(p @ v)
        draws = rng.multinomial(int(n.sum()), p, size=B) / n.sum()
        boot[k] = draws @ v
    return m, boot


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t0 = time.time()
    df, mask, counts, mort, support = load()
    svp = np.load(os.path.join(HERE, 'results', 'svp_policies.npy')) > 0
    ded = np.load(os.path.join(HERE, 'results', 'ded_policies.npy')) > 0
    zetas = np.arange(0.01, 1.00, 0.01)
    thetas = np.arange(0.01, 1.00, 0.01)

    # --- collect conflicting pairs that have an unflagged SVP alternative -----
    pair_alts = {}       # (s, a_conflict) -> set of alternative actions
    pair_hits = {}       # (s, a_conflict) -> number of (zeta, theta) settings
    instances = []
    for i in range(99):
        Si = svp[i] & mask
        for j in range(99):
            Dj = ded[j] & mask
            inter = Si & Dj
            ni, sz = inter.sum(1), Si.sum(1)
            for s in np.nonzero((ni > 0) & (ni < sz))[0]:
                alts = [int(a) for a in np.nonzero(Si[s])[0] if not Dj[s, a]]
                for a in np.nonzero(inter[s])[0]:
                    k = (int(s), int(a))
                    pair_alts.setdefault(k, set()).update(alts)
                    pair_hits[k] = pair_hits.get(k, 0) + 1
                    instances.append((round(zetas[i], 2), round(thetas[j], 2), k, tuple(alts)))
    keys = set(pair_alts)
    for (st, _), alts in pair_alts.items():
        for b in alts:
            keys.add((st, b))
    print(f"{len(instances)} instances -> {len(pair_alts)} unique conflicting (s,a) pairs "
          f"across {len({k[0] for k in pair_alts})} states  [{time.time()-t0:.0f}s]")

    m, boot = point_and_boot(support, mort, sorted(keys), B)
    print(f"bootstrapped {len(keys)} (s,a) pairs x {B}  [{time.time()-t0:.0f}s]")

    # --- per-pair comparison --------------------------------------------------
    rows, d_boot = [], []
    for k in sorted(pair_alts):
        s, a = k
        alts = sorted(pair_alts[k])
        av = [(s, b) for b in alts if not np.isnan(m.get((s, b), np.nan))]
        if np.isnan(m.get(k, np.nan)) or not av:
            continue
        m_c = m[k]
        m_alt = float(np.mean([m[x] for x in av]))
        m_best = float(np.min([m[x] for x in av]))
        rows.append(dict(state=s, action=a, n_obs=int(counts[s, a]),
                         n_settings=pair_hits[k], n_alts=len(av),
                         alts='|'.join(str(x[1]) for x in av),
                         m_conflict=m_c, m_alt_mean=m_alt, m_alt_best=m_best,
                         diff_mean=m_c - m_alt, diff_best=m_c - m_best))
        d_boot.append(boot[k] - np.mean([boot[x] for x in av], axis=0))
    d = pd.DataFrame(rows)
    D = np.array(d_boot)                       # (n_pairs, B)
    d.to_csv(os.path.join(OUT, 'conflict_actions_pairs.csv'), index=False)

    diff = d.diff_mean.values
    n = len(diff)
    mean_d = diff.mean()
    rep = D.mean(axis=0)                       # bootstrap of the mean difference
    ci95 = np.percentile(rep, [2.5, 97.5])
    ci90 = np.percentile(rep, [5, 95])
    t, p_t = stats.ttest_rel(d.m_conflict, d.m_alt_mean)
    p_w = stats.wilcoxon(d.m_conflict, d.m_alt_mean).pvalue
    delta_min = max(abs(ci90[0]), abs(ci90[1]))

    out = []
    out.append(f"unique conflicting (s,a) pairs analyzed: {n}  "
               f"(median {int(d.n_obs.median())} observations each, "
               f"median {int(d.n_settings.median())} hyperparameter settings each)")
    out.append(f"mean one-step mortality: conflicting action {d.m_conflict.mean():.4f}  "
               f"vs unflagged SVP alternatives {d.m_alt_mean.mean():.4f}  "
               f"(best alternative {d.m_alt_best.mean():.4f})")
    out.append(f"mean paired difference (conflicting - alternatives): {mean_d:+.4f}  "
               f"95% CI [{ci95[0]:+.4f}, {ci95[1]:+.4f}]")
    out.append(f"paired t-test p={p_t:.3g}; Wilcoxon signed-rank p={p_w:.3g}")
    out.append(f"conflicting action LOWER than the alternatives' mean in "
               f"{int((diff < 0).sum())}/{n} pairs ({100*(diff<0).mean():.1f}%); "
               f"lower than the BEST alternative in "
               f"{int((d.diff_best.values < 0).sum())}/{n} "
               f"({100*(d.diff_best.values<0).mean():.1f}%)")
    out.append(f"TOST equivalence: 90% CI [{ci90[0]:+.4f}, {ci90[1]:+.4f}] -> equivalent at "
               f"alpha=0.05 for any margin delta > {delta_min:.4f} "
               f"({100*delta_min:.2f} percentage points of one-step mortality)")
    for dl in (0.01, 0.02, 0.05):
        ok = (ci90[0] > -dl) and (ci90[1] < dl)
        out.append(f"   margin +/-{dl:.2f}: {'EQUIVALENT' if ok else 'not established'}")
    print('\n' + '\n'.join(out))

    with open(os.path.join(OUT, 'R5_log.md'), 'a') as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} — conflicting action vs "
                f"unflagged SVP alternatives (B={B}) ===\n")
        f.write('\n'.join(out) + '\n')
    print(f"\nwrote {OUT}/conflict_actions_pairs.csv  [{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()

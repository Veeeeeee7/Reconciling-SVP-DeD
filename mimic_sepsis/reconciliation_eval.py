"""
Evaluating the reconciliation ordering across the hyperparameter grid.

One-step experiment over CONFLICTING states, comparing conflict-resolution rules:

  (i)   ours        keep every SVP-recommended action, including conflicting ones
  (ii)  always-DeD  remove DeD-flagged actions from the SVP set; if the set empties,
                    fall back to the greedy optimal action   [as drafted]
  (ii') defer-DeD   remove DeD-flagged actions; if the set empties, fall back to the
                    available action DeD considers least risky (max Q_D, i.e. lowest
                    best-case probability of inevitable death)
  (iii) random      per conflicting state, a fair coin between (i) and (ii)

Rule (ii) as drafted cannot differ from (i) at a state whose SVP set is a single
DeD-flagged action, because the greedy fallback re-inserts that same action; (ii')
is included because it is the variant that actually defers to DeD there.

One-step estimate for state s and resolved action set A(s), exact under the
empirical transition model (no sampling):
    m(s) = mean_{a in A(s)} sum_{s'} P(s'|s,a) * mort(s')
    mort(s') = 1 if death, 0 if discharge, else the empirical mortality of s'
               (fraction of training visits to s' whose trajectory ended in death)

Modes:
  focal  one (zeta, theta_D): per-state table + paired tests over conflicting states
  grid   every (zeta, theta_D) on the 99x99 grid that has at least one REMOVABLE
         conflict (a conflicting state where the rules can differ at all), using the
         cached policy grids; paired tests with the hyperparameter pair as the unit

Usage:
    python3 reconciliation_eval.py focal 0.2 0.0973
    python3 reconciliation_eval.py grid
"""
import os
import sys
import contextlib
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
OUT_DIR = os.path.join(HERE, 'results', 'reconciliation')
C.nS, C.nA = nS, nA


def empirical_mortality(df):
    # NOTE: this dataset's trajectory id is the 'traj' column. additional_experiments.py
    # looked for 'icustayid'/'bloc', found neither, and fell back to df.index -- i.e. it
    # treated every ROW as its own trajectory, so its "empirical mortality" was the
    # one-step probability of dying, not the trajectory's eventual mortality.
    for cand in ('traj', 'icustayid', '_traj_id'):
        if cand in df.columns:
            traj_col = cand
            break
    else:
        traj_col = '_traj_id'
        df['_traj_id'] = (df['bloc'] == 1).cumsum() if 'bloc' in df.columns else df.index
    died = df.groupby(traj_col)['s:next_state'].apply(lambda x: int((x == S_DEATH).any()))
    df['_died'] = df[traj_col].map(died)
    by_visits = np.full(nS, np.nan)
    visits = np.zeros(nS, dtype=int)
    g = df.groupby('s:state')['_died']
    for s, v in g.mean().items():
        if 0 <= s < nS:
            by_visits[s] = v
    for s, n in g.size().items():
        if 0 <= s < nS:
            visits[s] = n
    return by_visits, visits


def one_step_matrix(P_arr, mort):
    """M[s,a] = expected mortality one step after taking a in s (nan if (s,a) unobserved)."""
    M = np.full((nS, nA), np.nan)
    ok = ~np.isnan(mort)
    for s in range(nS):
        for a in range(nA):
            row = P_arr[s, a]
            tot = np.nansum(row)
            if tot <= 0:
                continue
            p = np.nan_to_num(row, nan=0.0) / tot
            scored = p[S_DEATH] + p[S_SURVIVAL] + p[:nS][ok].sum()
            if scored <= 0:
                continue
            M[s, a] = (p[S_DEATH] + float((p[:nS][ok] * mort[ok]).sum())) / scored
    return M


def _mean(M, s, actions):
    v = [M[s, a] for a in actions if not np.isnan(M[s, a])]
    return float(np.mean(v)) if v else np.nan


def setup():
    df = pd.read_csv(DATA, dtype={"a:action": "Int64", 'a:next_action': "Int64"})
    _, SA_mask_df, _ = C.make_policy(df)
    SA_mask = SA_mask_df.values
    P_arr = C.make_transition_matrix(df, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)
    mort, visits = empirical_mortality(df)
    M = one_step_matrix(P_arr, mort)

    R_svp = np.zeros((nS_TOTAL, nA, nS_TOTAL))
    R_svp[:, :, S_SURVIVAL] = 1
    R_svp[:, :, S_DEATH] = -1
    gymP_svp = C.make_gymP(P_arr, R_svp, nS, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)
    R_ded = np.zeros((nS_TOTAL, nA, nS_TOTAL))
    R_ded[:, :, S_DEATH] = -1
    gymP_ded = C.make_gymP(P_arr, R_ded, nS, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn), \
            contextlib.redirect_stderr(dn):
        V_star, pi_star = C.value_iteration_masked(gymP_svp, nS, nA, SA_mask, 1.0, theta=1e-10)
        V_ded, _ = C.value_iteration_masked(gymP_ded, nS, nA, SA_mask, 1.0, theta=1e-10)
    Q_d = C.V2Q(gymP_ded, V_ded, nA, nS, SA_mask, 1.0, mode='ded')
    return dict(df=df, SA_mask=SA_mask, P_arr=P_arr, mort=mort, visits=visits, M=M,
                gymP_svp=gymP_svp, V_star=V_star, pi_star=pi_star, Q_d=Q_d, len_df=len(df))


def resolve(s, svp, bad, ctx):
    """Return the action sets for rules (i), (ii), (ii'), for a conflicting state."""
    M, SA_mask, pi_star, Q_d = ctx['M'], ctx['SA_mask'], ctx['pi_star'], ctx['Q_d']
    a_i = svp
    safe = [a for a in svp if a not in bad]
    greedy = [a for a in range(nA) if pi_star[s, a] > 0 and SA_mask[s, a]]
    a_ii = safe if safe else greedy
    if safe:
        a_iialt = safe
    else:
        avail = [a for a in range(nA) if SA_mask[s, a]]
        unflagged = [a for a in avail if a not in bad]
        pool = unflagged if unflagged else avail
        best = max(pool, key=lambda a: Q_d[s, a])      # least risky by DeD's own values
        a_iialt = [best]
    return a_i, a_ii, a_iialt


def paired(name_a, a, name_b, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    d = a - b
    out = {'comparison': f'{name_a} vs {name_b}', 'n': n,
           'mean_a': float(a.mean()) if n else np.nan,
           'mean_b': float(b.mean()) if n else np.nan,
           'mean_diff': float(d.mean()) if n else np.nan,
           'n_nonzero_diff': int((d != 0).sum()) if n else 0}
    if n >= 2 and np.any(d != 0):
        t, p = stats.ttest_rel(a, b)
        out['t'] = float(t); out['p_ttest'] = float(p)
        try:
            w, pw = stats.wilcoxon(a, b)
            out['p_wilcoxon'] = float(pw)
        except Exception:
            out['p_wilcoxon'] = np.nan
        out['a_lower_in'] = int((d < 0).sum())
        out['b_lower_in'] = int((d > 0).sum())
    else:
        out['t'] = np.nan; out['p_ttest'] = np.nan; out['p_wilcoxon'] = np.nan
        out['a_lower_in'] = int((d < 0).sum()) if n else 0
        out['b_lower_in'] = int((d > 0).sum()) if n else 0
    return out


def focal(ctx, zeta, theta):
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn), \
            contextlib.redirect_stderr(dn):
        _, pi_svp, _, _, st = C.svp_masked(ctx['gymP_svp'], ctx['V_star'], nS, nA,
                                           ctx['SA_mask'], gamma=1.0, zeta=zeta,
                                           theta=1e-10, return_status=True,
                                           optimal_policies=ctx['pi_star'])
        _, pi_raw, _, _, _ = C.svp_masked(ctx['gymP_svp'], ctx['V_star'], nS, nA,
                                          ctx['SA_mask'], gamma=1.0, zeta=zeta,
                                          theta=1e-10, return_status=True,
                                          optimal_policies=np.zeros((nS, nA)))
    pi_ded = C.ded_deadend(ctx['Q_d'], nS, nA, ctx['SA_mask'], threshold=theta)
    M, SA_mask = ctx['M'], ctx['SA_mask']
    rows = []
    for s in range(nS):
        svp = [a for a in range(nA) if pi_svp[s, a] > 0 and SA_mask[s, a]]
        bad = [a for a in range(nA) if pi_ded[s, a] > 0 and SA_mask[s, a]]
        conf = [a for a in svp if a in bad]
        if not conf:
            continue
        a_i, a_ii, a_iialt = resolve(s, svp, bad, ctx)
        m_i, m_ii, m_alt = (_mean(M, s, a_i), _mean(M, s, a_ii), _mean(M, s, a_iialt))
        rows.append(dict(
            zeta=zeta, theta_D=theta, state=s, visits=int(ctx['visits'][s]),
            V_star=float(ctx['V_star'][s]), n_svp=len(svp), n_conflict=len(conf),
            conflict_actions='|'.join(map(str, conf)),
            greedy_fallback=not any(pi_raw[s, a] > 0 and SA_mask[s, a] for a in range(nA)),
            removable=0 < len(conf) < len(svp),
            set_ours='|'.join(map(str, a_i)), set_ded='|'.join(map(str, a_ii)),
            set_ded_alt='|'.join(map(str, a_iialt)),
            m_ours=m_i, m_ded=m_ii, m_ded_alt=m_alt,
            m_random=np.nan if np.isnan(m_i) or np.isnan(m_ii) else 0.5 * (m_i + m_ii),
            m_random_alt=np.nan if np.isnan(m_i) or np.isnan(m_alt) else 0.5 * (m_i + m_alt)))
    d = pd.DataFrame(rows)
    tests = [paired('ours', d.m_ours, 'always-DeD', d.m_ded),
             paired('ours', d.m_ours, 'random', d.m_random),
             paired('ours', d.m_ours, 'defer-DeD', d.m_ded_alt),
             paired('ours', d.m_ours, 'random(defer-DeD)', d.m_random_alt)]
    return d, pd.DataFrame(tests), st


def grid(ctx):
    svp_grid = np.load(os.path.join(HERE, 'results', 'svp_policies.npy')) > 0
    ded_grid = np.load(os.path.join(HERE, 'results', 'ded_policies.npy')) > 0
    zetas = np.arange(0.01, 1.00, 0.01)
    thetas = np.arange(0.01, 1.00, 0.01)
    M, SA_mask = ctx['M'], ctx['SA_mask']
    mask = SA_mask
    pair_rows, state_rows = [], []
    for i in range(len(zetas)):
        svp_i = svp_grid[i] & mask
        for j in range(len(thetas)):
            ded_j = ded_grid[j] & mask
            inter = svp_i & ded_j
            ni = inter.sum(axis=1)
            sz = svp_i.sum(axis=1)
            conf_states = np.nonzero(ni > 0)[0]
            if conf_states.size == 0:
                continue
            rem = [s for s in conf_states if 0 < ni[s] < sz[s]]
            if not rem:
                continue
            mi, mii, malt = [], [], []
            for s in rem:
                svp = list(np.nonzero(svp_i[s])[0])
                bad = list(np.nonzero(ded_j[s])[0])
                a_i, a_ii, a_alt = resolve(s, svp, bad, ctx)
                v_i, v_ii, v_alt = (_mean(M, s, a_i), _mean(M, s, a_ii), _mean(M, s, a_alt))
                mi.append(v_i); mii.append(v_ii); malt.append(v_alt)
                state_rows.append(dict(zeta=round(zetas[i], 2), theta_D=round(thetas[j], 2),
                                       state=int(s), n_svp=int(sz[s]), n_conflict=int(ni[s]),
                                       m_ours=v_i, m_ded=v_ii, m_ded_alt=v_alt,
                                       m_random=0.5 * (v_i + v_ii)))
            pair_rows.append(dict(zeta=round(zetas[i], 2), theta_D=round(thetas[j], 2),
                                  n_removable=len(rem),
                                  m_ours=np.nanmean(mi), m_ded=np.nanmean(mii),
                                  m_ded_alt=np.nanmean(malt),
                                  m_random=np.nanmean([0.5 * (x + y) for x, y in zip(mi, mii)])))
    dp, ds = pd.DataFrame(pair_rows), pd.DataFrame(state_rows)
    tests = [paired('ours', dp.m_ours, 'always-DeD', dp.m_ded),
             paired('ours', dp.m_ours, 'random', dp.m_random),
             paired('ours', dp.m_ours, 'defer-DeD', dp.m_ded_alt)]
    tests_s = [paired('ours[state]', ds.m_ours, 'always-DeD[state]', ds.m_ded),
               paired('ours[state]', ds.m_ours, 'random[state]', ds.m_random),
               paired('ours[state]', ds.m_ours, 'defer-DeD[state]', ds.m_ded_alt)]
    return dp, ds, pd.DataFrame(tests + tests_s)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else 'focal'
    ctx = setup()
    print(f"setup: {ctx['len_df']} training rows")
    if mode == 'focal':
        zeta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
        theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0973
        d, t, st = focal(ctx, zeta, theta)
        tag = f"zeta{zeta:g}_theta{theta:g}"
        d.to_csv(os.path.join(OUT_DIR, f'focal_{tag}_states.csv'), index=False)
        t.to_csv(os.path.join(OUT_DIR, f'focal_{tag}_tests.csv'), index=False)
        print(f"\n=== focal (zeta={zeta:g}, theta_D={theta:g}); SVP {st['terminated_by']}, "
              f"{st['n_sweeps']} sweeps ===")
        print(f"conflicting states: {len(d)}  greedy-fallback: {int(d.greedy_fallback.sum())}  "
              f"removable (rules can differ): {int(d.removable.sum())}")
        with pd.option_context('display.width', 250, 'display.max_columns', 60):
            print(d[['state', 'visits', 'n_svp', 'conflict_actions', 'greedy_fallback',
                     'removable', 'set_ours', 'set_ded', 'set_ded_alt', 'm_ours', 'm_ded',
                     'm_ded_alt', 'm_random']].to_string(index=False))
            print("\nmeans over conflicting states:")
            for c in ['m_ours', 'm_ded', 'm_ded_alt', 'm_random', 'm_random_alt']:
                print(f"  {c:14s} {d[c].mean():.6f}")
            print("\npaired tests:")
            print(t.to_string(index=False))
    else:
        dp, ds, t = grid(ctx)
        dp.to_csv(os.path.join(OUT_DIR, 'grid_pairs.csv'), index=False)
        ds.to_csv(os.path.join(OUT_DIR, 'grid_states.csv'), index=False)
        t.to_csv(os.path.join(OUT_DIR, 'grid_tests.csv'), index=False)
        print(f"\n=== grid: {len(dp)} (zeta,theta) pairs with >=1 removable conflict, "
              f"{len(ds)} conflicting states total ===")
        with pd.option_context('display.width', 250, 'display.max_columns', 60):
            print("means over pairs:")
            for c in ['m_ours', 'm_ded', 'm_ded_alt', 'm_random']:
                print(f"  {c:12s} {dp[c].mean():.6f}")
            print("\npaired tests:")
            print(t.to_string(index=False))


if __name__ == '__main__':
    main()

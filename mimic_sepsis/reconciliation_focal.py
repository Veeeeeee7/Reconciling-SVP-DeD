"""
Evaluating the reconciliation ordering at the clinical hyperparameter setting.

One-step experiment restricted to the CONFLICTING state-action pairs, comparing
three conflict-resolution rules at a single (zeta, theta_D):

  (i)   ours        -- keep every SVP-recommended action, including conflicting ones
  (ii)  always-DeD  -- remove DeD-flagged actions from the SVP set; if that empties
                       the set, fall back to the greedy optimal action
  (iii) random      -- per conflicting state, a fair coin between (i) and (ii)

For a conflicting state s and a resolved action set A(s), the one-step estimate is
the exact expectation under the empirical transition model (no sampling):

    m(s) = mean_{a in A(s)}  sum_{s'} P(s'|s,a) * mort(s')
    mort(s') = 1 if s' is death, 0 if s' is discharge,
               else the empirical mortality of state s'

Two definitions of "empirical mortality of a state" are reported:
  visits : over every training visit to s', the fraction whose trajectory ended in
           death (the natural reading of "the mortality rate at s'"; default)
  starts : over trajectories whose FIRST state is s' (what additional_experiments.py
           used, kept for comparability with the earlier logs)

Stage 1 of the output separates conflicts that exist only because SVP's near-greedy
set was empty and the greedy fallback inserted the action ("greedy") from the rest
("non-greedy"), because rules (i) and (ii) cannot differ on a greedy-fallback state.

Usage:  python3 reconciliation_focal.py [zeta ...]      (default 0.2 0.02)
Writes results/reconciliation/focal_<zeta>_<theta>.csv and a summary log.
"""
import os
import sys
import json
import time
import contextlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import consistency as C

nS, nA = 750, 25
S_SURVIVAL, S_DEATH = 750, 751
nS_TOTAL = nS + 2
THETA_D = 0.0973
DATA = os.path.join(HERE, 'mimic_sepsis_data_2025', 'traj_shifted_train.csv')
OUT_DIR = os.path.join(HERE, 'results', 'reconciliation')
C.nS, C.nA = nS, nA


def empirical_mortality(df):
    """Return (by_visits, by_starts) arrays of per-state mortality, plus visit counts."""
    traj_col = 'icustayid' if 'icustayid' in df.columns else '_traj_id'
    if traj_col == '_traj_id':
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

    first = (df.sort_values('bloc').groupby(traj_col).first().reset_index()
             if 'bloc' in df.columns else df.groupby(traj_col).first().reset_index())
    by_starts = np.full(nS, np.nan)
    for s, v in first.groupby('s:state')['_died'].mean().items():
        if 0 <= s < nS:
            by_starts[s] = v
    return by_visits, by_starts, visits


def one_step(s, actions, P, mort):
    """Exact expected one-step mortality from s when sampling uniformly from `actions`."""
    vals = []
    for a in actions:
        row = P[s, a]
        tot = np.nansum(row)
        if tot <= 0:
            continue                      # unobserved (s, a): skipped
        p = np.nan_to_num(row, nan=0.0) / tot
        v = p[S_DEATH] * 1.0 + p[S_SURVIVAL] * 0.0
        rest = p[:nS]
        ok = ~np.isnan(mort)
        # renormalize over the support we can score
        scored = p[S_DEATH] + p[S_SURVIVAL] + rest[ok].sum()
        if scored <= 0:
            continue
        v += float((rest[ok] * np.nan_to_num(mort[ok], nan=0.0)).sum())
        vals.append(v / scored)
    return float(np.mean(vals)) if vals else np.nan


def run(zeta, gymP_svp, V_star, Q_d, SA_mask, P_arr, morts, visits):
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn), \
            contextlib.redirect_stderr(dn):
        _, pi_star = C.value_iteration_masked(gymP_svp, nS, nA, SA_mask, 1.0, theta=1e-10)
        _, pi_svp, _, _, st = C.svp_masked(gymP_svp, V_star, nS, nA, SA_mask, gamma=1.0,
                                           zeta=zeta, theta=1e-10, return_status=True,
                                           optimal_policies=pi_star)
        # same solve without the greedy fallback, to see which sets were empty
        _, pi_svp_raw, _, _, _ = C.svp_masked(gymP_svp, V_star, nS, nA, SA_mask, gamma=1.0,
                                              zeta=zeta, theta=1e-10, return_status=True,
                                              optimal_policies=np.zeros((nS, nA)))
    pi_ded = C.ded_deadend(Q_d, nS, nA, SA_mask, threshold=THETA_D)

    rows = []
    for s in range(nS):
        svp = [a for a in range(nA) if pi_svp[s, a] > 0 and SA_mask[s, a]]
        bad = [a for a in range(nA) if pi_ded[s, a] > 0 and SA_mask[s, a]]
        conflict = [a for a in svp if a in bad]
        if not conflict:
            continue
        was_empty = not any(pi_svp_raw[s, a] > 0 and SA_mask[s, a] for a in range(nA))
        safe = [a for a in svp if a not in bad]
        a_ii = safe if safe else [a for a in range(nA) if pi_star[s, a] > 0 and SA_mask[s, a]]
        rec = dict(zeta=round(float(zeta), 3), state=s,
                   n_svp=len(svp), n_conflict=len(conflict),
                   conflict_actions='|'.join(map(str, conflict)),
                   greedy_fallback=bool(was_empty),
                   same_set=(sorted(svp) == sorted(a_ii)),
                   visits=int(visits[s]), V_star=float(V_star[s]))
        for key, mort in morts.items():
            m_i = one_step(s, svp, P_arr, mort)
            m_ii = one_step(s, a_ii, P_arr, mort)
            rec[f'm_ours_{key}'] = m_i
            rec[f'm_ded_{key}'] = m_ii
            rec[f'm_random_{key}'] = np.nan if (np.isnan(m_i) or np.isnan(m_ii)) else 0.5 * (m_i + m_ii)
        rows.append(rec)
    return rows, st


def main():
    zetas = [float(x) for x in sys.argv[1:]] or [0.2, 0.02]
    t0 = time.time()
    df = pd.read_csv(DATA, dtype={"a:action": "Int64", 'a:next_action': "Int64"})
    pi_b, SA_mask_df, SA_count = C.make_policy(df)
    SA_mask = SA_mask_df.values
    P_arr = C.make_transition_matrix(df, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)
    by_visits, by_starts, visits = empirical_mortality(df)
    morts = {'visits': by_visits, 'starts': by_starts}

    R_svp = np.zeros((nS_TOTAL, nA, nS_TOTAL))
    R_svp[:, :, S_SURVIVAL] = 1
    R_svp[:, :, S_DEATH] = -1
    gymP_svp = C.make_gymP(P_arr, R_svp, nS, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)

    R_ded = np.zeros((nS_TOTAL, nA, nS_TOTAL))
    R_ded[:, :, S_DEATH] = -1
    gymP_ded = C.make_gymP(P_arr, R_ded, nS, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn), \
            contextlib.redirect_stderr(dn):
        V_star, _ = C.value_iteration_masked(gymP_svp, nS, nA, SA_mask, 1.0, theta=1e-10)
        V_ded, _ = C.value_iteration_masked(gymP_ded, nS, nA, SA_mask, 1.0, theta=1e-10)
    Q_d = C.V2Q(gymP_ded, V_ded, nA, nS, SA_mask, 1.0, mode='ded')
    print(f"setup {time.time() - t0:.0f}s; {len(df)} training rows")

    os.makedirs(OUT_DIR, exist_ok=True)
    for zeta in zetas:
        rows, st = run(zeta, gymP_svp, V_star, Q_d, SA_mask, P_arr, morts, visits)
        df_out = pd.DataFrame(rows)
        path = os.path.join(OUT_DIR, f'focal_zeta{zeta:g}_theta{THETA_D:g}.csv')
        df_out.to_csv(path, index=False)
        ng = df_out[~df_out.greedy_fallback]
        print(f"\n=== zeta={zeta:g}, theta_D={THETA_D} "
              f"(SVP {st['terminated_by']}, {st['n_sweeps']} sweeps) ===")
        print(f"conflicting states: {len(df_out)}  "
              f"(greedy-fallback: {int(df_out.greedy_fallback.sum())}, "
              f"non-greedy: {len(ng)}); "
              f"conflicting (s,a) pairs: {int(df_out.n_conflict.sum())} "
              f"(non-greedy: {int(ng.n_conflict.sum()) if len(ng) else 0})")
        print(f"states where rules (i) and (ii) give the SAME action set: "
              f"{int(df_out.same_set.sum())} of {len(df_out)}")
        with pd.option_context('display.width', 200, 'display.max_columns', 50):
            cols = ['state', 'n_svp', 'n_conflict', 'conflict_actions', 'greedy_fallback',
                    'same_set', 'visits', 'V_star', 'm_ours_visits', 'm_ded_visits',
                    'm_random_visits']
            print(df_out[cols].to_string(index=False))
        print(f"wrote {path}")


if __name__ == '__main__':
    main()

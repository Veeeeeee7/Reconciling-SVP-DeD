"""
summarize_state_level_stats.py
==============================

Standalone summary statistics for the LifeGate *state-level* SVP/DeD conflict
analysis.  Reads ONLY cached artifacts -- it never retrains policies and never
re-runs the (zeta, theta_D) hyperparameter sweep.

Provenance of the two per-state arrays (both reused verbatim from
`visualize_state_level.py`, so numbers match the heatmap figures):

  NF(s)  -- normalized conflict frequency.
            Source: results/trained_policies*.pkl -> results["incons_tensor"],
            shape (nS, |dt_vals|, |zeta_vals|), entry 1 iff
            pi_SVP^(h)(s) INTERSECT pi_DeD^(h)(s) != empty.
            Same reduction as `plot_state_frequency_heatmap()`:
                state_conflict_counts = incons_tensor.sum(axis=(1, 2))
            (see visualize_state_level.py, lines 215-226)

  IOU(s) -- mean intersection-over-union, 0 when the union is empty.
            Source: results/iou_map.npy, produced by `train_search_pair()`
            in visualize_state_level.py (lines 134-151) and cached there;
            the recompute call in that file's __main__ is commented out and
            the .npy is loaded instead.

Run from this directory:

    python summarize_state_level_stats.py                 # default: drag 0.4
    python summarize_state_level_stats.py --pkl results/trained_policies06.pkl

Requires numpy only.  scipy is used for the correlation p-values if present,
otherwise an equivalent pure-numpy implementation is used (the two agree; see
the SELF-CHECKS section of the output).
"""

import argparse
import os
import pickle

import numpy as np

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# The paper's main state-level configuration: natural drag 0.4, gamma = 1,
# cycle-detection fix active (svp.py::value_iter_near_greedy takes the
# intersection of the SVPs in a detected cycle, lines 174-188).
#
# NOTE ON FILE CHOICE.  visualize_state_level.py's __main__ loads
# "results/trained_policies.pkl" for the NF heatmap, but that file is
# BYTE-IDENTICAL to results/trained_policies06.pkl (drag = 0.6).  The drag-0.4
# sweep lives in results/trained_policies04.pkl.  We default to the 0.4 file
# and assert the mismatch loudly below.
DEFAULT_PKL = "results/trained_policies04.pkl"
IOU_NPY = "results/iou_map.npy"

GRID = 10
N_ACTIONS = 5  # env.nb_actions -- used only for a sanity bound on IOU
GAMMA = 1
DRAG = 0.4

# State-index convention (lifegate.py / MDP_lifegate):  s = y * width + x,
# i.e. s = row * 10 + col, row 0 at the top (imshow origin="upper").
#
# These four lists are copied verbatim from visualize_state_level.py __main__
# (lines 259-263) so that the mask matches the published heatmaps exactly.
BARRIER_STATES = [0, 1, 2, 3, 4, 51, 52, 53, 54]
LIFEGATE_STATES = [5, 6, 7]                                    # recoveries
DEAD_STATES = [8, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]       # deaths
DEAD_ENDS = [45, 46, 47, 48, 55, 56, 57, 58, 65, 66, 67, 68,
             75, 76, 77, 78, 85, 86, 87, 88, 95, 96, 97, 98]

# The *death column* proper is x = 9 (all rows).  env.main_deaths additionally
# contains [8, 0] -> s = 8, a death cell that is NOT in the column.
DEATH_COLUMN = [r * GRID + (GRID - 1) for r in range(GRID)]    # 9,19,...,99
DEATH_OFF_COLUMN = sorted(set(DEAD_STATES) - set(DEATH_COLUMN))  # -> [8]

ENV_TRUE_DEAD_ENDS = [r * GRID + c for r in range(5, 10) for c in range(5, 9)]


# --------------------------------------------------------------------------
# statistics helpers (pure numpy; scipy optional)
# --------------------------------------------------------------------------
def _betacf(a, b, x, itmax=300, eps=3.0e-16):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    from math import exp, lgamma
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b)
                + a * np.log(x) + b * np.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - exp(lgamma(a + b) - lgamma(a) - lgamma(b)
                     + b * np.log(1.0 - x) + a * np.log(x)) * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t, df):
    """Two-sided p-value for Student's t."""
    if not np.isfinite(t):
        return 0.0
    return _betainc(0.5 * df, 0.5, df / (df + t * t))


def _rankdata(a):
    """Average ranks, ties handled (equivalent to scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x, y):
    """Pearson r and two-sided p (t-approximation, same as scipy)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    r = float((xm * ym).sum() / denom)
    r = max(-1.0, min(1.0, r))
    df = n - 2
    if abs(r) == 1.0:
        return r, 0.0
    t = r * np.sqrt(df / (1.0 - r * r))
    return r, _t_two_sided_p(t, df)


def spearman(x, y):
    """Spearman rho and two-sided p (t-approximation, scipy's default)."""
    return pearson(_rankdata(x), _rankdata(y))


def sig(x, n=3):
    """Format to n significant figures."""
    if x == 0:
        return "0.000"
    if not np.isfinite(x):
        return "nan"
    from math import floor, log10
    d = n - 1 - int(floor(log10(abs(x))))
    if d < 0:
        d = 0
    return f"{round(x, d):.{d}f}"


def sig_p(p, n=3):
    """p-values: scientific notation when tiny."""
    if p == 0.0:
        return "< 1e-300"
    if p < 1e-4:
        return f"{p:.{n - 1}e}"
    return sig(p, n)


def rc(s):
    """State index -> (row, col)."""
    return (s // GRID, s % GRID)


# --------------------------------------------------------------------------
# adjacency (item 2)
# --------------------------------------------------------------------------
def orth_neighbors(s):
    """Orthogonal (4-neighbour) neighbours of state s, inside the grid."""
    r, c = rc(s)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < GRID and 0 <= cc < GRID:
            out.append(rr * GRID + cc)
    return out


def group_a(non_terminal, hazard_cells):
    """Group A = non-terminal states orthogonally adjacent to any hazard cell."""
    hz = set(hazard_cells)
    return [s for s in non_terminal if any(nb in hz for nb in orth_neighbors(s))]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=DEFAULT_PKL,
                    help="cached sweep results (default: drag 0.4)")
    ap.add_argument("--iou", default=IOU_NPY, help="cached per-state IOU map")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    pkl_path = args.pkl if os.path.isabs(args.pkl) else os.path.join(here, args.pkl)
    iou_path = args.iou if os.path.isabs(args.iou) else os.path.join(here, args.iou)

    # ---------------- load cached artifacts (no retraining) ----------------
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    zeta_vals = np.asarray(results["zeta_vals"])
    dt_vals = np.asarray(results["dt_vals"])
    incons_tensor = np.asarray(results["incons_tensor"])
    H = len(zeta_vals) * len(dt_vals)

    iou_map = np.load(iou_path).ravel()

    # NF: exact reduction used by plot_state_frequency_heatmap()
    conflict_counts = incons_tensor.sum(axis=(1, 2)).astype(np.float64)

    # ---------------- masks ----------------
    excluded = set(BARRIER_STATES) | set(LIFEGATE_STATES) | set(DEAD_STATES) | set(DEAD_ENDS)
    non_terminal = [s for s in range(GRID * GRID) if s not in excluded]
    S = np.array(non_terminal, dtype=int)

    counts = conflict_counts[S]
    nf_H = counts / H                       # paper definition: (1/|H|) * sum_h 1[...]
    nf_max = counts / counts.max()          # figure convention: divide by max count
    iou = iou_map[S]

    # ---------------- provenance / sign-convention audit ----------------
    print("=" * 78)
    print("PROVENANCE & CONFIG AUDIT")
    print("=" * 78)
    print(f"NF  source : {os.path.relpath(pkl_path, here)}  ->  results['incons_tensor']"
          f"  shape {incons_tensor.shape}")
    print(f"IOU source : {os.path.relpath(iou_path, here)}  ->  cached output of "
          f"train_search_pair() in visualize_state_level.py")
    print(f"Reduction  : identical to plot_state_frequency_heatmap() "
          f"(incons_tensor.sum(axis=(1,2))) -- nothing reimplemented.")
    print()
    print(f"zeta grid  : n={len(zeta_vals)}  [{zeta_vals.min():.2f} .. {zeta_vals.max():.2f}]"
          f"  step {np.diff(zeta_vals).mean():.2f}")
    print(f"theta_D grid: n={len(dt_vals)}  [{dt_vals.min():.2f} .. {dt_vals.max():.2f}]"
          f"  step {np.diff(dt_vals).mean():.2f}")
    print(f"|H| = {len(zeta_vals)} x {len(dt_vals)} = {H}")
    print()
    sign = "POSITIVE" if (dt_vals > 0).all() else "NEGATIVE"
    print(f"theta_D SIGN CONVENTION IN THE CACHED RESULTS: {sign}")
    print("  dt_vals are swept positive (0.01 .. 0.99) and the DeD rule in")
    print("  train_policies.py::bad_policies is  `if Q_d[s,a] <= -threshold`,")
    print("  i.e.  -Q_D(s,a) >= theta_D  with theta_D > 0.")
    print("  => matches the POSITIVE convention the paper writes. No sign flip needed.")
    print("  (The axis labels that used to read '$-\\theta_D$' in visualize_policy_level.py,")
    print("   mimic_sepsis/bootstrap.py and mimic_sepsis/visualizations.py have been")
    print("   corrected to '$\\theta_D$' to match the positive sweep.)")
    print()

    # cross-check: is the file visualize_state_level.py actually loads the 0.4 one?
    default_pkl = os.path.join(here, "results", "trained_policies.pkl")
    if os.path.exists(default_pkl):
        with open(default_pkl, "rb") as f:
            d0 = pickle.load(f)
        same_as_loaded = np.array_equal(np.asarray(d0["incons_map"]),
                                        np.asarray(results["incons_map"]))
        matches = []
        for tag in ("00", "02", "04", "06"):
            p = os.path.join(here, "results", f"trained_policies{tag}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    dd = pickle.load(f)
                if np.array_equal(np.asarray(d0["incons_map"]), np.asarray(dd["incons_map"])):
                    matches.append(tag)
        if not same_as_loaded:
            drag_tag = matches[0] if matches else "?"
            drag_val = f"0.{drag_tag[1]}" if len(drag_tag) == 2 else "?"
            print("!! WARNING -------------------------------------------------------------")
            print("!! visualize_state_level.py __main__ loads 'results/trained_policies.pkl'")
            print(f"!! for the NF heatmap, but that file is identical to "
                  f"trained_policies{'/'.join(matches) or '??'}.pkl (drag = {drag_val}),")
            print("!! NOT the drag-0.4 sweep.")
            print("!! The published state_conflict_frequency_heatmap.pdf therefore shows the")
            print("!! WRONG DRAG.  This script uses %s instead." % os.path.basename(pkl_path))
            print("!! ----------------------------------------------------------------------")
            print()

    # IOU/NF consistency: a state conflicts for h iff it contributes IOU>0 for h,
    # and per-h IOU is in [1/|A|, 1].  Hard two-sided bound tying the two arrays
    # to the SAME configuration.
    lo = counts / (N_ACTIONS * H)
    hi = counts / H
    viol = int(np.sum((iou < lo - 1e-12) | (iou > hi + 1e-12)))
    print(f"IOU/NF configuration cross-check: count/(|A||H|) <= IOU <= count/|H|"
          f"  -> {viol} violations over {len(S)} states"
          f"  {'[PASS: both arrays are the same config]' if viol == 0 else '[FAIL: MISMATCHED CONFIGS]'}")
    zero_nf = set(int(s) for s in S[counts == 0])
    zero_iou = set(int(s) for s in S[iou == 0])
    print(f"Zero-set check: {{NF=0}} == {{IOU=0}}  -> "
          f"{'PASS' if zero_nf == zero_iou else 'FAIL'}  ({sorted(zero_nf)} vs {sorted(zero_iou)})")
    print()

    # ---------------- grouping ----------------
    hazards = set(DEAD_STATES) | set(DEAD_ENDS)   # death column (+ [8,0]) and dead-end region
    A = group_a(non_terminal, hazards)
    B = [s for s in non_terminal if s not in set(A)]

    idx = {int(s): i for i, s in enumerate(S)}
    iA = np.array([idx[s] for s in A], dtype=int)
    iB = np.array([idx[s] for s in B], dtype=int)

    print("=" * 78)
    print("GROUPING (item 2)")
    print("=" * 78)
    print("No adjacency definition exists anywhere in the plotting code "
          "(grep: no 'neighbor'/'adjacent'/'orthogonal' in the repo),")
    print("so this is defined here: Group A = non-terminal states with at least one")
    print("ORTHOGONAL (4-neighbour) neighbour in the death set or the dead-end set.")
    print(f"  death set    : column x=9 -> {DEATH_COLUMN}")
    print(f"                 plus off-column death cell(s) {DEATH_OFF_COLUMN} (= env cell [8,0])")
    print(f"  dead-end set : {DEAD_ENDS}")
    print()
    print(f"GROUP A ({len(A)} states)  index: (row,col)")
    print("  " + "  ".join(f"{s}:{rc(s)}" for s in A))
    print()
    print(f"GROUP B ({len(B)} states)  index: (row,col)")
    for k in range(0, len(B), 8):
        print("  " + "  ".join(f"{s}:{rc(s)}" for s in B[k:k + 8]))
    print()

    # ASCII map
    print("ASCII MAP   A = adjacent group   b = remaining non-terminal")
    print("            # = barrier   R = recovery   X = death   D = dead-end")
    print("      " + " ".join(f"c{c}" for c in range(GRID)))
    setA = set(A)
    for r in range(GRID):
        row = []
        for c in range(GRID):
            s = r * GRID + c
            if s in BARRIER_STATES:
                ch = "#"
            elif s in LIFEGATE_STATES:
                ch = "R"
            elif s in DEAD_STATES:
                ch = "X"
            elif s in DEAD_ENDS:
                ch = "D"
            elif s in setA:
                ch = "A"
            else:
                ch = "b"
            row.append(ch)
        print(f"  r{r}  " + "  ".join(row))
    print()

    # ---------------- statistics ----------------
    rho, p_rho = spearman(nf_H, iou)
    r_p, p_r = pearson(nf_H, iou)
    # rank/linear correlations are invariant to the choice of NF normalization
    rho_max, _ = spearman(nf_max, iou)
    r_max, _ = pearson(nf_max, iou)

    try:
        from scipy import stats as _st  # optional cross-check
        s_rho, s_p = _st.spearmanr(nf_H, iou)
        s_r, s_pr = _st.pearsonr(nf_H, iou)
        scipy_line = (f"scipy present: spearman rho={s_rho:.10f} p={s_p:.3e} | "
                      f"pearson r={s_r:.10f} p={s_pr:.3e}  "
                      f"(max abs diff vs pure-numpy: "
                      f"{max(abs(s_rho - rho), abs(s_r - r_p)):.2e})")
    except Exception:
        scipy_line = "scipy not installed -- pure-numpy Spearman/Pearson used (t-approximation p-values)."

    print("=" * 78)
    print("SELF-CHECKS")
    print("=" * 78)
    print(f"  {scipy_line}")
    print(f"  normalization invariance: rho(NF/|H|) = {rho:.10f}, "
          f"rho(NF/max) = {rho_max:.10f}  -> {'identical' if abs(rho - rho_max) < 1e-12 else 'DIFFER'}")
    print(f"                            r  (NF/|H|) = {r_p:.10f}, "
          f"r  (NF/max) = {r_max:.10f}  -> {'identical' if abs(r_p - r_max) < 1e-12 else 'DIFFER'}")
    print(f"  |S| = {len(S)} = 100 - {len(BARRIER_STATES)} barriers - {len(LIFEGATE_STATES)} recoveries "
          f"- {len(DEAD_STATES)} deaths - {len(DEAD_ENDS)} dead-ends")
    print(f"  n_A + n_B = {len(A)} + {len(B)} = {len(A) + len(B)}  "
          f"{'[PASS]' if len(A) + len(B) == len(S) else '[FAIL]'}")
    print(f"  group-mean recombination: "
          f"({len(A)}*{nf_H[iA].mean():.6f} + {len(B)}*{nf_H[iB].mean():.6f}) / {len(S)} = "
          f"{(len(A) * nf_H[iA].mean() + len(B) * nf_H[iB].mean()) / len(S):.6f} "
          f"vs overall {nf_H.mean():.6f}")
    print()

    # ---------------- THE BLOCK ----------------
    zero_states = [rc(int(s)) for s in S[counts == 0]]

    print("=" * 78)
    print("PASTE-READY BLOCK  (NF normalized by |H|, i.e. the paper's definition)")
    print("=" * 78)
    print(f"CONFIG:      drag={DRAG}  gamma={GAMMA}  |H|={H}  theta_D sign convention={sign.lower()} "
          f"(swept 0.01..0.99, rule -Q_D >= theta_D)")
    print(f"|S| (non-terminal):        {len(S)}")
    print()
    print(f"SPEARMAN  rho={sig(rho)}  p={sig_p(p_rho)}        PEARSON r={sig(r_p)}  (p={sig_p(p_r)})")
    print()
    print(f"GROUP A (adjacent, n={len(A)}):   mean NF={sig(nf_H[iA].mean())}   mean IOU={sig(iou[iA].mean())}")
    print(f"GROUP B (remaining, n={len(B)}):  mean NF={sig(nf_H[iB].mean())}   mean IOU={sig(iou[iB].mean())}")
    print()
    print(f"NF   min={sig(nf_H.min())}  median={sig(np.median(nf_H))}  max={sig(nf_H.max())}")
    print(f"IOU  min={sig(iou.min())}  median={sig(np.median(iou))}  max={sig(iou.max())}")
    print()
    print(f"ZERO-CONFLICT STATES: {zero_states}")
    print()

    # secondary block under the figure's max-normalization
    print("-" * 78)
    print("SECONDARY: same numbers with NF normalized by max count "
          f"({int(counts.max())}), which is what")
    print("plot_state_frequency_heatmap() actually prints on the heatmap cells.")
    print("-" * 78)
    print(f"GROUP A (adjacent, n={len(A)}):   mean NF={sig(nf_max[iA].mean())}   mean IOU={sig(iou[iA].mean())}")
    print(f"GROUP B (remaining, n={len(B)}):  mean NF={sig(nf_max[iB].mean())}   mean IOU={sig(iou[iB].mean())}")
    print(f"NF   min={sig(nf_max.min())}  median={sig(np.median(nf_max))}  max={sig(nf_max.max())}")
    print(f"(Spearman/Pearson are unchanged: both normalizations are positive linear rescalings.)")
    print()

    # ---------------- LaTeX sentences ----------------
    print("=" * 78)
    print("LATEX SENTENCES")
    print("=" * 78)
    print(f"The two measures are strongly rank-correlated across states "
          f"(Spearman $\\rho = {sig(rho)}$), so states that conflict more often")
    print("also conflict more severely.")
    print()
    print(f"Conflict frequency and IOU are highest for states adjacent to the death column "
          f"and to the dead-end region")
    print(f"(mean frequency {sig(nf_H[iA].mean())}, mean IOU {sig(iou[iA].mean())}) and lowest for "
          f"states farther from the death zone")
    print(f"({sig(nf_H[iB].mean())}, {sig(iou[iB].mean())}).")
    print()

    # ---------------- per-state table ----------------
    print("=" * 78)
    print("PER-STATE TABLE (non-terminal only)")
    print("=" * 78)
    print(f"{'s':>4} {'(row,col)':>10} {'grp':>4} {'count':>7} {'NF=/|H|':>9} {'NF=/max':>9} {'IOU':>9}")
    for i, s in enumerate(S):
        s = int(s)
        print(f"{s:>4} {str(rc(s)):>10} {('A' if s in setA else 'B'):>4} "
              f"{int(counts[i]):>7} {nf_H[i]:>9.4f} {nf_max[i]:>9.4f} {iou[i]:>9.4f}")


if __name__ == "__main__":
    main()

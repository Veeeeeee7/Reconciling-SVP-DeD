"""
State-level figure: DeD Q-values against the SVP policy on MIMIC-III.

Regenerates the MIMIC-III state-level plot in two versions, using the same
solvers and the same state ordering as experiments.py (imported, not copied):

  figures/mimic_state_level_plot.pdf
      MAIN TEXT. Clinical setting only: DeD Q_D heatmap + the strict SVP policy
      at zeta=0.2 (circles = actions that met the near-optimality margin,
      black dots = actions forced by the greedy fallback) + DeD crosses.

  figures/mimic_state_level_plot_supplementary.pdf
      SUPPLEMENT. The previous two-policy version, adding the loose zeta=0.8
      policy and the two states (546, 47) that only conflict at that margin.

Run from mimic_sepsis/:  python3 state_level_figure.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import experiments as E   # guarded by __main__, so importing runs nothing

# make_policy() reads these as module-level globals (they are set inside
# experiments.py's __main__ block, which importing does not execute)
E.nS, E.nA = 750, 25

nS, nA = 750, 25
S_SURVIVAL, S_DEATH = 750, 751
nS_TOTAL = nS + 2
ZETA_STRICT, ZETA_LOOSE = 0.2, 0.8
N_MAIN_ROWS_MAIN = 10      # main text: down to State 604
N_MAIN_ROWS_SUPP = 16      # supplement keeps the longer tail
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# greedy-fallback (state, action) pairs at zeta=0.2; identical to the list in
# experiments.py. Recomputed below and asserted against this so it cannot rot.
FALLBACK_POINTS = [(211, 23), (323, 19), (444, 24), (490, 14), (531, 5),
                   (540, 14), (728, 24), (177, 13), (52, 19)]


def build():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mimic_sepsis_data_2025") + os.sep
    train_df = pd.read_csv(data_dir + "traj_shifted_train.csv",
                           dtype={"a:action": "Int64", "a:next_action": "Int64"})
    pi_b, SA_mask, SA_count = E.make_policy(train_df)
    P = E.make_transition_matrix(train_df, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)

    # theta_D = empirical mortality rate of the cohort
    deaths = sum(1 for t in train_df["traj"].unique()
                 if train_df[train_df["traj"] == t].iloc[-1]["s:next_state"] == S_DEATH)
    theta_D = deaths / train_df["traj"].nunique()

    R_svp = np.zeros((nS_TOTAL, nA, nS_TOTAL))
    R_svp[:, :, S_SURVIVAL] = 1
    R_svp[:, :, S_DEATH] = -1
    P_svp = E.make_gymP(P, R_svp, nS, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)
    V_star, pi_star = E.value_iteration_masked(P_svp, nS, nA, SA_mask.values,
                                               gamma=1.0, theta=1e-10)
    _, pi_svp, _, _ = E.svp_masked(P_svp, V_star, nS, nA, SA_mask.values,
                                   gamma=1.0, zeta=ZETA_STRICT, theta=1e-10)
    _, pi_svp_loose, _, _ = E.svp_masked(P_svp, V_star, nS, nA, SA_mask.values,
                                         gamma=1.0, zeta=ZETA_LOOSE, theta=1e-10)

    R_ded = np.zeros((nS_TOTAL, nA, nS_TOTAL))
    R_ded[:, :, S_DEATH] = -1
    P_ded = E.make_gymP(P, R_ded, nS, nA, nS_TOTAL, S_SURVIVAL, S_DEATH)
    V_ded, _ = E.value_iteration_masked(P_ded, nS, nA, SA_mask.values,
                                        gamma=1.0, theta=1e-10)
    Q_ded = E.V2Q(P_ded, V_ded, nA, nS, SA_mask.values, 1.0, mode="ded")

    sort_indices = np.argsort(V_ded)          # most severe (lowest V_D) first
    return dict(Q_ded=Q_ded, V_ded=V_ded, pi_svp=pi_svp, pi_svp_loose=pi_svp_loose,
                sort_indices=sort_indices, theta_D=theta_D, SA_mask=SA_mask.values)


def draw(d, supplementary=False):
    Q_ded, sort_indices, theta_D = d["Q_ded"], d["sort_indices"], d["theta_D"]
    Q_ord = Q_ded[sort_indices]
    pi_ord = d["pi_svp"][sort_indices]
    pi_loose_ord = d["pi_svp_loose"][sort_indices]

    n_rows_wanted = N_MAIN_ROWS_SUPP if supplementary else N_MAIN_ROWS_MAIN
    main_rows = list(range(min(n_rows_wanted, Q_ord.shape[0])))
    add_rows = []
    if supplementary:   # states that only conflict under the loose policy
        for s in (546, 47):
            idx = int(np.where(sort_indices == s)[0][0])
            if idx not in main_rows:
                add_rows.append(idx)

    SEP = None
    row_sel = main_rows + ([SEP] if add_rows else []) + add_rows
    orig_to_y, y = {}, 0
    for r in row_sel:
        if r is SEP:
            y += 1
            continue
        orig_to_y[r] = y
        y += 1

    Q_rows, Pi_rows, PiL_rows, ylab = [], [], [], []
    for r in row_sel:
        if r is SEP:
            Q_rows.append(np.full((nA,), np.nan))
            Pi_rows.append(np.zeros(nA, dtype=int))
            PiL_rows.append(np.zeros(nA, dtype=int))
            ylab.append("...")
        else:
            Q_rows.append(Q_ord[r, :])
            Pi_rows.append(pi_ord[r, :])
            PiL_rows.append(pi_loose_ord[r, :])
            ylab.append(f"State {sort_indices[r]}")
    Q, Pi, PiL = np.vstack(Q_rows), np.vstack(Pi_rows), np.vstack(PiL_rows)
    nrows = Q.shape[0]

    # fallback pairs, in plot coordinates
    fb = [(orig_to_y[int(np.where(sort_indices == s)[0][0])], a)
          for s, a in FALLBACK_POINTS
          if int(np.where(sort_indices == s)[0][0]) in orig_to_y]
    fb_set = set(fb)

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    # aspect="equal" makes every grid cell an exact square; the figure is given
    # enough height that the width stays the binding constraint, and the unused
    # vertical space is removed by bbox_inches="tight" at save time.
    fig, ax = plt.subplots(figsize=(14, 14.0 * nrows / nA + 3.0))
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.25)
    cmap = plt.get_cmap("Reds_r").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(Q, aspect="equal", interpolation="nearest", cmap=cmap)

    ax.set_xticks(np.arange(nA))
    ax.set_xticklabels([f"Action {j}" for j in range(nA)], rotation=45, ha="right", fontsize=20)
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(ylab, fontsize=20)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"DeD $Q$-values", fontsize=32)
    cbar.ax.tick_params(labelsize=20)

    # DeD eliminations
    ys, xs = np.where(Q < -theta_D)
    ax.scatter(xs, ys, marker="x", s=175, linewidths=2, c="red", zorder=3)

    if supplementary:   # loose policy underneath, as before
        ysl, xsl = np.where(PiL == 1)
        ax.scatter(xsl, ysl, marker="o", s=400, facecolors="none",
                   edgecolors="#6FA8FF", linewidths=2, zorder=2)

    # strict SVP policy. In the main figure every recommended action is drawn as
    # one blue circle, including the actions supplied by the greedy fallback, so
    # the figure carries a single marker type; which conflicts are fallback-driven
    # is stated in the text instead. The supplementary version keeps the original
    # black dots so the two policies remain distinguishable there.
    ys1, xs1 = np.where(Pi == 1)
    pts = {(int(yy), int(x)) for yy, x in zip(ys1, xs1)}
    if supplementary:
        pts -= fb_set
    else:
        pts |= fb_set
    if pts:
        ax.scatter([x for _, x in pts], [yy for yy, _ in pts], marker="o",
                   s=400 if not supplementary else 120, facecolors="none",
                   edgecolors="blue", linewidths=2, zorder=4)
    if supplementary and fb:
        ax.scatter([a for _, a in fb], [yy for yy, _ in fb], marker="o", s=75,
                   facecolors="black", edgecolors="black", linewidths=0, zorder=5)

    if SEP in row_sel:
        ax.hlines(row_sel.index(SEP), -0.5, nA - 0.5, colors="black",
                  linewidth=1.0, alpha=0.35)
    ax.set_xlim(-0.5, nA - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)

    # the square-cell constraint shrinks the axes inside its box, so re-seat the
    # colorbar on the heatmap's actual vertical extent
    fig.canvas.draw()
    pos, cpos = ax.get_position(), cax.get_position()
    cax.set_position([cpos.x0, pos.y0, cpos.width, pos.height])

    os.makedirs(OUT_DIR, exist_ok=True)
    name = "mimic_state_level_plot_supplementary.pdf" if supplementary \
        else "mimic_state_level_plot.pdf"
    out = os.path.join(OUT_DIR, name)
    plt.savefig(out, dpi=800, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return row_sel, orig_to_y


if __name__ == "__main__":
    d = build()
    print(f"theta_D = {d['theta_D']:.4f}")
    # sanity: the fallback list must still be the empty-set states at zeta=0.2
    conf = [(s, a) for s in range(nS) for a in range(nA)
            if d["pi_svp"][s, a] and -d["Q_ded"][s, a] >= d["theta_D"]]
    print("conflicting (state, action) pairs at zeta=0.2:", conf)
    order = d["sort_indices"][:N_MAIN_ROWS_MAIN]
    print(f"{N_MAIN_ROWS_MAIN} most severe states (main figure, top to bottom):", list(map(int, order)))
    print("  -V_D:", [round(float(-d["V_ded"][s]), 3) for s in order])
    draw(d, supplementary=False)
    draw(d, supplementary=True)

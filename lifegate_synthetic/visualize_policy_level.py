# visualize_search_pair.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os

def visualize_svp_and_bad_sizes(zeta_vals, dt_vals, svp_map, bad_map,
                                save_path="figures/svp_vs_bad_sizes_fixed.pdf"):
    """
    One plot, two y-axes:
      - Green (left y): SVP size vs zeta (mean over death_threshold)
      - Red   (right y): DeD eliminated size vs death_threshold (mean over zeta)

    Note: both curves share the same x-axis range (usually [0, 1]),
    but the x-values correspond to different hyperparameters.
    """
    svp_map = np.asarray(svp_map)
    bad_map = np.asarray(bad_map)

    if svp_map.shape != bad_map.shape:
        raise ValueError(f"svp_map shape {svp_map.shape} must match bad_map shape {bad_map.shape}")
    if svp_map.shape != (len(dt_vals), len(zeta_vals)):
        raise ValueError(
            f"Expected maps with shape (len(dt_vals), len(zeta_vals)) = "
            f"({len(dt_vals)}, {len(zeta_vals)}), got {svp_map.shape}"
        )

    # Use Times-like serif fonts + serif math (ensures θ_D subscripting works)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # SVP depends on zeta -> average over dt for each zeta
    svp_curve = svp_map.mean(axis=0)   # (len(zeta_vals),)

    # DeD depends on death_threshold -> average over zeta for each dt
    bad_curve = bad_map.mean(axis=1)   # (len(dt_vals),)

    fig, ax1 = plt.subplots(figsize=(16, 12))

    # Left axis: SVP (green)
    ax1.plot(zeta_vals, svp_curve, color="green", linewidth=2)
    ax1.set_xlabel(r"Hyperparameter Value ($\zeta$ for SVP, $\theta_D$ for DeD)", fontsize=32)
    ax1.set_ylabel("Avg SVP Recommended Actions per State", color="green", fontsize=32)

    # All axis tick labels fontsize 20
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelcolor="green", labelsize=20)

    # Right axis: DeD eliminated (red)
    ax2 = ax1.twinx()
    ax2.plot(dt_vals, bad_curve, color="red", linewidth=2)
    ax2.set_ylabel("Avg DeD Eliminated Actions per State", color="red", fontsize=32)
    ax2.tick_params(axis="y", labelcolor="red", labelsize=20)

    # Keep existing y-ticks (no change), but explicitly re-set if you want
    ax1.set_yticks(list(ax1.get_yticks()))
    ax2.set_yticks(list(ax2.get_yticks()))

    # Shared x-limits cover both ranges
    x_min = min(float(np.min(zeta_vals)), float(np.min(dt_vals)))
    x_max = max(float(np.max(zeta_vals)), float(np.max(dt_vals)))
    ax1.set_xlim(x_min, x_max)

    # Legend (green/red)
    legend_handles = [
        Line2D([0], [0], color="green", lw=2, label=r"SVP size vs. $\zeta$"),
        Line2D([0], [0], color="red",   lw=2, label=r"DeD size vs. $\theta_D$"),
    ]
    ax1.legend(handles=legend_handles, loc="best", fontsize=20)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=800)
    plt.show()

def analyze_paramwise_inconsistency(results, save_dir="figures/"):
    os.makedirs(save_dir, exist_ok=True)

    incons_map = results["incons_map"]  # shape: [len(theta), len(zeta)]
    zetas = results["zeta_vals"]
    thetas = results["dt_vals"]

    # Use Times-like serif fonts + serif math (ensures θ_D subscripting works)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Times-like math glyphs
    })

    # Generate and save heatmap
    plt.figure(figsize=(16, 12))
    plt.imshow(
        incons_map,
        cmap="RdYlGn_r",
        origin="lower",
        aspect="auto",
        extent=[zetas[0], zetas[-1], thetas[0], thetas[-1]],
    )

    cbar = plt.colorbar()
    cbar.set_label("Conflict Fraction", fontsize=32)
    cbar.ax.tick_params(labelsize=20)

    plt.xlabel(r"$\zeta$", fontsize=32)
    plt.ylabel(r"$\theta_D$", fontsize=32)

    plt.tick_params(labelsize=20)
    plt.tight_layout()

    heatmap_path = os.path.join(save_dir, "policy_level_inconsistency_heatmap.pdf")
    plt.savefig(heatmap_path, dpi=800, bbox_inches="tight")
    plt.show()
    plt.close()

    return heatmap_path

def compute_fraction_consistent_under_le1(results):
    incons_map = results["incons_map"]  # shape: [len(theta), len(zeta)]
    zeta_vals = np.tile(results["zeta_vals"], (incons_map.shape[0], 1))
    dt_vals = np.tile(results["dt_vals"].reshape(-1, 1), (1, incons_map.shape[1]))
    le1_incides = np.argwhere(zeta_vals + dt_vals <= 1.0)
    total_le1 = len(le1_incides)
    consistent_le1 = sum(1 for (ti, zi) in le1_incides if incons_map[ti, zi] == 0)
    fraction_consistent = consistent_le1 / total_le1 if total_le1 > 0 else 0.0
    print(f'Fraction of consistent policies under ζ + θ_D ≤ 1: {consistent_le1} / {total_le1} = {fraction_consistent:.4f} ')

def plot_4_inconsistency_heatmaps(
    data00, data02, data04, data06,
    titles=("Drag=0.0", "Drag=0.2", "Drag=0.4", "Drag=0.6"),
    save_dir="figures/",
    filename="policy_level_inconsistency_heatmaps_4panel.png",
    cmap="RdYlGn_r",
):
    """
    Create ONE figure with 4 subplots (1x4 row) + dedicated horizontal
    colorbar row at the bottom, with reduced extra whitespace.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)

    datas = [data00, data02, data04, data06]

    # Use Times-like serif fonts + serif math (ensures θ_D subscripting works)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # Validate + collect for shared color scale
    all_maps = []
    for i, d in enumerate(datas):
        for k in ("incons_map", "zeta_vals", "dt_vals"):
            if k not in d:
                raise KeyError(f"Dataset index {i} missing required key: '{k}'")

        inc = np.asarray(d["incons_map"])
        zetas = np.asarray(d["zeta_vals"])
        thetas = np.asarray(d["dt_vals"])

        if inc.shape != (len(thetas), len(zetas)):
            raise ValueError(
                f"Dataset index {i} incons_map shape mismatch: "
                f"expected ({len(thetas)}, {len(zetas)}), got {inc.shape}"
            )
        all_maps.append(inc)

    global_vmin = float(np.nanmin([np.nanmin(m) for m in all_maps]))
    global_vmax = float(np.nanmax([np.nanmax(m) for m in all_maps]))

    # --- GridSpec layout: 1x4 plots + 1 slim horizontal colorbar row ---
    fig = plt.figure(figsize=(28, 7))

    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        height_ratios=[1, 0.06],  # slim cbar row
        wspace=0.08, hspace=0.35
    )

    ax00 = fig.add_subplot(gs[0, 0])
    ax02 = fig.add_subplot(gs[0, 1], sharey=ax00)
    ax04 = fig.add_subplot(gs[0, 2], sharey=ax00)
    ax06 = fig.add_subplot(gs[0, 3], sharey=ax00)
    axs = [ax00, ax02, ax04, ax06]

    im = None
    for ax, d, title in zip(axs, datas, titles):
        incons_map = np.asarray(d["incons_map"])
        zetas = np.asarray(d["zeta_vals"])
        thetas = np.asarray(d["dt_vals"])

        im = ax.imshow(
            incons_map,
            cmap=cmap,
            origin="lower",
            aspect="auto",
            extent=[zetas[0], zetas[-1], thetas[0], thetas[-1]],
            vmin=global_vmin,
            vmax=global_vmax,
        )
        ax.set_title(title, fontsize=32)
        ax.set_xlabel(r"$\zeta$", fontsize=24)

    # Only the leftmost panel gets a y-label / y-tick labels
    ax00.set_ylabel(r"$\theta_D$", fontsize=24)
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)

    # Set tick label size
    for ax in axs:
        ax.tick_params(labelsize=16)

    # Horizontal colorbar at the bottom (spans all four columns)
    cax = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Conflict Fraction", fontsize=32)
    cbar.ax.tick_params(labelsize=16)

    # Tighten outer margins explicitly
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.12, top=0.90)

    # Save tightly cropped (dpi=800)
    fig.savefig(out_path, dpi=800, bbox_inches="tight", pad_inches=0.02)

    plt.show()
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    # load trained maps
    with open("results/trained_policies04.pkl","rb") as f:
        data = pickle.load(f)

    zeta_vals     = data['zeta_vals']
    dt_vals       = data['dt_vals']
    incons_map    = data['incons_map']
    svp_size_map  = data['svp_size_map']
    bad_size_map  = data['bad_size_map']

    visualize_svp_and_bad_sizes(
        zeta_vals, dt_vals,
        svp_size_map, bad_size_map,
        save_path="figures/svp_and_ded_sizes_vs_hyperparameters.pdf"
    )

    analyze_paramwise_inconsistency(data)
    # compute_fraction_consistent_under_le1(data)


    with open("results/trained_policies00.pkl","rb") as f:
        data00 = pickle.load(f)

    with open("results/trained_policies02.pkl","rb") as f:
        data02 = pickle.load(f)

    with open("results/trained_policies04.pkl","rb") as f:
        data04 = pickle.load(f)

    with open("results/trained_policies06.pkl","rb") as f:
        data06 = pickle.load(f)

    plot_4_inconsistency_heatmaps(
        data00, data02, data04, data06,
        titles=("Drag=0.0", "Drag=0.2", "Drag=0.4", "Drag=0.6"),
        save_dir="figures/",
        filename="drag_comparison.pdf"
    )

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_svp_size(svp_sizes, path):
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0.0, 1.0, len(svp_sizes)), svp_sizes, marker='o')
    plt.title('SVP Sizes vs. Zeta')
    plt.xlabel('Zeta')
    plt.ylabel('SVP Size')
    plt.grid(True)
    plt.savefig(path, dpi=500)
    plt.close()

def plot_ded_size(ded_sizes, path):
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0.0, 1.0, len(ded_sizes)), ded_sizes, marker='o', color='orange')
    plt.title('DeD Sizes vs. Death Threshold')
    plt.xlabel('Death Threshold')
    plt.ylabel('DeD Size')
    plt.grid(True)
    plt.savefig(path, dpi=500)
    plt.close()

def plot_conflict_fraction_heatmap(conflict_fractions, path):
    # Use Times-like serif fonts + serif math
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Times-like math
    })

    plt.figure(figsize=(16, 12))
    im = plt.imshow(conflict_fractions.T, extent=[0, 1, 0, 1], origin='lower',
                    aspect='auto', cmap='RdYlGn_r')

    cbar = plt.colorbar(im)
    cbar.set_label('Conflict Fraction', fontsize=32)
    cbar.ax.tick_params(labelsize=20)

    plt.xlabel(r'$\zeta$', fontsize=32)
    plt.ylabel(r'$\theta_D$', fontsize=32)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.axvline(x=0.2, color='blue', linestyle='--')
    plt.axhline(y=0.0973, color='blue', linestyle='--')

    # tight_layout() before the tight bbox crop: without it the axes stay at the
    # default subplot margins, the cropped page comes out ~15% smaller than the
    # LifeGate heatmaps, and LaTeX scales this figure up so identical fontsize=
    # values render larger. Matches visualize_policy_level.py.
    plt.tight_layout()
    plt.savefig(path, dpi=800, bbox_inches='tight')
    plt.close()


def plot_iou_heatmap(ious, path):
    plt.figure(figsize=(8, 6))
    plt.imshow(ious.T, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='RdYlGn_r')
    plt.colorbar(label='IoU')
    plt.title('IoU Heatmap')
    plt.xlabel('Zeta')
    plt.ylabel('Death Threshold')
    plt.savefig(path, dpi=500)
    plt.close()

def visualize_svp_and_bad_sizes(zeta_vals, dt_vals, svp_map, bad_map,
                                save_path="figures/svp_vs_bad_sizes_fixed.pdf"):
    """
    One plot, two y-axes:
      - Green (left y): SVP size vs zeta (mean over death_threshold)
      - Red   (right y): DeD eliminated size vs death_threshold (mean over zeta)
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

    # Times-like serif fonts + serif math (ensures θ_D subscripting works)
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

    # Tick label sizes (all axes tickers fontsize 20)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelcolor="green", labelsize=20)

    # Right axis: DeD eliminated (red)
    ax2 = ax1.twinx()
    ax2.plot(dt_vals, bad_curve, color="red", linewidth=2)
    ax2.set_ylabel("Avg DeD Eliminated Actions per State", color="red", fontsize=32)
    ax2.tick_params(axis="y", labelcolor="red", labelsize=20)

    # Re-set y-ticks explicitly (keeps current tick locations)
    ax1.set_yticks(list(ax1.get_yticks()))
    ax2.set_yticks(list(ax2.get_yticks()))


    x_min = min(float(np.min(zeta_vals)), float(np.min(dt_vals)))
    x_max = max(float(np.max(zeta_vals)), float(np.max(dt_vals)))
    ax1.set_xlim(x_min, x_max)

    # Legend (green/red) with math subscripts
    legend_handles = [
        Line2D([0], [0], color="green", lw=2, label=r"SVP size vs. $\zeta$"),
        Line2D([0], [0], color="red",   lw=2, label=r"DeD size vs. $\theta_D$"),
    ]
    ax1.legend(handles=legend_handles, loc="best", fontsize=20)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=800)
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    results_dir = 'results/'
    visualizations_dir = 'visualizations/'
    svp_sizes = np.load(results_dir + 'svp_sizes.npy')
    ded_sizes = np.load(results_dir + 'ded_sizes.npy')
    conflict_fractions = np.load(results_dir + 'conflict_fractions.npy')
    ious = np.load(results_dir + 'ious.npy')

    # plot_svp_size(svp_sizes, visualizations_dir + 'svp_sizes.png')
    # plot_ded_size(ded_sizes, visualizations_dir + 'ded_sizes.png')
    plot_conflict_fraction_heatmap(conflict_fractions, visualizations_dir + 'conflict_fraction_heatmap.pdf')
    # plot_iou_heatmap(ious, visualizations_dir + 'iou_heatmap.png')
    zeta_values = np.linspace(0.0, 1.0, svp_sizes.shape[0])
    death_thresholds = np.linspace(0.0, 1.0, ded_sizes.shape[0])
    visualize_svp_and_bad_sizes(
        zeta_vals=zeta_values,
        dt_vals=death_thresholds,
        svp_map=svp_sizes.reshape(1, -1).repeat(len(death_thresholds), axis=0),
        bad_map=ded_sizes.reshape(-1, 1).repeat(len(zeta_values), axis=1),
        save_path="visualizations/svp_vs_ded_sizes.pdf"
    )

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lifegate import LifeGate
from svp import value_iter, value_iter_near_greedy_prob, V2Q, value_iter_near_greedy


def visualize_lifegate(barrier_states, lifegate_states, dead_end, deads_states):
    grid_size = 10
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # draw a grid
    ax.set_xticks(np.arange(0, grid_size+1, 1))
    ax.set_yticks(np.arange(0, grid_size+1, 1))
    ax.grid(which='both', color='black', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    def draw_cells(states, color):
        for s in states:
            y = s // grid_size
            x = s % grid_size
            # draw each cell with a black border
            rect = patches.Rectangle(
                (x, y), 1, 1,
                facecolor=color,
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)

    draw_cells(barrier_states,  "gray")
    draw_cells(lifegate_states, "blue")
    draw_cells(dead_end,        "yellow")
    draw_cells(deads_states,    "red")

    plt.tight_layout()
    plt.savefig('figures/lifegate_environment.pdf', bbox_inches='tight', dpi=800)
    plt.show()


def visualize_v_grid(grid, title="V-Grid Heatmap", cmap="RdBu", vmin=None, vmax=None):
    plt.imshow(grid, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Value")
    plt.title(title)
    plt.axis("on")
    plt.savefig(f'figures/{title}.pdf', bbox_inches='tight')
    plt.show()


def visualize_svp(π, barrier_states, lifegate_states, dead_end, deads_states):
    grid_size = 10
    fig, ax = plt.subplots(figsize=(6, 6))
    arrow_offsets = {
        0: (0, 0),
        1: (0, -0.4),
        2: (0, 0.4),
        3: (-0.4, 0),
        4: (0.4, 0),
    }

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('on')

    # Draw color regions
    def draw_cells(states, color):
        for s in states:
            y = s // grid_size
            x = s % grid_size
            rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=color)
            ax.add_patch(rect)

    draw_cells(barrier_states, "gray")
    draw_cells(lifegate_states, "blue")
    draw_cells(dead_end, "yellow")
    draw_cells(deads_states, "red")

    # Draw arrows
    arrow_style = dict(arrowstyle="->", color="green", lw=2)
    for y in range(grid_size):
        for x in range(grid_size):
            s = y * grid_size + x
            if s in barrier_states:
                continue
            for a in range(5):
                if π[s, a] == 1:
                    dx, dy = arrow_offsets[a]
                    if a == 0:
                        ax.plot(x + 0.5, y + 0.5, 'o', markersize=4, color="green")
                    else:
                        x_end = x + 0.5 + dx
                        y_end = y + 0.5 + dy
                        ax.annotate("", xy=(x_end, y_end), xytext=(x + 0.5, y + 0.5), arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig('figures/zeta = 0.1 svp policy.pdf', bbox_inches='tight')
    plt.show()


def visualize_ded(Q_d, barrier_states, lifegate_states, dead_end, deads_states, death_threshold=0.7, title="Dead-End Policy"):
    grid_size = 10
    fig, ax = plt.subplots(figsize=(6, 6))
    arrow_offsets = {
        0: (0, 0),
        1: (0, -0.4),
        2: (0, 0.4),
        3: (-0.4, 0),
        4: (0.4, 0),
    }

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('on')

    # Draw color regions
    def draw_cells(states, color):
        for s in states:
            y = s // grid_size
            x = s % grid_size
            rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=color)
            ax.add_patch(rect)

    draw_cells(barrier_states, "gray")
    draw_cells(lifegate_states, "blue")
    draw_cells(dead_end, "yellow")
    draw_cells(deads_states, "red")

    # Draw arrows
    arrow_style = dict(arrowstyle="->", color="red", lw=2)
    for y in range(grid_size):
        for x in range(grid_size):
            s = y * grid_size + x
            for a in range(5):
                if Q_d[s, a] <= -death_threshold:
                    dx, dy = arrow_offsets[a]
                    if a == 0:
                        ax.plot(x + 0.5, y + 0.5, 'o', markersize=4, color="red")
                    else:
                        x_end = x + 0.5 + dx
                        y_end = y + 0.5 + dy
                        ax.annotate("", xy=(x_end, y_end), xytext=(x + 0.5, y + 0.5), arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig('figures/theta = 0.65 ded policy.pdf', bbox_inches='tight')
    plt.show()


def load_results(filename="trained_results.pkl"):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results


def main():


    barrier_states = [0, 1, 2, 3, 4, 51, 52, 53, 54]
    lifegate_states = [5, 6, 7]
    dead_ends = [45, 46, 47, 48, 55, 56, 57, 58, 65, 66, 67, 68, 75, 76, 77, 78, 85, 86, 87, 88, 95, 96, 97, 98]
    deads_states = [8, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

    # # Visualize Lifegate Environment
    visualize_lifegate(barrier_states, lifegate_states, dead_ends, deads_states)
    #
    # # Visualize V function grids
    # visualize_v_grid(V_SVP_Grid, title="V_SVP Grid", vmin=V_SVP.min(), vmax=V_SVP.max())
    # visualize_v_grid(V_r_Grid, title="V_R Grid", cmap="Blues", vmin=V_r.min(), vmax=V_r.max())
    # visualize_v_grid(V_d_Grid, title="V_D Grid", cmap="Reds_r", vmin=V_d.min(), vmax=V_d.max())

    # # Visualize policies
    # visualize_svp(π_SVP, barrier_states, lifegate_states, dead_ends, deads_states)
    # visualize_ded(Q_d, barrier_states, lifegate_states, dead_ends, deads_states, death_threshold=0.65)

if __name__ == "__main__":
    main()

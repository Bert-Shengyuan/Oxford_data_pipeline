"""
generate_subspace_schematics.py
================================

Illustrative (non-fitted) poster schematics of the two limiting regimes of
inter-regional communication-subspace geometry, anchored at a shared pivot
region (MOp), as used in the pCCA communication-subspace framework
(cf. Gonzalez et al. 2026).

Each figure is a 2-row x 1-column panel (stacked, for a tall narrow poster
column):

    Panel (0,0)  bipartite weight diagram  [top]
                 Three MOp pivot neurons (N1, N2, N3), arranged vertically
                 along a dashed central axis, each projecting with weight
                 w_L[i] to a "sens Thal"-side canonical variable and weight
                 w_R[i] to a "motor Thal"-side canonical variable. Line
                 width/opacity encode |w|.

    Panel (1,0)  3-D subspace-geometry cartoon  [bottom]
                 Two *vertical* planes embedded in the 3-D MOp neuron-space
                 (axes MOp N1, MOp N2, MOp N3), each plane being the linear
                 span of {e_z, e_theta} with e_theta = (cos theta, sin theta, 0).
                 The angle between the two planes' azimuths, |theta_L - theta_R|,
                 is the cartoon analogue of the principal angle theta between
                 the two fitted communication subspaces:

                     cos(theta) = |w_L . w_R| / (||w_L|| ||w_R||)

Figure 1 -- ORTHOGONAL regime
    Neuron 1 dominates the sens-Thal channel, neuron 2 dominates the
    motor-Thal channel (disjoint supports) -> w_L, w_R nearly orthogonal
    -> the two planes are drawn ~90 deg apart in azimuth.

Figure 2 -- PARALLEL (shared-channel) regime
    Neuron 1 dominates BOTH channels -> w_L, w_R nearly collinear -> the
    two planes are drawn at a small azimuthal offset (near-parallel).

Neuron 3 always carries small, independently-drawn random weight on each
side (a "noise" dimension irrelevant to the leading communication mode).

NOTE: all weights/angles below are illustrative constants chosen to make
the intended qualitative contrast (orthogonal vs. parallel) unambiguous;
they are NOT derived from a model fit and should not be read as data.

Outputs (PDF + PNG, 300 dpi) are written to /mnt/user-data/outputs/:
    fig1_orthogonal_subspaces.[pdf|png]
    fig2_parallel_subspaces.[pdf|png]
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' proj)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------------------
# Poster-scale constants
# ---------------------------------------------------------------------------
# Figures are sized for a tall, narrow column on an A0 poster (~ half of a
# three-column A0 layout). Font sizes are pre-multiplied so that, once the
# PDF is placed at its native size, text is legible at normal poster
# viewing distance (~1-2 m).
POSTER_SCALE = 2.6
_FS_FLOOR = 14.0

BLUE = "#3355A6"       # sens-Thal target / "MOp-sens Thal" subspace
RED = "#B0464F"        # motor-Thal target / "MOp-motor Thal" subspace
BLUE_LINE = "#6D8FD1"
RED_LINE = "#CC7E88"
NODE_EDGE = "black"

RNG = np.random.default_rng(42)  # fixed seed: reproducible schematic


def _fs(base):
    """Poster-scaled font size with a legibility floor."""
    return max(base * POSTER_SCALE, _FS_FLOOR)


# ---------------------------------------------------------------------------
# Panel (0,0): bipartite weight schematic
# ---------------------------------------------------------------------------
def draw_bipartite_panel(
    ax,
    left_weights,
    right_weights,
    left_label="sens Thal",
    right_label="motor Thal",
    pivot_label="MOp neurons",
    neuron_labels=("N1", "N2", "N3"),
):
    """
    Draw a 3-neuron bipartite communication-weight schematic.

    Three pivot neurons sit on a dashed vertical central axis (the pivot
    region). Neuron i sends a weighted line of width/opacity ~ |w_left[i]|
    to the left target node and ~ |w_right[i]| to the right target node.

    Parameters
    ----------
    left_weights, right_weights : array-like, shape (3,)
        Signed or unsigned canonical weights for neurons N1, N2, N3 toward
        the left- and right-side canonical variable, respectively.
    """
    left_weights = np.asarray(left_weights, dtype=float)
    right_weights = np.asarray(right_weights, dtype=float)
    n = len(left_weights)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.35, 1.65)
    ax.axis("off")
    ax.set_aspect("equal")

    x_left, x_right, x_pivot = -1.0, 1.0, 0.0
    y_target = 0.0
    y_pivot = np.linspace(0.62, -0.62, n)

    # header: arrow line on top, pivot-region name below it (reference order)
    ax.text(-0.05, 1.32, f"$\\leftarrow$ {left_label}", color="black",
            ha="right", va="bottom", fontsize=_fs(14), fontweight="bold")
    ax.text(0.05, 1.32, f"{right_label} $\\rightarrow$", color="black",
            ha="left", va="bottom", fontsize=_fs(14), fontweight="bold")
    ax.text(0, 1.06, pivot_label, ha="center", va="bottom",
            fontsize=_fs(14), fontweight="bold")

    # dashed central pivot axis
    ax.plot(
        [0, 0], [-0.95, 0.95],
        linestyle=(0, (2, 2)), color="black", linewidth=2.2, zorder=1,
    )

    # shared normalization so line styling is comparable within the panel
    wmax = np.abs(np.concatenate([left_weights, right_weights])).max()
    wmax = wmax if wmax > 0 else 1.0

    def _style(w):
        f = np.clip(abs(w) / wmax, 0.05, 1.0)
        return dict(linewidth=0.8 + 6.5 * f, alpha=0.18 + 0.78 * f)

    for i in range(n):
        yp = y_pivot[i]
        ax.plot(
            [x_left, x_pivot], [y_target, yp],
            color=BLUE_LINE, solid_capstyle="round", zorder=2,
            **_style(left_weights[i]),
        )
        ax.plot(
            [x_pivot, x_right], [yp, y_target],
            color=RED_LINE, solid_capstyle="round", zorder=2,
            **_style(right_weights[i]),
        )

    # pivot neuron markers + labels
    for i in range(n):
        ax.scatter(
            [x_pivot], [y_pivot[i]], s=110, color="black",
            edgecolor="white", linewidth=0.7, zorder=4,
        )
        ax.text(
            0.07, y_pivot[i], neuron_labels[i],
            fontsize=_fs(11), va="center", ha="left", zorder=5,
        )

    # target ("canonical variable") nodes
    ax.scatter([x_left], [y_target], s=2200, color=BLUE, edgecolor=NODE_EDGE,
               linewidth=1.5, zorder=3)
    ax.scatter([x_right], [y_target], s=2200, color=RED, edgecolor=NODE_EDGE,
               linewidth=1.5, zorder=3)
    ax.text(x_left, y_target - 0.30, left_label, color=BLUE,
            fontsize=_fs(11), fontweight="bold", ha="center", va="top")
    ax.text(x_right, y_target - 0.30, right_label, color=RED,
            fontsize=_fs(11), fontweight="bold", ha="center", va="top")


# ---------------------------------------------------------------------------
# Panel (0,1): 3-D subspace-plane cartoon
# ---------------------------------------------------------------------------
def draw_subspace_panel(
    ax,
    theta_left_deg,
    theta_right_deg,
    left_label="MOp\u2013sensory Thal",
    right_label="MOp\u2013motor Thal",
    axis_prefix="MOp",
):
    """
    Cartoon of two *vertical* communication-subspace planes in the 3-D
    neuron-state space of the pivot region.

    Each plane is span{e_z, e_theta}, e_theta = (cos theta, sin theta, 0).
    |theta_left - theta_right| plays the role of the principal angle
    between the two fitted subspaces: ~90 deg -> orthogonal (Figure 1),
    ~small -> parallel / shared channel (Figure 2).
    """
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    def _plane_verts(theta_deg, half_w=1.05, half_h=1.05):
        th = np.deg2rad(theta_deg)
        d = np.array([np.cos(th), np.sin(th), 0.0])
        z = np.array([0.0, 0.0, half_h])
        c = np.zeros(3)
        return [c - half_w * d - z, c + half_w * d - z,
                c + half_w * d + z, c - half_w * d + z]

    for theta, color in ((theta_left_deg, BLUE), (theta_right_deg, RED)):
        poly = Poly3DCollection(
            [_plane_verts(theta)], facecolor=color, edgecolor=color,
            linewidth=1.6, alpha=0.32,
        )
        ax.add_collection3d(poly)

    # a tilted closed neural trajectory threading both planes (purely
    # decorative, echoing the reference figure's population trajectory)
    t = np.linspace(0, 2 * np.pi, 240)
    loop = np.stack([0.55 * np.cos(t), 0.55 * np.sin(t), 0.16 * np.sin(2 * t)])
    rot = np.deg2rad(22)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rot), -np.sin(rot)],
                   [0, np.sin(rot), np.cos(rot)]])
    loop = Rx @ loop
    ax.plot(loop[0], loop[1], loop[2], color="black", linewidth=2.6, zorder=10)

    # Dim1 / Dim2 arrows on the blue (left) plane, echoing reference style
    th0 = np.deg2rad(theta_left_deg)
    d0 = np.array([np.cos(th0), np.sin(th0), 0.0])
    origin = np.array([0.0, 0.0, -0.12])
    ax.quiver(*origin, *(0.42 * np.array([0, 0, 1.0])), color="black",
              linewidth=1.7, arrow_length_ratio=0.22, zorder=11)
    ax.text(*(origin + 0.52 * np.array([0, 0, 1.0])), "Dim1",
            fontsize=_fs(12), ha="center", zorder=11)
    ax.quiver(*origin, *(0.42 * d0), color="black", linewidth=1.7,
              arrow_length_ratio=0.22, zorder=11)
    ax.text(*(origin + 0.52 * d0), "Dim2", fontsize=_fs(12), ha="center", zorder=11)

    ax.grid(False)  # 关闭网格线

    ax.set_xlabel(f"{axis_prefix} N1", fontsize=_fs(14), labelpad=8)
    ax.set_ylabel(f"{axis_prefix} N2", fontsize=_fs(14), labelpad=8)
    ax.set_zlabel(f"{axis_prefix} N3", fontsize=_fs(14), labelpad=4)

    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)
    ax.tick_params(labelsize=_fs(0))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=-55)

    proxy_blue = mpatches.Patch(color=BLUE, alpha=0.55, label=left_label)
    proxy_red = mpatches.Patch(color=RED, alpha=0.55, label=right_label)
    ax.legend(
        handles=[proxy_blue, proxy_red], loc="upper left",
        bbox_to_anchor=(-0.08, 1.06), fontsize=_fs(12), frameon=False,
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_figure(save_prefix, left_w, right_w, theta_left, theta_right, title):
    fig = plt.figure(figsize=(9.0, 16.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.12)

    ax0 = fig.add_subplot(gs[0, 0])
    draw_bipartite_panel(ax0, left_w, right_w)

    ax1 = fig.add_subplot(gs[1, 0], projection="3d")
    draw_subspace_panel(ax1, theta_left, theta_right)

    fig.suptitle(title, fontsize=_fs(13), fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.03)

    # fig.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = "/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Figure 1: orthogonal regime ----------------
    left_w1 = np.array([0.92, 0.15, RNG.uniform(0.08, 0.20)])
    right_w1 = np.array([0.15, 0.92, RNG.uniform(0.08, 0.20)])
    build_figure(
        os.path.join(out_dir, "fig1_orthogonal_subspaces"),
        left_w1, right_w1,
        theta_left=0.0, theta_right=90.0,
        title="Orthogonal private subspace",
    )

    # ---------------- Figure 2: parallel (shared) regime ----------------
    left_w2 = np.array([0.90, 0.16, RNG.uniform(0.06, 0.18)])
    right_w2 = np.array([0.85, 0.18, RNG.uniform(0.06, 0.18)])
    build_figure(
        os.path.join(out_dir, "fig2_parallel_subspaces"),
        left_w2, right_w2,
        theta_left=8.0, theta_right=24.0,
        title="Parallel private subspace",
    )

    print("Wrote:")
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("fig1_") or f.startswith("fig2_"):
            print(" ", os.path.join(out_dir, f))


if __name__ == "__main__":
    main()
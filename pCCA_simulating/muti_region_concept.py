"""
Reworked interaction diagram: Figure 1 (triangle A-B-C) -> Figure 2 style
(row of upper source points interacting with A and C).

Matches Figure 1's colour palette and circle-to-arrow size ratio, measured
directly from the reference images:
    R_MAIN / R_SMALL  ~ 31 px / 17.5 px ~ 0.56
    d(A,C) / R_MAIN    ~ 169.5 px / 31 px ~ 5.47
    arrow colour       #969797 (identical in both reference figures)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# --------------------------------------------------------------------------
# Style constants, taken from Figure 1 / Figure 2
# --------------------------------------------------------------------------
ORANGE     = "#E69636"   # region A
BLUE       = "#3265A9"   # region C
ARROW_GRAY = "#f5f5f5"   # bidirectional interaction arrows
ARROW_GRAY_main = "#969797"
DARK_GREEN = "#2E7D32"   # the two randomly "highlighted" source points

R_MAIN       = 0.9            # radius of A and C  (reference unit)
R_SMALL      = 0.66 * R_MAIN        # radius of upper source points
AC_DISTANCE  = 5.47 * R_MAIN        # A-C centre-to-centre distance
Y_TOP_OFFSET = 3.2  * R_MAIN        # vertical offset of the upper row

ARROW_LW   = 2.6      # linewidth, proportioned to R_MAIN = 1
ARROW_HEAD = 14        # mutation_scale (arrowhead size)
FONT_SIZE  = 30


def build_figure(n_points=6, seed=None, savepath="reworked_figure.png"):
    """
    n_points : number of upper source points interacting with A and C
    seed     : optional int, for reproducible random colour/selection draws
    savepath : output PNG path
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- positions ----------------
    xA, xC = 0.0, AC_DISTANCE
    y_main = 0.0
    x_mid = (xA + xC) / 2.0

    span = AC_DISTANCE + 4.0 * R_MAIN          # let the row overhang A and C
    xs_top = np.linspace(x_mid - span / 2, x_mid + span / 2, n_points)
    y_top = y_main + Y_TOP_OFFSET

    # ---------------- A and C circles ----------------
    circA = Circle((xA, y_main), R_MAIN, facecolor=ORANGE, edgecolor="none", zorder=3)
    circC = Circle((xC, y_main), R_MAIN, facecolor=BLUE, edgecolor="none", zorder=3)

    # ---------------- colour assignment for the upper points ----------------
    dark_idx = set(random.sample(range(n_points), k=min(2, n_points)))
    cmap = plt.get_cmap("Greens")

    top_circles = []
    for i, x in enumerate(xs_top):
        if i in dark_idx:
            color, alpha = DARK_GREEN, 1.0
        else:
            color = cmap(np.random.uniform(0.30, 0.65))
            alpha = np.random.uniform(0.35, 0.85)
        c = Circle((x, y_top), R_SMALL, facecolor=color, edgecolor="none",
                   alpha=alpha, zorder=3)
        top_circles.append(c)

    # ---------------- arrows (drawn first, clipped to circle boundaries) ----------------
    def add_arrow(pA, pB, patchA, patchB,color=ARROW_GRAY):
        arr = FancyArrowPatch(pA, pB, arrowstyle="<->", mutation_scale=ARROW_HEAD,
                               linewidth=ARROW_LW, color=color,
                               patchA=patchA, patchB=patchB,
                               shrinkA=2, shrinkB=2, zorder=1)
        ax.add_patch(arr)

    add_arrow((xA, y_main), (xC, y_main), circA, circC, color=ARROW_GRAY_main)          # A <-> C
    for c in top_circles:                                        # each point <-> A, <-> C
        add_arrow(c.center, (xA, y_main), c, circA)
        add_arrow(c.center, (xC, y_main), c, circC)

    # ---------------- draw circles on top of arrows ----------------
    ax.add_patch(circA)
    ax.add_patch(circC)
    for c in top_circles:
        ax.add_patch(c)

    # ---------------- labels ----------------
    ax.text(xA, y_main - R_MAIN - 0.45, "MOp", ha="center", va="top", fontsize=FONT_SIZE)
    ax.text(xC, y_main - R_MAIN - 0.45, "motor Thal", ha="center", va="top", fontsize=FONT_SIZE)

    # ---------------- limits ----------------
    ax.set_xlim(x_mid - span / 2 - 1.2, x_mid + span / 2 + 1.2)
    ax.set_ylim(y_main - R_MAIN - 1.0, y_top + R_SMALL + 0.6)

    fig.tight_layout()
    fig.savefig(savepath, dpi=300, transparent=True)
    plt.close(fig)
    return savepath


if __name__ == "__main__":
    build_figure(n_points=6, seed=None, savepath="/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/muti-region-concept")
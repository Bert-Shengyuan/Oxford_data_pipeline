#!/usr/bin/env python3
"""
visualize_reward_kernels.py
----------------------------
Self-contained visualisation of the two reward predictors
(reward_presence  and  reward_consumption) as they appear
within a single trial, using the same parameters and logic
as pCCA_latent_extrenal_variable_bar.py.

Run:
    python visualize_reward_kernels.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# ── Parameters (mirror pCCA_latent_extrenal_variable_bar.py) ──────────────
BEHAVIOR_FS               = 50.0          # Hz
BEHAVIOR_T_OFFSET         = -1.0          # t=0 is movement onset
BEHAVIOR_TIME_RANGE_S     = (-1.0, 2.0)  # full trial window

REWARD_PRESENCE_WINDOW_S    = (0.0, 0.5)
REWARD_CONSUMPTION_WINDOW_S = (0.5, 1.5)
N_REWARD_CONSUMPTION_BASIS  = 5

REWARD_SPLINE_DEGREE        = 2


# ── Core functions (copied verbatim from the main script) ──────────────────
def _bspline_basis_matrix(t: np.ndarray, n_basis: int,
                           degree: int = REWARD_SPLINE_DEGREE) -> np.ndarray:
    t_min, t_max = float(t.min()), float(t.max())
    n_basis= n_basis+2
    n_interior   = max(n_basis - degree - 1, 0)
    interior_knots = (np.linspace(t_min, t_max, n_interior + 2)[1:-1]
                      if n_interior > 0 else np.array([]))
    knots = np.concatenate([
        np.full(degree + 1, t_min),
        interior_knots,
        np.full(degree + 1, t_max),
    ])
    n_coef = len(knots) - degree - 1
    basis  = np.zeros((n_coef, t.size))
    for i in range(n_coef):
        c    = np.zeros(n_coef); c[i] = 1.3
        spl  = BSpline(knots, c, degree, extrapolate=False)
        basis[i] = np.nan_to_num(spl(t), nan=0.0)

    basis_end = basis[1:-1,:]
    return basis_end   # (n_basis, T_window)


def build_presence(t: np.ndarray) -> np.ndarray:
    lo, hi = REWARD_PRESENCE_WINDOW_S
    return ((t >= lo) & (t <= hi)).astype(float)   # (T,)


def build_consumption(t: np.ndarray) -> np.ndarray:
    lo, hi  = REWARD_CONSUMPTION_WINDOW_S
    n_basis = N_REWARD_CONSUMPTION_BASIS
    mask    = (t >= lo) & (t <= hi)
    out    = np.zeros((n_basis, t.size))
    if mask.sum() >= (n_basis + REWARD_SPLINE_DEGREE + 1):
        out[:, mask] = _bspline_basis_matrix(t[mask], n_basis)
    return out     # (5, T)


# ── Time axis ──────────────────────────────────────────────────────────────
T_total = int((BEHAVIOR_TIME_RANGE_S[1] - BEHAVIOR_TIME_RANGE_S[0]) * BEHAVIOR_FS) + 1
t = np.linspace(BEHAVIOR_TIME_RANGE_S[0], BEHAVIOR_TIME_RANGE_S[1], T_total)

presence    = build_presence(t)       # (T,)
consumption = build_consumption(t)    # (5, T)

# ── Knot positions for annotation ─────────────────────────────────────────
lo_c, hi_c = REWARD_CONSUMPTION_WINDOW_S
n_int = max(N_REWARD_CONSUMPTION_BASIS - REWARD_SPLINE_DEGREE - 1, 0)
interior_knots = (np.linspace(lo_c, hi_c, n_int + 2)[1:-1]
                  if n_int > 0 else np.array([]))

# ── Colour palette ─────────────────────────────────────────────────────────
COLOR_PRESENCE    = "#55A868"
# COLORS_SPLINE     = ["#6A5ACD", "#E07B39", "#2196A6", "#C44E52", "#8172B2"]
ALPHA_FILL        = 0.18

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 1, figsize=(9, 6),
    gridspec_kw=dict(hspace=0.52),
)

# ---- Panel A: reward_presence (Boxcar) ------------------------------------
ax = axes[0]
ax.fill_between(t, presence, color=COLOR_PRESENCE, alpha=ALPHA_FILL)
ax.plot(t, presence, color=COLOR_PRESENCE, lw=2.2, label="reward_presence")

ax.axvline(0,   color="#888", lw=1.0, ls="--")
ax.axvline(0.5, color="#aaa", lw=0.9, ls=":")
ax.set_xlim(t[0], t[-1])
ax.set_ylim(-0.08, 1.25)
ax.set_ylabel("Amplitude", fontsize=11)
ax.set_title("Reward_presence  (step / Boxcar predictor, $C=1$)",
             fontsize=12, fontweight="bold", loc="left")
ax.text(0.25, 1.08, f"[{REWARD_PRESENCE_WINDOW_S[0]}, {REWARD_PRESENCE_WINDOW_S[1]}] s",
        ha="center", va="bottom", fontsize=9.5, color=COLOR_PRESENCE)
# ax.text(0, -0.06, "t = 0\n(movement onset)", ha="center", va="top",
#         fontsize=8.5, color="#666")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.set_xlabel("Time relative to movement onset (s)", fontsize=10)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ---- Panel B: reward_consumption (B-Spline basis) -------------------------
ax = axes[1]
for i in range(N_REWARD_CONSUMPTION_BASIS):
    # ax.fill_between(t, consumption[i], alpha=ALPHA_FILL, color=COLORS_SPLINE[i])
    # ax.plot(t, consumption[i], lw=2.0, color=COLORS_SPLINE[i],
    #         label=f"$B_{{i={i+1},\\ d=3}}$")
    ax.fill_between(t, consumption[i], alpha=ALPHA_FILL)
    ax.plot(t, consumption[i], lw=2.0,
            label=f"$B_{{i={i+1},\\ d=3}}$")

# Mark boundary & interior knots
boundary_knots = [lo_c, hi_c]
for xk in boundary_knots:
    ax.axvline(xk, color="#aaa", lw=0.9, ls=":")
for xk in interior_knots:
    ax.axvline(xk, color="#bbb", lw=1.2, ls="--")
    ax.text(xk, -0.06, f"ξ={xk:.2f} s", ha="center", va="top",
            fontsize=8.5, color="#555")

ax.axvline(0, color="#888", lw=1.0, ls="--")
ax.set_xlim(t[0], t[-1])
ax.set_ylim(-0.10, 1.15)
ax.set_ylabel("Amplitude", fontsize=11)
ax.set_xlabel("Time relative to movement onset (s)", fontsize=10)
ax.set_title(
    f"Reward_consumption  "
    f"(cubic B-Spline basis, $K={N_REWARD_CONSUMPTION_BASIS}$, "
    f"$d={REWARD_SPLINE_DEGREE}$, $n_{{\\mathrm{{interior}}}}=1$)",
    fontsize=12, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=10, ncol=5,
          loc="upper right", bbox_to_anchor=(1.0, 1.02))
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# fig.suptitle("Reward Kernel Design Matrix — Single Trial View\n"
#              r"(trial window: $[-1.0,\ 2.0]$ s,  $f_s = 50$ Hz)",
#              fontsize=13, y=1.01)

# plt.savefig("/mnt/user-data/outputs/reward_kernels.png",
#             dpi=180, bbox_inches="tight")
# print("Saved: reward_kernels.png")
plt.show()
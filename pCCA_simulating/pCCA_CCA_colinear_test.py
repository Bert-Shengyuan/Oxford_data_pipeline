#!/usr/bin/env python3
"""
pcca_benchmark_simulations.py
===========================================
Benchmark suite for partial canonical correlation analysis (pCCA).

Part 1  Canonical correctness  (Simulations 1-3)
  Sim 1  True shared hub                -> expected angle ~  0 deg
  Sim 2  Disjoint communication ports   -> expected angle ~ 90 deg
  Sim 3  False-hub illusion (rho_BC!=0) -> angle collapses to 0 deg

Part 2  Confound characterisation & correction  (Tasks 2.1-2.3c)
  2.1  Shared internal noise in A (off-diagonal Sigma_AA)
  2.2  External stimulus drive
  2.3a PSTH-residual separation
  2.3b Explicit external conditioning in pCCA
  2.3c Time-lagged canonical correlation (synaptic delay detection)

Part 3  Cascade topologies (Simulations 4a-4c)
  4a  Z -> A -> B pure cascade
  4b  Cascade with direct Z-B confound coupling
  4c  Cascade with shared internal noise in A
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Global parameters
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
N = 5_000
SIGMA = 0.0
LAMBDA_REG = 1e-4
CCA_ALPHA = 1e-4

C_AB = '#2166ac'  # blue   – pCCA(A,B|C) canonical direction
C_AC = '#d6604d'  # red    – pCCA(A,C|B) canonical direction
C_RAW = '#969696'  # grey   – raw cross-covariance direction (reference arrow)
C_HUB = '#4dac26'  # green  – hub / shared signal
C_COMP = '#762a83'  # purple – compensatory artefact weight
C_STIM = '#f4a582'  # peach  – external stimulus
C_CAUS = '#1b7837'  # forest – causal delay
C_COMM = '#e08214'  # amber  – instantaneous common drive
C_CCA = '#4393c3'  # light blue – CCA unconditional baseline

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
})


# =============================================================================
# SECTION 1 -- CORE UTILITIES
# =============================================================================

def partial_residuals(X, Z, lam=LAMBDA_REG):
    q = Z.shape[1]
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(q), Z.T @ X)
    return X - Z @ beta


def cca_svd(X, Y, k=1, alpha=CCA_ALPHA):
    Nc = X.shape[0]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Sxx = Xc.T @ Xc / Nc
    Syy = Yc.T @ Yc / Nc
    Sxy = Xc.T @ Yc / Nc

    def inv_sqrt(S):
        lam, V = np.linalg.eigh(S)
        lam = np.maximum(lam, 0.0) + alpha
        return V @ np.diag(lam ** -0.5) @ V.T

    Six, Siy = inv_sqrt(Sxx), inv_sqrt(Syy)
    U, s, Vt = np.linalg.svd(Six @ Sxy @ Siy, full_matrices=False)
    ke = min(k, len(s))
    return Six @ U[:, :ke], Siy @ Vt.T[:, :ke], np.clip(s[:ke], 0.0, 1.0)


def pcca(X, Y, Z=None, k=1, lam=LAMBDA_REG, alpha=CCA_ALPHA):
    if Z is not None and Z.ndim == 2 and Z.shape[1] > 0:
        Xr, Yr = partial_residuals(X, Z, lam), partial_residuals(Y, Z, lam)
    else:
        Xr, Yr = X - X.mean(0), Y - Y.mean(0)
    Wx, Wy, rho = cca_svd(Xr, Yr, k=k, alpha=alpha)
    return Wx, Wy, rho, Xr, Yr


def pa_deg(w1, w2):
    d = np.linalg.norm(w1) * np.linalg.norm(w2)
    if d < 1e-12: return np.nan
    return float(np.degrees(np.arccos(np.clip(abs(float(w1 @ w2)) / d, 0.0, 1.0))))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def sigma_dir(A, B):
    return ((A - A.mean(0)).T @ (B - B.mean(0)) / A.shape[0]).ravel()


def cca_simple(X, Y, k=1):
    Wx, Wy, rho, _, _ = pcca(X, Y, Z=None, k=k)
    return Wx, Wy, rho


# =============================================================================
# SECTION 2 -- DATA GENERATORS
# =============================================================================

def _rng(seed): return np.random.default_rng(seed)


def gen_sim1_a(N=N, sigma=SIGMA, seed=SEED):
    rng = _rng(seed)
    h = rng.standard_normal(N)
    A = np.c_[h + sigma * rng.standard_normal(N), sigma * rng.standard_normal(N)]
    B = (h + sigma * rng.standard_normal(N))[:, None]
    C = (h + sigma * rng.standard_normal(N))[:, None]
    return A, B, C


def gen_sim1_b(N=N, sigma=SIGMA, seed=SEED):
    rng = _rng(seed)
    h = rng.standard_normal(N)
    B = (h + sigma * rng.standard_normal(N))[:, None]
    C = (h ** 2 + sigma * rng.standard_normal(N))[:, None]
    noise_A = np.zeros(N)
    A = np.c_[B + C, noise_A]
    return A, B, C


def gen_sim1_c(N=N, sigma=SIGMA, seed=SEED):
    rng = _rng(seed)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    B = (sB + sigma * rng.standard_normal(N))[:, None]
    C = (sC + sigma * rng.standard_normal(N))[:, None]
    A = np.c_[B + C, sigma * rng.standard_normal(N)]
    return A, B, C


def gen_sim2(N=N, sigma=SIGMA, seed=SEED):
    rng = _rng(seed)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    A = np.c_[sB + rng.standard_normal(N), sC + rng.standard_normal(N)]
    B = (sB + sigma * rng.standard_normal(N))[:, None]
    C = (sC + sigma * rng.standard_normal(N))[:, None]
    return A, B, C


def gen_sim3_a(N=N, sigma=SIGMA, rho_BC=0.8, seed=SEED):
    rng = _rng(seed)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    delta = rng.standard_normal(N)
    A = np.c_[sB + sigma * rng.standard_normal(N), sC + sigma * rng.standard_normal(N)]
    B = (sB + delta + sigma * rng.standard_normal(N))[:, None]
    C = (sC + rho_BC * delta + sigma * rng.standard_normal(N))[:, None]
    return A, B, C


def gen_sim3_b(N=N, sigma=SIGMA, rho_A=0.8, seed=SEED):
    rng = _rng(seed)
    h = rng.standard_normal(N)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    A = np.c_[sB + h, sC + rho_A * h]
    B = (sB + sigma * rng.standard_normal(N))[:, None]
    C = (sC + sigma * rng.standard_normal(N))[:, None]
    return A, B, C


def gen_task21(N=N, sigma_ind=SIGMA, sigma_shared=0.0, seed=SEED):
    rng = _rng(seed)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    eta = rng.standard_normal(N)
    A = np.c_[sB + sigma_shared * eta + sigma_ind * rng.standard_normal(N),
              sC + sigma_shared * eta + sigma_ind * rng.standard_normal(N)]
    B = (sB + sigma_ind * rng.standard_normal(N))[:, None]
    C = (sC + sigma_ind * rng.standard_normal(N))[:, None]
    return A, B, C

def gen_sim4a1(N=N, sigma=SIGMA, seed=SEED):
    rng = _rng(seed)
    sZ_G = rng.standard_normal(N)
    sZB = rng.standard_normal(N)
    sA1B,sA2B= rng.standard_normal(N),rng.standard_normal(N)

    Z = (sZB  +sZ_G+ sigma * rng.standard_normal(N))[:, None]

    A = np.c_[sZB +  sA1B + sigma * rng.standard_normal(N), sZB+ sA2B + sigma * rng.standard_normal(N)]

    B = (sZB  + sA2B + sigma * rng.standard_normal(N))[:, None]

    return A, B, Z


def gen_sim4a2(N=N, sigma=SIGMA, lam=1.0, seed=SEED):

    rng = np.random.default_rng(seed)

    sZB = rng.poisson(lam, N)
    sZA1 = rng.poisson(lam, N)
    sZA2 = rng.poisson(lam, N)

    sA1B = rng.poisson(lam, N)
    sA2B = rng.poisson(lam, N)


    Z = (sZB + sigma * rng.poisson(lam, N))[:, None]

    A = np.c_[sZB + sigma * rng.poisson(lam, N), sA2B + sigma * rng.poisson(lam, N)]

    B = (sZB + sA2B + sigma * rng.poisson(lam, N))[:, None]

    return A, B, Z


# def gen_sim4a(N=N, sigma=SIGMA, seed=SEED):
#     rng = _rng(seed)
#     sZB, sZA1, sZA2 = rng.standard_normal(N),  rng.standard_normal(N), rng.standard_normal(N)
#
#     sA1B,sA2B= rng.standard_normal(N),rng.standard_normal(N)
#
#     Z = (sZB + 1*sZA1 + 0*sZA2 +sigma * rng.standard_normal(N))[:, None]
#
#     A = np.c_[sZB +  sA1B + sigma * rng.standard_normal(N), sZB+ sA2B + sigma * rng.standard_normal(N)]
#
#     B = (sZB + 0*sA1B + 1*sA2B+ sigma * rng.standard_normal(N))[:, None]
#
#     return A, B, Z

# def gen_sim4a(N=N, sigma=SIGMA, seed=SEED):
#     rng = _rng(seed)
#     sZB, sZA1, sZA2 = rng.standard_normal(N),  rng.standard_normal(N), rng.standard_normal(N)
#
#     sA1B,sA2B= rng.standard_normal(N),rng.standard_normal(N)
#
#     Z = (sZB + 1*sZA1 + 0*sZA2 +sigma * rng.standard_normal(N))[:, None]
#
#     A = np.c_[sZA1 +  sA1B + sigma * rng.standard_normal(N), sZA2+ sA2B + sigma * rng.standard_normal(N)]
#
#     B = (sZB + 0*sA1B + 1*sA2B+ sigma * rng.standard_normal(N))[:, None]
#
#     return A, B, Z


def gen_sim4b(N=N, sigma=SIGMA, rho_BZ=0.1, seed=SEED):
    rng = _rng(seed)
    sZ, sB = rng.standard_normal(N), rng.standard_normal(N)
    delta = rng.standard_normal(N)

    Z = (sZ + delta + sigma * rng.standard_normal(N))[:, None]

    A = np.c_[sZ + sigma * rng.standard_normal(N), sB + sigma * rng.standard_normal(N)]

    B = (sB + rho_BZ * delta + sigma * rng.standard_normal(N))[:, None]

    return A, B, Z


def gen_sim4c(N=N, sigma=SIGMA, rho_A=0.8, seed=SEED):
    rng = _rng(seed)
    sZ, sB = rng.standard_normal(N), rng.standard_normal(N)
    h = rng.standard_normal(N)

    A = np.c_[sZ + h, sB + rho_A * h]
    Z = (sZ + sigma * rng.standard_normal(N))[:, None]
    B = (sB + sigma * rng.standard_normal(N))[:, None]
    return A, B, Z


# =============================================================================
# SECTION 3 -- VISUALISATION PRIMITIVES
# =============================================================================

def _draw_schematic(ax, mode):
    ax.set_xlim(0, 1);
    ax.set_ylim(0, 1);
    ax.axis('off')
    bs = dict(boxstyle='round,pad=0.15', lw=1.0, zorder=3)

    def node(x, y, lbl, fc='#e0e0e0', ec='#333', fs=9.5):
        ax.text(x, y, lbl, ha='center', va='center', fontsize=fs,
                fontweight='bold', zorder=4, bbox=dict(fc=fc, ec=ec, **bs))

    def arr(x0, y0, x1, y1, col='#333', lw=1.5, lbl='', ls='solid'):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                    linestyle=ls, mutation_scale=11), zorder=2)
        if lbl:
            ax.text((x0 + x1) / 2 + 0.02, (y0 + y1) / 2 + 0.04, lbl,
                    ha='center', fontsize=7.5, color=col, fontweight='bold')

    a1x, a1y = 0.43, 0.72
    a2x, a2y = 0.43, 0.28
    Bx, By = 0.83, 0.72
    Cx, Cy = 0.83, 0.28

    if mode == 'hub':
        hx, hy = 0.10, 0.50
        node(hx, hy, 'h', fc='#d9f0d3', ec=C_HUB, fs=10)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        arr(hx + 0.07, hy + 0.04, a1x - 0.06, a1y - 0.04, col=C_HUB, lbl='h')
        arr(a1x + 0.07, a1y, Bx - 0.06, By, col=C_AB)
        arr(a1x + 0.07, a1y - 0.02, Cx - 0.08, Cy + 0.04, col=C_AC, lw=1.2)

    elif mode == 'disjoint_to_A':
        sBx, sBy = 0.10, 0.72
        sCx, sCy = 0.10, 0.28
        node(sBx, sBy, 's_B', fc='#d9f0d3', ec=C_HUB, fs=9)
        node(sCx, sCy, 's_C', fc='#c7e9c0', ec='#238b45', fs=9)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        arr(sBx + 0.07, sBy, a1x - 0.06, a1y, col=C_HUB)
        arr(sCx + 0.07, sCy, a1x - 0.06, a1y, col='#238b45')
        arr(a1x + 0.07, a1y, Bx - 0.06, By, col=C_AB)
        arr(a1x + 0.07, a1y, Cx - 0.06, Cy, col=C_AC)

    elif mode == 'disjoint':
        sBx, sBy = 0.10, 0.72
        sCx, sCy = 0.10, 0.28
        node(sBx, sBy, 's_B', fc='#d9f0d3', ec=C_HUB, fs=9)
        node(sCx, sCy, 's_C', fc='#c7e9c0', ec='#238b45', fs=9)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        arr(sBx + 0.07, sBy, a1x - 0.06, a1y, col=C_HUB)
        arr(sCx + 0.07, sCy, a2x - 0.06, a2y, col='#238b45')
        arr(a1x + 0.07, a1y, Bx - 0.06, By, col=C_AB)
        arr(a2x + 0.07, a2y, Cx - 0.06, Cy, col=C_AC)

    elif mode == 'false_hub_a':
        sBx, sBy = 0.10, 0.72
        sCx, sCy = 0.10, 0.28
        dx, dy = 0.63, 0.50
        node(sBx, sBy, 's_B', fc='#d9f0d3', ec=C_HUB, fs=9)
        node(sCx, sCy, 's_C', fc='#c7e9c0', ec='#238b45', fs=9)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        node(dx, dy, '\u03b4', fc='#e0c3e6', ec=C_COMP, fs=10)
        arr(sBx + 0.07, sBy, a1x - 0.06, a1y, col=C_HUB)
        arr(sCx + 0.07, sCy, a2x - 0.06, a2y, col='#238b45')
        arr(a1x + 0.07, a1y, Bx - 0.06, By, col=C_AB)
        arr(a2x + 0.07, a2y, Cx - 0.06, Cy, col=C_AC)
        arr(dx + 0.05, dy + 0.04, Bx - 0.05, By - 0.06, col=C_COMP, lw=2.0)
        arr(dx + 0.05, dy - 0.04, Cx - 0.05, Cy + 0.06, col=C_COMP, lw=2.0)
        ax.text(0.50, 0.03, '\u03c1_BC >> 0', ha='center', fontsize=7.5, color=C_COMP, style='italic',
                fontweight='bold')

    elif mode == 'false_hub_b':
        sBx, sBy = 0.10, 0.72
        sCx, sCy = 0.10, 0.28
        dx, dy = 0.63, 0.50
        node(sBx, sBy, 's_B', fc='#d9f0d3', ec=C_HUB, fs=9)
        node(sCx, sCy, 's_C', fc='#c7e9c0', ec='#238b45', fs=9)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        node(dx, dy, '\u03b4', fc='#e0c3e6', ec=C_COMP, fs=10)
        arr(sBx + 0.07, sBy, a1x - 0.06, a1y, col=C_HUB)
        arr(sCx + 0.07, sCy, a2x - 0.06, a2y, col='#238b45')
        arr(a1x + 0.07, a1y, Bx - 0.06, By, col=C_AB)
        arr(a2x + 0.07, a2y, Cx - 0.06, Cy, col=C_AC)
        arr(dx - 0.05, dy + 0.04, a1x + 0.05, a1y - 0.06, col=C_COMP, lw=2.0)
        arr(dx - 0.05, dy - 0.04, a2x + 0.05, a2y + 0.06, col=C_COMP, lw=2.0)
        ax.text(0.50, 0.03, '\u03c1_ATA >> 0', ha='center', fontsize=7.5, color=C_COMP, style='italic',
                fontweight='bold')

    elif mode == 'sim4a1':
        Zx, Zy = 0.15, 0.50
        a1x, a1y = 0.50, 0.72
        a2x, a2y = 0.50, 0.28
        Bx, By = 0.85, 0.50
        node(Zx, Zy, 'Z', fc='#fdd0a2', ec='#8c2d04', fs=10)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        arr(Zx + 0.06, Zy + 0.04, a1x - 0.06, a1y - 0.04, col=C_AC)
        arr(Zx + 0.06, Zy - 0.04, a2x - 0.06, a2y + 0.04, col=C_AC)
        arr(Zx + 0.07, Zy, Bx - 0.07, By, col=C_AC)
        arr(a2x + 0.06, a2y + 0.04, Bx - 0.06, By - 0.04, col=C_AB)

    elif mode == 'sim4a2':
        Zx, Zy = 0.15, 0.50
        a1x, a1y = 0.50, 0.72
        a2x, a2y = 0.50, 0.28
        Bx, By = 0.85, 0.50
        node(Zx, Zy, 'Z', fc='#fdd0a2', ec='#8c2d04', fs=10)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        arr(Zx + 0.06, Zy + 0.04, a1x - 0.06, a1y - 0.04, col=C_AC)
        arr(Zx + 0.07, Zy, Bx - 0.07, By, col=C_AC)
        arr(a2x + 0.06, a2y + 0.04, Bx - 0.06, By - 0.04, col=C_AB)

    elif mode == 'sim4b':
        Zx, Zy = 0.15, 0.50
        a1x, a1y = 0.50, 0.72
        a2x, a2y = 0.50, 0.28
        Bx, By = 0.85, 0.50
        dx, dy = 0.50, 0.10
        node(Zx, Zy, 'Z', fc='#fdd0a2', ec='#8c2d04', fs=10)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(dx, dy, '\u03b4', fc='#e0c3e6', ec=C_COMP, fs=10)
        arr(Zx + 0.06, Zy + 0.04, a1x - 0.06, a1y - 0.04, col=C_AC)
        arr(a2x + 0.06, a2y + 0.04, Bx - 0.06, By - 0.04, col=C_AB)
        arr(dx - 0.05, dy + 0.03, Zx + 0.05, Zy - 0.06, col=C_COMP, lw=2.0)
        arr(dx + 0.05, dy + 0.03, Bx - 0.05, By - 0.06, col=C_COMP, lw=2.0)

    elif mode == 'sim4c':
        Zx, Zy = 0.15, 0.50
        a1x, a1y = 0.50, 0.72
        a2x, a2y = 0.50, 0.28
        Bx, By = 0.85, 0.50
        hx, hy = 0.50, 0.50
        node(Zx, Zy, 'Z', fc='#fdd0a2', ec='#8c2d04', fs=10)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(hx, hy, 'h', fc='#d9f0d3', ec=C_HUB, fs=10)
        arr(Zx + 0.06, Zy + 0.04, a1x - 0.06, a1y - 0.04, col=C_AC)
        arr(a2x + 0.06, a2y + 0.04, Bx - 0.06, By - 0.04, col=C_AB)
        arr(hx, hy + 0.05, a1x, a1y - 0.06, col=C_HUB)
        arr(hx, hy - 0.05, a2x, a2y + 0.06, col=C_HUB)


def _compass(ax, w_AB, w_AC, sig_AB=None, sig_AC=None, angle=None, title='', lbl_AC='w_AC'):
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#c0c0c0', lw=0.8, zorder=0)
    ax.axhline(0, color='#d0d0d0', lw=0.5)
    ax.axvline(0, color='#d0d0d0', lw=0.5)
    ax.set_aspect('equal')
    lm = 1.45
    ax.set_xlim(-lm, lm)
    ax.set_ylim(-lm, lm)
    ax.set_xlabel('A\u2081  weight', fontsize=8)
    ax.set_ylabel('A\u2082  weight', fontsize=8)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    for v in [sig_AB, sig_AC]:
        if v is not None:
            uv = unit(np.asarray(v, float))
            ax.annotate("", xy=(0.78 * uv[0], 0.78 * uv[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=C_RAW,
                                        lw=1.2, linestyle='dashed', mutation_scale=10))
            # if v==0:
            #     ax.text(1.22 * uv[0], 1.22 * uv[1], 'sig_AB', ha='center', va='center',
            #             fontsize=8.5, color=C_RAW, fontweight='bold')
            # elif v==1:
            #     ax.text(1.22 * uv[0], 1.22 * uv[1], 'sig_AZ', ha='center', va='center',
            #             fontsize=8.5, color=C_RAW, fontweight='bold')

    for w, col, lbl in [(w_AB, C_AB, 'w_AB'), (w_AC, C_AC, lbl_AC)]:
        uw = unit(np.asarray(w, float))
        ax.annotate("", xy=(uw[0], uw[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2.2, mutation_scale=12))
        ax.text(1.22 * uw[0], 1.22 * uw[1], lbl, ha='center', va='center',
                fontsize=8.5, color=col, fontweight='bold')

    hdr = title
    if angle is not None:
        hdr += ('\n' if title else '') + f'\u03b8 = {angle:.1f}\u00b0'
    if hdr:
        ax.set_title(hdr, fontsize=9, pad=3, fontweight='bold')


def _wt_bars(ax, w_AB, w_AC, sig_AB=None, sig_AC=None, lbl_AC='w_AC'):
    x, bw = np.array([0.0, 1.0]), 0.20

    if sig_AB is not None and sig_AC is not None:
        sAB = unit(np.asarray(sig_AB, float))
        sAC = unit(np.asarray(sig_AC, float))
        ax.bar(x - 1.5 * bw, sAB, bw, color=C_RAW, alpha=0.55, label='\u03a3_AB raw')
        ax.bar(x - 0.5 * bw, sAC, bw, color=C_RAW, alpha=0.30, label='\u03a3_AC raw', hatch='//')
        xs = 0.5 * bw
    else:
        xs = -bw

    nAB = unit(np.asarray(w_AB, float))
    nAC = unit(np.asarray(w_AC, float))
    ax.bar(x + xs, nAB, bw, color=C_AB, alpha=0.85, label='w_AB pCCA')
    ax.bar(x + xs + bw, nAC, bw, color=C_AC, alpha=0.85, label=f'{lbl_AC} pCCA')

    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks([0, 1]);
    ax.set_xticklabels(['A\u2081', 'A\u2082'])
    ax.set_ylabel('Normalised weight', fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_xlim(-0.6, 1.85)


def _wt_bars_compare(ax, w_AB_cca, w_AC_cca, w_AB_pcca, w_AC_pcca, show_legend=True, lbl_AC='w_AC'):
    x = np.array([0.0, 1.0])
    bw = 0.17
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw

    nAB_c = unit(np.asarray(w_AB_cca, float))
    nAC_c = unit(np.asarray(w_AC_cca, float))
    nAB_p = unit(np.asarray(w_AB_pcca, float))
    nAC_p = unit(np.asarray(w_AC_pcca, float))

    ax.bar(x + offsets[0], nAB_c, bw, color=C_AB, alpha=0.45, hatch='//', label='w_AB  CCA')
    ax.bar(x + offsets[1], nAC_c, bw, color=C_AC, alpha=0.45, hatch='//', label=f'{lbl_AC}  CCA')
    ax.bar(x + offsets[2], nAB_p, bw, color=C_AB, alpha=0.90, label='w_AB  pCCA')
    ax.bar(x + offsets[3], nAC_p, bw, color=C_AC, alpha=0.90, label=f'{lbl_AC}  pCCA')

    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['A\u2081', 'A\u2082'])
    ax.set_ylabel('Normalised weight', fontsize=8)
    if show_legend:
        ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_xlim(-0.55, 1.80)


def _angle_color(theta):
    if theta < 20.0:
        return C_HUB
    elif theta > 70.0:
        return C_AB
    else:
        return C_COMP


# =============================================================================
# SECTION 5 -- CCA vs pCCA COMPARISON FIGURES
# =============================================================================

def fig1_2_cca_vs_pcca(save=True):
    print("Building Figure 1-2: CCA vs pCCA comparison (all sims including 4)...")

    sims = [
        ('Sim 1a \u2013 Linear Shared Hub', gen_sim1_a(), 'hub'),
        ('Sim 1b \u2013 Non-linear Hub (Cov\u22480)', gen_sim1_b(), 'hub'),
        ('Sim 1c \u2013 Additive Ports', gen_sim1_c(), 'disjoint_to_A'),
        ('Sim 2  \u2013 Disjoint Ports', gen_sim2(), 'disjoint'),
        ('Sim 3a \u2013 False Hub (B\u2013C coupling)', gen_sim3_a(rho_BC=0.8), 'false_hub_a'),
        ('Sim 3b \u2013 False Hub (shared A noise)', gen_sim3_b(rho_A=0.8), 'false_hub_b'),
        ('Sim 4a \u2013 Z \u2192 A \u2192 B Cascade', gen_sim4a1(), 'sim4a1'),
        ('Sim 4a \u2013 Z \u2192 A \u2192 B Cascade', gen_sim4a2(), 'sim4a2'),
        ('Sim 4b \u2013 Cascade (Z\u2013B coupling)', gen_sim4b(rho_BZ=0.8), 'sim4b'),
        ('Sim 4c \u2013 Cascade (shared A noise)', gen_sim4c(rho_A=0.8), 'sim4c'),
    ]
    ncols = len(sims)

    fig = plt.figure(figsize=(4.6 * ncols, 15.5))
    gs = fig.add_gridspec(4, ncols, hspace=0.52, wspace=0.30, height_ratios=[1.0, 1.65, 1.65, 1.2])

    for col, (title, (A, B, Z_cond), mode) in enumerate(sims):

        lbl_AC = 'w_AZ' if 'Cascade' in title else 'w_AC'
        cov_matrix = np.cov(B.flatten(), Z_cond.flatten())
        cov_bc = abs(cov_matrix[0, 1])

        cov_matrix = np.cov(A.T)
        cov_aa = abs(cov_matrix[0, 1])

        sAB = sigma_dir(A, B)
        sAC = sigma_dir(A, Z_cond)

        Wx_AB_c, _, _ = cca_simple(A, B)
        Wx_AC_c, _, _ = cca_simple(A, Z_cond)
        w_AB_c = Wx_AB_c[:, 0]
        w_AC_c = Wx_AC_c[:, 0]
        theta_c = pa_deg(w_AB_c, w_AC_c)

        Wx_AB_p, _, _, _, _ = pcca(A, B, Z=Z_cond)
        Wx_AC_p, _, _, _, _ = pcca(A, Z_cond, Z=B)
        w_AB_p = Wx_AB_p[:, 0]
        w_AC_p = Wx_AC_p[:, 0]
        theta_p = pa_deg(w_AB_p, w_AC_p)

        ax0 = fig.add_subplot(gs[0, col])
        _draw_schematic(ax0, mode)
        new_title = f"{title}\nCov(B, Z(C)) = {cov_bc:.3f}     Cov(A, A) = {cov_aa:.3f}"
        ax0.set_title(new_title, fontsize=9, fontweight='bold', pad=4)

        ax1 = fig.add_subplot(gs[1, col])
        _compass(ax1, w_AB_c, w_AC_c, sig_AB=sAB, sig_AC=sAC, lbl_AC=lbl_AC)
        ax1.set_title(f'CCA   \u03b8 = {theta_c:.1f}\u00b0',
                      fontsize=9, fontweight='bold', color=_angle_color(theta_c), pad=3)

        ax2 = fig.add_subplot(gs[2, col])
        _compass(ax2, w_AB_p, w_AC_p, sig_AB=sAB, sig_AC=sAC, lbl_AC=lbl_AC)
        ax2.set_title(f'pCCA  \u03b8 = {theta_p:.1f}\u00b0',
                      fontsize=9, fontweight='bold', color=_angle_color(theta_p), pad=3)

        ax3 = fig.add_subplot(gs[3, col])
        _wt_bars_compare(ax3, w_AB_c, w_AC_c, w_AB_p, w_AC_p, show_legend=(col == 0), lbl_AC=lbl_AC)

        # Highlight columns with partialling/false-hub artefacts (Sim 3a, 3b, 4b, 4c)
        if col in [4, 5, 7, 8]:
            ax3.axhspan(-1.05, -0.02, alpha=0.06, color=C_COMP)
        ax3.set_ylim(-1.1, 1.15)

    row_labels = ['Circuit\ndiagram', 'CCA', 'pCCA', 'Weight\ncomparison']
    for y_frac, lbl in zip([0.882, 0.648, 0.394, 0.133], row_labels):
        fig.text(0.005, y_frac, lbl, va='center', ha='left',
                 fontsize=9, fontweight='bold', color='#444', rotation=90)

    if save:
        out = ('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/'
               'pCCA_simulation/fig1_2_cca_vs_pcca.png')
        fig.savefig(out, dpi=480, bbox_inches='tight')
        print('  Saved: fig1_2_cca_vs_pcca.png')
    plt.close(fig)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    import os
    os.makedirs(
        '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation',
        exist_ok=True)

    fig1_2_cca_vs_pcca()  # Extended rendering


if __name__ == '__main__':
    main()





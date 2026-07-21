#!/usr/bin/env python3
"""
pcca_benchmark_simulations.py
===========================================
Benchmark suite for partial canonical correlation analysis (pCCA).

Part 1  Canonical correctness  (Simulations 1-3)
  Sim 1  True shared hub                -> expected angle ~  0 deg
  Sim 2  Disjoint communication ports   -> expected angle ~ 90 deg
  Sim 3  False-hub illusion (rho_BC!=0) -> angle collapses to 0 deg

Population-limit derivation for Simulation 3 (disjoint ports + B-C coupling delta):
  Sigma_AB = [1, 0]',  Sigma_AC = [0, 1]'   (raw cross-cov: physically correct)
  w_AB  proportional to  [1, -1/rho]'        (compensatory negative A2 weight)
  w_AC  proportional to  [-rho, 1]'          (compensatory negative A1 weight)
  Note: w_AC = -rho * w_AB  =>  COLLINEAR  =>  theta = 0 deg for all rho != 0.

Part 2  Confound characterisation & correction  (Tasks 2.1-2.3c)
  2.1  Shared internal noise in A (off-diagonal Sigma_AA)
  2.2  External stimulus drive
  2.3a PSTH-residual separation
  2.3b Explicit external conditioning in pCCA
  2.3c Time-lagged canonical correlation (synaptic delay detection)

Nuisance regression (matching perform_session_pcca.m):
  X_res = X - Z (Z'Z + lambda*I)^{-1} Z' X    (lambda = 1e-4)

CCA via regularised SVD:
  M = Sxx^{-1/2} Sxy Syy^{-1/2} = U Sigma V'
  Wx = Sxx^{-1/2} U,   Wy = Syy^{-1/2} V,   rho = diag(Sigma)
  Tikhonov alpha = 1e-4 added to eigenvalues before whitening.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpmath.math2 import sqrt2
from mpmath.math2 import sqrt

# ─────────────────────────────────────────────────────────────────────────────
# Global parameters
# ─────────────────────────────────────────────────────────────────────────────
SEED       = 42
N          = 5_000
SIGMA      = 0.5
LAMBDA_REG = 1e-4
CCA_ALPHA  = 1e-4

C_AB    = '#2166ac'   # blue   - pCCA(A,B|C)
C_AC    = '#d6604d'   # red    - pCCA(A,C|B)
C_RAW   = '#969696'   # grey   - raw cross-covariance direction
C_HUB   = '#4dac26'   # green  - hub/shared signal
C_COMP  = '#762a83'   # purple - compensatory artefact weight
C_STIM  = '#f4a582'   # peach  - external stimulus
C_CAUS  = '#1b7837'   # forest - causal delay
C_COMM  = '#e08214'   # amber  - instantaneous common drive

mpl.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 9,
    'axes.labelsize'    : 9,
    'axes.titlesize'    : 10,
    'xtick.labelsize'   : 8,
    'ytick.labelsize'   : 8,
    'legend.fontsize'   : 8,
    'figure.dpi'        : 130,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : False,
})


# =============================================================================
# SECTION 1 -- CORE UTILITIES
# =============================================================================

def partial_residuals(X, Z, lam=LAMBDA_REG):
    """
    X_res = X - Z (Z'Z + lambda*I)^{-1} Z' X

    Mirrors perform_session_pcca.m exactly (lambda_reg = 1e-4 default).
    Uses np.linalg.solve to avoid forming the explicit inverse.
    """
    q    = Z.shape[1]
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(q), Z.T @ X)
    return X - Z @ beta


def cca_svd(X, Y, k=1, alpha=CCA_ALPHA):
    """
    CCA via regularised SVD of the cross-covariance operator.

    M = Sxx^{-1/2} Sxy Syy^{-1/2} = U S V'
    Wx = Sxx^{-1/2} U[:,: k]   (p x k)
    Wy = Syy^{-1/2} V[:, :k]   (q x k)
    rho = diag(S)[:k]           (k,)

    Tikhonov alpha is added to eigenvalues before taking the inverse
    square root, preventing division by zero in rank-deficient residuals.
    """
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
    """
    pCCA(X, Y | Z).  Reduces to standard CCA when Z is None.

    Returns Wx, Wy, rho, X_res, Y_res.
    """
    if Z is not None and Z.ndim == 2 and Z.shape[1] > 0:
        Xr, Yr = partial_residuals(X, Z, lam), partial_residuals(Y, Z, lam)
    else:
        Xr, Yr = X - X.mean(0), Y - Y.mean(0)
    Wx, Wy, rho = cca_svd(Xr, Yr, k=k, alpha=alpha)
    return Wx, Wy, rho, Xr, Yr


def pa_deg(w1, w2):
    """
    Principal angle in degrees between two 1-D subspaces in R^n.
    theta = arccos(|w1 . w2| / (||w1|| ||w2||))  in [0, 90].
    0 deg = collinear (shared hub);  90 deg = orthogonal (disjoint ports).
    """
    d = np.linalg.norm(w1) * np.linalg.norm(w2)
    if d < 1e-12:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(abs(float(w1 @ w2)) / d, 0.0, 1.0))))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def sigma_dir(A, B):
    """Cross-covariance direction Sigma_AB in A's space (shape: (n_A,))."""
    return ((A - A.mean(0)).T @ (B - B.mean(0)) / A.shape[0]).ravel()


# =============================================================================
# SECTION 2 -- DATA GENERATORS
# =============================================================================

def _rng(seed): return np.random.default_rng(seed)


def gen_sim1(N=N, sigma=SIGMA, seed=SEED):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(N)
    B = (h + sigma * rng.standard_normal(N))[:, None]
    C = (h ** 2 + sigma * rng.standard_normal(N))[:, None]
    noise_A = np.abs(rng.standard_normal(N))
    A = np.c_[B + C, noise_A]
    return A, B, C


def gen_sim1_a(N=N, sigma=SIGMA, seed=SEED):
    rng = np.random.default_rng(seed)
    h   = rng.standard_normal(N)
    A   = np.c_[h + sigma * rng.standard_normal(N),
                sigma * rng.standard_normal(N)]
    B   = (h + sigma * rng.standard_normal(N))[:, None]
    C   = (h + sigma * rng.standard_normal(N))[:, None]
    return A, B, C

def gen_sim1_b(N=N, sigma=SIGMA, seed=SEED):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(N)
    B = (h + sigma * rng.standard_normal(N))[:, None]
    C = (h ** 2 + sigma * rng.standard_normal(N))[:, None]
    noise_A = np.abs(rng.standard_normal(N))*0
    A = np.c_[B + C, noise_A]
    return A, B, C

def gen_sim1_c(N=N, sigma=SIGMA, seed=SEED):
    rng = np.random.default_rng(seed)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    B  = (sB + sigma * rng.standard_normal(N))[:, None]
    C  = (sC + sigma * rng.standard_normal(N))[:, None]
    A = np.c_[B+C, sigma * rng.standard_normal(N)]
    return A, B, C

def gen_sim2(N=N, sigma=SIGMA, seed=SEED):
    """Disjoint ports: sB->A1->B, sC->A2->C, no B-C link."""
    rng = _rng(seed)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    A  = np.c_[sB + rng.standard_normal(N),
               sC + rng.standard_normal(N)]
    B  = (sB + sigma * rng.standard_normal(N))[:, None]
    C  = (sC + sigma * rng.standard_normal(N))[:, None]
    return A, B, C


def gen_sim3_a(N=N, sigma=SIGMA, rho_BC=0.8, seed=SEED):
    """
    False-hub illusion.  Disjoint ports + shared B-C noise delta.
    Population-limit: w_AB proportional to [1,-1/rho], w_AC proportional to [-rho,1]
    These are COLLINEAR (theta=0) for any rho != 0.
    """
    rng   = _rng(seed)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    delta  = rng.standard_normal(N)
    A = np.c_[sB + sigma * rng.standard_normal(N),
              sC + sigma * rng.standard_normal(N)]
    B = (sB + delta          + sigma * rng.standard_normal(N))[:, None]
    C = (sC + rho_BC * delta + sigma * rng.standard_normal(N))[:, None]
    return A, B, C

def gen_sim3_b(N=N, sigma=SIGMA, rho_A=0.8,seed=SEED):
    """Disjoint ports: sB->A1->B, sC->A2->C, no B-C link."""
    rng = _rng(seed)
    h = rng.standard_normal(N)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    A  = np.c_[sB + h,
               sC + rho_A*h]
    B  = (sB + sigma * rng.standard_normal(N))[:, None]
    C  = (sC + sigma * rng.standard_normal(N))[:, None]
    return A, B, C

def gen_sim3(N=N, sigma=SIGMA, rho_BC=0.8, seed=SEED):
    """
    False-hub illusion.  Disjoint ports + shared B-C noise delta.
    Population-limit: w_AB proportional to [1,-1/rho], w_AC proportional to [-rho,1]
    These are COLLINEAR (theta=0) for any rho != 0.
    """
    rng   = _rng(seed)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    delta  = rng.standard_normal(N)
    A = np.c_[sB + sigma * rng.standard_normal(N),
              sC + sigma * rng.standard_normal(N)]
    B = (sB + delta          + sigma * rng.standard_normal(N))[:, None]
    C = (sC + rho_BC * delta + sigma * rng.standard_normal(N))[:, None]
    return A, B, C

def gen_task21(N=N, sigma_ind=SIGMA, sigma_shared=0.0, seed=SEED):
    """Disjoint ports + shared internal noise eta in A (off-diagonal Sigma_AA)."""
    rng = _rng(seed)
    sB, sC = rng.standard_normal(N), rng.standard_normal(N)
    eta    = rng.standard_normal(N)
    A = np.c_[sB + sigma_shared * eta + sigma_ind * rng.standard_normal(N),
              sC + sigma_shared * eta + sigma_ind * rng.standard_normal(N)]
    B = (sB + sigma_ind * rng.standard_normal(N))[:, None]
    C = (sC + sigma_ind * rng.standard_normal(N))[:, None]
    return A, B, C


def gen_task22(N=N, sigma=SIGMA, seed=SEED,
               alpha_coefs=(1.0, 0.8, 1.2, 0.9)):
    """External stimulus s drives A, B, C -- no direct synaptic connections."""
    rng = _rng(seed)
    a1, a2, aB, aC = alpha_coefs
    s = rng.standard_normal(N)
    A = np.c_[a1 * s + sigma * rng.standard_normal(N),
              a2 * s + sigma * rng.standard_normal(N)]
    B = (aB * s + sigma * rng.standard_normal(N))[:, None]
    C = (aC * s + sigma * rng.standard_normal(N))[:, None]
    return A, B, C, s[:, None]


def gen_task23a(n_trials=300, T=60, sigma_stim=1.0,
                sigma_noise=0.4, seed=SEED):
    """
    Multi-trial: sinusoidal PSTH + independent noise.
    Returns raw data, PSTH-subtracted residuals, and empirical PSTHs.
    """
    rng = _rng(seed)
    t   = np.linspace(0, 1, T)
    pA  = sigma_stim * np.c_[np.sin(2*np.pi*t), np.cos(2*np.pi*t)]
    pB  = sigma_stim * np.sin(2*np.pi*t + np.pi/4)
    pC  = sigma_stim * np.sin(2*np.pi*t + np.pi/2)

    NT    = n_trials * T
    A_raw = np.tile(pA, (n_trials, 1)) + sigma_noise * rng.standard_normal((NT, 2))
    B_raw = np.tile(pB, n_trials)[:, None] + sigma_noise * rng.standard_normal((NT, 1))
    C_raw = np.tile(pC, n_trials)[:, None] + sigma_noise * rng.standard_normal((NT, 1))

    A3 = A_raw.reshape(n_trials, T, 2)
    B3 = B_raw.reshape(n_trials, T, 1)
    C3 = C_raw.reshape(n_trials, T, 1)

    psth_A, psth_B, psth_C = A3.mean(0), B3.mean(0), C3.mean(0)

    A_res = (A3 - psth_A).reshape(NT, 2)
    B_res = (B3 - psth_B).reshape(NT, 1)
    C_res = (C3 - psth_C).reshape(NT, 1)

    return (A_raw, B_raw, C_raw,
            A_res, B_res, C_res,
            t, pA, pB, pC,
            psth_A, psth_B[:, 0], psth_C[:, 0])


def gen_task23c(N_T=20_000, delta_t=30, sigma=0.25,
                phi=0.92, seed=SEED):
    """
    Two time-series scenarios for lagged CCA.
    (i)  Causal: A1->B with delay delta_t.  rho(tau) peaks at tau=delta_t.
    (ii) Common drive: A,B share s(t) at tau=0.  rho(tau) peaks at tau=0.
    """
    rng = _rng(seed)

    def ar1(N, p):
        x, e = np.zeros(N), rng.standard_normal(N)
        for i in range(1, N):
            x[i] = p * x[i-1] + np.sqrt(1 - p**2) * e[i]
        return x

    sA1 = ar1(N_T, phi)
    sA2 = ar1(N_T, phi * 0.8)

    noise_B = sigma * rng.standard_normal(N_T)
    B_causal = np.empty(N_T)
    B_causal[:delta_t] = noise_B[:delta_t]
    B_causal[delta_t:] = 0.75 * sA1[:-delta_t] + noise_B[delta_t:]
    A_causal = np.c_[sA1 + sigma * rng.standard_normal(N_T),
                     sA2 + sigma * rng.standard_normal(N_T)]

    common   = ar1(N_T, phi)
    A_common = np.c_[common + sigma * rng.standard_normal(N_T),
                     common + sigma * rng.standard_normal(N_T)]
    B_common = (common + sigma * rng.standard_normal(N_T))[:, None]

    return A_causal, B_causal[:, None], A_common, B_common


def lagged_rho(A, B, tau_range, alpha=CCA_ALPHA):
    """
    rho(tau) for integer lags in tau_range.
    Convention: tau>0 means A leads B (A is the source).
    """
    NT = A.shape[0]
    out = []
    for tau in tau_range:
        if tau > 0:
            At, Bt = A[:-tau], B[tau:]
        elif tau < 0:
            At, Bt = A[-tau:], B[:NT + tau]
        else:
            At, Bt = A, B
        _, _, r = cca_svd(At, Bt, k=1, alpha=alpha)
        out.append(r[0] if len(r) > 0 else np.nan)
    return np.array(out)


# =============================================================================
# SECTION 3 -- VISUALISATION PRIMITIVES
# =============================================================================

def _draw_schematic(ax, mode):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    bs = dict(boxstyle='round,pad=0.15', lw=1.0, zorder=3)

    def node(x, y, lbl, fc='#e0e0e0', ec='#333', fs=9.5):
        ax.text(x, y, lbl, ha='center', va='center', fontsize=fs,
                fontweight='bold', zorder=4,
                bbox=dict(fc=fc, ec=ec, **bs))

    def arr(x0, y0, x1, y1, col='#333', lw=1.5, lbl='', ls='solid'):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=col,
                                    lw=lw, linestyle=ls, mutation_scale=11),
                    zorder=2)
        if lbl:
            ax.text((x0+x1)/2+0.02, (y0+y1)/2+0.04, lbl,
                    ha='center', fontsize=7.5, color=col, fontweight='bold')

    a1x, a1y = 0.43, 0.72
    a2x, a2y = 0.43, 0.28
    Bx,  By  = 0.83, 0.72
    Cx,  Cy  = 0.83, 0.28

    if mode == 'hub':
        hx, hy = 0.10, 0.50
        node(hx, hy, 'h',  fc='#d9f0d3', ec=C_HUB, fs=10)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        arr(hx+0.07, hy+0.04, a1x-0.06, a1y-0.04, col=C_HUB, lbl='h')
        # arr(hx+0.07, hy-0.04, a2x-0.06, a2y+0.04, col=C_HUB)
        arr(a1x+0.07, a1y, Bx-0.06, By, col=C_AB)
        arr(a1x+0.07, a1y-0.02, Cx-0.08, Cy+0.04, col=C_AC, lw=1.2)

    elif mode == 'disjoint_to_A':
        sBx, sBy = 0.10, 0.72
        sCx, sCy = 0.10, 0.28
        node(sBx, sBy, 's_B', fc='#d9f0d3', ec=C_HUB, fs=9)
        node(sCx, sCy, 's_C', fc='#c7e9c0', ec='#238b45', fs=9)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        arr(sBx+0.07, sBy, a1x-0.06, a1y, col=C_HUB)
        arr(sCx+0.07, sCy, a1x-0.06, a1y, col='#238b45')
        arr(a1x+0.07, a1y, Bx-0.06, By, col=C_AB)
        arr(a1x+0.07, a1y, Cx-0.06, Cy, col=C_AC)

    elif mode == 'disjoint':
        sBx, sBy = 0.10, 0.72
        sCx, sCy = 0.10, 0.28
        node(sBx, sBy, 's_B', fc='#d9f0d3', ec=C_HUB, fs=9)
        node(sCx, sCy, 's_C', fc='#c7e9c0', ec='#238b45', fs=9)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        arr(sBx+0.07, sBy, a1x-0.06, a1y, col=C_HUB)
        arr(sCx+0.07, sCy, a2x-0.06, a2y, col='#238b45')
        arr(a1x+0.07, a1y, Bx-0.06, By, col=C_AB)
        arr(a2x+0.07, a2y, Cx-0.06, Cy, col=C_AC)

    elif mode == 'false_hub_a':
        sBx, sBy = 0.10, 0.72
        sCx, sCy = 0.10, 0.28
        dx,  dy  = 0.63, 0.50
        node(sBx, sBy, 's_B', fc='#d9f0d3', ec=C_HUB, fs=9)
        node(sCx, sCy, 's_C', fc='#c7e9c0', ec='#238b45', fs=9)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        node(dx, dy,  '\u03b4', fc='#e0c3e6', ec=C_COMP, fs=10)
        arr(sBx+0.07, sBy, a1x-0.06, a1y, col=C_HUB)
        arr(sCx+0.07, sCy, a2x-0.06, a2y, col='#238b45')
        arr(a1x+0.07, a1y, Bx-0.06, By, col=C_AB)
        arr(a2x+0.07, a2y, Cx-0.06, Cy, col=C_AC)
        arr(dx+0.05, dy+0.04, Bx-0.05, By-0.06, col=C_COMP, lw=2.0)
        arr(dx+0.05, dy-0.04, Cx-0.05, Cy+0.06, col=C_COMP, lw=2.0)
        ax.text(0.50, 0.03, '\u03c1_BC >> 0',
                ha='center', fontsize=7.5, color=C_COMP, style='italic',
                fontweight='bold')

    elif mode == 'false_hub_b':
        sBx, sBy = 0.10, 0.72
        sCx, sCy = 0.10, 0.28
        dx,  dy  = 0.63, 0.50
        node(sBx, sBy, 's_B', fc='#d9f0d3', ec=C_HUB, fs=9)
        node(sCx, sCy, 's_C', fc='#c7e9c0', ec='#238b45', fs=9)
        node(a1x, a1y, 'A\u2081', fc='#c6dbef', ec='#084594')
        node(a2x, a2y, 'A\u2082', fc='#c6dbef', ec='#084594')
        node(Bx, By, 'B', fc='#fcbba1', ec='#a50f15', fs=10)
        node(Cx, Cy, 'C', fc='#fdd0a2', ec='#8c2d04', fs=10)
        node(dx, dy,  '\u03b4', fc='#e0c3e6', ec=C_COMP, fs=10)
        arr(sBx+0.07, sBy, a1x-0.06, a1y, col=C_HUB)
        arr(sCx+0.07, sCy, a2x-0.06, a2y, col='#238b45')
        arr(a1x+0.07, a1y, Bx-0.06, By, col=C_AB)
        arr(a2x+0.07, a2y, Cx-0.06, Cy, col=C_AC)
        arr(dx-0.05, dy+0.04, a1x+0.05, a1y-0.06, col=C_COMP, lw=2.0)
        arr(dx-0.05, dy-0.04, a2x+0.05, a2y+0.06, col=C_COMP, lw=2.0)
        ax.text(0.50, 0.03, '\u03c1_ATA >> 0',
                ha='center', fontsize=7.5, color=C_COMP, style='italic',
                fontweight='bold')


def _compass(ax, w_AB, w_AC, sig_AB=None, sig_AC=None, angle=None, title=''):
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#c0c0c0', lw=0.8, zorder=0)
    ax.axhline(0, color='#d0d0d0', lw=0.5); ax.axvline(0, color='#d0d0d0', lw=0.5)
    ax.set_aspect('equal'); lm = 1.45
    ax.set_xlim(-lm, lm); ax.set_ylim(-lm, lm)
    ax.set_xlabel('A\u2081  weight', fontsize=8)
    ax.set_ylabel('A\u2082  weight', fontsize=8)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])

    for v in [sig_AB, sig_AC]:
        if v is not None:
            uv = unit(np.asarray(v, float))
            ax.annotate("", xy=(0.78*uv[0], 0.78*uv[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=C_RAW,
                                        lw=1.2, linestyle='dashed', mutation_scale=10))

    for w, col, lbl in [(w_AB, C_AB, 'w_AB'), (w_AC, C_AC, 'w_AC')]:
        uw = unit(np.asarray(w, float))
        ax.annotate("", xy=(uw[0], uw[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2.2, mutation_scale=12))
        ax.text(1.22*uw[0], 1.22*uw[1], lbl, ha='center', va='center',
                fontsize=8.5, color=col, fontweight='bold')

    hdr = title
    if angle is not None:
        hdr += ('\n' if title else '') + f'\u03b8 = {angle:.1f}\u00b0'
    if hdr:
        ax.set_title(hdr, fontsize=9, pad=3, fontweight='bold')


def _wt_bars(ax, w_AB, w_AC, sig_AB=None, sig_AC=None):
    x, bw = np.array([0.0, 1.0]), 0.20

    if sig_AB is not None and sig_AC is not None:
        sAB = unit(np.asarray(sig_AB, float))
        sAC = unit(np.asarray(sig_AC, float))
        ax.bar(x - 1.5*bw, sAB, bw, color=C_RAW, alpha=0.55, label='\u03a3_AB raw')
        ax.bar(x - 0.5*bw, sAC, bw, color=C_RAW, alpha=0.30,
               label='\u03a3_AC raw', hatch='//')
        xs = 0.5 * bw
    else:
        xs = -bw

    nAB = unit(np.asarray(w_AB, float))
    nAC = unit(np.asarray(w_AC, float))
    ax.bar(x + xs,       nAB, bw, color=C_AB, alpha=0.85, label='w_AB pCCA')
    ax.bar(x + xs + bw,  nAC, bw, color=C_AC, alpha=0.85, label='w_AC pCCA')

    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['A\u2081', 'A\u2082'])
    ax.set_ylabel('Normalised weight', fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_xlim(-0.6, 1.85)


# =============================================================================
# SECTION 4 -- FIGURE FUNCTIONS
# =============================================================================

# ── Figure 1: Canonical validations (Sims 1, 2, 3) ───────────────────────────
import numpy as np
import matplotlib.pyplot as plt


def fig1_canonical_validations(save=True):
    print("Building Figure 1: Canonical validations (Extended 5-Column)...")

    sims = [
        ('Sim 1a -- Linear Shared Hub', gen_sim1_a(), 'hub'),
        ('Sim 1b -- Non-linear Hub (Cov=0)', gen_sim1_b(), 'hub'),
        ('Sim 1c -- Additive Ports', gen_sim1_c(), 'disjoint_to_A'),
        ('Sim 2 -- Disjoint Ports', gen_sim2(), 'disjoint'),
        ('Sim 3a -- False Hub Illusion by B and C', gen_sim3_a(rho_BC=0.8), 'false_hub_a'),
        ('Sim 3b -- False Hub Illusion by A', gen_sim3_b(rho_A=0.8), 'false_hub_b'),
    ]

    num_cols = len(sims)

    fig = plt.figure(figsize=(4.6 * num_cols, 10.5))

    gs = fig.add_gridspec(3, num_cols, hspace=0.42, wspace=0.30,
                          height_ratios=[1.0, 1.6, 1.2])

    for col, (title, (A, B, C), mode) in enumerate(sims):

        cov_matrix = np.cov(B.flatten(), C.flatten())
        cov_bc = abs(cov_matrix[0, 1])

        cov_matrix = np.cov(A.T)
        cov_aa = abs(cov_matrix[0, 1])

        Wx_AB, _, rho_AB, _, _ = pcca(A, B, Z=C, k=1)
        Wx_AC, _, rho_AC, _, _ = pcca(A, C, Z=B, k=1)
        w_AB = Wx_AB[:, 0]
        w_AC = Wx_AC[:, 0]
        angle = pa_deg(w_AB, w_AC)

        sAB = sigma_dir(A, B)
        sAC = sigma_dir(A, C)

        ax0 = fig.add_subplot(gs[0, col])
        _draw_schematic(ax0, mode)

        new_title = f"{title}\nCov(B, C) = {cov_bc:.3f}     Cov(A, A) = {cov_aa:.3f}"
        ax0.set_title(new_title, fontsize=9.5, fontweight='bold', pad=4)

        ax1 = fig.add_subplot(gs[1, col])

        show_raw = (col >= 0)

        _compass(ax1, w_AB, w_AC,
                 sig_AB=sAB if show_raw else None,
                 sig_AC=sAC if show_raw else None,
                 angle=angle)

        if col <= 1:
            ax1.set_title(f'\u03b8 = {angle:.1f}\u00b0', fontsize=9.5,
                          fontweight='bold', color=C_HUB, pad=3)
        elif col < num_cols - 2:
            ax1.set_title(f'\u03b8 = {angle:.1f}\u00b0', fontsize=9.5,
                          fontweight='bold', color=C_AB, pad=3)
        else:
            ax1.set_title(f'\u03b8 = {angle:.1f}\u00b0  (expected 90\u00b0 \u2192 COLLAPSE)',
                          fontsize=8.5, fontweight='bold', color=C_COMP, pad=3)

        ax2 = fig.add_subplot(gs[2, col])
        _wt_bars(ax2, w_AB, w_AC,
                 sig_AB=sAB if show_raw else None,
                 sig_AC=sAC if show_raw else None)

        if col >= num_cols - 2:
            ax2.axhspan(-1.05, -0.02, alpha=0.06, color=C_COMP)
            ax2.text(1.55, -0.55, 'spurious\nneg. weight',
                     ha='center', fontsize=7.5, color=C_COMP, style='italic')
        ax2.set_ylim(-1.1, 1.15)

    # fig.suptitle('pCCA Canonical Validation  --  Simulations (Extended)\n'
    #              '(N = 5 000,  \u03c3 = 0.5,  \u03bb = 1e-4)', fontsize=11)

    if save:
        fig.savefig(
            '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/fig1_canonical_validations.png',
            dpi=180, bbox_inches='tight')
        print("  Saved: fig1_canonical_validations.png")

    plt.close(fig)

# ── Figure 2: False-hub quantification (rho sweep + Task 2.1) ────────────────
def fig2_false_hub_quantification(save=True):
    """
    Left column  Simulation 3: principal angle and weight components vs rho_BC.
    Right column Task 2.1:     principal angle and weight components vs sigma_shared.

    The rho sweep demonstrates the discontinuous transition at rho=0 in the
    population limit: any nonzero B-C coupling causes angle collapse, while
    independent observation noise smooths this into a rapid finite-sample curve.
    """
    print("Building Figure 2: False-hub quantification...")

    # ── Sweep 1: rho_BC ──────────────────────────────────────────────────
    rho_vals = np.concatenate([np.array([0.0]),
                               np.linspace(0.05, 0.95, 19),
                               np.array([0.99])])
    angles_rho = []
    w_AB_comp_rho, w_AC_comp_rho = [], []

    for rho in rho_vals:
        A, B, C = gen_sim3_a(rho_BC=rho)
        Wx_AB, _, _, _, _ = pcca(A, B, Z=C)
        Wx_AC, _, _, _, _ = pcca(A, C, Z=B)
        w_AB = unit(Wx_AB[:, 0])
        w_AC = unit(Wx_AC[:, 0])
        angles_rho.append(pa_deg(w_AB, w_AC))
        w_AB_comp_rho.append(w_AB.copy())
        w_AC_comp_rho.append(w_AC.copy())

    angles_rho    = np.array(angles_rho)
    w_AB_comp_rho = np.array(w_AB_comp_rho)   # (n_rho, 2)
    w_AC_comp_rho = np.array(w_AC_comp_rho)

    # Population-limit prediction: theta=90 at rho=0, theta=0 else
    theta_theory = np.where(rho_vals == 0.0, 90.0, 0.0)

    # ── Sweep 2: sigma_shared (Task 2.1) ─────────────────────────────────
    ss_vals  = np.linspace(0.0, 2.5, 21)
    angles_ss = []
    w_AB_comp_ss, w_AC_comp_ss = [], []

    for ss in ss_vals:
        A, B, C = gen_task21(sigma_shared=ss)
        Wx_AB, _, _, _, _ = pcca(A, B, Z=C)
        Wx_AC, _, _, _, _ = pcca(A, C, Z=B)
        w_AB = unit(Wx_AB[:, 0])
        w_AC = unit(Wx_AC[:, 0])
        angles_ss.append(pa_deg(w_AB, w_AC))
        w_AB_comp_ss.append(w_AB.copy())
        w_AC_comp_ss.append(w_AC.copy())

    angles_ss    = np.array(angles_ss)
    w_AB_comp_ss = np.array(w_AB_comp_ss)
    w_AC_comp_ss = np.array(w_AC_comp_ss)

    # ── Build figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    # (0,0) Angle vs rho_BC
    ax = axes[0, 0]
    ax.plot(rho_vals, theta_theory, 'k--', lw=1.0, label='Population limit', alpha=0.6)
    ax.plot(rho_vals, angles_rho, 'o-', color=C_COMP, lw=1.8,
            ms=5, label='Simulated (N=5 000)')
    ax.axhline(90, color='#888', lw=0.6, ls=':')
    ax.axhline(0,  color='#888', lw=0.6, ls=':')
    ax.set_xlabel(r'$\rho_{BC}$  (B–C coupling coefficient)', fontsize=9)
    ax.set_ylabel('Principal angle  (\u00b0)', fontsize=9)
    ax.set_title('Sim 3a -- Angle collapse vs \u03c1_BC', fontsize=10)
    ax.set_ylim(-5, 100); ax.legend(fontsize=8, frameon=False)
    ax.annotate('90\u00b0 expected\n(disjoint ports)',
                xy=(0, 90), xytext=(0.25, 82),
                fontsize=8, color='#888', ha='center',
                arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8))

    # ax.annotate('0\u00b0 collapse\n(false hub)',
    #             xy=(0.5, angles_rho[10]), xytext=(0.65, 28),
    #             fontsize=8, color=C_COMP, ha='center',
    #             arrowprops=dict(arrowstyle='->', color=C_COMP, lw=0.8))

    # (1,0) Weight components vs rho_BC
    ax = axes[1, 0]
    ax.plot(rho_vals, w_AB_comp_rho[:, 0], '-', color=C_AB, lw=1.8,
            label='w_AB  A\u2081')
    ax.plot(rho_vals, w_AB_comp_rho[:, 1], '--', color=C_AB, lw=1.8,
            label='w_AB  A\u2082 (compensatory)')
    ax.plot(rho_vals, w_AC_comp_rho[:, 0], '--', color=C_AC, lw=1.8,
            label='w_AC  A\u2081 (compensatory)')
    ax.plot(rho_vals, w_AC_comp_rho[:, 1], '-', color=C_AC, lw=1.8,
            label='w_AC  A\u2082 component')
    ax.axhline(0, color='k', lw=0.5)
    ax.fill_between(rho_vals,
                    np.minimum(w_AB_comp_rho[:, 1], 0),
                    0, alpha=0.12, color=C_COMP)
    ax.fill_between(rho_vals,
                    np.minimum(w_AC_comp_rho[:, 0], 0),
                    0, alpha=0.12, color=C_COMP)
    ax.set_xlabel(r'$\rho_{BC}$', fontsize=9)
    ax.set_ylabel('Normalised weight component', fontsize=9)
    ax.set_title('Compensatory negative weights grow with \u03c1_BC', fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_ylim(-1.1, 1.2)
    ax.text(0.72, -0.22, 'spurious artefact', fontsize=8,
            color=C_COMP, ha='center', style='italic')

    # (0,1) Angle vs sigma_shared (Task 2.1)
    ax = axes[0, 1]
    ax.plot(rho_vals, theta_theory, 'k--', lw=1.0, label='Population limit', alpha=0.6)
    ax.plot(ss_vals, angles_ss, 'o-', color='#d95f02', lw=1.8, ms=5, label='Simulated (N=5 000)')
    ax.axhline(90, color='#888', lw=0.6, ls=':')
    ax.axhline(0,  color='#888', lw=0.6, ls=':')
    ax.set_xlabel(r'$\sigma_{\rm shared}$  (internal noise in A)', fontsize=9)
    ax.set_ylabel('Principal angle  (\u00b0)', fontsize=9)
    ax.set_title('Sim 3b -- Angle collapse vs shared noise', fontsize=10)
    ax.set_ylim(-5, 100),ax.legend(fontsize=8, frameon=False)
    ax.annotate('Disjoint ports:\n90\u00b0 at \u03c3_shared = 0',
                xy=(0.0, 90), xytext=(0.7, 80),
                fontsize=8, color='#888', ha='center',
                arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8))
    # ax.annotate('CCA whitening\nrotates weights',
    #             xy=(1.2, angles_ss[10]), xytext=(1.9, 45),
    #             fontsize=8, color='#d95f02', ha='center',
    #             arrowprops=dict(arrowstyle='->', color='#d95f02', lw=0.8))

    # (1,1) Weight components vs sigma_shared
    ax = axes[1, 1]
    ax.plot(ss_vals, w_AB_comp_ss[:, 0], '-', color=C_AB, lw=1.8,
            label='w_AB  A\u2081 (true port)')
    ax.plot(ss_vals, w_AB_comp_ss[:, 1], '--', color=C_AB, lw=1.8,
            label='w_AB  A\u2082 (leak)')
    ax.plot(ss_vals, w_AC_comp_ss[:, 0], '--', color=C_AC, lw=1.8,
            label='w_AC  A\u2081 (leak)')
    ax.plot(ss_vals, w_AC_comp_ss[:, 1], '-', color=C_AC, lw=1.8,
            label='w_AC  A\u2082 (true port)')
    ax.fill_between(ss_vals,
                    np.minimum(w_AB_comp_ss[:, 1], 0),
                    0, alpha=0.12, color=C_COMP)
    ax.fill_between(ss_vals,
                    np.minimum(w_AC_comp_ss[:, 0], 0),
                    0, alpha=0.12, color=C_COMP)
    ax.text(1.52, -0.22, 'spurious artefact', fontsize=8,
            color=C_COMP, ha='center', style='italic')

    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$\sigma_{\rm shared}$', fontsize=9)
    ax.set_ylabel('Normalised weight component', fontsize=9)
    ax.set_title('Weight leakage induced by off-diagonal \u03a3_AA', fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_ylim(-1.05, 1.15)

    fig.suptitle('Left: B–C coupling (Sim 3a)     Right: Internal A noise (Sim 3b)',
                 fontsize=11)
    if save:
        fig.savefig('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/fig2_false_hub_quantification.png',
                    dpi=180, bbox_inches='tight')
        print("  Saved: fig2_false_hub_quantification.png")
    plt.close(fig)


# ── Figure 3: External confound suite (Tasks 2.2) ────────────────────────────
def fig3_confound_suite(save=True):
    """
    Task 2.2: all apparent communication is stimulus-driven (no direct synapses).

    Three analysis strategies compared:
      (i)  Standard CCA(A, B)      -- spurious high rho
      (ii) pCCA(A, B | C)          -- C partially, but not fully, blocks stimulus
      (iii)pCCA(A, B | s)          -- s is the exact stimulus; rho -> near 0 (correct)

    Also shown: the compass directions for each strategy in A's 2-D space.
    """
    print("Building Figure 3: External confound suite (Task 2.2)...")

    A, B, C, s = gen_task22()

    # Three analysis strategies
    Wx_std, _, rho_std, _, _ = pcca(A, B, Z=None)
    Wx_pC,  _, rho_pC,  _, _ = pcca(A, B, Z=C)
    Wx_ps,  _, rho_ps,  _, _ = pcca(A, B, Z=s)

    # True structure: because A and B share only s, pCCA(A,B|s) should give ~0
    labels  = ['CCA(A,B)', 'pCCA(A,B|C)', 'pCCA(A,B|s)']
    rho_vals_bar = [rho_std[0], rho_pC[0], rho_ps[0]]
    cols    = [C_STIM, C_STIM, C_HUB]
    hatches = ['', '//', '']
    notes   = ['Spurious\n(stimulus)', 'Partial\ncorrection', 'Correct\n(s removed)']

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.38, top=0.84)

    # ── Left: bar chart of rho values ──
    ax = axes[0]
    bars = ax.bar(np.arange(3), rho_vals_bar, color=cols, alpha=0.85,
                  hatch=['', '//', ''], edgecolor='k', lw=0.8)
    bars[2].set_facecolor(C_HUB)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Canonical correlation  \u03c1', fontsize=9)
    ax.set_title('Canonical correlation\nfor each analysis strategy', fontsize=10)
    ax.set_ylim(0, 1.05)
    for i, (r, note) in enumerate(zip(rho_vals_bar, notes)):
        ax.text(i, r + 0.03, f'\u03c1={r:.3f}', ha='center', fontsize=8.5,
                fontweight='bold')
        ax.text(i, 0.05, note, ha='center', fontsize=7.5,
                color='k' if i < 2 else '#1b7837', style='italic')
    ax.text(1.0, 0.82, 'No direct A\u2194B synapse\nexists in simulation',
            ha='center', fontsize=8, color='#555', style='italic',
            bbox=dict(boxstyle='round', fc='#fffbe0', ec='#ccc', lw=0.6))

    # ── Middle: compass for pCCA(A,B|C) (partially corrected) ──
    ax = axes[1]
    sAB = sigma_dir(A, B)
    # For contrast also show the pCCA(A,B|s) weight
    _compass(ax, Wx_pC[:, 0], Wx_ps[:, 0],
             sig_AB=sAB, angle=None)
    ax.set_title('pCCA(A,B|C)  vs  pCCA(A,B|s)\ndirections in A-space',
                 fontsize=9)
    # Override the labels to distinguish the two
    ax.texts[-2].set_text('pCCA(A,B|C)')
    ax.texts[-2].set_color(C_AB)
    ax.texts[-1].set_text('pCCA(A,B|s)')
    ax.texts[-1].set_color(C_HUB)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Right: scatter of A canonical variate vs B for the three strategies ──
    ax = axes[2]
    # Project A onto each weight and scatter against B
    Bc = (B - B.mean(0)).ravel()

    def _proj(Wx):
        Ac = A - A.mean(0)
        return (Ac @ Wx[:, 0]).ravel()

    z_std = _proj(Wx_std)
    z_pC  = _proj(Wx_pC)
    z_ps  = _proj(Wx_ps)

    idx   = np.random.default_rng(0).integers(0, N, 600)
    ax.scatter(z_std[idx], Bc[idx], s=6, alpha=0.35, color=C_STIM,
               label=f'CCA  (\u03c1={rho_std[0]:.2f})', rasterized=True)
    ax.scatter(z_pC[idx],  Bc[idx], s=6, alpha=0.35, color=C_AB,
               label=f'pCCA|C (\u03c1={rho_pC[0]:.2f})', rasterized=True)
    ax.scatter(z_ps[idx],  Bc[idx], s=6, alpha=0.60, color=C_HUB,
               label=f'pCCA|s (\u03c1={rho_ps[0]:.2f})', rasterized=True)
    ax.set_xlabel('A canonical variate  (a.u.)', fontsize=9)
    ax.set_ylabel('B  (a.u.)', fontsize=9)
    ax.set_title('Scatter: A variate vs B\n(n=600 random samples shown)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle('Task 2.2 -- External Stimulus Drive\n'
                 'A, B, C share only s(t); no direct synaptic connections',
                 fontsize=11)
    if save:
        fig.savefig('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/fig3_confound_suite.png',
                    dpi=180, bbox_inches='tight')
        print("  Saved: fig3_confound_suite.png")
    plt.close(fig)


# ── Figure 4: Correction strategies (Tasks 2.3a and 2.3b) ────────────────────
def fig4_correction_strategies(save=True):
    """
    Task 2.3a  PSTH-residual separation.
    Task 2.3b  Explicit external conditioning (same data as Task 2.2).

    2.3a: compare pCCA on raw (PSTH-dominated) data vs PSTH-subtracted residuals.
    2.3b: confirm that pCCA(A,B|s) fully removes stimulus-driven variance.
    """
    print("Building Figure 4: Correction strategies (Tasks 2.3a, 2.3b)...")

    # ── Task 2.3a data ──
    (A_raw, B_raw, C_raw,
     A_res, B_res, C_res,
     t_vec, pA_true, pB_true, pC_true,
     psth_A, psth_B, psth_C) = gen_task23a()

    Wx_raw, _, rho_raw, _, _ = pcca(A_raw, B_raw, Z=C_raw)
    Wx_res, _, rho_res, _, _ = pcca(A_res, B_res, Z=C_res)

    # ── Task 2.3b data (reuse Task 2.2 generator) ──
    A22, B22, C22, s22 = gen_task22()
    Wx_pC22, _, rho_pC22, _, _ = pcca(A22, B22, Z=C22)
    Wx_ps22, _, rho_ps22, _, _ = pcca(A22, B22, Z=s22)

    # ── Figure layout ──
    fig = plt.figure(figsize=(14, 10.5))
    gs  = fig.add_gridspec(3, 4, hspace=0.50, wspace=0.38)

    # ─ Row 0: PSTH time courses ─────────────────────────────────────────
    ax_psth_A = fig.add_subplot(gs[0, 0:2])
    ax_psth_A.plot(t_vec, pA_true[:, 0], '-', color='#555', lw=1.5,
                   label='true \u03bc_A1 (A\u2081 neuron)', alpha=0.8)
    ax_psth_A.plot(t_vec, psth_A[:, 0], '--', color=C_AB, lw=1.5,
                   label='empirical PSTH (n=300 trials)')
    ax_psth_A.fill_between(t_vec, pA_true[:, 0] - 0.4, pA_true[:, 0] + 0.4,
                            alpha=0.10, color='k')
    ax_psth_A.set_xlabel('Normalised time', fontsize=8)
    ax_psth_A.set_ylabel('Activity  (a.u.)', fontsize=8)
    ax_psth_A.set_title('Task 2.3a -- True vs empirical PSTH  (region A\u2081)',
                        fontsize=9.5)
    ax_psth_A.legend(fontsize=7.5, frameon=False)

    ax_psth_B = fig.add_subplot(gs[0, 2:4])
    ax_psth_B.plot(t_vec, pB_true, '-', color='#555', lw=1.5,
                   label='true \u03bc_B', alpha=0.8)
    ax_psth_B.plot(t_vec, psth_B, '--', color=C_AC, lw=1.5,
                   label='empirical PSTH')
    ax_psth_B.set_xlabel('Normalised time', fontsize=8)
    ax_psth_B.set_ylabel('Activity  (a.u.)', fontsize=8)
    ax_psth_B.set_title('True vs empirical PSTH  (region B)', fontsize=9.5)
    ax_psth_B.legend(fontsize=7.5, frameon=False)

    # ─ Row 1: compass comparison (raw vs residuals) ─────────────────────
    ax_comp_raw = fig.add_subplot(gs[1, 0:2])
    _compass(ax_comp_raw, Wx_raw[:, 0],
             # For comparison: show the residual weight in grey overlay
             Wx_raw[:, 0],   # placeholder; will add residual manually
             angle=None)
    # Manually add residual weight arrow in a different colour
    uw_res = unit(Wx_res[:, 0])
    ax_comp_raw.annotate("", xy=(uw_res[0], uw_res[1]), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color='#d95f02',
                                          lw=2.2, mutation_scale=12))
    ax_comp_raw.text(1.22*uw_res[0], 1.22*uw_res[1], 'w (resid.)',
                     ha='center', fontsize=8, color='#d95f02', fontweight='bold')
    ax_comp_raw.set_title(
        f'A-space directions\nraw \u03c1={rho_raw[0]:.3f}  (blue)   '
        f'residuals \u03c1={rho_res[0]:.3f}  (orange)',
        fontsize=8.5)
    # Fix duplicate label from reuse
    for txt in ax_comp_raw.texts:
        if txt.get_text() == 'w_AC':
            txt.set_text('w_AB (raw)')
            break

    ax_bar_23a = fig.add_subplot(gs[1, 2:4])
    conditions  = ['pCCA\nraw data', 'pCCA\nresiduals']
    rho_23a     = [rho_raw[0], rho_res[0]]
    bar_cols    = [C_STIM, '#d95f02']
    bars        = ax_bar_23a.bar(conditions, rho_23a, color=bar_cols, alpha=0.85,
                                  edgecolor='k', lw=0.8)
    for i, r in enumerate(rho_23a):
        ax_bar_23a.text(i, r + 0.015, f'\u03c1 = {r:.3f}',
                        ha='center', fontsize=9, fontweight='bold')
    ax_bar_23a.set_ylabel('Canonical correlation  \u03c1', fontsize=9)
    ax_bar_23a.set_title('Task 2.3a -- pCCA: raw vs residual data\n'
                         'High rho on raw = stimulus; low on residuals = intrinsic',
                         fontsize=9.5)
    ax_bar_23a.set_ylim(0, 1.05)
    ax_bar_23a.axhline(0, color='k', lw=0.5)

    # ─ Row 2: Task 2.3b -- explicit conditioning ─────────────────────────
    ax_bar_23b = fig.add_subplot(gs[2, 0:2])
    conds   = ['pCCA(A,B|C)', 'pCCA(A,B|s)']
    rho_23b = [rho_pC22[0], rho_ps22[0]]
    b_cols  = [C_AB, C_HUB]
    ax_bar_23b.bar(conds, rho_23b, color=b_cols, alpha=0.85, edgecolor='k', lw=0.8)
    for i, r in enumerate(rho_23b):
        ax_bar_23b.text(i, r + 0.015, f'\u03c1 = {r:.3f}',
                        ha='center', fontsize=9, fontweight='bold')
    ax_bar_23b.set_ylabel('Canonical correlation  \u03c1', fontsize=9)
    ax_bar_23b.set_title('Task 2.3b -- Explicit conditioning on s(t)\n'
                         'Partialling C removes ~ but not all stimulus variance;\n'
                         'partialling s directly eliminates it entirely',
                         fontsize=8.5)
    ax_bar_23b.set_ylim(0, 1.05)

    ax_comp_23b = fig.add_subplot(gs[2, 2:4])
    _compass(ax_comp_23b, Wx_pC22[:, 0], Wx_ps22[:, 0], angle=None)
    ax_comp_23b.set_title(
        f'Task 2.3b -- A-space directions\n'
        f'pCCA(A,B|C): \u03c1={rho_pC22[0]:.3f}  (blue)   '
        f'pCCA(A,B|s): \u03c1={rho_ps22[0]:.3f}  (red)',
        fontsize=8.5)
    # Rename legend labels
    for txt in ax_comp_23b.texts:
        if txt.get_text() == 'w_AB':
            txt.set_text('pCCA|C')
        elif txt.get_text() == 'w_AC':
            txt.set_text('pCCA|s')
            txt.set_color(C_HUB)

    fig.suptitle('Correction Strategies -- Tasks 2.3a and 2.3b', fontsize=11)
    if save:
        fig.savefig('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/fig4_correction_strategies.png',
                    dpi=180, bbox_inches='tight')
        print("  Saved: fig4_correction_strategies.png")
    plt.close(fig)


# ── Figure 5: Time-lagged canonical correlation (Task 2.3c) ──────────────────
def fig5_temporal_analysis(save=True):
    """
    Task 2.3c -- Time-lagged canonical correlation reveals synaptic delay.

    Two scenarios:
      Causal:  A1(t) -> B(t) with delay delta_t = 30 samples.
               rho(tau) should peak sharply at tau = 30.
      Common:  A(t) and B(t) both driven by s(t) with no delay.
               rho(tau) should peak at tau = 0 and decay with the AR autocorrelation.

    The contrast in peak location is the key diagnostic: a peak at tau>0 is
    evidence for directed communication (A drives B); a peak at tau=0 suggests
    instantaneous common input.
    """
    print("Building Figure 5: Temporal analysis (Task 2.3c)...")

    DELTA_T  = 30
    A_caus, B_caus, A_comm, B_comm = gen_task23c(delta_t=DELTA_T)

    tau_range = np.arange(-60, 121, 3)   # lags in samples

    print("  Computing lagged rho (causal)...")
    rho_caus = lagged_rho(A_caus, B_caus, tau_range)
    print("  Computing lagged rho (common drive)...")
    rho_comm = lagged_rho(A_comm, B_comm, tau_range)

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    fig.subplots_adjust(wspace=0.35, top=0.84)

    for ax, rhos, col, lbl, expected_peak, title in [
        (axes[0], rho_caus, C_CAUS, 'Causal  A\u2192B  (\u0394t=30 samples)',
         DELTA_T,
         'Task 2.3c -- Causal Transmission\nA\u2081(t) drives B(t) with delay \u0394t'),
        (axes[1], rho_comm, C_COMM, 'Common drive  (no delay)',
         0,
         'Task 2.3c -- Common Drive\nA and B share s(t) with zero delay'),
    ]:
        ax.fill_between(tau_range, 0, rhos, alpha=0.18, color=col)
        ax.plot(tau_range, rhos, '-', color=col, lw=2.0, label=lbl)
        ax.axvline(0, color='#888', lw=0.8, ls='--', alpha=0.6, label=r'$\tau$ = 0')
        ax.axvline(expected_peak, color=col, lw=1.5, ls=':', alpha=0.8,
                   label=f'Expected peak: \u03c4 = {expected_peak} samples')

        peak_idx = np.nanargmax(rhos)
        peak_tau = tau_range[peak_idx]
        peak_rho = rhos[peak_idx]
        ax.annotate(
            f'Peak at \u03c4={peak_tau}\n\u03c1={peak_rho:.3f}',
            xy=(peak_tau, peak_rho), xytext=(peak_tau + 18, peak_rho - 0.06),
            fontsize=9, color=col, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

        ax.set_xlabel('Lag \u03c4  (samples, 1 sample \u2248 1 ms)', fontsize=9)
        ax.set_ylabel('Canonical correlation  \u03c1(\u03c4)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.set_ylim(bottom=0)
        ax.set_xlim(tau_range[0], tau_range[-1])

    fig.suptitle('Time-Lagged Canonical Correlation\n'
                 'Causal delay (left) vs instantaneous common drive (right)',
                 fontsize=11)
    if save:
        fig.savefig('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/fig5_temporal_analysis.png',
                    dpi=180, bbox_inches='tight')
        print("  Saved: fig5_temporal_analysis.png")
    plt.close(fig)


# =============================================================================
# SECTION 5 -- SUMMARY TABLE (printed to stdout)
# =============================================================================

def print_summary_table():
    """Print a concise numerical summary of all seven simulations."""
    print('\n' + '='*72)
    print('pCCA BENCHMARK SUMMARY')
    print('='*72)
    print(f'{"Simulation":<30}  {"rho_AB":>7}  {"rho_AC":>7}  {"angle":>8}  {"Expected":>10}')
    print('-'*72)

    rows = [
        ('Sim 1 -- True hub',           gen_sim1(),        '~  0 deg'),
        ('Sim 2 -- Disjoint ports',     gen_sim2(),        '~ 90 deg'),
        ('Sim 3 -- False hub (rho=0.8)',gen_sim3(rho_BC=0.8),'~  0 deg (COLLAPSE)'),
        ('Task 2.1 (sig_shr=0)',        gen_task21(sigma_shared=0.0), '~ 90 deg'),
        ('Task 2.1 (sig_shr=1.0)',      gen_task21(sigma_shared=1.0), 'degraded'),
        ('Task 2.1 (sig_shr=2.0)',      gen_task21(sigma_shared=2.0), 'collapsed'),
    ]

    for lbl, (A, B, C), exp in rows:
        Wx_AB, _, rAB, _, _ = pcca(A, B, Z=C)
        Wx_AC, _, rAC, _, _ = pcca(A, C, Z=B)
        ang = pa_deg(Wx_AB[:, 0], Wx_AC[:, 0])
        print(f'{lbl:<30}  {rAB[0]:>7.3f}  {rAC[0]:>7.3f}  {ang:>7.1f}  {exp:>10}')

    # Task 2.2
    A22, B22, C22, s22 = gen_task22()
    _, _, r_std, _, _  = pcca(A22, B22, Z=None)
    _, _, r_pC,  _, _  = pcca(A22, B22, Z=C22)
    _, _, r_ps,  _, _  = pcca(A22, B22, Z=s22)
    print('-'*72)
    print(f'{"Task 2.2 -- CCA(A,B)":<30}  {r_std[0]:>7.3f}  {"--":>7}  {"--":>8}  spurious')
    print(f'{"Task 2.2 -- pCCA(A,B|C)":<30}  {r_pC[0]:>7.3f}  {"--":>7}  {"--":>8}  partial fix')
    print(f'{"Task 2.2 -- pCCA(A,B|s)":<30}  {r_ps[0]:>7.3f}  {"--":>7}  {"--":>8}  ~0 (correct)')
    print('='*72 + '\n')


# =============================================================================
# MAIN
# =============================================================================

def main():
    import os
    os.makedirs('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation', exist_ok=True)

    print_summary_table()
    fig1_canonical_validations()
    fig2_false_hub_quantification()
    fig3_confound_suite()
    fig4_correction_strategies()
    fig5_temporal_analysis()
    print("\nAll five figures saved to /Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/")


if __name__ == '__main__':
    main()
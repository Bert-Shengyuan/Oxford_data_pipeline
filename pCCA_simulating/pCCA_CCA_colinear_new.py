#!/usr/bin/env python3
"""
pcca_hublateral_simulations.py
===========================================
Benchmark suite for the "Global Hub + Lateral Coupling" simulation family
(full generator set: Sim a, b1-b6, c1-c5; 12 simulations total).

Circuit primitives shared across the family
--------------------------------------------
  Global hub          s_ZG  -> drives a growing subset of {Z, A1, A2, B}
  Lateral coupling     s_A2B -> drives {A2, B}  (the genuine "private" A-B
                                                  channel that pCCA(A,B|Z)
                                                  should recover even though
                                                  Z is conditioned out)
  Private noise terms  s_ZP, s_A1P, s_A2P, s_BP -> idiosyncratic, node-local

Two parallel branches
----------------------
  "b" branch -- hub reaches {Z, A1, B} but NOT A2 directly:
    Sim a   -- minimal hub: s_ZG -> {Z, A1} only; B driven purely by s_A2B
    Sim b1  -- hub now also reaches B: s_ZG -> {Z, A1, B}
    Sim b2  -- b1 + private noise on A2          (s_A2P)
    Sim b3  -- b1 + private noise on Z           (s_ZP)
    Sim b4  -- b1 + private noise on Z and A2    (s_ZP, s_A2P)
    Sim b5  -- b1 + private noise on Z, A1, A2   (s_ZP, s_A1P, s_A2P)
    Sim b6  -- b1 fully privatised: s_ZP, s_A1P, s_A2P, s_BP

  "c" branch -- hub reaches {Z, A1, A2, B} ("full hub"):
    Sim c1  -- full hub, no private noise
    Sim c2  -- c1 + private noise on Z           (s_ZP)
    Sim c3  -- c1 + private noise on Z and A2    (s_ZP, s_A2P)
    Sim c4  -- c1 + private noise on Z, A1, A2   (s_ZP, s_A1P, s_A2P)
    Sim c5  -- c1 fully privatised: s_ZP, s_A1P, s_A2P, s_BP  (fully
               specified circuit)

For every simulation we compute, in parallel:
  CCA(A,B)         vs  pCCA(A,B|Z)
  CCA(A,Z)         vs  pCCA(A,Z|B)
and compare the canonical weight directions w_AB and w_AZ (subspace angle
theta) under both methods.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Global parameters (kept identical to pcca_benchmark_simulations.py)
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
N = 5_000
SIGMA = 0.0
LAMBDA_REG = 1e-4
CCA_ALPHA = 1e-4
POISSON_LAM = 1.0

C_AB = '#2166ac'   # blue   - pCCA(A,B|Z) canonical direction
C_AZ = '#d6604d'   # red    - pCCA(A,Z|B) canonical direction
C_RAW = '#969696'  # grey   - raw cross-covariance direction (reference arrow)
C_HUB = '#4dac26'  # green  - global hub s_ZG
C_COMP = '#762a83' # purple - private / compensatory noise
C_STIM = '#f4a582' # peach  - external stimulus (unused here, kept for palette parity)
C_CAUS = '#1b7837' # forest - causal delay (unused here, kept for palette parity)
C_COMM = '#e08214' # amber  - lateral common-drive s_A2B
C_CCA = '#4393c3'  # light blue - CCA unconditional baseline

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
# SECTION 1 -- CORE UTILITIES  (unchanged from pcca_benchmark_simulations.py)
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


def _rng(seed): return np.random.default_rng(seed)


# =============================================================================
# SECTION 2 -- DATA GENERATORS  (Sim a, b1-b6, c1-c5)
# =============================================================================

def gen_sima(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Minimal hub: s_ZG -> {Z, A1} only; B driven purely by the lateral term.
    s_ZG -> {Z, A1} ;  s_A2B -> {A2, B}
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)
    sA2B = rng.poisson(lam, N)

    Z = (sZG + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sigma * rng.poisson(lam, N), sA2B + sigma * rng.poisson(lam, N)]
    B = (sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simb1(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Hub now also drives B; no private noise.
    s_ZG -> {Z, A1, B} ;  s_A2B -> {A2, B}
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)
    sA2B = rng.poisson(lam, N)

    Z = (sZG + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sigma * rng.poisson(lam, N), sA2B + sigma * rng.poisson(lam, N)]
    B = (sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simb2(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Sim b1 + private noise on A2 (s_A2P).
    s_ZG -> {Z, A1, B} ;  s_A2B -> {A2, B} ;  s_A2P -> A2
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1 and b global
    sA2B = rng.poisson(lam, N)           # a2 to b
    sA2P = rng.poisson(lam, N)           # a2 private

    Z = (sZG + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sigma * rng.poisson(lam, N), sA2B + sA2P + sigma * rng.poisson(lam, N)]
    B = (sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simb3(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Sim b1 + private noise on Z (s_ZP).
    s_ZG -> {Z, A1, B} ;  s_A2B -> {A2, B} ;  s_ZP -> Z
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1 and b global
    sA2B = rng.poisson(lam, N)           # a2 to b
    sZP = rng.poisson(lam, N)            # z private

    Z = (sZG + sZP + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sigma * rng.poisson(lam, N), sA2B + sigma * rng.poisson(lam, N)]
    B = (sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simb4(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Sim b1 + private noise on Z and A2 (s_ZP, s_A2P).
    s_ZG -> {Z, A1, B} ;  s_A2B -> {A2, B} ;  s_ZP -> Z, s_A2P -> A2
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1 and b global
    sA2B = rng.poisson(lam, N)           # a2 to b
    sZP, sA2P = rng.poisson(lam, N), rng.poisson(lam, N)  # z, a2 private

    Z = (sZG + sZP + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sigma * rng.poisson(lam, N), sA2B + sA2P + sigma * rng.poisson(lam, N)]
    B = (sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simb5(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Sim b1 + private noise on Z, A1, and A2 (s_ZP, s_A1P, s_A2P).
    s_ZG -> {Z, A1, B} ;  s_A2B -> {A2, B} ;  s_ZP -> Z, s_A1P -> A1, s_A2P -> A2
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1 and b global
    sA2B = rng.poisson(lam, N)           # a2 to b
    sZP, sA1P, sA2P = rng.poisson(lam, N), rng.poisson(lam, N), rng.poisson(lam, N)  # z, a1, a2 private

    Z = (sZG + sZP + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sA1P + sigma * rng.poisson(lam, N), sA2B + sA2P + sigma * rng.poisson(lam, N)]
    B = (sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simb6(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Sim b1, fully privatised: private noise on Z, A1, A2, and B.
    s_ZG -> {Z, A1, B} ;  s_A2B -> {A2, B}
    private: s_ZP -> Z ; s_A1P -> A1 ; s_A2P -> A2 ; s_BP -> B
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1 and b global
    sA2B = rng.poisson(lam, N)           # a2 to b
    sZP, sA1P, sA2P, sBP = (rng.poisson(lam, N), rng.poisson(lam, N),
                            rng.poisson(lam, N), rng.poisson(lam, N))  # z, a1, a2, b private

    Z = (sZG + sZP + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sA1P + sigma * rng.poisson(lam, N), sA2B + sA2P + sigma * rng.poisson(lam, N)]
    B = (sBP + sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simc1(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Hub now also drives A2 directly (full hub on {Z,A1,A2,B}); no private
    noise.
    s_ZG -> {Z, A1, A2, B} ;  s_A2B -> {A2, B}
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1, a2, and b global
    sA2B = rng.poisson(lam, N)           # a2 to b

    Z = (sZG + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sigma * rng.poisson(lam, N), sZG + sA2B + sigma * rng.poisson(lam, N)]
    B = (sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simc2(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Sim c1 + private noise on Z (s_ZP).
    s_ZG -> {Z, A1, A2, B} ;  s_A2B -> {A2, B} ;  s_ZP -> Z
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1, a2, and b global
    sA2B = rng.poisson(lam, N)           # a2 to b
    sZP = rng.poisson(lam, N)            # z private

    Z = (sZG + sZP + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sigma * rng.poisson(lam, N), sZG + sA2B + sigma * rng.poisson(lam, N)]
    B = (sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simc3(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Sim c1 + private noise on Z and A2 (s_ZP, s_A2P).
    s_ZG -> {Z, A1, A2, B} ;  s_A2B -> {A2, B} ;  s_ZP -> Z ; s_A2P -> A2
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1, a2, and b global
    sA2B = rng.poisson(lam, N)           # a2 to b
    sZP, sA2P = rng.poisson(lam, N), rng.poisson(lam, N)  # z, a2 private

    Z = (sZG + sZP + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sigma * rng.poisson(lam, N), sZG + sA2B + sA2P + sigma * rng.poisson(lam, N)]
    B = (sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simc4(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Sim c1 + private noise on Z, A1, and A2 (s_ZP, s_A1P, s_A2P).
    s_ZG -> {Z, A1, A2, B} ;  s_A2B -> {A2, B}
    private: s_ZP -> Z ; s_A1P -> A1 ; s_A2P -> A2
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1, a2, and b global
    sA2B = rng.poisson(lam, N)           # a2 to b
    sZP, sA1P, sA2P = rng.poisson(lam, N), rng.poisson(lam, N), rng.poisson(lam, N)  # z, a1, a2 private

    Z = (sZG + sZP + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sA1P + sigma * rng.poisson(lam, N), sZG + sA2B + sA2P + sigma * rng.poisson(lam, N)]
    B = (sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


def gen_simc5(N=N, sigma=SIGMA, lam=POISSON_LAM, seed=SEED):
    """Sim c1, fully privatised: private noise on Z, A1, A2, and B -- fully
    specified circuit.
    s_ZG -> {Z, A1, A2, B} ;  s_A2B -> {A2, B}
    private: s_ZP -> Z ; s_A1P -> A1 ; s_A2P -> A2 ; s_BP -> B
    """
    rng = _rng(seed)
    sZG = rng.poisson(lam, N)            # z to a1, a2, and b global
    sA2B = rng.poisson(lam, N)           # a2 to b
    sZP, sA1P, sA2P, sBP = (rng.poisson(lam, N), rng.poisson(lam, N),
                            rng.poisson(lam, N), rng.poisson(lam, N))  # z, a1, a2, b private

    Z = (sZG + sZP + sigma * rng.poisson(lam, N))[:, None]
    A = np.c_[sZG + sA1P + sigma * rng.poisson(lam, N), sZG + sA2B + sA2P + sigma * rng.poisson(lam, N)]
    B = (sBP + sZG + sA2B + sigma * rng.poisson(lam, N))[:, None]
    return A, B, Z


# =============================================================================
# SECTION 3 -- SCHEMATIC RENDERER (generic, config-driven)
# =============================================================================

# Fixed node layout shared by every simulation in this family.
_NODE_POS = {
    'Z':  (0.12, 0.50),
    'A1': (0.50, 0.80),
    'A2': (0.50, 0.20),
    'B':  (0.90, 0.50),
}
_NODE_STYLE = {
    'Z':  dict(fc='#fdd0a2', ec='#8c2d04', fs=10),
    'A1': dict(fc='#c6dbef', ec='#084594', fs=9.5),
    'A2': dict(fc='#c6dbef', ec='#084594', fs=9.5),
    'B':  dict(fc='#fcbba1', ec='#a50f15', fs=10),
}
_NODE_LABEL = {'Z': 'Z', 'A1': 'A\u2081', 'A2': 'A\u2082', 'B': 'B'}

_HUB_POS = (0.30, 0.50)
_LAT_POS = (0.68, 0.27)

# Private-noise satellite nodes: position + which direction the arrow enters from.
_PRIV_POS = {
    'Z':  (0.12, 0.92),
    'A1': (0.66, 0.97),
    'A2': (0.66, 0.03),
    'B':  (0.90, 0.92),
}
_PRIV_LABEL = {'Z': 's_ZP', 'A1': 's_A1P', 'A2': 's_A2P', 'B': 's_BP'}


def _draw_schematic_hub_lateral(ax, config):
    """Generic schematic for the hub + lateral-coupling simulation family.

    config keys:
      hub_targets     : list of node names driven by the global hub s_ZG
                        (subset of {'Z','A1','A2','B'})
      lateral_targets : list of node names driven by the lateral source
                        s_A2B (normally ['A2','B'])
      private         : list of node names that additionally receive a
                        node-local private-noise input
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    bs = dict(boxstyle='round,pad=0.15', lw=1.0, zorder=3)

    def node(xy, lbl, style, fs=None):
        ax.text(xy[0], xy[1], lbl, ha='center', va='center',
                fontsize=fs if fs is not None else style.get('fs', 9.5),
                fontweight='bold', zorder=4,
                bbox=dict(fc=style['fc'], ec=style['ec'], **bs))

    def arr(p0, p1, col='#333', lw=1.5, ls='solid'):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                    linestyle=ls, mutation_scale=10), zorder=2)

    # --- main circuit nodes -------------------------------------------------
    for key, xy in _NODE_POS.items():
        node(xy, _NODE_LABEL[key], _NODE_STYLE[key])

    # --- global hub s_ZG -----------------------------------------------------
    node(_HUB_POS, 's_ZG', dict(fc='#d9f0d3', ec=C_HUB), fs=8.5)
    for tgt in config['hub_targets']:
        tx, ty = _NODE_POS[tgt]
        hx, hy = _HUB_POS
        # short inset so the arrowhead doesn't sit on top of the node box
        dx, dy = tx - hx, ty - hy
        norm = np.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        arr((hx + 0.06 * ux, hy + 0.06 * uy), (tx - 0.07 * ux, ty - 0.07 * uy),
            col=C_HUB, lw=1.4)

    # --- lateral common drive s_A2B ------------------------------------------
    node(_LAT_POS, 's_A2B', dict(fc='#fdeacb', ec=C_COMM), fs=8.5)
    for tgt in config.get('lateral_targets', ['A2', 'B']):
        tx, ty = _NODE_POS[tgt]
        lx, ly = _LAT_POS
        dx, dy = tx - lx, ty - ly
        norm = np.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        arr((lx + 0.06 * ux, ly + 0.06 * uy), (tx - 0.07 * ux, ty - 0.07 * uy),
            col=C_COMM, lw=1.4)

    # --- private noise satellites --------------------------------------------
    for tgt in config.get('private', []):
        px, py = _PRIV_POS[tgt]
        node((px, py), _PRIV_LABEL[tgt], dict(fc='#e6d6ec', ec=C_COMP), fs=7.5)
        tx, ty = _NODE_POS[tgt]
        dx, dy = tx - px, ty - py
        norm = np.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        arr((px + 0.05 * ux, py + 0.05 * uy), (tx - 0.07 * ux, ty - 0.07 * uy),
            col=C_COMP, lw=1.3, ls='dashed')


# =============================================================================
# SECTION 4 -- VISUALISATION PRIMITIVES (unchanged from
#              pcca_benchmark_simulations.py: _compass, _wt_bars_compare,
#              _angle_color)
# =============================================================================

def _compass(ax, w_AB, w_AZ, sig_AB=None, sig_AZ=None, angle=None, title='', lbl_AZ='w_AZ'):
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

    for v in [sig_AB, sig_AZ]:
        if v is not None:
            uv = unit(np.asarray(v, float))
            ax.annotate("", xy=(0.78 * uv[0], 0.78 * uv[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=C_RAW,
                                        lw=1.2, linestyle='dashed', mutation_scale=10))

    for w, col, lbl in [(w_AB, C_AB, 'w_AB'), (w_AZ, C_AZ, lbl_AZ)]:
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


def _wt_bars_compare(ax, w_AB_cca, w_AZ_cca, w_AB_pcca, w_AZ_pcca, show_legend=True, lbl_AZ='w_AZ'):
    x = np.array([0.0, 1.0])
    bw = 0.17
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw

    nAB_c = unit(np.asarray(w_AB_cca, float))
    nAZ_c = unit(np.asarray(w_AZ_cca, float))
    nAB_p = unit(np.asarray(w_AB_pcca, float))
    nAZ_p = unit(np.asarray(w_AZ_pcca, float))

    ax.bar(x + offsets[0], nAB_c, bw, color=C_AB, alpha=0.45, hatch='//', label='w_AB  CCA')
    ax.bar(x + offsets[1], nAZ_c, bw, color=C_AZ, alpha=0.45, hatch='//', label=f'{lbl_AZ}  CCA')
    ax.bar(x + offsets[2], nAB_p, bw, color=C_AB, alpha=0.90, label='w_AB  pCCA')
    ax.bar(x + offsets[3], nAZ_p, bw, color=C_AZ, alpha=0.90, label=f'{lbl_AZ}  pCCA')

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
# SECTION 5 -- SIMULATION REGISTRY
# =============================================================================
# Each entry: (display title, generator function, schematic config)

EXT_SIMS = [
    ('Sim a \u2013 Minimal Hub\n(s_ZG \u2192 Z, A\u2081 only)',
     gen_sima,
     dict(hub_targets=['Z', 'A1'], lateral_targets=['A2', 'B'], private=[])),

    ('Sim b1 \u2013 Hub Reaches B\n(s_ZG \u2192 Z, A\u2081, B)',
     gen_simb1,
     dict(hub_targets=['Z', 'A1', 'B'], lateral_targets=['A2', 'B'], private=[])),

    ('Sim b2 \u2013 + A\u2082 private noise\n(s_A2P)',
     gen_simb2,
     dict(hub_targets=['Z', 'A1', 'B'], lateral_targets=['A2', 'B'], private=['A2'])),

    ('Sim b3 \u2013 + Z private noise\n(s_ZP)',
     gen_simb3,
     dict(hub_targets=['Z', 'A1', 'B'], lateral_targets=['A2', 'B'], private=['Z'])),

    ('Sim b4 \u2013 + Z, A\u2082 private noise\n(s_ZP, s_A2P)',
     gen_simb4,
     dict(hub_targets=['Z', 'A1', 'B'], lateral_targets=['A2', 'B'], private=['Z', 'A2'])),

    ('Sim b5 \u2013 + Z, A\u2081, A\u2082 private noise\n(s_ZP, s_A1P, s_A2P)',
     gen_simb5,
     dict(hub_targets=['Z', 'A1', 'B'], lateral_targets=['A2', 'B'], private=['Z', 'A1', 'A2'])),

    ('Sim b6 \u2013 Fully privatised\n(s_ZP, s_A1P, s_A2P, s_BP)',
     gen_simb6,
     dict(hub_targets=['Z', 'A1', 'B'], lateral_targets=['A2', 'B'], private=['Z', 'A1', 'A2', 'B'])),

    ('Sim c1 \u2013 Full Hub\n(s_ZG \u2192 Z, A\u2081, A\u2082, B), no private noise',
     gen_simc1,
     dict(hub_targets=['Z', 'A1', 'A2', 'B'], lateral_targets=['A2', 'B'], private=[])),

    ('Sim c2 \u2013 + Z private noise\n(s_ZP)',
     gen_simc2,
     dict(hub_targets=['Z', 'A1', 'A2', 'B'], lateral_targets=['A2', 'B'], private=['Z'])),

    ('Sim c3 \u2013 + Z, A\u2082 private noise\n(s_ZP, s_A2P)',
     gen_simc3,
     dict(hub_targets=['Z', 'A1', 'A2', 'B'], lateral_targets=['A2', 'B'], private=['Z', 'A2'])),

    ('Sim c4 \u2013 + Z, A\u2081, A\u2082 private noise\n(s_ZP, s_A1P, s_A2P)',
     gen_simc4,
     dict(hub_targets=['Z', 'A1', 'A2', 'B'], lateral_targets=['A2', 'B'], private=['Z', 'A1', 'A2'])),

    ('Sim c5 \u2013 Fully privatised\n(s_ZP, s_A1P, s_A2P, s_BP) \u2013 fully specified circuit',
     gen_simc5,
     dict(hub_targets=['Z', 'A1', 'A2', 'B'], lateral_targets=['A2', 'B'], private=['Z', 'A1', 'A2', 'B'])),
]


# =============================================================================
# SECTION 6 -- CCA vs pCCA COMPARISON FIGURE
# =============================================================================

def fig_hublateral_cca_vs_pcca(save=True):
    print("Building hub+lateral-coupling CCA vs pCCA comparison figure...")

    ncols = len(EXT_SIMS)
    lbl_AZ = 'w_AZ'

    fig = plt.figure(figsize=(4.6 * ncols, 15.5))
    gs = fig.add_gridspec(4, ncols, hspace=0.55, wspace=0.30, height_ratios=[1.0, 1.65, 1.65, 1.2])

    for col, (title, gen_fn, schem_cfg) in enumerate(EXT_SIMS):
        A, B, Z = gen_fn()

        # diagnostic covariances printed in the schematic header
        cov_bz = abs(np.cov(B.flatten(), Z.flatten())[0, 1])
        cov_a1a2 = abs(np.cov(A.T)[0, 1])

        sAB = sigma_dir(A, B)
        sAZ = sigma_dir(A, Z)

        # --- unconditional CCA ---------------------------------------------
        Wx_AB_c, _, _ = cca_simple(A, B)
        Wx_AZ_c, _, _ = cca_simple(A, Z)
        w_AB_c = Wx_AB_c[:, 0]
        w_AZ_c = Wx_AZ_c[:, 0]
        theta_c = pa_deg(w_AB_c, w_AZ_c)

        # --- partial CCA, each conditioned on the other regressor -----------
        Wx_AB_p, _, _, _, _ = pcca(A, B, Z=Z)
        Wx_AZ_p, _, _, _, _ = pcca(A, Z, Z=B)
        w_AB_p = Wx_AB_p[:, 0]
        w_AZ_p = Wx_AZ_p[:, 0]
        theta_p = pa_deg(w_AB_p, w_AZ_p)

        # --- row 0: schematic -------------------------------------------------
        ax0 = fig.add_subplot(gs[0, col])
        _draw_schematic_hub_lateral(ax0, schem_cfg)
        ax0.set_title(f"{title}\nCov(B,Z) = {cov_bz:.3f}     Cov(A\u2081,A\u2082) = {cov_a1a2:.3f}",
                       fontsize=9, fontweight='bold', pad=4)

        # --- row 1: CCA compass -------------------------------------------------
        ax1 = fig.add_subplot(gs[1, col])
        _compass(ax1, w_AB_c, w_AZ_c, sig_AB=sAB, sig_AZ=sAZ, lbl_AZ=lbl_AZ)
        ax1.set_title(f'CCA   \u03b8 = {theta_c:.1f}\u00b0',
                       fontsize=9, fontweight='bold', color=_angle_color(theta_c), pad=3)

        # --- row 2: pCCA compass -------------------------------------------------
        ax2 = fig.add_subplot(gs[2, col])
        _compass(ax2, w_AB_p, w_AZ_p, sig_AB=sAB, sig_AZ=sAZ, lbl_AZ=lbl_AZ)
        ax2.set_title(f'pCCA  \u03b8 = {theta_p:.1f}\u00b0',
                       fontsize=9, fontweight='bold', color=_angle_color(theta_p), pad=3)

        # --- row 3: weight comparison bars ----------------------------------
        ax3 = fig.add_subplot(gs[3, col])
        _wt_bars_compare(ax3, w_AB_c, w_AZ_c, w_AB_p, w_AZ_p, show_legend=(col == 0), lbl_AZ=lbl_AZ)
        ax3.set_ylim(-1.1, 1.15)

    row_labels = ['Circuit\ndiagram', 'CCA', 'pCCA', 'Weight\ncomparison']
    for y_frac, lbl in zip([0.882, 0.648, 0.394, 0.133], row_labels):
        fig.text(0.005, y_frac, lbl, va='center', ha='left',
                 fontsize=9, fontweight='bold', color='#444', rotation=90)

    if save:
        out = ('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/'
               'pCCA_simulation/fig3_hublateral_cca_vs_pcca.png')
        import os
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=480, bbox_inches='tight')
        print('  Saved: fig3_hublateral_cca_vs_pcca.png')
    plt.close(fig)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    fig_hublateral_cca_vs_pcca()


if __name__ == '__main__':
    main()
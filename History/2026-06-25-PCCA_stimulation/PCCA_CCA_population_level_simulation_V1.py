#!/usr/bin/env python3
"""
pcca_hublateral_population_simulations.py  (v2 — regulated)
=============================================================
Population-level benchmark, b-topology:

    s_ZG  -> {Z, A1, B}         (hub; does NOT project to A2)
    s_A2B -> {A2, B}             (the genuine A2 <-> B private lateral channel)

with optional Poisson private noise on every pool.

Visualization conventions are consistent with ``pcca_single_session.py``
throughout:
  - PSTH panels   : RdBu_r imshow, onset dashed line, colorbar
  - Weight bars   : horizontal barh, C3/C0 (positive/negative pCCA),
                    amber/purple (CCA), green (Z-to-region beta norms)
  - Latent z(t)  : thin translucent per-trial lines + mean +/- SEM overlay
  - Figure layout : 2 rows (region A / region B) x 7 columns, mirroring
                    pcca_sequential_ablation.py:
                    [PSTH | pCCA w | pCCA z(t) | CCA w | CCA z(t) |
                     Z PSTH | Z-to-region beta]

Key structural property of the b-topology
------------------------------------------
Ground-truth loading vectors in A's neuron space are exactly orthogonal:

    u_hub_A  = [g_ZG_A1  ;  0_{n_A2}]      -- A1-subspace only
    u_lat_A  = [0_{n_A1} ;  g_A2B_A2]      -- A2-subspace only

so that

    Sigma_AB = sigma^2_ZG  * u_hub_A u_hub_B^T
             + sigma^2_A2B * u_lat_A u_lat_B^T

is rank-2 with orthogonal terms (unlike the c-topology).

Contents
--------
  S0   Imports and global constants
  S1   Core pCCA / CCA utilities
  S2   Continuous b-topology generator   (N i.i.d. observations)
  S2b  Trial-structured b-topology generator   (n_trials x T peristimulus)
  S3   Ground-truth communication-plane utilities
  S4   Compass + weight-bar visualization primitives   (original)
  S5   PSTH / latent-trajectory visualization primitives
         (new; mirrors pcca_single_session.py)
  S6   Figure A -- Ablation-style PSTH + latent-trajectory figure   (new)
  S7   Figure B -- Compass + weight bars across population size
  S8   Figure C -- Subspace recovery vs population size
  S9   Figure D -- 3-D pCCA latent trajectories (noise-free)
  S10  Entry point
"""

from __future__ import annotations
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from scipy.linalg import subspace_angles

# =============================================================================
# S0  GLOBAL CONSTANTS
# =============================================================================

SEED        = 42
N           = 5_000
LAMBDA_REG  = 1e-4
CCA_ALPHA   = 1e-4
POISSON_LAM = 1.0
SIGMA_PRIV  = 0.0          # default private-noise magnitude (noisy setting)

# Trial-structure defaults (S2b)
N_TRIALS = 80
T_BINS   = 150             # 150 bins @ 30 ms/bin -> [-1.50, +2.97] s
T_ONSET  = 50              # bin index of reach onset -> t = 0 s
DT_SEC   = 0.030           # 30 ms per bin

# Colour palette
C_AB      = '#2166ac'   # pCCA(A,B|Z) / CCA(A,B)  (blue)
C_AZ      = '#d6604d'   # pCCA(A,Z|B) / CCA(A,Z)  (red-orange)
C_RAW     = '#969696'   # empirical cross-cov       (grey)
C_HUB     = '#4dac26'   # global hub s_ZG           (green)
C_COMP    = '#762a83'   # private noise             (purple)
C_COMM    = '#e08214'   # lateral source s_A2B      (amber)
C_CCA     = '#4393c3'   # CCA baseline              (light-blue)

# Ablation-figure row colours (mirroring pcca_sequential_ablation.py)
_A_PCCA = '#8B0000'    # dark red  -- pCCA z(t) region A
_A_CCA  = '#e08214'    # amber     -- CCA  z(t) region A
_B_PCCA = '#1f77b4'    # blue      -- pCCA z(t) region B
_B_CCA  = '#4393c3'    # lt-blue   -- CCA  z(t) region B

mpl.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    10,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        130,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

OUT_DIR = '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation'


# =============================================================================
# S1  CORE pCCA / CCA UTILITIES
# =============================================================================

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def pa_deg(w1: np.ndarray, w2: np.ndarray) -> float:
    """Principal angle between two 1-D weight vectors, in degrees."""
    d = np.linalg.norm(w1) * np.linalg.norm(w2)
    if d < 1e-12:
        return np.nan
    return float(np.degrees(np.arccos(
        np.clip(abs(float(w1 @ w2)) / d, 0., 1.))))


def partial_residuals(
    X: np.ndarray, Z: np.ndarray, lam: float = LAMBDA_REG
) -> np.ndarray:
    """X_tilde = X - Z (Z^T Z + lam I)^{-1} Z^T X."""
    q    = Z.shape[1]
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(q), Z.T @ X)
    test = Z @ beta
    return X - Z @ beta


def cca_svd(
    X: np.ndarray, Y: np.ndarray, k: int = 1, alpha: float = CCA_ALPHA
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regularised CCA via symmetric whitening + SVD.
    Returns (W_x, W_y, rho) with shapes (d_x, k), (d_y, k), (k,)."""
    Nc  = X.shape[0]
    Xc  = X - X.mean(0)
    Yc  = Y - Y.mean(0)
    Sxx = Xc.T @ Xc / Nc
    Syy = Yc.T @ Yc / Nc
    Sxy = Xc.T @ Yc / Nc

    def _inv_sqrt(S):
        lam_, V = np.linalg.eigh(S)
        lam_    = np.maximum(lam_, 0.) + alpha
        return V @ np.diag(lam_ ** -0.5) @ V.T

    Six, Siy  = _inv_sqrt(Sxx), _inv_sqrt(Syy)
    U, s, Vt  = np.linalg.svd(Six @ Sxy @ Siy, full_matrices=False)
    ke         = min(k, len(s))
    return Six @ U[:, :ke], Siy @ Vt.T[:, :ke], np.clip(s[:ke], 0., 1.)


def pcca(
    X: np.ndarray, Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
    k: int = 1, lam: float = LAMBDA_REG, alpha: float = CCA_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partial CCA.  Returns (Wx, Wy, rho, X_resid, Y_resid)."""
    if Z is not None and Z.ndim == 2 and Z.shape[1] > 0:
        Xr = partial_residuals(X, Z, lam)
        Yr = partial_residuals(Y, Z, lam)
    else:
        Xr = X - X.mean(0)
        Yr = Y - Y.mean(0)
    Wx, Wy, rho = cca_svd(Xr, Yr, k=k, alpha=alpha)
    return Wx, Wy, rho, Xr, Yr


def cca_simple(
    X: np.ndarray, Y: np.ndarray, k: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unconditional CCA (no nuisance).  Returns (Wx, Wy, rho)."""
    Wx, Wy, rho, _, _ = pcca(X, Y, Z=None, k=k)
    return Wx, Wy, rho


def raw_cross_dir(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Leading left singular vector of the empirical cross-covariance."""
    Ac, Bc  = A - A.mean(0), B - B.mean(0)
    S       = Ac.T @ Bc / A.shape[0]
    U, _, _ = np.linalg.svd(S, full_matrices=False)
    return U[:, 0]


def gini(x: np.ndarray) -> float:
    x   = np.sort(np.abs(np.asarray(x, float)))
    n   = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-12)) / n)


def make_gini_loadings(d: int, rng: np.random.Generator, shape: float = 1.5) -> np.ndarray:
    """Gamma-distributed loading weights, unit-normalised.  Gini ~ 0.35."""
    raw = rng.gamma(shape=shape, scale=1., size=d)
    return raw / np.linalg.norm(raw)


# =============================================================================
# S2  CONTINUOUS B-TOPOLOGY GENERATOR   (N i.i.d. observations)
# =============================================================================

def gen_population_b(
    n_A1: int, n_A2: int, n_B: int, n_Z: int,
    N: int = N, seed: int = SEED,
    lam: float = POISSON_LAM, sigma_priv: float = SIGMA_PRIV,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Continuous b-topology generator with trial-concatenated time-series.

    The latent source rates follow the same peristimulus profile as
    ``gen_population_b_trials`` — a Gaussian bump per trial of length T_BINS
    — tiled ceil(N / T_BINS) times and truncated to N observations.  This
    gives the continuous generator the identical temporal structure as the
    trial-structured version, enabling the 3-D trajectory figure to display
    coherent within-trial paths rather than i.i.d. noise.

    Hub    s_ZG  : peak at (T_ONSET - 10) bins  => t = -0.30 s  (pre-movement)
    Lateral s_A2B : peak at (T_ONSET + 10) bins  => t = +0.30 s  (post-movement)
    Both use sigma = 8 bins (= 0.24 s), identical to gen_population_b_trials.

    Returns A (N, n_A1+n_A2), B (N, n_B), Z (N, n_Z), gt dict.
    """
    rng = _rng(seed)

    # Per-trial Poisson rate profiles (length T_BINS)
    t_trial   = np.arange(T_BINS, dtype=float)
    sigma_s   = 8.0
    rate_hub  = 1.0 + 4.0 * np.exp(-0.5 * ((t_trial - (T_ONSET - 10.0)) / sigma_s) ** 2)
    rate_lat  = 1.0 + 3.0 * np.exp(-0.5 * ((t_trial - (T_ONSET + 10.0)) / sigma_s) ** 2)

    # Tile to N samples (last incomplete trial is truncated)
    reps         = int(np.ceil(N / T_BINS))
    rate_ZG_all  = np.tile(rate_hub, reps)[:N]
    rate_A2B_all = np.tile(rate_lat, reps)[:N]

    sZG  = rng.poisson(rate_ZG_all ).astype(float)   # (N,)  hub source
    sA2B = rng.poisson(rate_A2B_all).astype(float)   # (N,)  lateral source

    g_ZG_A1  = make_gini_loadings(n_A1, rng)
    g_A2B_A2 = make_gini_loadings(n_A2, rng)
    g_ZG_B   = make_gini_loadings(n_B,  rng)
    g_A2B_B = g_ZG_B
    #g_A2B_B  = make_gini_loadings(n_B,  rng)
    g_ZG_Z   = make_gini_loadings(n_Z,  rng)

    A1 = np.outer(sZG,  g_ZG_A1)  + sigma_priv * rng.poisson(lam, (N, n_A1))
    A2 = np.outer(sA2B, g_A2B_A2) + sigma_priv * rng.poisson(lam, (N, n_A2))
    A  = np.hstack([A1, A2])
    B  = (np.outer(sZG, g_ZG_B) + np.outer(sA2B, g_A2B_B)
          + sigma_priv * rng.poisson(lam, (N, n_B)))
    Z  = np.outer(sZG, g_ZG_Z) + sigma_priv * rng.poisson(lam, (N, n_Z))

    u_hub_A = np.concatenate([g_ZG_A1,        np.zeros(n_A2)])
    u_lat_A = np.concatenate([np.zeros(n_A1),  g_A2B_A2     ])

    gt = dict(
        u_hub_A    = u_hub_A,
        u_lat_A    = u_lat_A,
        sZG        = sZG,
        sA2B       = sA2B,
        rate_hub   = rate_hub,     # single-trial profile (T_BINS,)
        rate_lat   = rate_lat,
        sigma2_ZG  = lam,
        sigma2_A2B = lam,
        gini_A1    = gini(g_ZG_A1),
        gini_A2    = gini(g_A2B_A2),
        topology='hub'
    )
    return A, B, Z, gt


# =============================================================================
# S2b  TRIAL-STRUCTURED B-TOPOLOGY GENERATOR   (noise-free Poisson default)
# =============================================================================

def gen_population_b_trials(
    n_A1: int, n_A2: int, n_B: int, n_Z: int,
    n_trials:   int   = N_TRIALS,
    T:          int   = T_BINS,
    t_onset:    int   = T_ONSET,
    sigma_priv: float = 0.0,
    seed:       int   = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Trial-structured b-topology generator with time-varying Poisson profiles.

    Latent source rate profiles
    ---------------------------
    Hub s_ZG    : broad Gaussian ramp centred at onset, sigma_hub = 12 bins.
                  Analogous to a preparatory motor-cortical broadband signal.
    Lateral s_A2B: sharper burst, peaking 5 bins post-onset, sigma_lat = 8 bins.
                  Analogous to a phasic thalamo-cortical execution signal.

    The two profiles are temporally overlapping but distinct, producing
    heterogeneous PSTH structure inside region A:
        A1 neurons: broad early response (hub profile)
        A2 neurons: sharp slightly-later response (lateral profile)

    Noise-free Poisson (sigma_priv = 0)
    ------------------------------------
    All trial-to-trial variability derives from Poisson sampling of the
    latent sources.  In this regime:
        pCCA(A, B | Z) recovers  w proportional to u_lat_A  (A2 block only)
        CCA(A, B)      recovers a mixture of u_hub_A and u_lat_A
        Z->A beta-norms are large only for A1 neurons (hub-driven)

    Returns
    -------
    X_A      : (n_trials, n_A1+n_A2, T)
    X_B      : (n_trials, n_B, T)
    X_Z      : (n_trials, n_Z, T)
    time_vec : (T,)  seconds relative to reach onset
    gt       : dict  -- loading vectors, rate profiles, pool boundary
    """
    rng = _rng(seed)
    t   = np.arange(T, dtype=float)
    time_vec = (t - t_onset) * DT_SEC

    # Time-varying Poisson rate profiles
    # Hub peaks 10 bins PRE-onset  (t = -0.30 s): broad preparatory signal
    # Lateral peaks 10 bins POST-onset (t = +0.30 s): phasic execution signal
    # Both profiles use sigma = 8 bins (0.24 s), matching gen_population_b.
    sigma_s  = 8.0
    rate_ZG  = 1.0 + 4.0 * np.exp(-0.5 * ((t - (t_onset - 10.0)) / sigma_s) ** 2)
    rate_A2B = 1.0 + 3.0 * np.exp(-0.5 * ((t - (t_onset + 10.0)) / sigma_s) ** 2)

    # Loading vectors
    g_ZG_A1  = make_gini_loadings(n_A1, rng)
    g_A2B_A2 = make_gini_loadings(n_A2, rng)
    g_ZG_B   = make_gini_loadings(n_B,  rng)
    g_A2B_B = g_ZG_B
    #g_A2B_B  = make_gini_loadings(n_B,  rng)
    g_ZG_Z   = make_gini_loadings(n_Z,  rng)

    X_A1 = np.zeros((n_trials, n_A1, T))
    X_A2 = np.zeros((n_trials, n_A2, T))
    X_B  = np.zeros((n_trials, n_B,  T))
    X_Z  = np.zeros((n_trials, n_Z,  T))

    for r in range(n_trials):
        sZG_r  = rng.poisson(rate_ZG ).astype(float)   # (T,)
        sA2B_r = rng.poisson(rate_A2B).astype(float)   # (T,)

        X_A1[r] = np.outer(g_ZG_A1,  sZG_r )
        X_A2[r] = np.outer(g_A2B_A2, sA2B_r)
        X_B[r]  = np.outer(g_ZG_B,   sZG_r ) + np.outer(g_A2B_B, sA2B_r)
        X_Z[r]  = np.outer(g_ZG_Z,   sZG_r )

        if sigma_priv > 0.0:
            X_A1[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_A1, T))
            X_A2[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_A2, T))
            X_B[r]  += sigma_priv * rng.poisson(POISSON_LAM, (n_B,  T))
            X_Z[r]  += sigma_priv * rng.poisson(POISSON_LAM, (n_Z,  T))

    X_A = np.concatenate([X_A1, X_A2], axis=1)   # (n_trials, n_A1+n_A2, T)

    u_hub_A = np.concatenate([g_ZG_A1,        np.zeros(n_A2)])
    u_lat_A = np.concatenate([np.zeros(n_A1),  g_A2B_A2     ])

    gt = dict(
        u_hub_A  = u_hub_A,
        u_lat_A  = u_lat_A,
        g_ZG_A1  = g_ZG_A1,
        g_A2B_A2 = g_A2B_A2,
        g_ZG_B   = g_ZG_B,
        g_A2B_B  = g_A2B_B,
        g_ZG_Z   = g_ZG_Z,
        rate_ZG  = rate_ZG,
        rate_A2B = rate_A2B,
        n_A1     = n_A1,
        n_A2     = n_A2,
        gini_A1  = gini(g_ZG_A1),
        gini_A2  = gini(g_A2B_A2),
    )
    return X_A, X_B, X_Z, time_vec, gt
# =============================================================================
# S2c  COLLIDER TOPOLOGY  (Mode 2: X → Z ← Y)
# =============================================================================

def gen_population_collider_trials(
    n_X: int, n_Y: int, n_Z: int,
    n_trials:   int   = N_TRIALS,
    T:          int   = T_BINS,
    t_onset:    int   = T_ONSET,
    sigma_priv: float = 0.0,
    seed:       int   = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Mode 2 — Collider (X → Z ← Y).

    X is driven by an independent private source s_X; Y by an independent
    s_Y.  Z receives from BOTH.  There is NO direct X-Y coupling.

    Statistical derivation of the spurious pCCA correlation
    -------------------------------------------------------
    Population cross-covariance of pCCA residuals (annihilator M_Z applied):

        Cov(M_Z X, M_Z Y) = Sigma_XY - Sigma_XZ Sigma_ZZ^{-1} Sigma_ZY

    With Sigma_XY = 0, Sigma_XZ = sigma_X^2 g_X g_ZX^T, Sigma_ZY = sigma_Y^2 g_ZY g_Y^T:

        Cov = -sigma_X^2 sigma_Y^2 g_X [g_ZX^T Sigma_ZZ^{-1} g_ZY] g_Y^T

    The scalar bracket is nonzero whenever g_ZX and g_ZY are non-orthogonal —
    which holds generically for random Gini loadings on a finite Z pool.
    Thus pCCA rho_1 > CCA rho_1 ≈ 0 in the POPULATION LIMIT, not merely
    as a finite-sample artefact.  The magnitude decreases as n_Z grows and
    the random inner product g_ZX · g_ZY concentrates toward 0.

    Key signatures:  CCA rho ≈ 0,  pCCA rho > 0,  theta large,  kappa ≈ 1.
    kappa ≈ 1 because X has a single loading direction g_X — the direction
    coupling to Y (via the collider) and the direction coupling to Z (via s_X)
    are the same vector in X-space.

    Source temporal profiles (mirrors b-topology timing)
    -----------------------------------------------------
    s_X : pre-onset peak  (t = -0.30 s)
    s_Y : post-onset peak (t = +0.30 s)
    Z PSTH shows a characteristic two-humped profile (pre + post).
    """
    rng = _rng(seed)
    t        = np.arange(T, dtype=float)
    time_vec = (t - t_onset) * DT_SEC
    sigma_s  = 8.0

    rate_X = 1.0 + 4.0 * np.exp(-0.5 * ((t - (t_onset - 10.)) / sigma_s) ** 2)
    rate_Y = 1.0 + 3.0 * np.exp(-0.5 * ((t - (t_onset + 10.)) / sigma_s) ** 2)

    g_X  = make_gini_loadings(n_X, rng)
    g_Y  = make_gini_loadings(n_Y, rng)
    g_ZX = make_gini_loadings(n_Z, rng)   # Z's loading from s_X
    g_ZY = make_gini_loadings(n_Z, rng)   # Z's loading from s_Y  (independent)

    X_out = np.zeros((n_trials, n_X, T))
    Y_out = np.zeros((n_trials, n_Y, T))
    Z_out = np.zeros((n_trials, n_Z, T))

    for r in range(n_trials):
        sX = rng.poisson(rate_X).astype(float)
        sY = rng.poisson(rate_Y).astype(float)
        X_out[r] = np.outer(g_X,  sX)
        Y_out[r] = np.outer(g_Y,  sY)
        Z_out[r] = np.outer(g_ZX, sX) + np.outer(g_ZY, sY)   # collider
        if sigma_priv > 0.0:
            X_out[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_X, T))
            Y_out[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_Y, T))
            Z_out[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_Z, T))

    gt = dict(
        g_X=g_X, g_Y=g_Y, g_ZX=g_ZX, g_ZY=g_ZY,
        rate_X=rate_X, rate_Y=rate_Y,
        gini_X=gini(g_X), gini_Y=gini(g_Y),
        topology='collider',
        # No shared lateral ground truth — X and Y are independent
    )
    return X_out, Y_out, Z_out, time_vec, gt


# =============================================================================
# S2d  PARTIAL MEDIATOR TOPOLOGY  (Mode 3: relay through Z + direct bypass)
# =============================================================================

def gen_population_mediator_trials(
    n_X: int, n_Y: int, n_Z: int,
    n_trials:       int   = N_TRIALS,
    T:              int   = T_BINS,
    t_onset:        int   = T_ONSET,
    sigma_priv:     float = 0.0,
    seed:           int   = SEED,
    relay_fraction: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Mode 3 — Partial mediator (relay + bypass).

    s_relay  : drives X (direction g_X), Z (direction g_Z), and Y
               (direction g_Y_relay) — the relay path X → Z → Y.
    s_bypass : drives X (SAME direction g_X) and Y (direction g_Y_bypass)
               — the direct path X → Y that bypasses Z.

    Critical design: both sources use THE SAME loading direction g_X in X.
    This is the structural property that distinguishes Mode 3 from Mode 1:

      Mode 1 (b-topology): A1 neurons (hub) and A2 neurons (lateral) are
          DISJOINT pools → kappa ≈ 0.  The hub axis and the lateral axis in
          X are exactly orthogonal by construction.

      Mode 3 (mediator): relay and bypass engage THE SAME neuron pool →
          kappa → 1.  The direction in X that correlates with Y (via bypass)
          and the direction that correlates with Z (relay) are both g_X.

    This means pCCA(X, Y | Z) finds the bypass direction (still g_X after
    relay removal), while pCCA(X, Z | Y) also finds g_X (relay direction).
    Hence kappa = |cos(g_X, g_X)| = 1 identically.

    Biological interpretation: in Mode 3, the SAME neurons in X participate
    in both communication pathways (relay and direct).  In Mode 1, different
    neuron populations carry the hub and lateral signals.

    relay_fraction : fraction of X-Y coupling carried by the relay.
        Higher values → more severe over-removal by pCCA.
        At relay_fraction=1.0 (pure relay, no bypass):
            CCA rho ≫ 0,  pCCA rho ≈ 0  (total over-removal).
        At relay_fraction=0.0 (pure bypass, Mode 4 equivalent):
            CCA rho ≈ pCCA rho  (Z irrelevant for X-Y coupling).

    Key signatures: pCCA rho < CCA rho, large theta, kappa ≈ 1, large r2_nuis.
    """
    rng = _rng(seed)
    t        = np.arange(T, dtype=float)
    time_vec = (t - t_onset) * DT_SEC
    sigma_s  = 8.0

    # Relay peaks slightly PRE-onset; bypass peaks slightly POST-onset.
    # This gives the Z PSTH a distinct temporal profile from Y, making
    # the over-removal visually legible in the PSTH panels.
    amp_relay  = 4.0 * relay_fraction
    amp_bypass = 3.0 * (1.0 - relay_fraction + 0.25)

    rate_relay  = 1.0 + amp_relay  * np.exp(-0.5 * ((t - (t_onset - 5.)) / sigma_s) ** 2)
    rate_bypass = 1.0 + amp_bypass * np.exp(-0.5 * ((t - (t_onset + 5.)) / sigma_s) ** 2)

    # CRITICAL: single g_X for both relay and bypass in X → kappa = 1
    g_X        = make_gini_loadings(n_X, rng)
    g_Z        = make_gini_loadings(n_Z, rng)
    g_Y_relay  = make_gini_loadings(n_Y, rng)
    g_Y_bypass = make_gini_loadings(n_Y, rng)

    X_out = np.zeros((n_trials, n_X, T))
    Y_out = np.zeros((n_trials, n_Y, T))
    Z_out = np.zeros((n_trials, n_Z, T))

    for r in range(n_trials):
        s_rel = rng.poisson(rate_relay ).astype(float)
        s_byp = rng.poisson(rate_bypass).astype(float)

        X_out[r] = np.outer(g_X, s_rel + s_byp)           # same direction for both
        Z_out[r] = np.outer(g_Z, s_rel)                   # relay only
        Y_out[r] = (np.outer(g_Y_relay,  s_rel)
                  + np.outer(g_Y_bypass, s_byp))

        if sigma_priv > 0.0:
            X_out[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_X, T))
            Y_out[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_Y, T))
            Z_out[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_Z, T))

    gt = dict(
        g_X=g_X, g_Z=g_Z, g_Y_relay=g_Y_relay, g_Y_bypass=g_Y_bypass,
        rate_relay=rate_relay, rate_bypass=rate_bypass,
        relay_fraction=relay_fraction,
        gini_X=gini(g_X), gini_Y=gini(g_Y_relay),
        topology='mediator',
        # Ground-truth lateral direction in X: g_X (single pool)
        # pCCA should find g_X (bypass); CCA finds g_X (relay+bypass mixture)
    )
    return X_out, Y_out, Z_out, time_vec, gt


# =============================================================================
# S2e  INDEPENDENT CONFOUND TOPOLOGY  (Mode 4: Z exogenous)
# =============================================================================

def gen_population_independent_trials(
    n_X: int, n_Y: int, n_Z: int,
    n_trials:   int   = N_TRIALS,
    T:          int   = T_BINS,
    t_onset:    int   = T_ONSET,
    sigma_priv: float = 0.0,
    seed:       int   = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Mode 4 — Independent confounding (Z exogenous, unrelated to X-Y coupling).

    s_XY : shared source driving X (g_X) and Y (g_Y) — the genuine coupling.
           Peak at t_onset + 5 bins (post-onset, like the lateral source).
    s_Z  : independent source driving Z (g_Z) ONLY.
           Peak at t_onset - 10 bins (pre-onset, like ORB-type broadband signal).

    Population-limit result:
        Cov(M_Z X, M_Z Y) = Sigma_XY - Sigma_XZ Sigma_ZZ^{-1} Sigma_ZY
                          = sigma_XY^2 g_X g_Y^T - 0  (Sigma_XZ = 0, Sigma_ZY = 0)
                          = Sigma_XY

    The annihilator M_Z leaves Cov(X, Y) completely unchanged because Z shares
    no source with X or Y.  pCCA rho = CCA rho exactly in the population limit.
    Finite-sample deviations arise only from the ridge regularisation term in
    the hat matrix and are O(lambda / n).

    Key signatures: pCCA rho ≈ CCA rho, theta ≈ 0°, kappa ≈ 0, r2_nuis ≈ 0.
    kappa ≈ 0 because pCCA(X, Z | Y) finds effectively noise in X (Z is
    unrelated to X after removing Y), so w_IZ is a random direction; its
    cosine with w_IJ (which correctly recovers g_X) is negligible.
    """
    rng = _rng(seed)
    t        = np.arange(T, dtype=float)
    time_vec = (t - t_onset) * DT_SEC
    sigma_s  = 8.0

    rate_XY = 1.0 + 4.0 * np.exp(-0.5 * ((t - (t_onset + 5.)) / sigma_s) ** 2)
    rate_Z  = 1.0 + 3.0 * np.exp(-0.5 * ((t - (t_onset - 10.)) / sigma_s) ** 2)

    g_X = make_gini_loadings(n_X, rng)
    g_Y = make_gini_loadings(n_Y, rng)
    g_Z = make_gini_loadings(n_Z, rng)   # entirely independent

    X_out = np.zeros((n_trials, n_X, T))
    Y_out = np.zeros((n_trials, n_Y, T))
    Z_out = np.zeros((n_trials, n_Z, T))

    for r in range(n_trials):
        s_XY = rng.poisson(rate_XY).astype(float)
        s_Z  = rng.poisson(rate_Z ).astype(float)
        X_out[r] = np.outer(g_X, s_XY)
        Y_out[r] = np.outer(g_Y, s_XY)
        Z_out[r] = np.outer(g_Z, s_Z)              # no causal link to X or Y
        if sigma_priv > 0.0:
            X_out[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_X, T))
            Y_out[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_Y, T))
            Z_out[r] += sigma_priv * rng.poisson(POISSON_LAM, (n_Z, T))

    gt = dict(
        g_X=g_X, g_Y=g_Y, g_Z=g_Z,
        rate_XY=rate_XY, rate_Z=rate_Z,
        gini_X=gini(g_X), gini_Y=gini(g_Y),
        topology='independent',
        u_true_X=g_X,   # pCCA should recover g_X unchanged (theta ≈ 0)
        u_true_Y=g_Y,
    )
    return X_out, Y_out, Z_out, time_vec, gt

# =============================================================================
# S3  GROUND-TRUTH COMMUNICATION-PLANE UTILITIES
# =============================================================================

def orthonormal_plane(u1: np.ndarray, u2: np.ndarray) -> Tuple[np.ndarray, float]:
    """Orthonormal basis E (d x 2) for span{u1, u2} via economy QR.
    Returns (E, phi) where phi = angle(u1, u2) in degrees.
    For the b-topology phi = 90 deg by construction."""
    M    = np.stack([u1, u2], axis=1)
    Q, _ = np.linalg.qr(M)
    phi  = float(np.degrees(np.arccos(
        np.clip(abs(unit(u1) @ unit(u2)), 0., 1.))))
    return Q[:, :2], phi


# =============================================================================
# S4  COMPASS + WEIGHT-BAR VISUALIZATION PRIMITIVES   (original, unchanged)
# =============================================================================

def _compass_population(ax, w_AB, w_AZ, E, raw_dir=None, angle=None, title='',
                         basis_labels=('hub-aligned axis', 'lateral-residual axis')):
    """Unit-circle compass of pCCA/CCA weight vectors projected onto the
    ground-truth communication plane."""
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#c0c0c0', lw=0.8, zorder=0)
    ax.axhline(0, color='#d0d0d0', lw=0.5)
    ax.axvline(0, color='#d0d0d0', lw=0.5)
    ax.set_aspect('equal')
    lm = 1.45
    ax.set_xlim(-lm, lm);  ax.set_ylim(-lm, lm)
    ax.set_xlabel(basis_labels[0], fontsize=8)
    ax.set_ylabel(basis_labels[1], fontsize=8)
    ax.set_xticks([-1, 0, 1]);  ax.set_yticks([-1, 0, 1])

    if raw_dir is not None:
        p = E.T @ raw_dir
        if np.linalg.norm(p) > 1e-12:
            up = unit(p)
            ax.annotate('', xy=(.78*up[0], .78*up[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=C_RAW,
                                        lw=1.2, linestyle='dashed',
                                        mutation_scale=10))

    in_plane = {}
    for w, col, lbl in [(w_AB, C_AB, 'w_AB'), (w_AZ, C_AZ, 'w_AZ')]:
        w  = np.asarray(w, float)
        p  = E.T @ w
        in_plane[lbl] = float(np.linalg.norm(p) / (np.linalg.norm(w) + 1e-12))
        up = unit(p)
        ax.annotate('', xy=(up[0], up[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=col,
                                    lw=2.2, mutation_scale=12))
        ax.text(1.22*up[0], 1.22*up[1], lbl, ha='center', va='center',
                fontsize=8.5, color=col, fontweight='bold')

    hdr = title
    if angle is not None:
        hdr += ('\n' if title else '') + f'theta = {angle:.1f} deg'
    hdr += (f"\nin-plane: w_AB {in_plane['w_AB']*100:.0f}%,"
            f" w_AZ {in_plane['w_AZ']*100:.0f}%")
    ax.set_title(hdr, fontsize=8.5, pad=3, fontweight='bold')


def _wt_bars_population(ax, w_AB_cca, w_AZ_cca, w_AB_pcca, w_AZ_pcca, E,
                         show_legend=True, cats=None):
    """Grouped bar chart: CCA vs pCCA weight projections onto hub / lateral /
    out-of-plane components."""
    if cats is None:
        cats = ['hub axis', 'lateral axis', 'out-of-plane']
    x       = np.arange(3)
    bw      = 0.17
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw

    def _proj3(w):
        w   = unit(np.asarray(w, float))
        p   = E.T @ w
        oop = float(np.sqrt(max(0., 1. - p @ p)))
        return np.array([p[0], p[1], oop])

    vAB_c, vAZ_c = _proj3(w_AB_cca),  _proj3(w_AZ_cca)
    vAB_p, vAZ_p = _proj3(w_AB_pcca), _proj3(w_AZ_pcca)

    ax.bar(x + offsets[0], vAB_c, bw, color=C_AB, alpha=0.45, hatch='//', label='w_AB  CCA')
    ax.bar(x + offsets[1], vAZ_c, bw, color=C_AZ, alpha=0.45, hatch='//', label='w_AZ  CCA')
    ax.bar(x + offsets[2], vAB_p, bw, color=C_AB, alpha=0.90,              label='w_AB  pCCA')
    ax.bar(x + offsets[3], vAZ_p, bw, color=C_AZ, alpha=0.90,              label='w_AZ  pCCA')

    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x);  ax.set_xticklabels(cats)
    ax.set_ylabel('Projected component', fontsize=8)
    if show_legend:
        ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_ylim(-0.15, 1.15)



# =============================================================================
# S5  PSTH / LATENT-TRAJECTORY VISUALIZATION PRIMITIVES
#     Mirrors pcca_single_session.py conventions throughout.
# =============================================================================

def _zscore_sim(X: np.ndarray) -> np.ndarray:
    """Z-score each neuron across all (trial x time) samples.

    Mirrors OxfordPCCASessionVisualizer._zscore_per_neuron():
    the full (N_flat,) trace per neuron is normalised to zero mean,
    unit variance before computing the trial-averaged PSTH.

    X   : (n_trials, n_neurons, T)
    Returns (n_trials, n_neurons, T)
    """
    n_trials, n, T = X.shape
    flat_raw = X.transpose(1, 0, 2).reshape(n, n_trials * T)   # (n, N_flat)
    mu   = flat_raw.mean(axis=1, keepdims=True)
    std  = flat_raw.std( axis=1, keepdims=True)
    std[std < 1e-7] = 1.0
    flat = (flat_raw - mu)/std
    test = flat.reshape(n, n_trials,T).transpose(1, 0, 2)
    return flat.reshape(n, n_trials,T).transpose(1, 0, 2) # (n_trials, n, T)


def _sim_neuron_sort_A(gt: dict) -> Tuple[np.ndarray, int]:
    """Ground-truth neuron ordering for region A.

    A1 neurons (hub-responsive, indices 0..n_A1-1) are placed first, sorted
    by g_ZG_A1 descending; A2 neurons (lateral-responsive, indices
    n_A1..n_A-1) follow, sorted by g_A2B_A2 descending.

    This ordering makes the PSTH split between the two pools immediately
    visible, and ensures the weight-bar panels are spatially aligned:
    pCCA weights are concentrated in the lower (A2) block, while CCA
    weights spread across both blocks.

    Returns
    -------
    sort_idx     : (n_A,) index array for PSTH / weight-bar subsampling
    boundary_idx : int    n_A1 -- absolute position of the A1/A2 boundary
    """
    n_A1    = gt['n_A1']
    sort_A1 = np.argsort(-gt['g_ZG_A1'])                  # within A1 block
    sort_A2 = np.argsort(-gt['g_A2B_A2']) + n_A1          # offset to A2 block
    return np.concatenate([sort_A1, sort_A2]), n_A1


def _draw_sim_rastermap_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    X_z: np.ndarray,
    sort_idx: np.ndarray,
    time_vec: np.ndarray,
    region_name: str,
    n_show: int = 60,
    boundary_idx: Optional[int] = None,
) -> None:
    """Trial-averaged PSTH imshow panel.

    Mirrors _draw_rastermap_panel() from pcca_single_session.py:
      - RdBu_r colourmap, vmax = 99th percentile of |PSTH|
      - Vertical dashed line at t = 0 (reach onset)
      - Optional dotted horizontal line + pool labels at A1/A2 boundary

    Parameters
    ----------
    X_z          : (n_trials, n_neurons, T)  z-scored spike data
    sort_idx     : (n_neurons,) neuron ordering (ground-truth or Rastermap)
    boundary_idx : absolute neuron index marking the A1/A2 pool boundary;
                   None => no annotation (use for region B and Z)
    """
    n_neurons = X_z.shape[1]
    if sort_idx.size != n_neurons:
        sort_idx = np.arange(n_neurons)

    step  = max(1, n_neurons // n_show)
    sel   = sort_idx[::step][:n_show]
    n_sel = len(sel)

    psth = X_z.mean(axis=0)[sel]        # (n_sel, T) trial-averaged PSTH
    vmax = max(float(np.nanpercentile(np.abs(psth), 99)), 0.5)

    im = ax.imshow(
        psth, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        extent=[time_vec[0], time_vec[-1], n_sel, 0], origin='upper',
    )
    ax.axvline(0, color='k', linestyle='--', lw=1.5, alpha=0.85, label='Onset')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.35)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Neurons (sorted)', fontsize=9)
    ax.set_title(f'{region_name} PSTH', fontsize=10, fontweight='bold')

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('Z-scored rate', fontsize=7)

    # A1 / A2 pool-boundary annotation (region A only)
    if boundary_idx is not None:
        n_A1_in_sel = int(np.sum(sel < boundary_idx))
        if 0 < n_A1_in_sel < n_sel:
            ax.axhline(n_A1_in_sel, color='#222222', lw=1.0,
                       linestyle=':', alpha=0.80)
            t_lbl = time_vec[0] + 0.05 * (time_vec[-1] - time_vec[0])
            ax.text(t_lbl, n_A1_in_sel / 2.0,
                    'A1 (hub)', fontsize=7, va='center',
                    color='#333333', style='italic',
                    bbox=dict(fc='white', ec='none', alpha=0.65, pad=1))
            ax.text(t_lbl, n_A1_in_sel + (n_sel - n_A1_in_sel) / 2.0,
                    'A2 (lat.)', fontsize=7, va='center',
                    color='#333333', style='italic',
                    bbox=dict(fc='white', ec='none', alpha=0.65, pad=1))


def _draw_sim_weight_bar(
    ax: plt.Axes,
    sort_idx: np.ndarray,
    weight: np.ndarray,
    bar_label: str,
    n_show: int = 60,
    pos_color: str = 'C3',
    neg_color: str = 'C0',
    rho: Optional[float] = None,
) -> None:
    """Horizontal weight barh panel.

    Mirrors the barh sub-panel inside _draw_rastermap_panel() from
    pcca_single_session.py.  The same sort_idx as the corresponding PSTH
    panel is applied to maintain neuron-row alignment between PSTH and bars.

    Parameters
    ----------
    weight    : (n_neurons,) canonical weight or beta-norm vector
    bar_label : title text;  rho_1 is appended when rho is not None
    pos_color : colour for positive weights  (C3=red, pCCA convention)
    neg_color : colour for negative weights  (C0=blue)
    """
    n_neurons = weight.size
    step  = max(1, n_neurons // n_show)
    sel   = sort_idx[::step][:n_show]
    n_sel = len(sel)

    ypos   = np.arange(n_sel) + 0.5
    w_sel  = weight[sel].ravel()
    colors = [pos_color if w >= 0 else neg_color for w in w_sel]

    ax.barh(ypos, w_sel, height=0.8, color=colors, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8, alpha=0.6)
    ax.set_ylim(n_sel, 0)
    title_str = bar_label if rho is None else f'{bar_label}\nrho1={rho:.3f}'
    ax.set_title(title_str, fontsize=8, fontweight='bold')
    ax.set_xlabel('weight', fontsize=7)
    ax.tick_params(axis='both', labelsize=7)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    plt.setp(ax.get_yticklabels(), visible=False)
# =============================================================================
# S5b  LATENT SIGN ALIGNMENT UTILITY
# =============================================================================
# =============================================================================
# S5b  LATENT SIGN ALIGNMENT UTILITY
# =============================================================================

# --- place align_projection_signs here (unchanged) ---
def align_projection_signs(
    projections: dict,
    n_components: int,
    reference_type: str,
    time_bins: Optional[np.ndarray] = None
) -> dict:
    """
    Align canonical variate signs across trial types via Z2 spectral
    synchronisation. Adapted to operate as a standalone function for the
    cross-session pipeline.

    Algorithm
    ---------
    For each CCA component k:

    1. Build pairwise Pearson-correlation matrices.
       Stack the mean time-courses into U of shape (C, T) and compute
       C_u = corrcoef(U) in R^{C x C}. Same for V.

    2. Z2 synchronisation via leading eigenvector.
       s = sign(q_max), where q_max is the leading eigenvector of C
       (last column returned by np.linalg.eigh).

    3. Apply per-trial-type signs.

    4. Anchor global orientation to the reference condition.
       The epoch-mean of the reference trial type in [0, 1.5s] must be
       positive.

    5. Fine correction via signed peak.
       The value at the time of maximum absolute deviation of the
       reference epoch must be positive.

    6. Write back aligned means and propagate flips to per-trial data.
       STD and SEM are sign-invariant and are left unchanged.

    Parameters
    ----------
    projections : dict
        Nested dictionary with structure:
        {trial_type: {'u_mean': ndarray, 'v_mean': ndarray,
                      'u_trials': ndarray, 'v_trials': ndarray}}
    n_components : int
        Number of CCA/pCCA components to align.
    reference_type : str
        The trial type key to use as the positive anchor.
    time_bins : np.ndarray, optional
        The continuous time axis to resolve epoch bins.

    Returns
    -------
    flip_decisions : dict
        {trial_type -> {comp_idx -> {'u_flip': bool, 'v_flip': bool}}}
    """
    trial_types = list(projections.keys())

    # Resolve task-epoch bin indices from the continuous time axis
    if time_bins is not None:
        t_start = int(np.searchsorted(time_bins, 0.0))
        t_end   = int(np.searchsorted(time_bins, 1.5))
    else:
        t_start, t_end = 75, 151  # fallback for 226-bin default axis

    from typing import Dict
    flip_decisions: Dict[str, Dict] = {
        tt: {k: {'u_flip': False, 'v_flip': False} for k in range(n_components)}
        for tt in trial_types
    }

    # Stack mean time-courses: (n_types, n_time, n_components)
    u_stack = np.stack(
        [projections[tt]['u_mean'] for tt in trial_types], axis=0
    )
    v_stack = np.stack(
        [projections[tt]['v_mean'] for tt in trial_types], axis=0
    )

    ref_idx = trial_types.index(reference_type)

    for comp_idx in range(n_components):

        # ---------------------------------------------------------------- #
        # Step 1  Build pairwise Pearson-correlation matrices               #
        # ---------------------------------------------------------------- #
        U = u_stack[:, :, comp_idx]   # (n_types, n_time)
        V = v_stack[:, :, comp_idx]

        C_u = np.atleast_2d(np.corrcoef(U))
        C_v = np.atleast_2d(np.corrcoef(V))
        # ---------------------------------------------------------------- #
        # Step 2  Z2 synchronisation — leading eigenvector                  #
        # ---------------------------------------------------------------- #
        _, evecs_u = np.linalg.eigh(C_u)
        s_u = np.sign(evecs_u[:, -1])

        _, evecs_v = np.linalg.eigh(C_v)
        s_v = np.sign(evecs_v[:, -1])

        # Guard: eigh can return 0.0 for degenerate entries (very rare)
        s_u[s_u == 0] = 1
        s_v[s_v == 0] = 1

        # ---------------------------------------------------------------- #
        # Step 3  Apply per-trial-type signs                                #
        # ---------------------------------------------------------------- #
        u_stack[:, :, comp_idx] = s_u[:, np.newaxis] * U
        v_stack[:, :, comp_idx] = s_v[:, np.newaxis] * V

        # ---------------------------------------------------------------- #
        # Step 4  Global orientation — epoch mean of reference must be +   #
        # ---------------------------------------------------------------- #
        u_ref_mean = u_stack[ref_idx, t_start:t_end, comp_idx].mean()
        v_ref_mean = v_stack[ref_idx, t_start:t_end, comp_idx].mean()

        if u_ref_mean < 0:
            s_u *= -1
            u_stack[:, :, comp_idx] *= -1

        if v_ref_mean < 0:
            s_v *= -1
            v_stack[:, :, comp_idx] *= -1

        # ---------------------------------------------------------------- #
        # Step 5  Fine correction — signed peak of reference epoch must +  #
        # ---------------------------------------------------------------- #
        u_ref_epoch = u_stack[ref_idx, t_start:t_end, comp_idx]
        v_ref_epoch = v_stack[ref_idx, t_start:t_end, comp_idx]

        u_peak_val = u_ref_epoch[np.argmax(np.abs(u_ref_epoch))]
        v_peak_val = v_ref_epoch[np.argmax(np.abs(v_ref_epoch))]

        if u_peak_val < 0:
            s_u *= -1
            u_stack[:, :, comp_idx] *= -1

        if v_peak_val < 0:
            s_v *= -1
            v_stack[:, :, comp_idx] *= -1

        # ---------------------------------------------------------------- #
        # Step 6  Record net flip decisions                                 #
        # ---------------------------------------------------------------- #
        for tt_idx, tt in enumerate(trial_types):
            flip_decisions[tt][comp_idx] = {
                'u_flip': bool(s_u[tt_idx] < 0),
                'v_flip': bool(s_v[tt_idx] < 0),
            }

    # Write aligned means back; propagate flips to per-trial data
    for tt_idx, tt in enumerate(trial_types):
        proj = projections[tt]
        for comp_idx in range(n_components):
            fd = flip_decisions[tt][comp_idx]
            if fd['u_flip']:
                proj['u_trials'][:, :, comp_idx] *= -1
            if fd['v_flip']:
                proj['v_trials'][:, :, comp_idx] *= -1
        proj['u_mean'] = u_stack[tt_idx]
        proj['v_mean'] = v_stack[tt_idx]

    print(f"    Sign alignment complete ({n_components} components, "
          f"{len(trial_types)} trial types).")
    return flip_decisions

def _apply_sign_alignment_to_sim_results(
    res: dict,
    time_vec: np.ndarray,
) -> None:
    """Apply Z2-spectral sign alignment to simulation latent trajectories.

    Adapts ``align_projection_signs`` to operate on the flat ``res`` dict
    returned by ``_compute_sim_pair_results``.  The simulation has a single
    condition ('sim'), so Steps 1-3 of the Z2 algorithm are degenerate
    (a 1×1 correlation matrix has a trivially positive leading eigenvector);
    the substantive work is done by Steps 4-5:

      Step 4  Global anchor  — epoch mean of u / v over [0, 1.5 s] must be ≥ 0.
      Step 5  Fine correction — signed peak within that epoch must be ≥ 0.

    Both pCCA and CCA latent pairs are aligned independently:

        (z_A_p, z_B_p)  with weight propagation to  (Wx_p, Wy_p)
        (z_A_c, z_B_c)  with weight propagation to  (Wx_c, Wy_c)

    Propagating the flip decision to the canonical weight vectors is
    essential: a sign flip on z_A = X_resid @ Wx is equivalent to negating
    Wx, so the weight-bar panel (col 1 / col 3) must be updated to remain
    spatially consistent with the PSTH panel (col 0).

    Parameters
    ----------
    res      : dict returned by ``_compute_sim_pair_results``; modified
               **in-place**.
    time_vec : (T,) time axis in seconds, passed directly to
               ``align_projection_signs`` so the epoch bounds [0, 1.5 s]
               are resolved correctly irrespective of T or t_onset.
    """
    for z_u_key, z_v_key, w_u_key, w_v_key, label in [
        ('z_A_p', 'z_B_p', 'Wx_p', 'Wy_p', 'pCCA'),
        ('z_A_c', 'z_B_c', 'Wx_c', 'Wy_c', 'CCA'),
    ]:
        z_u = res[z_u_key]   # (n_trials, T)
        z_v = res[z_v_key]   # (n_trials, T)

        # ── Pack into the projections schema ─────────────────────────────────
        # align_projection_signs expects:
        #   u_mean   : (T, n_components)   — trial-averaged latent
        #   v_mean   : (T, n_components)
        #   u_trials : (n_trials, T, n_components)
        #   v_trials : (n_trials, T, n_components)
        # We use n_components = 1, trial_type key = 'sim'.
        projections = {
            'sim': {
                'u_mean':   z_u.mean(axis=0)[:, np.newaxis],   # (T, 1)
                'v_mean':   z_v.mean(axis=0)[:, np.newaxis],   # (T, 1)
                'u_trials': z_u[:, :, np.newaxis],             # (n_trials, T, 1)
                'v_trials': z_v[:, :, np.newaxis],             # (n_trials, T, 1)
            }
        }

        flip_decisions = align_projection_signs(
            projections,
            n_components=1,
            reference_type='sim',
            time_bins=time_vec,
        )

        fd = flip_decisions['sim'][0]   # {'u_flip': bool, 'v_flip': bool}

        # ── Unpack aligned arrays back into res ──────────────────────────────
        res[z_u_key] = projections['sim']['u_trials'][:, :, 0]   # (n_trials, T)
        res[z_v_key] = projections['sim']['v_trials'][:, :, 0]

        # ── Propagate flips to canonical weight vectors ───────────────────────
        # z = X_resid @ w  =>  flipping z is equivalent to negating w.
        # Without this step the weight-bar panel is inconsistent with z(t).
        if fd['u_flip']:
            res[w_u_key] = -res[w_u_key]
        if fd['v_flip']:
            res[w_v_key] = -res[w_v_key]

        print(f"    [Sign align | {label:4s}]  "
              f"u_flip = {fd['u_flip']},  v_flip = {fd['v_flip']}")



def _plot_sim_latent_trials(
    ax: plt.Axes,
    time_vec: np.ndarray,
    z_trials: np.ndarray,
    color: str,
    title: str = '',
    alpha_trial: float = 0.06,
    lw_trial:    float = 0.28,
    gt_profile:  Optional[np.ndarray] = None,
    gt_label:    Optional[str]        = None,
) -> None:
    """Per-trial latent traces (thin, translucent) + mean +/- SEM overlay.

    Mirrors OxfordPCCASessionVisualizer._plot_with_trials().

    Parameters
    ----------
    z_trials   : (n_trials, T)
    gt_profile : (T,) optional ground-truth Poisson rate profile.
                 Scaled to match the mean amplitude and overlaid as a thin
                 dashed black line.  Allows the reader to verify:
                   - pCCA z_A(t) tracks rate_A2B  (lateral source only)
                   - CCA  z_A(t) tracks rate_ZG + rate_A2B  (mixture)
    """
    # Individual trial traces (rasterised for PDF efficiency)
    for tr in z_trials:
        ax.plot(time_vec, tr, color=color, lw=lw_trial,
                alpha=alpha_trial, rasterized=True)

    # Mean +/- SEM
    mean = np.nanmean(z_trials, axis=0)
    sem  = np.nanstd(z_trials,  axis=0) / np.sqrt(z_trials.shape[0])
    ax.plot(time_vec, mean, color=color, lw=2.2, zorder=3)
    ax.fill_between(time_vec, mean - sem, mean + sem,
                    color=color, alpha=0.25, zorder=2)

    # Ground-truth source-rate overlay
    if gt_profile is not None:
        amp  = np.abs(mean).max() + 1e-12
        span = gt_profile.max() - gt_profile.min() + 1e-12
        p_n  = (gt_profile - gt_profile.mean()) / span * amp * 0.9
        ax.plot(time_vec, p_n, color='#222222', lw=1.0,
                ls='--', alpha=0.70, zorder=4,
                label=gt_label if gt_label else 'GT source')
        ax.legend(fontsize=6.5, frameon=False, loc='upper right')

    ax.axvline(0, color='k', ls='--', lw=0.9, alpha=0.45)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylim(-3, 3)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def _compute_sim_pair_results(
    X_A: np.ndarray,   # (n_trials, n_A, T)
    X_B: np.ndarray,   # (n_trials, n_B, T)
    X_Z: np.ndarray,   # (n_trials, n_Z, T)
    k: int = 1,
) -> dict:
    """Run pCCA(A,B|Z) and CCA(A,B) on trial-structured data.

    Flattens (n_trials, n_neurons, T) -> (n_trials*T, n_neurons) so that
    every (trial, time) pair is treated as one IID observation.  The same
    convention is used by perform_session_pcca.m for the real Oxford data.

    Returns
    -------
    dict with keys:
      Wx_p, Wy_p, rho_p  --  pCCA canonical weight (n_A,), (n_B,) and
                               first canonical correlation
      Wx_c, Wy_c, rho_c  --  CCA  canonical weight and correlation
      z_A_p, z_B_p        --  pCCA latent trajectories  (n_trials, T)
      z_A_c, z_B_c        --  CCA  latent trajectories  (n_trials, T)
      beta_A_norm          --  ||Z->A beta||_2 per neuron  (n_A,)
      beta_B_norm          --  ||Z->B beta||_2 per neuron  (n_B,)
    """
    n_trials, n_A, T = X_A.shape
    n_B = X_B.shape[1]
    n_Z = X_Z.shape[1]

    # Flatten: (n_trials, n, T) -> (n_trials*T, n)
    # Trial r occupies rows r*T ... (r+1)*T - 1.
    A_flat = X_A.transpose(0, 2, 1).reshape(n_trials * T, n_A)
    B_flat = X_B.transpose(0, 2, 1).reshape(n_trials * T, n_B)
    Z_flat = X_Z.transpose(0, 2, 1).reshape(n_trials * T, n_Z)

    # pCCA(A, B | Z)
    Wx_p, Wy_p, rho_p, A_resid, B_resid = pcca(A_flat, B_flat, Z=Z_flat, k=k)
    z_A_p = (A_resid @ Wx_p[:, 0]).reshape(n_trials, T)
    z_B_p = (B_resid @ Wy_p[:, 0]).reshape(n_trials, T)

    # CCA(A, B) -- unconditional baseline
    Wx_c, Wy_c, rho_c = cca_simple(A_flat, B_flat, k=k)
    A_c   = A_flat - A_flat.mean(0)
    B_c   = B_flat - B_flat.mean(0)
    z_A_c = (A_c @ Wx_c[:, 0]).reshape(n_trials, T)
    z_B_c = (B_c @ Wy_c[:, 0]).reshape(n_trials, T)

    # Z -> {A, B} regression weights  (column-norms of beta matrix)
    gram_Z      = Z_flat.T @ Z_flat + LAMBDA_REG * np.eye(n_Z)
    beta_ZA     = np.linalg.solve(gram_Z, Z_flat.T @ A_flat)   # (n_Z, n_A)
    beta_ZB     = np.linalg.solve(gram_Z, Z_flat.T @ B_flat)   # (n_Z, n_B)
    beta_A_norm = np.linalg.norm(beta_ZA, axis=0)              # (n_A,)
    beta_B_norm = np.linalg.norm(beta_ZB, axis=0)              # (n_B,)

    # Reshape flat residuals back to trial structure for residual-PSTH display.
    # (n_trials*T, n)  ->  (n_trials, T, n)  ->  transpose to (n_trials, n, T)
    # Convention matches the (n_trials, n_neurons, T) input layout so that
    # _zscore_sim and _draw_sim_rastermap_panel can be called directly.
    X_A_resid = A_resid.reshape(n_trials, T, n_A).transpose(0, 2, 1)
    X_B_resid = B_resid.reshape(n_trials, T, n_B).transpose(0, 2, 1)

    return dict(
        Wx_p=Wx_p[:, 0], Wy_p=Wy_p[:, 0], rho_p=float(rho_p[0]),
        Wx_c=Wx_c[:, 0], Wy_c=Wy_c[:, 0], rho_c=float(rho_c[0]),
        z_A_p=z_A_p, z_B_p=z_B_p,
        z_A_c=z_A_c, z_B_c=z_B_c,
        beta_A_norm=beta_A_norm,
        beta_B_norm=beta_B_norm,
        X_A_resid=X_A_resid,   # (n_trials, n_A, T) -- tgt1 residual after nuis. regression
        X_B_resid=X_B_resid,   # (n_trials, n_B, T) -- tgt2 residual after nuis. regression
    )



# =============================================================================
# S6  FIGURE A -- ABLATION-STYLE PSTH + LATENT-TRAJECTORY FIGURE
# =============================================================================

def fig_simulation_psth_latents(
    n_pool:          int   = 20,
    n_trials:        int   = N_TRIALS,
    T:               int   = T_BINS,
    t_onset:         int   = T_ONSET,
    sigma_priv:      float = 0.0,
    n_show:          int   = 50,
    show_gt_overlay: bool  = True,
    nuisance:        str   = 'Z',    # 'Z' => pCCA(A,B|Z);  'B' => pCCA(A,Z|B)
    save:            bool  = True,
) -> None:
    """Ablation-style PSTH + latent-trajectory figure, b-topology simulation.

    A single function handles both conditioning choices via ``nuisance``:

      nuisance = 'Z'  (default)
          Target pair  : pCCA(A, B | Z)  -- recovers the lateral channel
          Row 0 / Row 1: Region A (A1+A2) / Region B
          pCCA z(t)    should track  rate_A2B  (lateral Poisson source)
          β-norms      large for A1 neurons  (hub driven, captured by Z)

      nuisance = 'B'
          Target pair  : pCCA(A, Z | B)  -- recovers the hub channel
          Row 0 / Row 1: Region A (A1+A2) / Region Z
          pCCA z(t)    should track  rate_ZG   (hub Poisson source)
          β-norms      large for A2 neurons  (lateral driven, captured by B)

    Layout  (2 rows x 8 columns)
    -------
      col 0       col 1     col 2        col 3     col 4       col 5         col 6   col 7
      ──────────────────────────────────────────────────────────────────────────────────────
      Raw PSTH  │ pCCA w │ pCCA z(t)  │ CCA w  │ CCA z(t) │ Nuisance   │ beta  │ Residual
      (RdBu_r)  │ barh   │ +trials    │ barh   │ +trials  │ PSTH       │ norms │ PSTH
                │ C3/C0  │ +SEM       │ amb/pur│ +SEM     │ (RdBu_r)   │ green │ (RdBu_r)

    Column 7 — residual PSTH
    ------------------------
    Shows activity remaining in each target region after the nuisance
    signal has been regressed out, i.e. the input to the CCA step of pCCA.
    In the noise-free b-topology with nuisance='Z':
      · A1 sub-block  → flat zero   (hub fully explained by Z)
      · A2 sub-block  → full signal (lateral not captured by Z)
    This makes the mechanism of pCCA directly legible: the pCCA weight bar
    (col 1) concentrates precisely where the residual PSTH (col 7) is non-zero.
    """
    if nuisance not in ('Z', 'B'):
        raise ValueError(f"nuisance must be 'Z' or 'B', got '{nuisance!r}'")

    print(f"[Fig A (nuisance={nuisance})]  "
          f"Simulation PSTH + latents  "
          f"(n={n_pool}/pool, sigma_priv={sigma_priv}, "
          f"{n_trials} trials, T={T}) ...")

    # ── Data generation ───────────────────────────────────────────────────────
    X_A, X_B, X_Z, time_vec, gt = gen_population_b_trials(
        n_pool, n_pool, n_pool, n_pool,
        n_trials=n_trials, T=T, t_onset=t_onset, sigma_priv=sigma_priv,
    )

    # ── Route target pair and nuisance region based on parameter ──────────────
    if nuisance == 'Z':
        X_tgt1, X_tgt2, X_nuis = X_A, X_B, X_Z
        tgt1_name  = 'Region A  (A1 + A2)'
        tgt2_name  = 'Region B'
        nuis_name  = 'Region Z'
        sort_nuis  = np.argsort(-gt['g_ZG_Z'])
        beta_lbl1, beta_lbl2 = 'Z->A', 'Z->B'
        gt_p_prof  = gt['rate_A2B']                       # pCCA(A,B|Z) tracks lateral
        gt_p2_prof = gt['rate_A2B']
        gt_p_lbl   = 'GT: lat. rate'
        gt_c_prof  = gt['rate_ZG'] + gt['rate_A2B']      # CCA(A,B): hub+lat mixture
        gt_c_lbl   = 'GT: hub+lat.'
    else:  # 'B'
        X_tgt1, X_tgt2, X_nuis = X_A, X_Z, X_B
        tgt1_name  = 'Region A  (A1 + A2)'
        tgt2_name  = 'Region Z'
        nuis_name  = 'Region B'
        sort_nuis  = np.argsort(-(np.abs(gt['g_ZG_B']) + np.abs(gt['g_A2B_B'])))
        beta_lbl1, beta_lbl2 = 'B->A', 'B->Z'
        gt_p_prof  = -gt['rate_ZG'] +gt['rate_A2B']                   # pCCA(A,Z|B) tracks hub
        gt_p2_prof = -gt['rate_ZG'] +gt['rate_A2B']
        gt_p_lbl   = 'GT: hub rate'
        gt_c_prof  = gt['rate_ZG']                        # CCA(A,Z): predominantly hub
        gt_c_lbl   = 'GT: hub rate'

    # ── pCCA + CCA ────────────────────────────────────────────────────────────
    res = _compute_sim_pair_results(X_tgt1, X_tgt2, X_nuis, k=5)

    _apply_sign_alignment_to_sim_results(res, time_vec)

    # ── Z-score for PSTH display ──────────────────────────────────────────────
    X_nuis_z = _zscore_sim(X_nuis)

    # ── Neuron orderings ─────────────────────────────────────────────────────
    sort_A, boundary_A = _sim_neuron_sort_A(gt)

    if nuisance == 'Z':
        sort_tgt1, bound1 = sort_A, boundary_A
        sort_tgt2 = np.argsort(-(np.abs(gt['g_ZG_B']) + np.abs(gt['g_A2B_B'])))
        X_tgt1_z  = _zscore_sim(X_A)
        X_tgt2_z  = _zscore_sim(X_B)
    else:
        sort_tgt1, bound1 = sort_A, boundary_A
        sort_tgt2 = np.argsort(-gt['g_ZG_Z'])
        X_tgt1_z  = _zscore_sim(X_A)
        X_tgt2_z  = _zscore_sim(X_Z)

    # Z-score the nuisance-residual arrays (for col 7)
    X_tgt1_resid_z = _zscore_sim(res['X_A_resid'])
    X_tgt2_resid_z = _zscore_sim(res['X_B_resid'])

    # ── GT profile helpers ────────────────────────────────────────────────────
    def _gt(prof): return prof if show_gt_overlay else None

    gt_mix = gt['rate_ZG'] + gt['rate_A2B']

    # ── Per-row data bundles ──────────────────────────────────────────────────
    rows = [
        dict(
            name         = tgt1_name,
            X_z          = X_tgt1_z,
            X_z_resid    = X_tgt1_resid_z,
            sort_idx     = sort_tgt1,
            boundary     = bound1,
            Wx_p         = res['Wx_p'],
            Wx_c         = res['Wx_c'],
            z_p          = res['z_A_p'],
            z_c          = res['z_A_c'],
            beta_n       = res['beta_A_norm'],
            pcca_col     = _A_PCCA,
            cca_col      = _A_CCA,
            gt_p         = _gt(gt_p_prof),
            gt_c         = _gt(gt_c_prof),
            gt_lbl_p     = gt_p_lbl,
            gt_lbl_c     = gt_c_lbl,
            beta_lbl     = beta_lbl1,
        ),
        dict(
            name         = tgt2_name,
            X_z          = X_tgt2_z,
            X_z_resid    = X_tgt2_resid_z,
            sort_idx     = sort_tgt2,
            boundary     = None,
            Wx_p         = res['Wy_p'],
            Wx_c         = res['Wy_c'],
            z_p          = res['z_B_p'],
            z_c          = res['z_B_c'],
            beta_n       = res['beta_B_norm'],
            pcca_col     = _B_PCCA,
            cca_col      = _B_CCA,
            gt_p         = _gt(gt_p2_prof),
            gt_c         = _gt(gt_c_prof),
            gt_lbl_p     = gt_p_lbl,
            gt_lbl_c     = gt_c_lbl,
            beta_lbl     = beta_lbl2,
        ),
    ]

    # ── Figure layout: 8 columns ──────────────────────────────────────────────
    # col 7 (residual PSTH) mirrors col 0 in width.
    # figsize is widened from 17.5 -> 20.0 to accommodate the extra column.
    wr  = [2.8, 0.6, 2.8,2.8, 0.6, 2.8, 0.6, 2.8]
    fig = plt.figure(figsize=(20.0, 9.0))
    gs  = fig.add_gridspec(
        nrows=2, ncols=8,
        width_ratios=wr,
        hspace=0.52, wspace=0.30,
        left=0.035, right=0.98, top=0.88, bottom=0.07,
    )

    for row_idx, rd in enumerate(rows):

        # col 0 -- raw PSTH
        ax_psth = fig.add_subplot(gs[row_idx, 0])
        _draw_sim_rastermap_panel(
            fig, ax_psth,
            rd['X_z'], rd['sort_idx'], time_vec, rd['name'],
            n_show=n_show, boundary_idx=rd['boundary'],
        )


        # col 3 -- CCA weight barh  (amber/purple)
        ax_wc = fig.add_subplot(gs[row_idx, 1])
        _draw_sim_weight_bar(
            ax_wc, rd['sort_idx'], rd['Wx_c'],
            bar_label='CCA w', n_show=n_show,
            pos_color=C_COMM, neg_color=C_COMP, rho=res['rho_c'],
        )

        # col 4 -- CCA z(t)
        ax_zc = fig.add_subplot(gs[row_idx, 2])
        _plot_sim_latent_trials(
            ax_zc, time_vec, rd['z_c'], color=rd['cca_col'],
            title=f"CCA z(t)   rho1 = {res['rho_c']:.3f}",
            gt_profile=rd['gt_c'], gt_label=rd['gt_lbl_c'],
        )

        # col 5 -- nuisance PSTH (identical for both rows)
        ax_npsth = fig.add_subplot(gs[row_idx, 3])
        _draw_sim_rastermap_panel(
            fig, ax_npsth, X_nuis_z, sort_nuis, time_vec,
            f'Nuisance: {nuis_name}', n_show=n_show, boundary_idx=None,
        )

        # col 6 -- nuisance -> region beta norms
        ax_beta = fig.add_subplot(gs[row_idx, 4])
        _draw_sim_weight_bar(
            ax_beta, rd['sort_idx'], rd['beta_n'],
            bar_label=rd['beta_lbl'] + '\nbeta||.||2',
            n_show=n_show,
            pos_color=C_HUB, neg_color=C_RAW, rho=None,
        )


        # col 7 -- residual PSTH (activity remaining after nuisance regression)
        # For row 0 (region A) the A1/A2 boundary is preserved so the reader
        # can immediately see that A1 neurons are zeroed out (noise-free case)
        # while A2 neurons retain their full lateral signal.
        ax_resid = fig.add_subplot(gs[row_idx, 5])
        region_short = rd['name'].split()[1]            # 'A', 'B', or 'Z'
        _draw_sim_rastermap_panel(
            fig, ax_resid,
            rd['X_z_resid'], rd['sort_idx'], time_vec,
            f'{region_short} resid | {nuis_name}',
            n_show=n_show, boundary_idx=rd['boundary'],
        )

        # col 1 -- pCCA weight barh  (C3/C0: positive/negative)
        ax_wp = fig.add_subplot(gs[row_idx, 6])
        _draw_sim_weight_bar(
            ax_wp, rd['sort_idx'], rd['Wx_p'],
            bar_label='pCCA w', n_show=n_show,
            pos_color='C3', neg_color='C0', rho=res['rho_p'],
        )

        # col 2 -- pCCA z(t)
        ax_zp = fig.add_subplot(gs[row_idx, 7])
        _plot_sim_latent_trials(
            ax_zp, time_vec, rd['z_p'], color=rd['pcca_col'],
            title=f"pCCA z(t)   rho1 = {res['rho_p']:.3f}",
            gt_profile=rd['gt_p'], gt_label=rd['gt_lbl_p'],
        )
        if row_idx == 0:
            ax_zp.set_ylabel('Latent projection', fontsize=9)


    # ── Supertitle ────────────────────────────────────────────────────────────
    noise_str = ('noise-free Poisson' if sigma_priv == 0.0
                 else f'sigma_priv = {sigma_priv:.1f}')
    tgt_pair  = (f'Region A (A1+A2) <-> Region B' if nuisance == 'Z'
                 else f'Region A (A1+A2) <-> Region Z')
    fig.suptitle(
        f'[Simulation -- b-topology,  {noise_str}]   '
        f'{tgt_pair}  |  Z = {{{nuis_name}}}\n'
        f'pCCA rho1 = {res["rho_p"]:.4f}   |   CCA rho1 = {res["rho_c"]:.4f}   |   '
        f'n = {n_pool}/pool,  {n_trials} trials,  T = {T} bins '
        f'({int(DT_SEC * 1e3)} ms/bin)   |   '
        f'Gini: A1={gt["gini_A1"]:.2f}, A2={gt["gini_A2"]:.2f}',
        fontsize=10.5, fontweight='bold', y=0.97,
    )

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = (f'npool{n_pool}_ntrials{n_trials}'
               f'_spriv{sigma_priv:.1f}_nuis{nuisance}')
        out = os.path.join(OUT_DIR, f'fig_sim_psth_latents_{tag}.png')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig)


    """Ablation-style two-row PSTH + latent-trajectory figure for the
    b-topology simulation.

    Layout  (2 rows x 7 columns, mirroring pcca_sequential_ablation.py)
    -------

      col 0       col 1      col 2         col 3      col 4        col 5          col 6
      ------------------------------------------------------------------------------------------
      PSTH A/B  | pCCA w   | pCCA z(t)   | CCA w   | CCA z(t)  | Z PSTH       | Z->region beta
      RdBu_r    | barh C3/ | trials+mean | barh    | trials+   | nuisance     | barh
                | C0       | +/-SEM      | amber/  | mean      | RdBu_r       | green
                |          |             | purple  | +/-SEM    |              |

    Row 0 : region A  (A1 hub-pool + A2 lateral-pool neurons)
    Row 1 : region B  (receives both hub and lateral inputs)

    Key diagnostic contrasts
    ------------------------
    1. Weight bars (cols 1 vs 3):  pCCA concentrates loading in the A2
       sub-block; CCA spreads across both A1 and A2.  This is the direct
       visualisation of pCCA's selective recovery of the lateral channel
       after hub confound removal.

    2. Z->A beta norms (col 6):  large for A1 neurons (hub-driven, captured
       by Z), near-zero for A2 neurons.  This explains *why* pCCA
       down-weights A1 -- because Z already explains those neurons.

    3. Latent z(t) (cols 2 vs 4):  pCCA z_A(t) tracks rate_A2B (GT overlay);
       CCA z_A(t) tracks a broader hub-lateral mixture.

    Parameters
    ----------
    n_pool          neurons per pool (n_A1 = n_A2 = n_B = n_Z = n_pool)
    sigma_priv      0.0 for noise-free Poisson (default)
    n_show          neurons shown per PSTH / weight-bar panel
    show_gt_overlay if True, overlay the GT source-rate profile on z(t) panels
    """


# =============================================================================
# S7  FIGURE B -- COMPASS + WEIGHT BARS ACROSS POPULATION SIZE
# =============================================================================

POP_SIZES = [20]

# =============================================================================
# S7  FIGURE B — FOUR-REGIME CONNECTIVITY COMPARISON
#     Replaces the population-size sweep.
# =============================================================================

def fig_regime_comparison(
    n_pool:     int   = 20,
    sigma_priv: float = 2.0,
    n_trials:   int   = N_TRIALS,
    T:          int   = T_BINS,
    t_onset:    int   = T_ONSET,
    save:       bool  = True,
) -> None:
    """Four-column figure contrasting the four interaction modes.

    Each column corresponds to one connectivity topology; each row probes a
    different layer of the pCCA–CCA relationship.

    ┌────┬──────────────────────────────────────────────────────────────┐
    │ R0 │ CCA compass: w_AB_cca (blue) and w_AZ_cca (red-orange)     │
    │    │ projected onto the regime-natural reference plane.           │
    │    │ Angle theta = ∠(w_AB_cca, w_AZ_cca) printed in title.      │
    ├────┼──────────────────────────────────────────────────────────────┤
    │ R1 │ pCCA compass: w_AB_pcca and w_AZ_pcca on the same plane.   │
    │    │ Comparing R0 and R1 shows how partialling rotates the axes. │
    ├────┼──────────────────────────────────────────────────────────────┤
    │ R2 │ Rho bar chart (PRIMARY DIAGNOSTIC): side-by-side bars for   │
    │    │ rho_CCA (light) and rho_pCCA (dark) with directional arrow. │
    │    │  Mode 1: pCCA < CCA  (hub correctly removed)                │
    │    │  Mode 2: pCCA > CCA  (collider inflates residual rho)       │
    │    │  Mode 3: pCCA < CCA  (relay over-removed)                   │
    │    │  Mode 4: pCCA ≈ CCA  (exogenous Z; no effect)               │
    └────┴──────────────────────────────────────────────────────────────┘

    Reference plane per regime
    --------------------------
    Mode 1 (b-topology): ground-truth plane span{u_hub_A, u_lat_A}.
    Modes 2–4:           data-driven plane span{w_AB_cca, w_AZ_cca}.
    In all cases the plane is computed via orthonormal_plane() so the
    Gram-Schmidt basis and in-plane fractions reported by _compass_population
    are comparable across columns.

    kappa per regime (shown in R1 title)
    ------------------------------------
    kappa = |cos∠(w_AB_pcca, w_AZ_pcca)| — the cross-analysis collinearity.
      Mode 1: ≈ 0  (disjoint A1/A2 pools in X; orthogonal axes)
      Mode 2: ≈ 1  (single g_X drives X; both pCCA analyses find g_X)
      Mode 3: ≈ 1  (same g_X for relay and bypass; kappa = 1 by construction)
      Mode 4: ≈ 0  (pCCA(X,Z|Y) returns noise when Z is independent)
    """
    print(f"[Fig B]  Four-regime comparison  "
          f"(n={n_pool}/pool, sigma_priv={sigma_priv}) ...")

    # ── Regime catalogue ──────────────────────────────────────────────────────
    _REGIMES = [
        dict(
            label  = 'Mode 1\nHub-lateral\n(b-topology)',
            topo   = 'hub',
            gen    = lambda: gen_population_b_trials(
                n_pool, n_pool, n_pool, n_pool,
                n_trials=n_trials, T=T, t_onset=t_onset, sigma_priv=sigma_priv,
            ),
            cats   = ['hub axis', 'lateral axis', 'OOP'],
        ),
        dict(
            label  = 'Mode 2\nCollider\n(X→Z←Y)',
            topo   = 'collider',
            gen    = lambda: gen_population_collider_trials(
                n_pool, n_pool, n_pool,
                n_trials=n_trials, T=T, t_onset=t_onset, sigma_priv=sigma_priv,
            ),
            cats   = ['CCA dir.', 'X–Z dir.', 'OOP'],
        ),
        dict(
            label  = 'Mode 3\nMediator\n(relay+bypass)',
            topo   = 'mediator',
            gen    = lambda: gen_population_mediator_trials(
                n_pool, n_pool, n_pool,
                n_trials=n_trials, T=T, t_onset=t_onset, sigma_priv=sigma_priv,
            ),
            cats   = ['CCA dir.', 'X–Z dir.', 'OOP'],
        ),
        dict(
            label  = 'Mode 4\nIndependent Z\n(exogenous)',
            topo   = 'independent',
            gen    = lambda: gen_population_independent_trials(
                n_pool, n_pool, n_pool,
                n_trials=n_trials, T=T, t_onset=t_onset, sigma_priv=sigma_priv,
            ),
            cats   = ['CCA dir.', 'X–Z dir.', 'OOP'],
        ),
    ]

    ncols = len(_REGIMES)
    fig   = plt.figure(figsize=(5.5 * ncols, 13.0))
    gs    = fig.add_gridspec(3, ncols, hspace=0.58, wspace=0.30,
                              height_ratios=[1.7, 1.7, 1.0])

    rho_cca_all  = []
    rho_pcca_all = []

    for col, reg in enumerate(_REGIMES):

        # ── Generate data and flatten ─────────────────────────────────────────
        out = reg['gen']()
        X_A, X_B, X_Z, time_vec, gt = out
        n_A = X_A.shape[1];  n_B = X_B.shape[1];  n_Z = X_Z.shape[1]

        A_flat = X_A.transpose(0, 2, 1).reshape(n_trials * T, n_A)
        B_flat = X_B.transpose(0, 2, 1).reshape(n_trials * T, n_B)
        Z_flat = X_Z.transpose(0, 2, 1).reshape(n_trials * T, n_Z)

        # ── CCA and pCCA (X↔Y) ────────────────────────────────────────────────
        Wx_AB_c, _, rho_c   = cca_simple(A_flat, B_flat, k=1)
        Wx_AB_p, _, rho_p, _, _ = pcca(A_flat, B_flat, Z=Z_flat, k=1)
        w_AB_c = Wx_AB_c[:, 0];  w_AB_p = Wx_AB_p[:, 0]
        rho_cca_all.append(float(rho_c[0]))
        rho_pcca_all.append(float(rho_p[0]))

        # ── CCA and pCCA (X↔Z) ────────────────────────────────────────────────
        Wx_AZ_c, _, _      = cca_simple(A_flat, Z_flat, k=1)
        Wx_AZ_p, _, _, _, _ = pcca(A_flat, Z_flat, Z=B_flat, k=1)
        w_AZ_c = Wx_AZ_c[:, 0];  w_AZ_p = Wx_AZ_p[:, 0]

        theta_c = pa_deg(w_AB_c, w_AZ_c)
        theta_p = pa_deg(w_AB_p, w_AZ_p)
        kappa   = float(np.abs(np.dot(unit(w_AB_p), unit(w_AZ_p))))

        # ── Reference plane ───────────────────────────────────────────────────
        if reg['topo'] == 'hub':
            E, phi = orthonormal_plane(gt['u_hub_A'], gt['u_lat_A'])
        else:
            # Data-driven: span{w_AB_cca, w_AZ_cca}
            E, phi = orthonormal_plane(w_AB_c, w_AZ_c)

        raw = raw_cross_dir(A_flat, B_flat)

        # ── Row 0: CCA compass ────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        _compass_population(
            ax0, w_AB_c, w_AZ_c, E,
            raw_dir=raw,
            angle=theta_c,
            title=(f'{reg["label"]}\nCCA\n'
                   f'phi(plane axes) = {phi:.1f}°'),
        )

        # ── Row 1: pCCA compass ───────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        _compass_population(
            ax1, w_AB_p, w_AZ_p, E,
            angle=theta_p,
            title=f'pCCA\ntheta(pCCA) = {theta_p:.1f}°\nkappa = {kappa:.2f}',
        )

        # ── Row 2: rho comparison ─────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        bw  = 0.28
        rc  = rho_cca_all[-1]
        rp  = rho_pcca_all[-1]

        ax2.bar([0.0], [rc], bw, color=C_CCA, alpha=0.82,
                label=f'CCA  ρ={rc:.3f}')
        ax2.bar([0.35], [rp], bw, color=C_AZ,  alpha=0.88,
                label=f'pCCA ρ={rp:.3f}')

        # Directional annotation: the PRIMARY DIAGNOSTIC
        if rp > rc + 0.02:
            arrow, acolor = '↑  pCCA > CCA', '#e74c3c'
        elif rp < rc - 0.02:
            arrow, acolor = '↓  pCCA < CCA', '#27ae60'
        else:
            arrow, acolor = '≈  pCCA ≈ CCA', '#7f8c8d'
        ax2.text(0.175, max(rc, rp) + 0.04, arrow,
                 ha='center', va='bottom', fontsize=9,
                 color=acolor, fontweight='bold')

        ax2.axhline(0, color='k', lw=0.6)
        ax2.set_xticks([0.0, 0.35])
        ax2.set_xticklabels(['CCA', 'pCCA'], fontsize=9)
        ax2.set_ylabel('ρ₁', fontsize=9)
        ax2.set_title('rho comparison\n(primary diagnostic)', fontsize=8.5)
        ax2.set_ylim(-0.05, max(rc, rp, 0.1) + 0.22)
        ax2.legend(fontsize=7, frameon=False, loc='upper right')
        for sp in ('top', 'right'):
            ax2.spines[sp].set_visible(False)

    # ── Row labels ────────────────────────────────────────────────────────────
    for y_frac, lbl in zip([0.82, 0.50, 0.10], ['CCA\ncompass', 'pCCA\ncompass', 'rho\ncompare']):
        fig.text(0.003, y_frac, lbl, va='center', ha='left',
                 fontsize=9, fontweight='bold', color='#444', rotation=90)

    fig.suptitle(
        f'Four-regime connectivity comparison\n'
        f'n = {n_pool} neurons/pool,  sigma_priv = {sigma_priv:.1f},  '
        f'{n_trials} trials,  T = {T} bins\n'
        f'Expected: Mode2 pCCA↑CCA,  Mode1/3 pCCA↓CCA,  Mode4 pCCA≈CCA',
        fontsize=11, fontweight='bold', y=1.02,
    )

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(
            OUT_DIR,
            f'fig_regime_comparison_n{n_pool}_spriv{sigma_priv:.1f}.png',
        )
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig)

# # =============================================================================
# # S8  FIGURE C -- SUBSPACE RECOVERY VS POPULATION SIZE
# # =============================================================================
#
# SWEEP_SIZES = [1, 2, 3, 5, 8, 13, 20, 30, 50,100]
#
#
# def fig_subspace_recovery_vs_size(save: bool = True) -> None:
#     """Subspace recovery diagnostics as a function of pool size n.
#
#     Left panel  : principal angle between CCA top-2 subspace and the
#                   ground-truth communication plane (requires n >= 2).
#     Right panel : CCA theta(w_AB, w_AZ),  pCCA theta(w_AB, w_AZ), and
#                   CCA-1 deviation from the pure hub axis -- three curves
#                   capturing contamination angle as n scales.
#     """
#     print("[Fig C]  Subspace-recovery-vs-size figure (b-topology) ...")
#     sub_angle, purity_cca, theta_cca, theta_pcca = [], [], [], []
#
#     for n in SWEEP_SIZES:
#         A, B, Z, gt = gen_population_b(n, n, n, n)
#         E, _         = orthonormal_plane(gt['u_hub_A'], gt['u_lat_A'])
#
#         k2 = min(2, n)
#         Wx_AB_c, _, _   = cca_simple(A, B, k=k2)
#         Wx_AZ_c, _, _   = cca_simple(A, Z, k=1)
#
#         if k2 == 2:
#             ang = subspace_angles(Wx_AB_c, E)
#             sub_angle.append(float(np.degrees(np.max(ang))))
#         else:
#             sub_angle.append(np.nan)
#
#         theta_cca.append(pa_deg(Wx_AB_c[:, 0], Wx_AZ_c[:, 0]))
#
#         Wx_AB_p, _, _, _, _ = pcca(A, B, Z=Z, k=1)
#         Wx_AZ_p, _, _, _, _ = pcca(A, Z, Z=B, k=1)
#         theta_pcca.append(pa_deg(Wx_AB_p[:, 0], Wx_AZ_p[:, 0]))
#
#         p = E.T @ unit(Wx_AB_c[:, 0])
#         purity_cca.append(float(np.degrees(
#             np.arccos(np.clip(abs(p[0]) / (np.linalg.norm(p) + 1e-12), 0., 1.)))))
#
#     fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
#
#     axes[0].plot(SWEEP_SIZES, sub_angle, 'o-', color=C_CCA)
#     axes[0].set_xlabel('neurons per pool (n)')
#     axes[0].set_ylabel('subspace angle (deg)')
#     axes[0].set_title('CCA top-2 subspace vs.\nground-truth communication plane')
#     axes[0].set_ylim(bottom=0)
#
#     axes[1].plot(SWEEP_SIZES, theta_cca,  'o-',  color=C_AB,   label='CCA  theta(w_AB, w_AZ)')
#     axes[1].plot(SWEEP_SIZES, theta_pcca, 's--', color=C_HUB,  label='pCCA theta(w_AB, w_AZ)')
#     axes[1].plot(SWEEP_SIZES, purity_cca, '^:',  color=C_COMP, label='CCA-1 vs. pure hub axis')
#     axes[1].set_xlabel('neurons per pool (n)')
#     axes[1].set_ylabel('angle (deg)')
#     axes[1].set_title(
#         'Contamination angle: scale dependence\n'
#         '(b-topology: u_hub perp. u_lat by construction)')
#     axes[1].legend(fontsize=7, frameon=False)
#     axes[1].set_ylim(0, 95)
#
#     fig.tight_layout()
#     if save:
#         os.makedirs(OUT_DIR, exist_ok=True)
#         out = os.path.join(OUT_DIR, 'fig_subspace_recovery_vs_size.png')
#         fig.savefig(out, dpi=300, bbox_inches='tight')
#         print(f'  Saved: {out}')
#     plt.close(fig)


# =============================================================================
# S9  FIGURE D -- 3-D pCCA LATENT TRAJECTORIES  (noise-free Poisson)
#     and standalone 2-D latent trajectory comparison  (new)
# =============================================================================

N_3D = 20    # neurons per pool
K_3D = 2     # canonical components (CC1, CC2 in x-y; trial index on z-axis)


def _latent_3d(
    ax: plt.Axes,
    L: np.ndarray,
    source: np.ndarray,
    source_name: str,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    cc_color: str,
    t_bins: int,
) -> None:
    """Render one 3-D trajectory panel.

    Each trial is drawn as a connected 2-D path in (CC1, CC2) space,
    stacked upward along the z-axis (trial index).  Timepoints within
    the trial are coloured by the ground-truth source value at that bin,
    so the colour encodes *what* the latent coordinate is tracking.

    Parameters
    ----------
    L        : (N_exact, 2)  canonical latent projections (CC1, CC2)
    source   : (N_exact,)    ground-truth source value (sZG or sA2B)
    t_bins   : length of one trial in time-bins  (= T_BINS = 150)
    """
    n_trials = len(L) // t_bins
    # Reshape to (n_trials, t_bins, 2) and (n_trials, t_bins)
    L_t  = L[:n_trials * t_bins].reshape(n_trials, t_bins, 2)
    src  = source[:n_trials * t_bins].reshape(n_trials, t_bins)

    sc = None
    for tr in range(n_trials):
        z_vals = np.full(t_bins, float(tr))   # constant z = trial index

        # Connecting path (grey skeleton)
        ax.plot(L_t[tr, :, 0], L_t[tr, :, 1], zs=z_vals,
                color='#aaaaaa', alpha=0.25, lw=0.6, zorder=1)

        # Source-coloured scatter at each timepoint
        sc = ax.scatter(L_t[tr, :, 0], L_t[tr, :, 1], z_vals,
                        c=src[tr], cmap=cmap, vmin=vmin, vmax=vmax,
                        s=6, alpha=0.70, linewidths=0, rasterized=True, zorder=2)

    ax.set_xlabel('$CC_1$', fontsize=8, labelpad=2, color=cc_color)
    ax.set_ylabel('$CC_2$', fontsize=8, labelpad=2, color=cc_color)
    ax.set_zlabel('Trial index', fontsize=7, labelpad=2, color='#444444')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=6)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#dddddd')
    ax.grid(False)

    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.50, pad=0.08, orientation='vertical')
        cbar.set_label(source_name, fontsize=7)
        cbar.ax.tick_params(labelsize=7)


def fig_latent_trajectories_3d(
    n: int = N_3D, k: int = K_3D,
    sigma_priv: float = 0.0,
    save: bool = True,
) -> None:
    """3-D trial-trajectory figure for pCCA region-A latent space (k = 2).

    Each trial's (CC1, CC2) path is drawn as a connected 2-D curve stacked
    along the z-axis (trial index).  Colour encodes the ground-truth Poisson
    source value at each timepoint:

    Left   pCCA(A, B | Z)  — colour = s_A2B (lateral source).
           After Z-residualisation the hub is removed; the surviving
           cross-covariance is driven by s_A2B.  If pCCA works correctly,
           the trajectory should be orderly (one colour band per trial).

    Right  pCCA(A, Z | B)  — colour = s_ZG (hub source).
           After B-residualisation the hub channel dominates.

    Uses gen_population_b with N_exact = floor(N/T_BINS)*T_BINS so that
    the concatenated time-series maps cleanly onto complete trials.
    """
    N_exact = (N // T_BINS) * T_BINS    # 4950 observations = 33 complete trials
    n_trials_eff = N_exact // T_BINS

    print(f"[Fig D]  3-D latent trajectory figure (k=2, n={n}/pool, "
          f"{n_trials_eff} trials, sigma_priv={sigma_priv}) ...")

    A, B, Z, gt = gen_population_b(n, n, n, n, N=N_exact, sigma_priv=sigma_priv)
    sZG  = gt['sZG']
    sA2B = gt['sA2B']

    k_eff = min(k, n)

    # pCCA(A, B | Z) — lateral channel
    W_A_AB, _, rho_AB, A_resid_Z, _ = pcca(A, B, Z=Z, k=k_eff)
    L_AB = A_resid_Z @ W_A_AB          # (N_exact, k_eff)

    # pCCA(A, Z | B) — hub channel
    W_A_AZ, _, rho_AZ, A_resid_B, _ = pcca(A, Z, Z=B, k=k_eff)
    L_AZ = A_resid_B @ W_A_AZ          # (N_exact, k_eff)

    # Pad to exactly 2 columns if k_eff < 2  (only when n = 1)
    def _pad2(M):
        if M.shape[1] < 2:
            return np.hstack([M, np.zeros((M.shape[0], 2 - M.shape[1]))])
        return M[:, :2]

    L_AB = _pad2(L_AB)
    L_AZ = _pad2(L_AZ)

    vmin_lat = int(np.percentile(sA2B, 2));  vmax_lat = int(np.percentile(sA2B, 98))
    vmin_hub = int(np.percentile(sZG,  2));  vmax_hub = int(np.percentile(sZG,  98))

    fig = plt.figure(figsize=(13.0, 5.5))

    ax_AB = fig.add_subplot(1, 2, 1, projection='3d')
    _latent_3d(
        ax_AB, L_AB, sA2B, '$s_{A_2B}$ (lateral)',
        cmap='YlOrBr', vmin=vmin_lat, vmax=vmax_lat,
        title=(f'pCCA$(A,B\\mid Z)$   $CC_1, CC_2$ of $W_A$\n'
               f'Colour = lateral source $s_{{A_2B}}$\n'
               f'$\\rho = [{", ".join(f"{r:.3f}" for r in rho_AB[:k_eff])}]$'),
        cc_color=C_AB, t_bins=T_BINS,
    )

    ax_AZ = fig.add_subplot(1, 2, 2, projection='3d')
    _latent_3d(
        ax_AZ, L_AZ, sZG, '$s_{ZG}$ (hub)',
        cmap='PuBuGn', vmin=vmin_hub, vmax=vmax_hub,
        title=(f'pCCA$(A,Z\\mid B)$   $CC_1, CC_2$ of $W_A$\n'
               f'Colour = hub source $s_{{ZG}}$\n'
               f'$\\rho = [{", ".join(f"{r:.3f}" for r in rho_AZ[:k_eff])}]$'),
        cc_color=C_AZ, t_bins=T_BINS,
    )

    noise_str = 'noise-free Poisson' if sigma_priv == 0.0 else f'$\\sigma_{{priv}}={sigma_priv}$'
    fig.suptitle(
        f'3-D pCCA latent trajectories  —  b-topology   ({noise_str})\n'
        f'n = {n} neurons/pool,  {n_trials_eff} trials (z-axis),  '
        f'$s_{{ZG}} \\to \\{{Z,A_1,B\\}}$,  $s_{{A_2B}} \\to \\{{A_2,B\\}}$',
        fontsize=10, fontweight='bold', y=1.02,
    )
    fig.tight_layout()

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = f'n{n}_k{k_eff}_spriv{sigma_priv:.1f}'
        out = os.path.join(OUT_DIR, f'fig_latent_trajectories_3d_{tag}.png')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig)


def fig_pcca_latent_trajectories_2d(
    n: int = N_3D,
    n_trials_show: int = 12,
    sigma_priv: float = 0.0,
    save: bool = True,
) -> None:
    """Standalone 2-D latent trajectory comparison for region A.

    Two panels side by side in (CC1, CC2) canonical-component space:

      Left   pCCA(A, B | Z) — the lateral communication subspace.
             After partialling out Z, the hub contribution to A is removed;
             canonical weights concentrate in the A2 pool.  The trajectory
             should follow the lateral source profile (peak at t = +0.30 s).

      Right  pCCA(A, Z | B) — the hub communication subspace.
             After partialling out B, the dominant residual A-Z covariance
             is hub-driven; weights concentrate in A1.  Trajectory should
             peak at t = -0.30 s (pre-movement).

    Each of n_trials_show trials is drawn as a LineCollection coloured by
    time within the trial (plasma: dark = pre-onset, bright = post-onset).
    The mean trajectory across all trials is overlaid as a bold curve.
    A filled circle marks reach onset (t = 0 s) on the mean.

    Data source: gen_population_b_trials (same API as fig_simulation_psth_latents).
    """
    print(f"[Fig D2]  2-D pCCA latent trajectory comparison "
          f"(n={n}/pool, sigma_priv={sigma_priv}, "
          f"{n_trials_show} trials shown) ...")

    # ── Data generation ───────────────────────────────────────────────────────
    X_A, X_B, X_Z, time_vec, gt = gen_population_b_trials(
        n, n, n, n, sigma_priv=sigma_priv,
    )
    n_trials, n_A, T = X_A.shape

    A_flat = X_A.transpose(0, 2, 1).reshape(n_trials * T, n_A)
    B_flat = X_B.transpose(0, 2, 1).reshape(n_trials * T, X_B.shape[1])
    Z_flat = X_Z.transpose(0, 2, 1).reshape(n_trials * T, X_Z.shape[1])

    k = 2
    # pCCA(A, B | Z) — isolates the lateral channel
    Wx_AB, _, rho_AB, A_resid_Z, _ = pcca(A_flat, B_flat, Z=Z_flat, k=k)
    L_AB = (A_resid_Z @ Wx_AB).reshape(n_trials, T, k)   # (n_trials, T, 2)

    # pCCA(A, Z | B) — isolates the hub channel
    Wx_AZ, _, rho_AZ, A_resid_B, _ = pcca(A_flat, Z_flat, Z=B_flat, k=k)
    L_AZ = (A_resid_B @ Wx_AZ).reshape(n_trials, T, k)   # (n_trials, T, 2)

    # ── Sign convention: positive activation in onset window ─────────────────
    onset_sl = slice(T_ONSET, min(T_ONSET + 10, T))
    for L in (L_AB, L_AZ):
        for d in range(k):
            if L[:, onset_sl, d].mean() < 0:
                L[:, :, d] *= -1

    # ── Colourmap: plasma, time in seconds ────────────────────────────────────
    cmap_t = mpl.cm.plasma
    norm_t = mpl.colors.Normalize(vmin=time_vec[0], vmax=time_vec[-1])

    # Equispaced subset of trials
    trial_sel = np.linspace(0, n_trials - 1, min(n_trials_show, n_trials),
                             dtype=int)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5),
                              gridspec_kw={'wspace': 0.28})

    for ax, L, rho, panel_title, cc_col in [
        (axes[0], L_AB, rho_AB,
         f'pCCA$(A,B\\mid Z)$  —  lateral subspace\n'
         f'$\\rho_1={rho_AB[0]:.3f}$,  $\\rho_2={rho_AB[1]:.3f}$',
         C_AB),
        (axes[1], L_AZ, rho_AZ,
         f'pCCA$(A,Z\\mid B)$  —  hub subspace\n'
         f'$\\rho_1={rho_AZ[0]:.3f}$,  $\\rho_2={rho_AZ[1]:.3f}$',
         C_AZ),
    ]:
        # Individual trial trajectories (thin, colour = time)
        for tr in trial_sel:
            traj = L[tr]                               # (T, 2)
            pts  = traj.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)  # (T-1, 2, 2)
            lc = LineCollection(segs, cmap=cmap_t, norm=norm_t,
                                 lw=0.85, alpha=0.35)
            lc.set_array(time_vec[:-1])
            ax.add_collection(lc)

        # Mean trajectory (bold)
        mean_traj = L.mean(axis=0)                     # (T, 2)
        pts  = mean_traj.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc_m = LineCollection(segs, cmap=cmap_t, norm=norm_t,
                               lw=2.8, alpha=0.95, zorder=4)
        lc_m.set_array(time_vec[:-1])
        ax.add_collection(lc_m)

        # Onset marker on mean trajectory
        ax.scatter(mean_traj[T_ONSET, 0], mean_traj[T_ONSET, 1],
                   c='k', s=70, marker='o', zorder=5, label='Onset  $t=0$')

        # Axis limits from the subsetted trial data
        all_pts = L[trial_sel].reshape(-1, 2)
        pad = 0.15
        xr  = all_pts[:, 0].ptp();  yr = all_pts[:, 1].ptp()
        ax.set_xlim(all_pts[:, 0].min() - pad * xr,
                     all_pts[:, 0].max() + pad * xr)
        ax.set_ylim(all_pts[:, 1].min() - pad * yr,
                     all_pts[:, 1].max() + pad * yr)

        ax.set_xlabel('$CC_1$', fontsize=10)
        ax.set_ylabel('$CC_2$', fontsize=10)
        ax.set_title(panel_title, fontsize=10, fontweight='bold', color=cc_col)
        ax.legend(fontsize=8, frameon=False, loc='lower right')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    # Shared time colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap_t, norm=norm_t)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                         shrink=0.70, pad=0.02, orientation='vertical')
    cbar.set_label('Time (s)', fontsize=9)
    # Mark t = 0 on the colorbar
    cbar.ax.axhline(0.0, color='k', lw=1.2, alpha=0.7, linestyle='--')
    cbar.ax.text(1.25, 0.0, '$t=0$', transform=cbar.ax.get_yaxis_transform(),
                 fontsize=7, va='center', color='k')

    noise_str = 'noise-free Poisson' if sigma_priv == 0.0 else f'$\\sigma_{{priv}}={sigma_priv}$'
    fig.suptitle(
        f'2-D pCCA latent trajectories  —  Region A   ({noise_str})\n'
        f'n = {n} neurons/pool,  {len(trial_sel)} of {n_trials} trials shown,  '
        f'b-topology  (hub peak $t = -0.30$ s,  lateral peak $t = +0.30$ s)',
        fontsize=11, fontweight='bold', y=1.02,
    )

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR,
                           f'fig_pcca_latent_2d_n{n}_spriv{sigma_priv:.1f}.png')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig)

# =============================================================================
# S11  SIMULATION SUPPLEMENTARY DIAGNOSTICS
#      Adapts plot_supplementary_panel from pcca_sequential_ablation.py
#      to compare the four interaction modes in a single 2×3 figure.
# =============================================================================

def _sim_trial_rows(trial_idx: np.ndarray, T: int) -> np.ndarray:
    """Row indices in the simulation flat matrix for given trial indices.

    Simulation flat layout (from _compute_sim_pair_results):
        row = trial * T + time   (trial is the outer index)

    This is the OPPOSITE of the real-data flat layout in pcca_sequential_ablation
    (row = time * n_trials + trial), so _trial_rows from that module cannot be
    reused here directly.
    """
    return np.concatenate([np.arange(r * T, (r + 1) * T) for r in trial_idx])


def compute_sim_supplementary_metrics(
    *,
    A_flat:    np.ndarray,     # (n_trials*T, n_A)  z-scored flat
    B_flat:    np.ndarray,     # (n_trials*T, n_B)
    Z_flat:    np.ndarray,     # (n_trials*T, n_Z)
    Wx_pcca:   np.ndarray,     # (n_A,)  pCCA(A,B|Z) weight, first component
    Wy_pcca:   np.ndarray,     # (n_B,)
    Wx_cca:    np.ndarray,     # (n_A,)  CCA(A,B)    weight
    Wy_cca:    np.ndarray,     # (n_B,)
    rho_pcca:  float,
    rho_cca:   float,
    z_A_p:     np.ndarray,     # (n_trials, T) sign-corrected pCCA latent A
    z_B_p:     np.ndarray,     # (n_trials, T)
    n_trials:  int,
    T:         int,
    time_vec:  np.ndarray,
    label:     str,
    n_cv_folds: int = 5,
) -> dict:
    """Compute the six supplementary diagnostic metrics for one simulation regime.

    All formulas are identical to compute_supplementary_metrics in
    pcca_sequential_ablation.py; only the flat-matrix row convention differs.

    Returns a plain dict (not a dataclass) to avoid cross-file imports.

    Keys
    ----
    label, theta_A_deg, theta_B_deg, kappa_A, kappa_B,
    rho_pcca, rho_cca, rho_cv_mean, rho_cv_sem,
    r2_nuis_A, r2_comm_A, r2_nuis_B, r2_comm_B,
    lag_peak_ms, lag_corr_at_peak, lag_axis_ms, xcorr_curve
    """
    # ── 1. CCA–pCCA rotation angles theta ─────────────────────────────────
    def _cos_sim(a, b):
        na = np.linalg.norm(a);  nb = np.linalg.norm(b)
        return float(np.abs(np.dot(a, b)) / (na * nb + 1e-12))

    theta_A = float(np.degrees(np.arccos(np.clip(_cos_sim(Wx_pcca, Wx_cca), 0., 1.))))
    theta_B = float(np.degrees(np.arccos(np.clip(_cos_sim(Wy_pcca, Wy_cca), 0., 1.))))

    # ── 2. Cross-analysis collinearity kappa ──────────────────────────────
    # pCCA(A, Z | B): what does A covary with in Z, after removing B?
    A_res_wrtB = partial_residuals(A_flat, B_flat)
    Z_res_wrtB = partial_residuals(Z_flat, B_flat)
    Wx_AZ, _, _ = cca_svd(A_res_wrtB, Z_res_wrtB, k=1)
    kappa_A = float(_cos_sim(Wx_pcca, Wx_AZ[:, 0]))

    # pCCA(B, Z | A): same for B
    B_res_wrtA = partial_residuals(B_flat, A_flat)
    Z_res_wrtA = partial_residuals(Z_flat, A_flat)
    Wy_BZ, _, _ = cca_svd(B_res_wrtA, Z_res_wrtA, k=1)
    kappa_B = float(_cos_sim(Wy_pcca, Wy_BZ[:, 0]))

    # ── 3. Cross-validated rho_1 ──────────────────────────────────────────
    fold_size  = n_trials // n_cv_folds
    trial_perm = np.arange(n_trials)
    rhos_cv    = []

    for fold in range(n_cv_folds):
        te_trials = trial_perm[fold * fold_size: (fold + 1) * fold_size]
        tr_trials = np.concatenate([
            trial_perm[: fold * fold_size],
            trial_perm[(fold + 1) * fold_size:],
        ])
        tr_rows = _sim_trial_rows(tr_trials, T)
        te_rows = _sim_trial_rows(te_trials, T)

        A_tr, B_tr, Z_tr = A_flat[tr_rows], B_flat[tr_rows], Z_flat[tr_rows]
        A_te, B_te, Z_te = A_flat[te_rows], B_flat[te_rows], Z_flat[te_rows]

        n_tr = len(tr_rows)
        ZtZ_tr = Z_tr.T @ Z_tr + LAMBDA_REG * n_tr * np.eye(Z_flat.shape[1])
        Beta_A  = np.linalg.solve(ZtZ_tr, Z_tr.T @ A_tr)
        Beta_B  = np.linalg.solve(ZtZ_tr, Z_tr.T @ B_tr)

        A_tr_res = A_tr - Z_tr @ Beta_A
        B_tr_res = B_tr - Z_tr @ Beta_B
        A_te_res = A_te - Z_te @ Beta_A   # training beta applied to test
        B_te_res = B_te - Z_te @ Beta_B

        Wx_cv, Wy_cv, _ = cca_svd(A_tr_res, B_tr_res, k=1)
        zA_te = A_te_res @ Wx_cv[:, 0]
        zB_te = B_te_res @ Wy_cv[:, 0]

        if np.std(zA_te) < 1e-9 or np.std(zB_te) < 1e-9:
            rhos_cv.append(0.0)
        else:
            from scipy.stats import pearsonr
            rhos_cv.append(float(np.clip(pearsonr(zA_te, zB_te)[0], -1., 1.)))

    rho_cv_mean = float(np.mean(rhos_cv))
    rho_cv_sem  = float(np.std(rhos_cv) / np.sqrt(n_cv_folds))

    # ── 4. Variance partition ─────────────────────────────────────────────
    A_res = partial_residuals(A_flat, Z_flat)
    B_res = partial_residuals(B_flat, Z_flat)

    denom_A = float(np.sum(A_flat ** 2)) + 1e-12
    denom_B = float(np.sum(B_flat ** 2)) + 1e-12

    A_hat = A_flat - A_res;  B_hat = B_flat - B_res
    r2_nuis_A = float(np.sum(A_hat ** 2)) / denom_A
    r2_nuis_B = float(np.sum(B_hat ** 2)) / denom_B

    wA_u = Wx_pcca / (np.linalg.norm(Wx_pcca) + 1e-12)
    wB_u = Wy_pcca / (np.linalg.norm(Wy_pcca) + 1e-12)
    r2_comm_A = float(np.sum((A_res @ wA_u) ** 2)) / denom_A
    r2_comm_B = float(np.sum((B_res @ wB_u) ** 2)) / denom_B

    # ── 5. Temporal lead-lag cross-correlation ────────────────────────────
    mean_A = z_A_p.mean(axis=0);  mean_A -= mean_A.mean()
    mean_B = z_B_p.mean(axis=0);  mean_B -= mean_B.mean()
    std_A  = float(np.std(mean_A)) + 1e-12
    std_B  = float(np.std(mean_B)) + 1e-12

    xcorr  = np.correlate(mean_A / std_A, mean_B / std_B, mode='full') / T
    lag_bins     = np.arange(-(T - 1), T)
    dt_ms        = float(time_vec[1] - time_vec[0]) * 1000.
    lag_axis_ms  = lag_bins.astype(float) * dt_ms

    max_lag_bins = min(int(150. / dt_ms), T - 1)
    search_mask  = np.abs(lag_bins) <= max_lag_bins
    peak_rel     = int(np.argmax(np.abs(xcorr[search_mask])))
    peak_abs     = int(np.where(search_mask)[0][peak_rel])

    return dict(
        label=label,
        theta_A_deg=theta_A, theta_B_deg=theta_B,
        kappa_A=kappa_A,     kappa_B=kappa_B,
        rho_pcca=rho_pcca,   rho_cca=rho_cca,
        rho_cv_mean=rho_cv_mean, rho_cv_sem=rho_cv_sem,
        r2_nuis_A=r2_nuis_A, r2_comm_A=r2_comm_A,
        r2_nuis_B=r2_nuis_B, r2_comm_B=r2_comm_B,
        lag_peak_ms=float(lag_axis_ms[peak_abs]),
        lag_corr_at_peak=float(xcorr[peak_abs]),
        lag_axis_ms=lag_axis_ms,
        xcorr_curve=xcorr,
    )


# Colour palette for the four regimes (distinct categorical colours)
_REGIME_COLORS = {
    'hub':         '#1f77b4',   # blue
    'collider':    '#d62728',   # red
    'mediator':    '#ff7f0e',   # orange
    'independent': '#2ca02c',   # green
}
_REGIME_LABELS = {
    'hub':         'Mode 1\n(hub-lateral)',
    'collider':    'Mode 2\n(collider)',
    'mediator':    'Mode 3\n(mediator)',
    'independent': 'Mode 4\n(independent)',
}


def fig_sim_supplementary_comparison(
    n_pool:     int   = 20,
    sigma_priv: float = 2.0,
    n_trials:   int   = N_TRIALS,
    T:          int   = T_BINS,
    t_onset:    int   = T_ONSET,
    save:       bool  = True,
) -> None:
    """Six-panel supplementary diagnostic figure comparing all four regimes.

    Layout mirrors plot_supplementary_panel in pcca_sequential_ablation.py
    but with mode (not ablation step) on the x-axis.

    Row 0 — Solution stability (can we trust the pCCA result?):
      (A) In-sample rho_1 vs. 5-fold CV rho_1 + CCA baseline.
          Large CV gap → overfitting to the nuisance regression step.
      (B) CCA–pCCA rotation angle theta for regions A and B.
          theta → 90° : nuisance removal uncovers an orthogonal axis.
      (C) Cross-analysis collinearity kappa for regions A and B.
          kappa → 1 : same neuron population couples to both Y and Z.

    Row 1 — Solution structure (what does the pCCA axis encode?):
      (D) Variance partition — Region A.
          Stacked bar: nuisance r² (grey) + communication r² (coloured).
      (E) Variance partition — Region B.
      (F) pCCA latent cross-correlation (lag axis up to ±300 ms).
          All four regime curves overlaid; dots mark peak lag.

    Expected pattern per regime
    ---------------------------
    Mode 1: rho_pCCA < rho_CCA;  moderate theta;  kappa ≈ 0;  moderate r2_nuis
    Mode 2: rho_pCCA > rho_CCA;  large theta;     kappa ≈ 1;  small r2_nuis
    Mode 3: rho_pCCA < rho_CCA;  large theta;     kappa ≈ 1;  large r2_nuis
    Mode 4: rho_pCCA ≈ rho_CCA;  theta ≈ 0°;     kappa ≈ 0;  r2_nuis ≈ 0
    """
    print(f"[Fig Supp]  Simulation supplementary 6-panel  "
          f"(n={n_pool}/pool, sigma_priv={sigma_priv}) ...")

    _REGIME_GENS = [
        ('hub',
         lambda: gen_population_b_trials(
             n_pool, n_pool, n_pool, n_pool,
             n_trials=n_trials, T=T, t_onset=t_onset, sigma_priv=sigma_priv)),
        ('collider',
         lambda: gen_population_collider_trials(
             n_pool, n_pool, n_pool,
             n_trials=n_trials, T=T, t_onset=t_onset, sigma_priv=sigma_priv)),
        ('mediator',
         lambda: gen_population_mediator_trials(
             n_pool, n_pool, n_pool,
             n_trials=n_trials, T=T, t_onset=t_onset, sigma_priv=sigma_priv)),
        ('independent',
         lambda: gen_population_independent_trials(
             n_pool, n_pool, n_pool,
             n_trials=n_trials, T=T, t_onset=t_onset, sigma_priv=sigma_priv)),
    ]

    metrics_list = []

    for topo, gen_fn in _REGIME_GENS:
        X_A, X_B, X_Z, time_vec, gt = gen_fn()
        n_A = X_A.shape[1];  n_B = X_B.shape[1];  n_Z = X_Z.shape[1]

        A_flat = X_A.transpose(0, 2, 1).reshape(n_trials * T, n_A)
        B_flat = X_B.transpose(0, 2, 1).reshape(n_trials * T, n_B)
        Z_flat = X_Z.transpose(0, 2, 1).reshape(n_trials * T, n_Z)

        # pCCA and CCA
        Wx_p_mat, Wy_p_mat, rho_p, A_res, B_res = pcca(A_flat, B_flat, Z=Z_flat, k=1)
        Wx_c_mat, Wy_c_mat, rho_c               = cca_simple(A_flat, B_flat, k=1)

        Wx_p = Wx_p_mat[:, 0];  Wy_p = Wy_p_mat[:, 0]
        Wx_c = Wx_c_mat[:, 0];  Wy_c = Wy_c_mat[:, 0]

        # Sign-align the latent projections (Steps 4–5 of Z2 algorithm)
        z_A_p = (A_res @ Wx_p).reshape(n_trials, T)
        z_B_p = (B_res @ Wy_p).reshape(n_trials, T)

        # Simple sign anchor: epoch mean [t=0, t=1.5s] must be positive in A
        t_start = int(np.searchsorted(time_vec, 0.0))
        t_end   = int(np.searchsorted(time_vec, 1.5))
        if z_A_p[:, t_start:t_end].mean() < 0:
            z_A_p *= -1;  Wx_p = -Wx_p
        if z_B_p[:, t_start:t_end].mean() < 0:
            z_B_p *= -1;  Wy_p = -Wy_p

        m = compute_sim_supplementary_metrics(
            A_flat=A_flat, B_flat=B_flat, Z_flat=Z_flat,
            Wx_pcca=Wx_p, Wy_pcca=Wy_p,
            Wx_cca=Wx_c,  Wy_cca=Wy_c,
            rho_pcca=float(rho_p[0]), rho_cca=float(rho_c[0]),
            z_A_p=z_A_p, z_B_p=z_B_p,
            n_trials=n_trials, T=T, time_vec=time_vec,
            label=_REGIME_LABELS[topo],
        )
        m['topo']  = topo
        m['color'] = _REGIME_COLORS[topo]
        metrics_list.append(m)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 3, figsize=(19, 9),
        gridspec_kw={'hspace': 0.58, 'wspace': 0.36},
    )

    xlabels = [m['label'] for m in metrics_list]
    colors  = [m['color'] for m in metrics_list]
    x       = np.arange(len(metrics_list))

    def _xax(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=8)
        ax.grid(alpha=0.22, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    # ── Panel A: rho in-sample vs. CV ─────────────────────────────────────
    ax = axes[0, 0]
    rho_pcca_vals = [m['rho_pcca']    for m in metrics_list]
    rho_cca_vals  = [m['rho_cca']     for m in metrics_list]
    rho_cv_vals   = [m['rho_cv_mean'] for m in metrics_list]
    rho_cv_sems   = [m['rho_cv_sem']  for m in metrics_list]

    ax.plot(x, rho_pcca_vals, 'o-', color='#922B21', lw=1.8, ms=6,
            label='pCCA  ρ₁  in-sample')
    ax.plot(x, rho_cca_vals,  's--', color='#7F8C8D', lw=1.4, ms=6,
            alpha=0.75, label='CCA  ρ₁  (Z = ∅)')
    ax.errorbar(x, rho_cv_vals, yerr=rho_cv_sems,
                fmt='^:', color='#2471A3', lw=1.5, ms=6, capsize=3,
                label='pCCA  ρ₁  5-fold CV')
    # Shade the key claim per mode
    for i, m in enumerate(metrics_list):
        rc, rp = m['rho_cca'], m['rho_pcca']
        ax.annotate(
            '↑' if rp > rc + 0.02 else ('↓' if rp < rc - 0.02 else '≈'),
            xy=(i, max(rc, rp) + 0.06), ha='center', fontsize=11,
            color=('#e74c3c' if rp > rc + 0.02 else
                   '#27ae60' if rp < rc - 0.02 else '#95a5a6'),
            fontweight='bold',
        )
    ax.set_ylabel('Canonical correlation  ρ₁', fontsize=9)
    ax.set_title('(A)  In-sample vs. CV  ρ₁\n'
                 '↑ Mode2 (collider)  ↓ Mode1/3  ≈ Mode4', fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(-0.08, max(rho_pcca_vals + rho_cca_vals) + 0.22)
    _xax(ax)

    # ── Panel B: CCA–pCCA rotation angle theta ─────────────────────────────
    ax = axes[0, 1]
    theta_A = [m['theta_A_deg'] for m in metrics_list]
    theta_B = [m['theta_B_deg'] for m in metrics_list]
    ax.plot(x, theta_A, 's-', color='#922B21', lw=1.8, ms=6, label='Region A')
    ax.plot(x, theta_B, '^-', color='#2471A3', lw=1.8, ms=6, label='Region B')
    ax.axhline(90, color='#AAB7B8', ls='--', lw=0.9, alpha=0.6,
               label='90° (orthogonal)')
    ax.axhline(0,  color='#AAB7B8', ls=':', lw=0.7, alpha=0.4)
    ax.set_ylabel('θ  CCA–pCCA weight angle  (°)', fontsize=9)
    ax.set_ylim(-3, 95)
    ax.set_title('(B)  CCA–pCCA rotation angle θ\n'
                 'Mode4 ≈ 0°;  Modes 1–3 large', fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    _xax(ax)

    # ── Panel C: Cross-analysis collinearity kappa ─────────────────────────
    ax = axes[0, 2]
    kappa_A = [m['kappa_A'] for m in metrics_list]
    kappa_B = [m['kappa_B'] for m in metrics_list]
    ax.plot(x, kappa_A, 's-', color='#922B21', lw=1.8, ms=6, label='Region A')
    ax.plot(x, kappa_B, '^-', color='#2471A3', lw=1.8, ms=6, label='Region B')
    ax.axhline(1.0, color='#E74C3C', ls='--', lw=0.9, alpha=0.55,
               label='κ=1 (Type-II collapse)')
    ax.axhline(0.0, color='#27AE60', ls='--', lw=0.9, alpha=0.55,
               label='κ=0 (orthogonal axes)')
    ax.fill_between(x, [0.8]*len(x), [1.0]*len(x),
                    color='#FADBD8', alpha=0.35, zorder=0)
    ax.set_ylabel('κ  cross-analysis collinearity', fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.set_title('(C)  Weight collinearity κ\n'
                 'Mode1/4 ≈ 0;  Mode2/3 ≈ 1  (single pool)', fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False)
    _xax(ax)

    # ── Panel D: Variance partition — Region A ─────────────────────────────
    ax = axes[1, 0]
    rn_A = [m['r2_nuis_A'] for m in metrics_list]
    rc_A = [m['r2_comm_A'] for m in metrics_list]
    ax.bar(x, rn_A, color='#5D6D7E', alpha=0.88, label='Nuisance  r²')
    ax.bar(x, rc_A, bottom=rn_A, color='#922B21', alpha=0.88,
           label='Communication  r²  (pCCA axis)')
    ax.set_ylabel('Fraction of total variance', fontsize=9)
    ax.set_title('(D)  Variance partition — Region A\n'
                 'Mode3 large nuisance;  Mode4 near-zero', fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    _xax(ax)

    # ── Panel E: Variance partition — Region B ─────────────────────────────
    ax = axes[1, 1]
    rn_B = [m['r2_nuis_B'] for m in metrics_list]
    rc_B = [m['r2_comm_B'] for m in metrics_list]
    ax.bar(x, rn_B, color='#5D6D7E', alpha=0.88, label='Nuisance  r²')
    ax.bar(x, rc_B, bottom=rn_B, color='#2471A3', alpha=0.88,
           label='Communication  r²  (pCCA axis)')
    ax.set_ylabel('Fraction of total variance', fontsize=9)
    ax.set_title('(E)  Variance partition — Region B', fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    _xax(ax)

    # ── Panel F: pCCA latent cross-correlation ─────────────────────────────
    ax = axes[1, 2]
    mask_300 = np.abs(metrics_list[0]['lag_axis_ms']) <= 300
    lag_ax   = metrics_list[0]['lag_axis_ms'][mask_300]

    for m in metrics_list:
        ax.plot(lag_ax, m['xcorr_curve'][mask_300],
                color=m['color'], lw=2.0, alpha=0.90,
                label=m['label'].replace('\n', ' '))
        ax.scatter([m['lag_peak_ms']], [m['lag_corr_at_peak']],
                   color=m['color'], s=40, zorder=5)

    ax.axvline(0, color='k', ls=':', lw=0.8, alpha=0.45)
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.25)
    ax.set_xlabel('Lag (ms)   [+ = A leads B]', fontsize=9)
    ax.set_ylabel('Normalised cross-correlation', fontsize=9)
    ax.set_title('(F)  pCCA latent cross-correlation\nall four regimes overlaid;  dots = peak lag',
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    fig.suptitle(
        f'Simulation supplementary diagnostics — four-regime comparison\n'
        f'n = {n_pool} neurons/pool,  sigma_priv = {sigma_priv:.1f},  '
        f'{n_trials} trials,  T = {T} bins',
        fontsize=11, fontweight='bold',
    )

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(
            OUT_DIR,
            f'fig_sim_supp_comparison_n{n_pool}_spriv{sigma_priv:.1f}.png',
        )
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig)
# =============================================================================
# S10  ENTRY POINT
# =============================================================================

def main() -> None:
    """Run all figures.

    Fig A (nuisance='Z') -- pCCA(A,B|Z), recovers lateral channel;
                            8-column layout with residual PSTH in col 7.
    Fig A (nuisance='B') -- pCCA(A,Z|B), recovers hub channel;
                            complement diagnostic to the above.
    Fig B                -- compass + weight bars across population sizes.
    Fig C                -- subspace-recovery angle vs pool size.
    Fig D                -- 3-D pCCA latent trajectories (trials as z-axis).
    Fig D2               -- standalone 2-D pCCA latent trajectory comparison.
    """
    print("=" * 64)
    print("  pcca_hublateral_population_simulations.py  (v2)")
    print("=" * 64)

    fig_simulation_psth_latents(
        n_pool=20, sigma_priv=2.0,
        n_trials=N_TRIALS, T=T_BINS, t_onset=T_ONSET,
        show_gt_overlay=True, nuisance='B',
    )

    fig_simulation_psth_latents(
        n_pool=20, sigma_priv=2.0,
        n_trials=N_TRIALS, T=T_BINS, t_onset=T_ONSET,
        show_gt_overlay=True, nuisance='Z',
    )

    # fig_population_sweep()
    #fig_subspace_recovery_vs_size()
    fig_regime_comparison(n_pool=20, sigma_priv=0)
    fig_sim_supplementary_comparison(n_pool=20, sigma_priv=0)

    # fig_latent_trajectories_3d(sigma_priv=0.01)
    # fig_pcca_latent_trajectories_2d(sigma_priv=0.01)

    print("\nAll figures complete.")


if __name__ == '__main__':
    main()
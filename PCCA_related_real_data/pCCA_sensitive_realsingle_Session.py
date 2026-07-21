"""
pcca_sequential_ablation.py
===========================

Single-session pCCA sensitivity analysis for the MOs ↔ VPMPO region pair,
implemented entirely in Python.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — Sequential cumulative removal
    Nuisance regions are added to Z one at a time following ANATOMICAL_ORDER
    (regions not present in the session are silently skipped; TARGET_I and
    TARGET_J are never added).

    Analysis step k:
        Z_k  = concat([X_{r_1}, …, X_{r_k}])  for regions r_1,…,r_k ∈ order
        pCCA_k = pCCA(X_MOs, X_VPMPO | Z_k)
        CCA    = ridge_cca(X_MOs, X_VPMPO)      ← fixed baseline, no Z

    Final step (k = M, all other present regions in Z) is the REFERENCE.
    Summary statistics and a four-panel figure identify which intermediate
    step most closely resembles — and which diverges most from — the
    reference.

    Clarification: by construction pCCA_M is identical (up to
    floating-point rounding) to the result obtained by removing all
    remaining regions at once, because the same joint Z is formed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — Single-region ablation
    For each region r in ANATOMICAL_ORDER (present, ≠ TARGET), we set Z = X_r
    alone and refit pCCA.  This isolates each region's individual confounding
    contribution.

    Two reference comparisons are reported for each ablation step:
        (a) vs. CCA baseline  (Z = ∅):       measures how much removing r
            alone shifts the solution from no-partialling at all.
        (b) vs. full pCCA reference (Z = all): measures how close a single-
            region removal comes to the joint full-partialling result, i.e.
            how much marginal confounding work r carries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIGURE LAYOUT — per analysis step (both parts):

    2 rows × 5 columns
        Row 0 : MOs  (TARGET_I)
        Row 1 : VPMPO (TARGET_J)

        Col 0 : Rastermap / peak-time-ordered PSTH  (z-scored, RdBu_r)
                ── fixed across all steps; drawn once and cached for speed.
        Col 1 : Dominant pCCA canonical weight vector  (barh, signed)
                ── redrawn at every step; reflects updated Z.
        Col 2 : pCCA latent z(t)  mean ± SEM across trials
                ── projection onto Col-1 weights, using *residualized* data.
        Col 3 : Dominant CCA canonical weight vector   (barh, no nuisance)
                ── fixed across all steps in Part 1; re-used as a visual anchor.
        Col 4 : CCA latent z(t)   mean ± SEM across trials
                ── projection onto Col-3 weights, original (non-residualized) data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dependencies:  mat73, numpy, scipy, matplotlib, (optionally) rastermap
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore, pearsonr

try:
    import mat73
except ImportError as exc:
    raise SystemExit("mat73 is required: pip install mat73") from exc

try:
    from rastermap import Rastermap
    _RASTERMAP_OK = True
except ImportError:
    _RASTERMAP_OK = False
    warnings.warn(
        "rastermap not found; falling back to peak-time neuron ordering."
    )

from Useful_definition import ANATOMICAL_ORDER, safe_array

# =============================================================================
# 0.  Configuration
# =============================================================================

TARGET_I        = "MOs"
TARGET_J        = "VPMPO"

LAMBDA_CCA      = 1e-4   # ridge coefficient added to Cxx and Cyy in whitening
LAMBDA_HAT      = 1e-4   # ridge on Z'Z in hat matrix (scaled by n inside pcca)
N_COMPONENTS    = 5      # canonical dimensions to retain
N_NEURONS_SHOW  = 60     # neurons displayed in PSTH and weight-bar panels
TIME_RANGE_S    = (-1.5, 3.0)

# ── Colour palette ─────────────────────────────────────────────────────────
_CI_PCCA = "#C0392B"   # MOs  pCCA latent
_CJ_PCCA = "#2471A3"   # VPMPO pCCA latent
_CI_CCA  = "#E59866"   # MOs  CCA latent  (warm orange, same hue family)
_CJ_CCA  = "#5DADE2"   # VPMPO CCA latent (sky blue)
_C_POS   = "#922B21"   # positive weight bar
_C_NEG   = "#1A5276"   # negative weight bar


# =============================================================================
# 1.  Core mathematics
# =============================================================================

def _zscore_flat(X: np.ndarray) -> np.ndarray:
    """
    Flatten (n_trials, n_neurons, T) → (T·n_trials, n_neurons) and z-score
    each neuron across all T·n_trials samples.

    Row-index convention
    --------------------
    Row  t · n_trials + r   ←→   (timepoint t, trial r).

    This convention is the inverse of MATLAB's column-major ordering but is
    self-consistent throughout this module.  The inverse reshape is
    always  proj.reshape(T, n_trials).T  (see latent_projections).
    """
    n_trials, n, T = X.shape
    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)   # (n, T·n_trials)
    flat = zscore(flat, axis=1, nan_policy="omit")
    np.nan_to_num(flat, nan=0.0, copy=False)
    return flat.T                                            # (T·n_trials, n)


def _zscore_3d(X: np.ndarray) -> np.ndarray:
    """
    Same z-scoring as _zscore_flat, but returns (n_trials, n_neurons, T).
    Used for PSTH display so the trial structure is preserved.
    """
    n_trials, n, T = X.shape
    flat = _zscore_flat(X)                                   # (T·n_trials, n)
    # Invert: flat.T is (n, T·n_trials) with layout (n, T, n_trials) → transpose
    return flat.T.reshape(n, T, n_trials).transpose(2, 0, 1)


def _ridge_inv_sqrt(C: np.ndarray, lam: float) -> np.ndarray:
    """
    Compute (C + λI)^{-1/2} via eigh.

    Eigenvalues are clamped to ≥ 1e-12 to guard against numerical singularity.
    The result is the unique symmetric positive-definite matrix square-root
    inverse, used for whitening in the CCA derivation.
    """
    vals, vecs = np.linalg.eigh(C + lam * np.eye(C.shape[0]))
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def ridge_cca(
    X: np.ndarray,
    Y: np.ndarray,
    lam: float = LAMBDA_CCA,
    n_components: int = N_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ridge-regularised CCA.

    Parameters
    ----------
    X : (n_samples, p)
    Y : (n_samples, q)   — both pre-centred / z-scored.
    lam : ridge coefficient λ, added identically to Cxx and Cyy.

    Returns
    -------
    Wx  : (p, k)   canonical weight matrix for X
    Wy  : (q, k)   canonical weight matrix for Y
    rho : (k,)     canonical correlations (descending), clipped to [0, 1]

    Derivation
    ----------
    Let  A = (Cxx + λI)^{-1/2},  B = (Cyy + λI)^{-1/2}.
    Form  M = A Cxy B  and decompose  M = U S V^T.
    Then  Wx = A U[:, :k],  Wy = B V[:, :k].
    The columns of Wx maximise the regularised criterion
        max  Wx^T Cxy Wy
    subject to  Wx^T (Cxx + λI) Wx = I,  Wy^T (Cyy + λI) Wy = I.
    """
    n, p = X.shape
    q    = Y.shape[1]
    k    = min(n_components, p, q, n - 1)

    Cxx = X.T @ X / (n - 1)
    Cyy = Y.T @ Y / (n - 1)
    Cxy = X.T @ Y / (n - 1)

    A = _ridge_inv_sqrt(Cxx, lam)
    B = _ridge_inv_sqrt(Cyy, lam)

    U, S, Vt = np.linalg.svd(A @ Cxy @ B, full_matrices=False)
    k = min(k, len(S))

    Wx  = A @ U[:, :k]
    Wy  = B @ Vt[:k].T
    rho = np.clip(S[:k], 0.0, 1.0)
    return Wx, Wy, rho


def residualize(
    X_flat: np.ndarray,
    Z_flat: Optional[np.ndarray],
    lam_hat: float = LAMBDA_HAT,
) -> np.ndarray:
    """
    Project the nuisance matrix Z out of X via the regularised hat matrix.

    X_res = (I − H_Z) X,   H_Z = Z (Z'Z + λ·n·I)^{−1} Z'

    The ridge coefficient is scaled by n so the regularisation is invariant
    to sample count (equivalent to a fixed signal-to-noise prior).

    If Z_flat is None or empty, returns a copy of X_flat unchanged.

    Note
    ----
    Using the joint Z matrix (all nuisance regions concatenated) is strictly
    preferable to per-region sequential regression when nuisance regions share
    variance: the full off-diagonal block structure of Z'Z correctly accounts
    for inter-regional multicollinearity, whereas sequential regression
    double-counts shared variance.
    """
    if Z_flat is None or Z_flat.ndim < 2 or Z_flat.shape[1] == 0:
        return X_flat.copy()
    n, m  = Z_flat.shape
    ZtZ   = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    Beta  = np.linalg.solve(ZtZ, Z_flat.T)    # (m, n) = (Z'Z + λnI)^{-1} Z'
    return X_flat - Z_flat @ (Beta @ X_flat)


def pcca(
    X_flat: np.ndarray,
    Y_flat: np.ndarray,
    Z_flat: Optional[np.ndarray],
    lam_cca: float = LAMBDA_CCA,
    lam_hat: float = LAMBDA_HAT,
    n_components: int = N_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partial CCA.

    Projects the nuisance matrix Z out of both X and Y, then runs
    ridge_cca on the residuals.

    Returns
    -------
    Wx        : (p, k)          canonical weights for X (in *residual* space)
    Wy        : (q, k)
    rho       : (k,)
    X_res     : (n_samples, p)  residualized X  (needed for latent projection)
    Y_res     : (n_samples, q)  residualized Y
    """
    X_res = residualize(X_flat, Z_flat, lam_hat)
    Y_res = residualize(Y_flat, Z_flat, lam_hat)
    Wx, Wy, rho = ridge_cca(X_res, Y_res, lam_cca, n_components)
    return Wx, Wy, rho, X_res, Y_res


def latent_projections(
    X_flat: np.ndarray,
    w: np.ndarray,
    n_trials: int,
    T: int,
) -> np.ndarray:
    """
    Project a (T·n_trials, n) spike matrix onto one canonical weight vector.

    Parameters
    ----------
    X_flat  : (T·n_trials, n)   — output of _zscore_flat *or* residualize.
              For pCCA latents pass X_res; for CCA latents pass X_flat.
    w       : (n,)               canonical weight (dominant dimension).

    Returns
    -------
    (n_trials, T)

    Derivation of reshape
    ---------------------
    _zscore_flat produces row index  t · n_trials + r  (time-major, trial-minor).
    Therefore  proj.reshape(T, n_trials)  groups samples by timepoint;
    transposing gives the trial-by-time matrix.
    """
    proj = X_flat @ w                      # (T·n_trials,)
    return proj.reshape(T, n_trials).T     # (n_trials, T)


def _cos_sim_abs(a: np.ndarray, b: np.ndarray) -> float:
    """
    Absolute cosine similarity  |cos θ|  between two vectors.
    Sign-flip invariant; returns 0 when either vector is zero.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.abs(np.dot(a, b)) / (na * nb))


# =============================================================================
# 2.  Data loading
# =============================================================================

def load_region_spikes(
    session_path: str,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """
    Load a pCCA session .mat (HDF5 v7.3) via mat73.

    Returns
    -------
    region_spikes : {region_name: (n_trials, n_neurons, T)}
                    Region-level selected_neurons mask applied.
    n_trials      : number of trials (from the first valid region).
    T             : number of time-bins.
    """
    data  = mat73.loadmat(session_path)
    rd    = data.get("region_data", {})
    regs  = rd.get("regions", {})

    region_spikes: Dict[str, np.ndarray] = {}
    n_trials_out = T_out = None

    for rname, info in regs.items():
        if not isinstance(info, dict):
            continue
        sd = safe_array(info.get("spike_data"))
        if sd is None or sd.ndim != 3:
            continue
        sel = safe_array(info.get("selected_neurons"))
        if sel is not None and sel.size > 0:
            sd = sd[:, sel.ravel().astype(int) - 1, :]
        region_spikes[rname] = sd.astype(np.float32)
        if n_trials_out is None:
            n_trials_out, _, T_out = sd.shape

    print(
        f"  [load_region_spikes]  {len(region_spikes)} regions loaded  "
        f"| n_trials={n_trials_out}  T={T_out}"
    )
    return region_spikes, int(n_trials_out), int(T_out)


# =============================================================================
# 3.  Neuron ordering
# =============================================================================

def get_neuron_order(X: np.ndarray) -> np.ndarray:
    """
    Compute a 1-D neuron ordering for PSTH display.

    Preference: Rastermap (if available and ≥ 5 neurons).
    Fallback: sort by time of peak firing in trial-averaged PSTH.

    X : (n_trials, n_neurons, T)  — raw (not z-scored).
    Returns index array of shape (n_neurons,).
    """
    n_neurons = X.shape[1]
    if _RASTERMAP_OK and n_neurons >= 5:
        try:
            mat = X.transpose(1, 2, 0).reshape(n_neurons, -1).astype(np.float64)
            mat = zscore(mat, axis=1, nan_policy="omit")
            np.nan_to_num(mat, nan=0.0, copy=False)
            mdl = Rastermap(
                n_PCs=min(50, n_neurons),
                locality=0.0,
                grid_upsample=5,
            )
            mdl.fit(mat)
            return mdl.isort
        except Exception as exc:
            warnings.warn(f"Rastermap failed ({exc}); using peak-time ordering.")
    # Fallback
    psth = X.mean(axis=0)      # (n_neurons, T)
    return np.argsort(np.argmax(psth, axis=1))


# =============================================================================
# 4.  Figure primitives
# =============================================================================

def _draw_psth(
    ax: plt.Axes,
    X_z3d: np.ndarray,        # (n_trials, n_neurons, T)  z-scored
    sort_idx: np.ndarray,
    time_vec: np.ndarray,
    region_name: str,
    n_show: int,
) -> None:
    """Rastermap-ordered trial-averaged PSTH as a heatmap."""
    n_neurons = X_z3d.shape[1]
    step = max(1, n_neurons // n_show)
    sel  = sort_idx[::step][:n_show]
    psth = X_z3d.mean(axis=0)[sel]          # (n_sel, T)
    vmax = max(float(np.nanpercentile(np.abs(psth), 99)), 0.5)
    ax.imshow(
        psth, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[time_vec[0], time_vec[-1], len(sel), 0],
        origin="upper",
    )
    ax.axvline(0.0, color="k", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Neurons (sorted)", fontsize=7)
    ax.set_title(f"{region_name}  PSTH", fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)


def _draw_weight_bar(
    ax: plt.Axes,
    weight: np.ndarray,        # (n_neurons,)
    sort_idx: np.ndarray,
    n_show: int,
    title: str,
) -> None:
    """Horizontal bar chart of one canonical weight vector."""
    n_neurons = len(weight)
    step = max(1, n_neurons // n_show)
    sel  = sort_idx[::step][:n_show]
    w    = weight[sel]
    ypos = np.arange(len(sel)) + 0.5
    colors = [_C_POS if v >= 0 else _C_NEG for v in w]
    ax.barh(ypos, w, height=0.82, color=colors, alpha=0.83)
    ax.axvline(0.0, color="k", lw=0.7, alpha=0.45)
    ax.set_ylim(len(sel), 0)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=5)
    plt.setp(ax.get_yticklabels(), visible=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def _draw_latent(
    ax: plt.Axes,
    trials: np.ndarray,        # (n_trials, T)
    time_vec: np.ndarray,
    color: str,
    title: str,
    n_trace_max: int = 40,
    alpha_trial: float = 0.07,
) -> None:
    """Mean ± SEM latent projection with a subsample of thin trial traces."""
    n_trials = trials.shape[0]
    stride   = max(1, n_trials // n_trace_max)
    for tr in trials[::stride]:
        ax.plot(
            time_vec, tr, color=color, lw=0.2, alpha=alpha_trial,
            rasterized=True,
        )
    mean = np.nanmean(trials, axis=0)
    sem  = np.nanstd(trials, axis=0) / np.sqrt(max(n_trials, 1))
    ax.plot(time_vec, mean, color=color, lw=1.8, zorder=4)
    ax.fill_between(
        time_vec, mean - sem, mean + sem,
        color=color, alpha=0.30, zorder=3,
    )
    ax.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# =============================================================================
# 5.  Main 2-row × 5-column panel
# =============================================================================

def plot_step_panel(
    *,
    # Z-scored 3-D data for PSTH (fixed across steps)
    X_i_z3d: np.ndarray,        # (n_trials, n_i, T)
    X_j_z3d: np.ndarray,
    # Flat z-scored data for CCA latents (original, fixed)
    X_i_flat: np.ndarray,        # (T·n_trials, n_i)
    X_j_flat: np.ndarray,
    # Residualized flat data for pCCA latents (changes with Z)
    X_i_res: np.ndarray,         # (T·n_trials, n_i)
    X_j_res: np.ndarray,
    # Neuron orderings (fixed after first call)
    sort_i: np.ndarray,
    sort_j: np.ndarray,
    # pCCA results (change with Z at each step)
    Wx_pcca: np.ndarray,         # (n_i, k)
    Wy_pcca: np.ndarray,         # (n_j, k)
    rho_pcca: np.ndarray,        # (k,)
    # CCA results (fixed; no nuisance)
    Wx_cca: np.ndarray,
    Wy_cca: np.ndarray,
    rho_cca: np.ndarray,
    # Metadata
    time_vec: np.ndarray,
    n_trials: int,
    T: int,
    fig_title: str,
    n_show: int = N_NEURONS_SHOW,
) -> plt.Figure:
    """
    2-row × 5-column diagnostic panel for one analysis step.

    Columns
    -------
    0 : Rastermap-ordered PSTH            (z-scored activity, RdBu_r)
    1 : Dominant pCCA weight vector        (barh, colour-coded by sign)
    2 : pCCA latent z(t) mean ± SEM        (projected on residualized data)
    3 : Dominant CCA weight vector         (barh; plain CCA, no nuisance)
    4 : CCA latent z(t) mean ± SEM         (projected on original data)
    """
    fig, axes = plt.subplots(
        2, 5,
        figsize=(20, 7.5),
        gridspec_kw={
            "width_ratios": [3.5, 1.2, 2.5, 1.2, 2.5],
            "hspace": 0.52,
            "wspace": 0.32,
        },
    )

    rho0_p = float(rho_pcca[0]) if len(rho_pcca) > 0 else float("nan")
    rho0_c = float(rho_cca[0])  if len(rho_cca)  > 0 else float("nan")

    row_cfg = [
        # (region, X_z3d, X_flat, X_res, sort, w_pcca, w_cca, c_pcca, c_cca)
        (
            TARGET_I, X_i_z3d, X_i_flat, X_i_res, sort_i,
            Wx_pcca[:, 0], Wx_cca[:, 0], _CI_PCCA, _CI_CCA,
        ),
        (
            TARGET_J, X_j_z3d, X_j_flat, X_j_res, sort_j,
            Wy_pcca[:, 0], Wy_cca[:, 0], _CJ_PCCA, _CJ_CCA,
        ),
    ]

    for row, (
        rname, X_z3d, X_flat, X_res, sort_idx,
        w_pcca, w_cca, c_pcca, c_cca,
    ) in enumerate(row_cfg):

        # Col 0 — PSTH (fixed)
        _draw_psth(axes[row, 0], X_z3d, sort_idx, time_vec, rname, n_show)

        # Col 1 — pCCA weight (changes each step)
        _draw_weight_bar(
            axes[row, 1], w_pcca, sort_idx, n_show,
            f"pCCA weight\nρ₁ = {rho0_p:.3f}",
        )

        # Col 2 — pCCA latent (projected on residualized data)
        lat_p = latent_projections(X_res, w_pcca, n_trials, T)
        _draw_latent(
            axes[row, 2], lat_p, time_vec, c_pcca,
            f"pCCA  z(t)   ρ₁={rho0_p:.3f}",
        )

        # Col 3 — CCA weight (fixed baseline)
        _draw_weight_bar(
            axes[row, 3], w_cca, sort_idx, n_show,
            f"CCA weight\nρ₁ = {rho0_c:.3f}",
        )

        # Col 4 — CCA latent (projected on original z-scored data)
        lat_c = latent_projections(X_flat, w_cca, n_trials, T)
        _draw_latent(
            axes[row, 4], lat_c, time_vec, c_cca,
            f"CCA  z(t)   ρ₁={rho0_c:.3f}",
        )

    fig.suptitle(fig_title, fontsize=11, fontweight="bold", y=1.02)
    return fig


# =============================================================================
# 6.  Step-result container and summary statistics
# =============================================================================

class StepResult:
    """
    Stores all quantities computed at one analysis step for later comparison.

    Attributes
    ----------
    label            : human-readable step label (e.g. "step02_+MOs")
    nuisance_regions : list of region names currently in Z
    rho_pcca         : dominant pCCA canonical correlation ρ₁
    Wx               : (n_i, k)  pCCA canonical weights for TARGET_I
    Wy               : (n_j, k)  pCCA canonical weights for TARGET_J
    z_i_mean         : (T,)      trial-averaged pCCA latent for TARGET_I
    z_j_mean         : (T,)      same for TARGET_J
    """
    __slots__ = [
        "label", "nuisance_regions", "rho_pcca",
        "Wx", "Wy", "z_i_mean", "z_j_mean",
    ]

    def __init__(
        self,
        label: str,
        nuisance_regions: List[str],
        rho_pcca: float,
        Wx: np.ndarray,
        Wy: np.ndarray,
        z_i_mean: np.ndarray,
        z_j_mean: np.ndarray,
    ) -> None:
        self.label            = label
        self.nuisance_regions = list(nuisance_regions)
        self.rho_pcca         = float(rho_pcca)
        self.Wx               = Wx
        self.Wy               = Wy
        self.z_i_mean         = z_i_mean
        self.z_j_mean         = z_j_mean


def compute_similarity(s: StepResult, ref: StepResult) -> Dict[str, float]:
    """
    Quantify similarity between step s and a reference step.

    Metrics
    -------
    cos_sim_i    : |cos θ| between Wx[:,0] of s and ref  (sign-flip invariant).
    cos_sim_j    : same for Wy[:,0].
    latent_r_i   : Pearson r between z_i_mean of s and ref.
    latent_r_j   : same for TARGET_J.
    rho_abs_diff : |ρ₁(s) − ρ₁(ref)|.
    divergence   : composite score D ∈ [0, 5].  D = 0 ↔ s identical to ref.

    Divergence definition
    ---------------------
    D = (1 − |cos θ_i|) + (1 − |cos θ_j|)
      + (1 − max(r_i, −1)) + (1 − max(r_j, −1))
      + |ρ₁(s) − ρ₁(ref)|

    The first four terms are bounded in [0, 1] each; the fifth is bounded in
    [0, 1] because ρ₁ ∈ [0, 1].  Hence D ∈ [0, 5].
    Smaller D → more similar to reference.
    """
    cos_i = _cos_sim_abs(s.Wx[:, 0], ref.Wx[:, 0])
    cos_j = _cos_sim_abs(s.Wy[:, 0], ref.Wy[:, 0])

    r_i = float(pearsonr(s.z_i_mean, ref.z_i_mean)[0])
    r_j = float(pearsonr(s.z_j_mean, ref.z_j_mean)[0])

    rho_diff = abs(s.rho_pcca - ref.rho_pcca)

    divergence = (
        (1 - cos_i)
        + (1 - cos_j)
        + (1 - max(r_i, -1.0))
        + (1 - max(r_j, -1.0))
        + rho_diff
    )

    return dict(
        cos_sim_i    = cos_i,
        cos_sim_j    = cos_j,
        latent_r_i   = r_i,
        latent_r_j   = r_j,
        rho_abs_diff = rho_diff,
        divergence   = divergence,
    )


def identify_extremes(
    step_results: List[StepResult],
    ref: StepResult,
    exclude_ref: bool = True,
) -> Tuple[int, int, List[Dict[str, float]]]:
    """
    Compare every step to ref and identify the most-similar and most-divergent.

    Parameters
    ----------
    exclude_ref : if True, the last element of step_results (assumed to equal
                  ref itself) is excluded from the search — it would trivially
                  win as "most similar" with D = 0.

    Returns
    -------
    idx_closest  : index into step_results with minimum divergence
    idx_furthest : index with maximum divergence
    sims         : list of similarity dicts, one per step
    """
    sims = [compute_similarity(s, ref) for s in step_results]
    candidate_idx = list(range(len(step_results) - 1) if exclude_ref
                         else range(len(step_results)))
    if not candidate_idx:
        return 0, 0, sims

    divs         = [sims[i]["divergence"] for i in candidate_idx]
    idx_closest  = candidate_idx[int(np.argmin(divs))]
    idx_furthest = candidate_idx[int(np.argmax(divs))]
    return idx_closest, idx_furthest, sims


def print_similarity_table(
    step_results: List[StepResult],
    sims: List[Dict[str, float]],
    ref_label: str,
    idx_closest: int,
    idx_furthest: int,
) -> None:
    """Print a formatted summary table to stdout."""
    hdr = (
        f"\n{'Step':<35}  {'ρ₁':>6}  {'|cosθ|_i':>8}  {'|cosθ|_j':>8}  "
        f"{'r_i':>6}  {'r_j':>6}  {'|Δρ|':>6}  {'D':>6}  {'Note'}"
    )
    print(hdr)
    print("─" * len(hdr))
    for k, (s, sim) in enumerate(zip(step_results, sims)):
        note = ""
        if k == idx_closest:
            note = "◀ MOST SIMILAR to reference"
        elif k == idx_furthest:
            note = "◀ MOST DIVERGENT from reference"
        elif k == len(step_results) - 1:
            note = "(reference)"
        print(
            f"  {s.label:<33}  {s.rho_pcca:6.3f}  "
            f"{sim['cos_sim_i']:8.3f}  {sim['cos_sim_j']:8.3f}  "
            f"{sim['latent_r_i']:6.3f}  {sim['latent_r_j']:6.3f}  "
            f"{sim['rho_abs_diff']:6.3f}  {sim['divergence']:6.3f}  {note}"
        )
    print(f"\n  Reference: {ref_label}\n")


# =============================================================================
# 7.  Four-panel summary figure (shared by Parts 1 and 2)
# =============================================================================

def plot_summary_figure(
    step_results: List[StepResult],
    sims: List[Dict[str, float]],
    idx_closest: int,
    idx_furthest: int,
    ref_label: str,
    title: str,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Four-panel summary figure.

    Panel A : Dominant pCCA ρ₁ vs. analysis step
    Panel B : |cos θ| similarity to reference (Wx and Wy separately)
    Panel C : Mean latent Pearson r to reference (TARGET_I and TARGET_J)
    Panel D : Composite divergence D bar chart
              Green bar = most similar, red bar = most divergent.
    """
    labels = [s.label for s in step_results]
    rhos   = [s.rho_pcca for s in step_results]
    cos_i  = [d["cos_sim_i"]    for d in sims]
    cos_j  = [d["cos_sim_j"]    for d in sims]
    lat_i  = [d["latent_r_i"]   for d in sims]
    lat_j  = [d["latent_r_j"]   for d in sims]
    div    = [d["divergence"]    for d in sims]
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 8),
        gridspec_kw={"hspace": 0.52, "wspace": 0.35},
    )

    # ── Panel A: ρ₁ ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, rhos, "o-", color="#922B21", lw=1.8, ms=5)
    ax.axhline(rhos[-1], color="gray", ls="--", lw=1.0, alpha=0.6,
               label=f"Reference ({ref_label})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel(r"Dominant pCCA  $\rho_1$", fontsize=9)
    ax.set_title(r"(A)  Dominant canonical correlation", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # ── Panel B: cosine similarity ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, cos_i, "s-", color=_CI_PCCA, lw=1.8, ms=5, label=TARGET_I)
    ax.plot(x, cos_j, "^-", color=_CJ_PCCA, lw=1.8, ms=5, label=TARGET_J)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel(r"$|\cos\theta|$  vs reference weight", fontsize=9)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(r"(B)  Weight-vector cosine similarity to reference", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # ── Panel C: latent Pearson r ────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(x, lat_i, "s-", color=_CI_PCCA, lw=1.8, ms=5, label=TARGET_I)
    ax.plot(x, lat_j, "^-", color=_CJ_PCCA, lw=1.8, ms=5, label=TARGET_J)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("Pearson  r  (mean latent vs reference)", fontsize=9)
    ax.set_ylim(-0.15, 1.08)
    ax.set_title(r"(C)  Mean latent time-series correlation to reference", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # ── Panel D: divergence bar ──────────────────────────────────────────────
    ax = axes[1, 1]
    bar_colors = ["#CACFD2"] * len(x)
    if idx_closest  < len(bar_colors): bar_colors[idx_closest]  = "#27AE60"
    if idx_furthest < len(bar_colors): bar_colors[idx_furthest] = "#E74C3C"
    ax.bar(x, div, color=bar_colors, alpha=0.88, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("Composite divergence  D  (max = 5)", fontsize=9)
    ax.set_title(
        "(D)  Divergence from reference\n"
        "  ■ green = most similar   ■ red = most divergent",
        fontsize=9,
    )
    ax.grid(alpha=0.2)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [summary] saved: {output_path}")

    return fig


# =============================================================================
# 8.  Part 1 — Sequential cumulative removal
# =============================================================================

def run_sequential_removal(
    region_spikes: Dict[str, np.ndarray],
    n_trials: int,
    T: int,
    output_dir: Path,
    session_name: str,
    n_show: int = N_NEURONS_SHOW,
) -> None:
    """
    Cumulative pCCA sweep for the TARGET_I ↔ TARGET_J pair.

    Algorithm
    ---------
    Filter ANATOMICAL_ORDER to regions that are (a) present in the session
    and (b) neither TARGET_I nor TARGET_J.  Call this list [r_1, …, r_M].

    Step 0  : Z = ∅             → plain CCA (pCCA baseline; W identical to CCA)
    Step k  : Z = [X_{r_1}, …, X_{r_k}]
    Step M  : Z = all other regions  ← REFERENCE

    For every step one 2×5-column figure is saved.
    After all steps a four-panel summary figure is produced.
    """
    out_part1 = output_dir / "part1_sequential"
    out_part1.mkdir(parents=True, exist_ok=True)

    # ── Validate targets present ─────────────────────────────────────────────
    for t in (TARGET_I, TARGET_J):
        if t not in region_spikes:
            raise RuntimeError(
                f"Target region '{t}' not found in session.  "
                f"Available: {sorted(region_spikes)}"
            )

    X_i_raw = region_spikes[TARGET_I]   # (n_trials, n_i, T)
    X_j_raw = region_spikes[TARGET_J]

    # ── Flat z-scored data (fixed) ────────────────────────────────────────────
    X_i_flat = _zscore_flat(X_i_raw)    # (T·n_trials, n_i)
    X_j_flat = _zscore_flat(X_j_raw)

    # ── Z-scored 3-D (for PSTH display; fixed) ───────────────────────────────
    X_i_z3d  = _zscore_3d(X_i_raw)     # (n_trials, n_i, T)
    X_j_z3d  = _zscore_3d(X_j_raw)

    # ── Neuron orderings (computed once) ─────────────────────────────────────
    sort_i = get_neuron_order(X_i_raw)
    sort_j = get_neuron_order(X_j_raw)

    # ── Plain CCA (fixed baseline, no nuisance) ───────────────────────────────
    Wx_cca, Wy_cca, rho_cca = ridge_cca(X_i_flat, X_j_flat)
    print(f"  [Part 1] CCA baseline  ρ₁ = {rho_cca[0]:.4f}")

    # ── Build nuisance queue following ANATOMICAL_ORDER ──────────────────────
    nuisance_queue: List[str] = [
        r for r in ANATOMICAL_ORDER
        if r in region_spikes and r not in (TARGET_I, TARGET_J)
    ]
    print(f"  [Part 1] Nuisance queue ({len(nuisance_queue)} regions): "
          f"{nuisance_queue}")

    # ── Precompute flat z-scored matrices for nuisance regions ───────────────
    nuisance_flat: Dict[str, np.ndarray] = {
        r: _zscore_flat(region_spikes[r]) for r in nuisance_queue
    }

    # ── Time axis ─────────────────────────────────────────────────────────────
    time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)

    # ── Iterate steps ─────────────────────────────────────────────────────────
    step_results: List[StepResult] = []
    accumulated_nuisance: List[str] = []

    # Step 0: Z = ∅ (same as CCA)
    steps_iter = [None] + nuisance_queue   # None = step 0 (add nothing)

    for step_idx, add_region in enumerate(steps_iter):
        if add_region is not None:
            accumulated_nuisance.append(add_region)

        # Build joint Z matrix
        if accumulated_nuisance:
            Z_flat = np.concatenate(
                [nuisance_flat[r] for r in accumulated_nuisance], axis=1
            )
        else:
            Z_flat = None

        # Compute pCCA
        Wx_p, Wy_p, rho_p, X_i_res, X_j_res = pcca(
            X_i_flat, X_j_flat, Z_flat,
        )

        # Mean latent traces (projected on residualized data)
        z_i_mean = latent_projections(X_i_res, Wx_p[:, 0], n_trials, T).mean(axis=0)
        z_j_mean = latent_projections(X_j_res, Wy_p[:, 0], n_trials, T).mean(axis=0)

        # Step label
        if accumulated_nuisance:
            label = f"step{step_idx:02d}_+{add_region if add_region else '∅'}"
        else:
            label = "step00_CCA_baseline"

        step_results.append(StepResult(
            label            = label,
            nuisance_regions = list(accumulated_nuisance),
            rho_pcca         = float(rho_p[0]),
            Wx               = Wx_p,
            Wy               = Wy_p,
            z_i_mean         = z_i_mean,
            z_j_mean         = z_j_mean,
        ))

        # Build nuisance description for title
        if not accumulated_nuisance:
            nuis_str = "Z = ∅  (plain CCA)"
        else:
            nuis_str = "Z = {" + ", ".join(accumulated_nuisance) + "}"

        fig_title = (
            f"[Part 1 — Step {step_idx:02d}]  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}  |  {nuis_str}\n"
            f"pCCA ρ₁ = {rho_p[0]:.4f}   |   CCA ρ₁ = {rho_cca[0]:.4f}"
        )

        fig = plot_step_panel(
            X_i_z3d  = X_i_z3d,
            X_j_z3d  = X_j_z3d,
            X_i_flat = X_i_flat,
            X_j_flat = X_j_flat,
            X_i_res  = X_i_res,
            X_j_res  = X_j_res,
            sort_i   = sort_i,
            sort_j   = sort_j,
            Wx_pcca  = Wx_p,
            Wy_pcca  = Wy_p,
            rho_pcca = rho_p,
            Wx_cca   = Wx_cca,
            Wy_cca   = Wy_cca,
            rho_cca  = rho_cca,
            time_vec = time_vec,
            n_trials = n_trials,
            T        = T,
            fig_title= fig_title,
            n_show   = n_show,
        )

        fig_path = out_part1 / (
            f"{session_name}_part1_step{step_idx:02d}"
            f"_{'_'.join(accumulated_nuisance) if accumulated_nuisance else 'baseline'}"
            ".png"
        )
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(
            f"  [Part 1] step {step_idx:02d}  "
            f"nuisance={accumulated_nuisance}  "
            f"ρ₁={rho_p[0]:.4f}  saved: {fig_path.name}"
        )

    # ── Reference = final step (all nuisance in Z) ───────────────────────────
    ref = step_results[-1]

    # ── Summary statistics ────────────────────────────────────────────────────
    idx_closest, idx_furthest, sims = identify_extremes(step_results, ref)

    print("\n" + "=" * 90)
    print(f"Part 1  [{TARGET_I} ↔ {TARGET_J}]  Summary: similarity to reference")
    print(f"Reference = {ref.label}  (ρ₁ = {ref.rho_pcca:.4f})")
    print_similarity_table(step_results, sims, ref.label, idx_closest, idx_furthest)
    print(
        f"  ▶  Most similar  : step {idx_closest:02d}  ({step_results[idx_closest].label})\n"
        f"  ▶  Most divergent: step {idx_furthest:02d}  ({step_results[idx_furthest].label})\n"
    )

    # ── Summary figure ─────────────────────────────────────────────────────────
    summary_fig = plot_summary_figure(
        step_results = step_results,
        sims         = sims,
        idx_closest  = idx_closest,
        idx_furthest = idx_furthest,
        ref_label    = ref.label,
        title        = (
            f"Part 1 — Sequential cumulative removal  |  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}   reference = {ref.label}"
        ),
        output_path  = out_part1 / f"{session_name}_part1_summary.png",
    )
    plt.close(summary_fig)


# =============================================================================
# 9.  Part 2 — Single-region ablation
# =============================================================================

def run_single_ablation(
    region_spikes: Dict[str, np.ndarray],
    n_trials: int,
    T: int,
    output_dir: Path,
    session_name: str,
    n_show: int = N_NEURONS_SHOW,
) -> None:
    """
    Single-region ablation pCCA sweep for the TARGET_I ↔ TARGET_J pair.

    For each candidate nuisance region r (present in session, ≠ TARGET),
    we set  Z = X_r  alone and refit pCCA.  This isolates each region's
    independent confounding contribution.

    Two reference comparisons
    -------------------------
    (a) vs. CCA baseline (Z = ∅):
        Quantifies how much removing r alone shifts the solution away from
        the no-partialling baseline — i.e. the individual confounding effect.
    (b) vs. full pCCA reference (Z = all other regions):
        Quantifies how close a single-region removal comes to the joint
        full-partialling result — i.e. which region carries most of the
        marginal confounding work within the joint Z.

    Figures
    -------
    One 2×5-column panel per ablation step (same layout as Part 1).
    Two four-panel summary figures (one per reference comparison).
    """
    out_part2 = output_dir / "part2_ablation"
    out_part2.mkdir(parents=True, exist_ok=True)

    # ── Validate ─────────────────────────────────────────────────────────────
    for t in (TARGET_I, TARGET_J):
        if t not in region_spikes:
            raise RuntimeError(f"Target region '{t}' not in session.")

    X_i_raw  = region_spikes[TARGET_I]
    X_j_raw  = region_spikes[TARGET_J]
    X_i_flat = _zscore_flat(X_i_raw)
    X_j_flat = _zscore_flat(X_j_raw)
    X_i_z3d  = _zscore_3d(X_i_raw)
    X_j_z3d  = _zscore_3d(X_j_raw)
    sort_i   = get_neuron_order(X_i_raw)
    sort_j   = get_neuron_order(X_j_raw)

    time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)

    # ── CCA baseline (Z = ∅) ──────────────────────────────────────────────────
    Wx_cca, Wy_cca, rho_cca = ridge_cca(X_i_flat, X_j_flat)
    print(f"  [Part 2] CCA baseline  ρ₁ = {rho_cca[0]:.4f}")

    # ── Full pCCA reference (Z = all nuisance regions) ────────────────────────
    nuisance_all: List[str] = [
        r for r in ANATOMICAL_ORDER
        if r in region_spikes and r not in (TARGET_I, TARGET_J)
    ]
    nuisance_flat: Dict[str, np.ndarray] = {
        r: _zscore_flat(region_spikes[r]) for r in nuisance_all
    }

    if nuisance_all:
        Z_full = np.concatenate([nuisance_flat[r] for r in nuisance_all], axis=1)
        Wx_ref, Wy_ref, rho_ref, X_i_ref_res, X_j_ref_res = pcca(
            X_i_flat, X_j_flat, Z_full,
        )
        zi_ref = latent_projections(X_i_ref_res, Wx_ref[:, 0], n_trials, T).mean(0)
        zj_ref = latent_projections(X_j_ref_res, Wy_ref[:, 0], n_trials, T).mean(0)
        full_pcca_ref = StepResult(
            label            = "full_pCCA_reference",
            nuisance_regions = nuisance_all,
            rho_pcca         = float(rho_ref[0]),
            Wx               = Wx_ref,
            Wy               = Wy_ref,
            z_i_mean         = zi_ref,
            z_j_mean         = zj_ref,
        )
        print(f"  [Part 2] Full pCCA reference  ρ₁ = {rho_ref[0]:.4f}")
    else:
        warnings.warn("No nuisance regions found; Part 2 has no ablation targets.")
        return

    # ── CCA baseline as StepResult (for structured comparison) ───────────────
    X_i_cca_res = X_i_flat.copy()   # Z = ∅ → residual = original
    X_j_cca_res = X_j_flat.copy()
    zi_cca = latent_projections(X_i_cca_res, Wx_cca[:, 0], n_trials, T).mean(0)
    zj_cca = latent_projections(X_j_cca_res, Wy_cca[:, 0], n_trials, T).mean(0)
    cca_baseline_ref = StepResult(
        label            = "CCA_baseline",
        nuisance_regions = [],
        rho_pcca         = float(rho_cca[0]),
        Wx               = Wx_cca,
        Wy               = Wy_cca,
        z_i_mean         = zi_cca,
        z_j_mean         = zj_cca,
    )

    # ── Ablation loop ─────────────────────────────────────────────────────────
    ablation_results: List[StepResult] = []

    for abl_idx, region in enumerate(nuisance_all):
        Z_single = nuisance_flat[region]    # (T·n_trials, n_r)

        Wx_p, Wy_p, rho_p, X_i_res, X_j_res = pcca(
            X_i_flat, X_j_flat, Z_single,
        )

        z_i_mean = latent_projections(X_i_res, Wx_p[:, 0], n_trials, T).mean(0)
        z_j_mean = latent_projections(X_j_res, Wy_p[:, 0], n_trials, T).mean(0)

        label = f"abl{abl_idx:02d}_{region}"
        ablation_results.append(StepResult(
            label            = label,
            nuisance_regions = [region],
            rho_pcca         = float(rho_p[0]),
            Wx               = Wx_p,
            Wy               = Wy_p,
            z_i_mean         = z_i_mean,
            z_j_mean         = z_j_mean,
        ))

        fig_title = (
            f"[Part 2 — Ablation {abl_idx:02d}]  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}  |  Z = {{{region}}}\n"
            f"pCCA ρ₁ = {rho_p[0]:.4f}   |   CCA ρ₁ = {rho_cca[0]:.4f}"
        )

        fig = plot_step_panel(
            X_i_z3d  = X_i_z3d,
            X_j_z3d  = X_j_z3d,
            X_i_flat = X_i_flat,
            X_j_flat = X_j_flat,
            X_i_res  = X_i_res,
            X_j_res  = X_j_res,
            sort_i   = sort_i,
            sort_j   = sort_j,
            Wx_pcca  = Wx_p,
            Wy_pcca  = Wy_p,
            rho_pcca = rho_p,
            Wx_cca   = Wx_cca,
            Wy_cca   = Wy_cca,
            rho_cca  = rho_cca,
            time_vec = time_vec,
            n_trials = n_trials,
            T        = T,
            fig_title= fig_title,
            n_show   = n_show,
        )

        fig_path = out_part2 / f"{session_name}_part2_abl{abl_idx:02d}_{region}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(
            f"  [Part 2] abl {abl_idx:02d}  Z = {{{region:<8}}}  "
            f"ρ₁={rho_p[0]:.4f}  saved: {fig_path.name}"
        )

    # ── Summary statistics — comparison (a): vs. CCA baseline ────────────────
    print("\n" + "=" * 90)
    print(f"Part 2  [{TARGET_I} ↔ {TARGET_J}]  "
          f"Summary (a): similarity to CCA baseline  (Z = ∅)")
    sims_vs_cca = [compute_similarity(s, cca_baseline_ref) for s in ablation_results]
    idx_c_a, idx_f_a, _ = identify_extremes(
        ablation_results, cca_baseline_ref, exclude_ref=False,
    )
    print_similarity_table(
        ablation_results, sims_vs_cca,
        cca_baseline_ref.label, idx_c_a, idx_f_a,
    )

    # ── Summary statistics — comparison (b): vs. full pCCA reference ─────────
    print("=" * 90)
    print(f"Part 2  [{TARGET_I} ↔ {TARGET_J}]  "
          f"Summary (b): similarity to full pCCA  (Z = all)")
    sims_vs_full = [compute_similarity(s, full_pcca_ref) for s in ablation_results]
    idx_c_b, idx_f_b, _ = identify_extremes(
        ablation_results, full_pcca_ref, exclude_ref=False,
    )
    print_similarity_table(
        ablation_results, sims_vs_full,
        full_pcca_ref.label, idx_c_b, idx_f_b,
    )
    print(
        f"  ▶  Region whose removal most closely approximates full pCCA:\n"
        f"       {ablation_results[idx_c_b].nuisance_regions[0]}  "
        f"(D = {sims_vs_full[idx_c_b]['divergence']:.3f})\n"
        f"  ▶  Region with least marginal confounding effect:\n"
        f"       {ablation_results[idx_f_b].nuisance_regions[0]}  "
        f"(D = {sims_vs_full[idx_f_b]['divergence']:.3f})\n"
    )

    # ── Summary figure (a): vs. CCA baseline ─────────────────────────────────
    fig_a = plot_summary_figure(
        step_results = ablation_results,
        sims         = sims_vs_cca,
        idx_closest  = idx_c_a,
        idx_furthest = idx_f_a,
        ref_label    = "CCA baseline  (Z = ∅)",
        title        = (
            f"Part 2 — Single-region ablation  |  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}   reference: CCA baseline"
        ),
        output_path  = out_part2 / f"{session_name}_part2_summary_vs_CCA.png",
    )
    plt.close(fig_a)

    # ── Summary figure (b): vs. full pCCA ─────────────────────────────────────
    fig_b = plot_summary_figure(
        step_results = ablation_results,
        sims         = sims_vs_full,
        idx_closest  = idx_c_b,
        idx_furthest = idx_f_b,
        ref_label    = "full pCCA  (Z = all)",
        title        = (
            f"Part 2 — Single-region ablation  |  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}   reference: full pCCA"
        ),
        output_path  = out_part2 / f"{session_name}_part2_summary_vs_fullpCCA.png",
    )
    plt.close(fig_b)


# =============================================================================
# 10.  Entry point
# =============================================================================

def main() -> None:
    # ── Path configuration ───────────────────────────────────────────────────
    BASE_DIR     = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    SESSION_NAME = "yp020_220331"
    SESSION_FILE = (
        BASE_DIR
        / "pcca_sessions_cued_hit_long_results"
        / f"{SESSION_NAME}_analysis_results.mat"
    )
    OUTPUT_DIR   = BASE_DIR / "Paper_output" / "pcca_ablation" / SESSION_NAME

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"pCCA Sequential Ablation Analysis")
    print(f"  Session : {SESSION_NAME}")
    print(f"  Pair    : {TARGET_I} ↔ {TARGET_J}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    region_spikes, n_trials, T = load_region_spikes(str(SESSION_FILE))

    # ── Part 1: sequential cumulative removal ────────────────────────────────
    print("\n── Part 1: Sequential cumulative removal ──────────────────────────")
    run_sequential_removal(
        region_spikes = region_spikes,
        n_trials      = n_trials,
        T             = T,
        output_dir    = OUTPUT_DIR,
        session_name  = SESSION_NAME,
    )

    # ── Part 2: single-region ablation ──────────────────────────────────────
    print("\n── Part 2: Single-region ablation ─────────────────────────────────")
    run_single_ablation(
        region_spikes = region_spikes,
        n_trials      = n_trials,
        T             = T,
        output_dir    = OUTPUT_DIR,
        session_name  = SESSION_NAME,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
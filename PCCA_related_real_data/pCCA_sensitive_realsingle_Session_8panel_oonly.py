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

    2 rows × 7 columns
        Row 0 : MOs  (TARGET_I)
        Row 1 : VPMPO (TARGET_J)

        Col 0 : Rastermap / peak-time-ordered PSTH  (z-scored, RdBu_r)
        Col 1 : Dominant pCCA canonical weight vector  (barh, signed)
        Col 2 : pCCA latent z(t)  mean ± SEM across trials
        Col 3 : Dominant CCA canonical weight vector   (barh, no nuisance)
        Col 4 : CCA latent z(t)   mean ± SEM across trials
        Col 5 : Nuisance region Rastermap (centered, spanning both rows)
        Col 6 : Nuisance → target regression weight  (β·w₁, barh, signed)

    NOTE on neuron ordering (cols 0, 1, 3, 5, 6):
        A single Rastermap model is fitted on all regions' activity
        concatenated.  Each region's per-region sort index is then extracted
        from the resulting global embedding, so the ordering is consistent
        across all display panels.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dependencies:  mat73, numpy, scipy, matplotlib, (optionally) rastermap
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as _manim
import numpy as np
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field as _field
from Useful_definition import (
    ANATOMICAL_ORDER,
    safe_array,
    apply_latent_sign_correction,   # ← new
)
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

TARGET_I = "MOs"
TARGET_J = "VPMPO"

SUBTRACT_PSTH:  bool = False
SHUFFLE_TRIALS: bool = False


# Build the trial permutation once so flat and 3-D views are always in sync.
# A single permutation is applied to TARGET_I only; TARGET_J is left intact,
# which is what breaks the cross-regional trial pairing.
_rng  = np.random.default_rng(42)
# which is what breaks the cross-regional trial pairing.
_rng2  = np.random.default_rng(43)


LAMBDA_CCA  = 1e-4   # ridge coefficient added to Cxx and Cyy in whitening
LAMBDA_HAT  = 1e-4   # ridge on Z'Z in hat matrix (scaled by n inside pcca)
N_COMPONENTS = 5     # canonical dimensions to retain
N_NEURONS_SHOW = 60  # neurons displayed in PSTH and weight-bar panels
TIME_RANGE_S = (-1.5, 3.0)

# ── Colour palette ─────────────────────────────────────────────────────────
_CI_PCCA  = "#C0392B"   # MOs  pCCA latent
_CJ_PCCA  = "#2471A3"   # VPMPO pCCA latent
_CI_CCA   = "#E59866"   # MOs  CCA latent  (warm orange, same hue family)
_CJ_CCA   = "#5DADE2"   # VPMPO CCA latent (sky blue)
_C_POS    = "#922B21"   # positive weight bar
_C_NEG    = "#1A5276"   # negative weight bar
_C_CCA_POS = '#E08214'    # amber — positive CCA weight (mirrors C_COMM)
_C_CCA_NEG = '#762A83'    # purple — negative CCA weight (mirrors C_COMP)
_C_BETA_POS = '#4DAC26'   # green  — positive nuisance β·w₁
_C_BETA_NEG = '#969696'   # grey   — negative nuisance β·w₁


# =============================================================================
# 1.  Core mathematics
# =============================================================================
def _zscore_flat(
    X: np.ndarray,
    *,
    subtract_psth: bool = False,
    shuffle_trials: bool = False,
    rng: Optional[np.random.Generator] = None,
    perm: Optional[np.ndarray] = None,
) -> np.ndarray:

    n_trials, n, T = X.shape
    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
    flat = zscore(flat, axis=1, nan_policy="omit")
    np.nan_to_num(flat, nan=0.0, copy=False)

    X= flat.reshape(n, T, n_trials).transpose(2, 0, 1)

    # ── 1. PSTH subtraction ───────────────────────────────────────────────
    # μ has shape (1, n, T); broadcasting removes it from every trial block.
    if subtract_psth:
        X = X - X.mean(axis=0, keepdims=True)

    # ── 2. Trial-level shuffle ────────────────────────────────────────────
    if shuffle_trials:
        if perm is not None:
            if perm.shape != (n_trials,):
                raise ValueError(
                    f"perm must have shape ({n_trials},); got {perm.shape}"
                )
            X = X[perm]
        else:
            if rng is None:
                rng = np.random.default_rng()
            X = X[rng.permutation(n_trials)]

    # ── 3. Flatten → (n, T * n_trials), z-score per neuron, transpose ─────
    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
    # flat = zscore(flat, axis=1, nan_policy="omit")
    # np.nan_to_num(flat, nan=0.0, copy=False)
    return flat.T   # (T * n_trials, n)


def _zscore_3d(
    X: np.ndarray,
    *,
    subtract_psth: bool = False,
    shuffle_trials: bool = False,
    rng: Optional[np.random.Generator] = None,
    perm: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_trials, n, T = X.shape
    flat = _zscore_flat(
        X,
        subtract_psth=subtract_psth,
        shuffle_trials=shuffle_trials,
        rng=rng,
        perm=perm,
    )
    return flat.T.reshape(n, T, n_trials).transpose(2, 0, 1)


def _ridge_inv_sqrt(C: np.ndarray, lam: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(C + lam * np.eye(C.shape[0]))
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def ridge_cca(
        X: np.ndarray,
        Y: np.ndarray,
        lam: float = LAMBDA_CCA,
        n_components: int = N_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if Z_flat is None or Z_flat.ndim < 2 or Z_flat.shape[1] == 0:
        return X_flat.copy()
    n, m  = Z_flat.shape
    ZtZ   = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    Beta  = np.linalg.solve(ZtZ, Z_flat.T)
    return X_flat - Z_flat @ (Beta @ X_flat)


def pcca(
        X_flat: np.ndarray,
        Y_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_cca: float = LAMBDA_CCA,
        lam_hat: float = LAMBDA_HAT,
        n_components: int = N_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_res = residualize(X_flat, Z_flat, lam_hat)
    Y_res = residualize(Y_flat, Z_flat, lam_hat)
    Wx, Wy, rho = ridge_cca(X_res, Y_res, lam_cca, n_components)
    return Wx, Wy, rho, X_res, Y_res


def _compute_nuisance_weight(
        Z_flat: np.ndarray,
        X_flat: np.ndarray,
        w: np.ndarray,
        lam_hat: float = LAMBDA_HAT,
) -> np.ndarray:

    n, m  = Z_flat.shape
    ZtZ   = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    Beta  = np.linalg.solve(ZtZ, Z_flat.T)
    Beta_z = (Beta @ X_flat)
    Beta_A_B =  (Beta @ X_flat) @ w
    test = (Beta @ X_flat) @ w[-50:]
    test2= Beta_A_B[-50:]
    #return (Beta @ X_flat)[-50:,0]
    #return (Beta @ X_flat) @ w[-50:]
    return Beta_A_B[-50:]                 # (m,)


def latent_projections(
        X_flat: np.ndarray,
        w: np.ndarray,
        n_trials: int,
        T: int,
) -> np.ndarray:
    proj = X_flat @ w
    return proj.reshape(T, n_trials).T


def _cos_sim_abs(a: np.ndarray, b: np.ndarray) -> float:
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
    data = mat73.loadmat(session_path)
    rd   = data.get("region_data", {})
    regs = rd.get("regions", {})

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
    """Peak-time / single-region Rastermap fallback (used internally)."""
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
    psth = X.mean(axis=0)
    return np.argsort(np.argmax(psth, axis=1))


def compute_global_neuron_order(
        region_spikes: Dict[str, np.ndarray],
        all_region_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Fit a single Rastermap model on the concatenated activity of all
    specified regions, then extract per-region sort indices from the
    global embedding.

    This guarantees that neuron orderings are derived from a common
    manifold estimate rather than independent per-region fits, so
    panels displaying different regions share a geometrically consistent
    basis for visual comparison.

    Parameters
    ----------
    region_spikes     : mapping from region name → raw spike array (n_trials, n, T)
    all_region_names  : ordered list of region names to include
                        (targets + all nuisance regions present in the session)

    Returns
    -------
    per_region_order  : mapping from region name → sort index array
                        Each array is a permutation of [0, n_neurons_r) derived
                        from the global Rastermap embedding.  If Rastermap fails
                        the fallback is independent peak-time ordering per region.
    """
    # ── Build per-region flat matrices and record neuron offsets ───────────
    flat_rows: List[np.ndarray] = []   # each entry: (n_neurons_r, n_obs)
    offsets:   Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for rname in all_region_names:
        X         = region_spikes[rname]
        n_neurons = X.shape[1]
        flat_r    = _zscore_flat(X).T          # (n_neurons_r, n_obs)
        flat_rows.append(flat_r)
        offsets[rname] = (cursor, cursor + n_neurons)
        cursor += n_neurons

    combined = np.concatenate(flat_rows, axis=0)  # (total_neurons, n_obs)
    total_n  = combined.shape[0]

    # ── Attempt global Rastermap ───────────────────────────────────────────
    global_isort: Optional[np.ndarray] = None
    if _RASTERMAP_OK and total_n >= 5:
        try:
            mdl = Rastermap(
                n_PCs=min(50, total_n),
                locality=0.0,
                time_lag_window=10,
                grid_upsample=10,
            )
            mdl.fit(combined)
            global_isort = mdl.isort          # (total_neurons,)
            print(
                f"  [global Rastermap]  fit on {total_n} neurons "
                f"({len(all_region_names)} regions)"
            )
        except Exception as exc:
            warnings.warn(
                f"Global Rastermap failed ({exc}); "
                "falling back to per-region peak-time ordering."
            )

    # ── Extract per-region indices from global order ───────────────────────
    per_region_order: Dict[str, np.ndarray] = {}
    for rname in all_region_names:
        start, end = offsets[rname]
        if global_isort is not None:
            # Positions in global_isort that belong to this region
            mask        = (global_isort >= start) & (global_isort < end)
            local_order = global_isort[mask] - start   # map to [0, n_neurons_r)
            per_region_order[rname] = local_order
        else:
            per_region_order[rname] = get_neuron_order(region_spikes[rname])

    return per_region_order


# =============================================================================
# 4.  Figure primitives
# =============================================================================

def _draw_psth(
        ax: plt.Axes,
        X_z3d: np.ndarray,
        sort_idx: np.ndarray,
        time_vec: np.ndarray,
        region_name: str,
        n_show: int,
        vmax: Optional[float] = None,
        show_colorbar: bool = False,# ← add this
) -> None:
    n_neurons = X_z3d.shape[1]
    step = max(1, n_neurons // n_show)
    sel  = sort_idx[::step][:n_show]
    psth = X_z3d.mean(axis=0)[sel]
    if vmax is None:                                     # ← replace the
        vmax =float(np.nanpercentile(              #   existing single
                       np.abs(X_z3d.mean(axis=0)), 99))  #   vmax line with

    im = ax.imshow(
        psth, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[time_vec[0], time_vec[-1], len(sel), 0],
        origin="upper",
    )

    if show_colorbar:
        # ax.figure.colorbar 可以自适应当前所在的 ax 子图
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=6)  # 调整 colorbar 的刻度字体大小以适应你的整体风格

    ax.axvline(0.0, color="k", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Neurons (sorted)", fontsize=7)
    ax.set_title(f"{region_name} PSTH", fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)


def _draw_weight_bar(
        ax: plt.Axes,
        weight: np.ndarray,
        sort_idx: np.ndarray,
        n_show: int,
        title: str,
        pos_color: str = _C_POS,  # default: original red constant
        neg_color: str = _C_NEG,  # default: original blue constant
) -> None:
    """Horizontal signed weight-bar panel.

    Parameters
    ----------
    weight    : (n_neurons,) canonical weight or β·w₁ vector.
    sort_idx  : global Rastermap sort order for this region.
    pos_color : bar colour for positive values  (default: pCCA dark-red).
    neg_color : bar colour for negative values  (default: pCCA dark-blue).

    Passing different palettes for CCA vs pCCA bars keeps the two
    estimators visually distinct while maintaining the same panel geometry.
    """
    n_neurons = len(weight)
    step = max(1, n_neurons // n_show)
    sel = sort_idx[::step][:n_show]
    w = weight[sel]
    ypos = np.arange(len(sel)) + 0.5
    colors = [pos_color if v >= 0 else neg_color for v in w]

    ax.barh(ypos, w, height=0.82, color=colors, alpha=0.83)
    ax.axvline(0.0, color='k', lw=0.7, alpha=0.45)
    ax.set_ylim(len(sel), 0)
    ax.set_title(title, fontsize=7, fontweight='bold')
    ax.tick_params(labelsize=5)
    plt.setp(ax.get_yticklabels(), visible=False)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)


def _draw_latent(
        ax: plt.Axes,
        trials: np.ndarray,
        time_vec: np.ndarray,
        color: str,
        title: str,
        alpha_trial: float = 0.06,
        lw_trial: float = 0.28,
) -> None:
    """Per-trial latent traces (thin, rasterised) + mean ± SEM overlay.

    Matches the aesthetics of ``_plot_sim_latent_trials`` in
    ``PCCA_CCA_population_level.py``:

      - All trials drawn (not strided) at ``lw_trial`` / ``alpha_trial``.
      - Mean: ``lw = 2.2``, SEM fill: ``alpha = 0.25``.
      - Y-axis fixed to [-3, 3] so panels are directly comparable across
        ablation steps.
      - Onset dashed line, fontsize 9, bold title (mirrors simulation).

    Note
    ----
    This function expects sign-corrected ``trials`` (output of
    ``apply_latent_sign_correction``).  Passing uncorrected data may
    produce downward-deflecting traces.

    Parameters
    ----------
    trials      : (n_trials, T) — sign-corrected latent projections.
    time_vec    : (T,)          — time axis in seconds.
    color       : trace / fill colour (consistent with the weight-bar panel).
    title       : axis title string (typically includes ρ₁).
    """
    n_trials = trials.shape[0]

    # Individual trial traces (rasterised for PDF efficiency)
    for tr in trials:
        ax.plot(
            time_vec, tr, color=color, lw=lw_trial,
            alpha=alpha_trial, rasterized=True,
        )

    # Mean ± SEM overlay
    mean = np.nanmean(trials, axis=0)
    sem = np.nanstd(trials, axis=0) / np.sqrt(max(n_trials, 1))
    ax.plot(time_vec, mean, color=color, lw=2.2, zorder=3)
    ax.fill_between(
        time_vec, mean - sem, mean + sem,
        color=color, alpha=0.25, zorder=2,
    )

    # Onset marker
    ax.axvline(0.0, color='k', ls='--', lw=0.9, alpha=0.45)

    ax.set_xlabel('Time (s)', fontsize=9)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylim(-3, 4)
    ax.tick_params(labelsize=7)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

# =============================================================================
# 5.  Main 2-row × 7-column panel
# =============================================================================

def plot_step_panel(
        *,
        X_i_z3d: np.ndarray,
        X_j_z3d: np.ndarray,
        X_i_flat: np.ndarray,
        X_j_flat: np.ndarray,
        X_i_res: np.ndarray,
        X_j_res: np.ndarray,
        sort_i: np.ndarray,
        sort_j: np.ndarray,
        Wx_pcca: np.ndarray,
        Wy_pcca: np.ndarray,
        rho_pcca: np.ndarray,
        Wx_cca: np.ndarray,
        Wy_cca: np.ndarray,
        rho_cca: np.ndarray,
        time_vec: np.ndarray,
        n_trials: int,
        T: int,
        fig_title: str,
        n_show: int = N_NEURONS_SHOW,
        Z_z3d: Optional[np.ndarray] = None,
        sort_z: Optional[np.ndarray] = None,
        Z_name: Optional[str] = None,
        nu_w_i: Optional[np.ndarray] = None,
        nu_w_j: Optional[np.ndarray] = None,
        psth_vmax: Optional[float] = None,
) -> plt.Figure:
    """2-row × 8-column diagnostic panel for one pCCA ablation step.

    Column layout (mirrors ``fig_simulation_psth_latents``, 0-indexed):

        col 0  Raw PSTH              z-scored RdBu_r (Rastermap order)
        col 1  CCA canonical weight  barh; amber/purple
        col 2  CCA latent z(t)       trials + mean ± SEM; fixed y ∈ [-3, 3]
        col 3  Nuisance PSTH         z-scored RdBu_r
        col 4  Nuisance → target β·w₁  barh; green/grey
        col 5  pCCA canonical weight barh; C3/C0
        col 6  Residual PSTH         z-scored residual after nuisance regression
        col 7  pCCA latent z(t)      trials + mean ± SEM; fixed y ∈ [-3, 3]

    Row 0: TARGET_I (MOs)  |  Row 1: TARGET_J (VPMPO)

    Sign correction (applied before any drawing)
    --------------------------------------------
    ``apply_latent_sign_correction`` enforces Steps 4–5 of the Z2
    synchronisation algorithm on each canonical-variate pair independently.
    Resulting flips are propagated to the weight vectors so that cols 1/5
    are spatially consistent with cols 2/7 respectively:

        z = X_res @ w   ⟹   flip(z) ≡ X_res @ (−w)

    Residual PSTH reconstruction
    ----------------------------
    ``X_i_res`` arrives as a flat matrix of shape ``(T * n_trials, n)``,
    with row index = ``t * n_trials + trial``.  Reconstruction to
    ``(n_trials, n, T)`` follows:

        X_i_res.reshape(T, n_trials, n).transpose(1, 2, 0)

    The result is z-scored via ``_zscore_3d`` and displayed with an
    independent per-panel vmax (``vmax=None`` in ``_draw_psth``) so that
    the reduced post-regression amplitude is visible rather than compressed.

    Parameters
    ----------
    X_i_flat / X_j_flat  : (T·n_trials, n)  z-scored flat matrices.
    X_i_res  / X_j_res   : (T·n_trials, n)  nuisance-residualised flat matrices.
    X_i_z3d / X_j_z3d   : (n_trials, n, T)  z-scored 3-D spike tensors.
    Wx_pcca / Wy_pcca    : (n, K)  pCCA canonical weight matrices (K components).
    Wx_cca  / Wy_cca     : (n, K)  CCA  canonical weight matrices.
    rho_pcca / rho_cca   : (K,)  canonical correlations.
    Z_z3d                : (n_trials, n_Z, T) nuisance 3-D tensor, or None.
    sort_z               : global Rastermap sort index for the nuisance region.
    Z_name               : display name of the nuisance region (or None).
    nu_w_i / nu_w_j      : (n_Z,)  nuisance regression projection β·w₁, or None.
    psth_vmax            : shared colour scale for raw and nuisance PSTH panels.
                           The residual PSTH always uses its own per-panel scale.
    """

    # ── Residual PSTH: reconstruct 3-D from flat ─────────────────────────────
    # Flat layout: row = t * n_trials + trial  (time is outer loop).
    # Inverse: reshape(T, n_trials, n) → transpose(1, 2, 0) → (n_trials, n, T)
    n_i = X_i_res.shape[1]
    n_j = X_j_res.shape[1]
    X_i_res_z3d = _zscore_3d(
        X_i_res.reshape(T, n_trials, n_i).transpose(1, 2, 0)
    )  # (n_trials, n_i, T)
    X_j_res_z3d = _zscore_3d(
        X_j_res.reshape(T, n_trials, n_j).transpose(1, 2, 0)
    )  # (n_trials, n_j, T)

    # ── Pre-compute latent projections ────────────────────────────────────────
    # pCCA: project residualised data;  CCA: project raw z-scored data.
    z_i_p = latent_projections(X_i_res, Wx_pcca[:, 0], n_trials, T)
    z_j_p = latent_projections(X_j_res, Wy_pcca[:, 0], n_trials, T)
    z_i_c = latent_projections(X_i_flat, Wx_cca[:, 0], n_trials, T)
    z_j_c = latent_projections(X_j_flat, Wy_cca[:, 0], n_trials, T)

    # ── Sign correction (Steps 4–5 of Z2 synchronisation) ────────────────────
    z_i_p, z_j_p, flip_ip, flip_jp = apply_latent_sign_correction(
        z_i_p, z_j_p, time_vec
    )
    z_i_c, z_j_c, flip_ic, flip_jc = apply_latent_sign_correction(
        z_i_c, z_j_c, time_vec
    )

    # ── Propagate flips to canonical weight vectors ───────────────────────────
    # z = X_res @ w  ⟹  flip(z) ≡ X_res @ (−w)
    w_pcca_i = Wx_pcca[:, 0] * (-1.0 if flip_ip else 1.0)
    w_pcca_j = Wy_pcca[:, 0] * (-1.0 if flip_jp else 1.0)
    w_cca_i = Wx_cca[:, 0] * (-1.0 if flip_ic else 1.0)
    w_cca_j = Wy_cca[:, 0] * (-1.0 if flip_jc else 1.0)

    # ── Figure scaffold (8 columns) ───────────────────────────────────────────
    # Width ratios mirror the simulation figure:
    #   PSTH / residual PSTH  →  3.5
    #   weight bars           →  0.9
    #   latent z(t)           →  2.8
    fig, axes = plt.subplots(
        2, 8,
        figsize=(33.0, 8.0),
        gridspec_kw={
            'width_ratios': [4.0, 0.9, 2.8, 3.5, 0.9, 3.5, 0.9, 2.8],
            'hspace': 0.52,
            'wspace': 0.26,
        },
    )

    rho0_p = float(rho_pcca[0]) if len(rho_pcca) > 0 else float('nan')
    rho0_c = float(rho_cca[0]) if len(rho_cca) > 0 else float('nan')
    z_label = Z_name if Z_name is not None else 'Z'

    # ── Per-row configuration ─────────────────────────────────────────────────
    row_cfg = [
        dict(
            name=TARGET_I,
            X_z3d=X_i_z3d,
            X_res_z3d=X_i_res_z3d,  # residual PSTH (col 6)
            sort_idx=sort_i,
            w_pcca=w_pcca_i,
            w_cca=w_cca_i,
            z_p=z_i_p,
            z_c=z_i_c,
            c_pcca=_CI_PCCA,  # dark red
            c_cca=_CI_CCA,  # warm orange
            nu_w=nu_w_i,
        ),
        dict(
            name=TARGET_J,
            X_z3d=X_j_z3d,
            X_res_z3d=X_j_res_z3d,
            sort_idx=sort_j,
            w_pcca=w_pcca_j,
            w_cca=w_cca_j,
            z_p=z_j_p,
            z_c=z_j_c,
            c_pcca=_CJ_PCCA,  # dark blue
            c_cca=_CJ_CCA,  # sky blue
            nu_w=nu_w_j,
        ),
    ]

    for row, rd in enumerate(row_cfg):

        # ── Col 0  Raw PSTH ───────────────────────────────────────────────────
        # Shared colour scale (psth_vmax) so amplitude is comparable
        # across all ablation steps.
        _draw_psth(
            axes[row, 0], rd['X_z3d'], rd['sort_idx'],
            time_vec, rd['name'], n_show, vmax=psth_vmax,show_colorbar=True
        )

        # ── Col 1  CCA canonical weight (amber / purple) ──────────────────────
        _draw_weight_bar(
            axes[row, 1], rd['w_cca'], rd['sort_idx'], n_show,
            f"CCA weight\nρ₁ = {rho0_c:.3f}",
            pos_color=_C_CCA_POS,
            neg_color=_C_CCA_NEG,
        )

        # ── Col 2  CCA latent z(t) ────────────────────────────────────────────
        _draw_latent(
            axes[row, 2], rd['z_c'], time_vec,
            rd['c_cca'],
            f"CCA  z(t)   ρ₁={rho0_c:.3f}",
        )
        if row == 0:
            axes[row, 2].set_ylabel('Latent projection', fontsize=9)

        # ── Col 3  Nuisance PSTH ──────────────────────────────────────────────
        if Z_z3d is not None and sort_z is not None and Z_name is not None:
            _draw_psth(
                axes[row, 3], Z_z3d, sort_z,
                time_vec, f'Nuisance: {z_label}', n_show, vmax=psth_vmax,
            )
        else:
            axes[row, 3].text(
                0.5, 0.5, 'No nuisance\nregion',
                ha='center', va='center', fontsize=9, color='gray',
                transform=axes[row, 3].transAxes,
            )
            axes[row, 3].axis('off')

        # ── Col 4  Nuisance → target β·w₁ ────────────────────────────────────
        if rd['nu_w'] is not None and sort_z is not None:
            _draw_weight_bar(
                axes[row, 4], rd['nu_w'], sort_z, n_show,
                f"{z_label}→{rd['name']}\nβ·w₁",
                pos_color=_C_BETA_POS,  # green
                neg_color=_C_BETA_NEG,  # grey
            )
        else:
            axes[row, 4].text(
                0.5, 0.5, '—',
                ha='center', va='center', fontsize=12, color='#AAAAAA',
                transform=axes[row, 4].transAxes,
            )
            axes[row, 4].set_title(
                f"{z_label}→{rd['name']}\nβ·w₁",
                fontsize=7, color='#AAAAAA',
            )
            axes[row, 4].axis('off')
        # ── Col 6  Residual PSTH ──────────────────────────────────────────────
        # Activity remaining after nuisance regression.  The same sort_idx as
        # col 0 is applied so that neuron ordering is consistent, making the
        # weight-bar panel (col 5) directly readable against the residual.
        # vmax=None: independent per-panel scale because the residual amplitude
        # is generically smaller than the raw PSTH after regression.

        region_short = rd['name'].split()[0]  # 'MOs' or 'VPMPO'
        _draw_psth(
            axes[row, 5], rd['X_res_z3d'], rd['sort_idx'],
            time_vec, f'{region_short} resid | {z_label}', n_show,
            vmax=None,  # auto-scale per panel
        )


        # ── Col 5  pCCA canonical weight (C3/C0 red-blue) ────────────────────
        _draw_weight_bar(
            axes[row, 6], rd['w_pcca'], rd['sort_idx'], n_show,
            f"pCCA weight\nρ₁ = {rho0_p:.3f}",
            pos_color=_C_POS,
            neg_color=_C_NEG,
        )


        # ── Col 7  pCCA latent z(t) ───────────────────────────────────────────
        _draw_latent(
            axes[row, 7], rd['z_p'], time_vec,
            rd['c_pcca'],
            f"pCCA  z(t)   ρ₁={rho0_p:.3f}",
        )
        if row == 0:
            axes[row, 7].set_ylabel('Latent projection', fontsize=9)

    fig.suptitle(fig_title, fontsize=11, fontweight='bold', y=1.02)
    return fig
# =============================================================================
# 6.  Step-result container and summary statistics
# =============================================================================

def _try_save(
        anim_obj: "_manim.FuncAnimation",
        output_path: Path,
        fps: float,
        dpi: int,
        fig_title: str,
) -> None:
    """Save animation; fall back from MP4 to GIF on FFMpeg failure."""
    ext = output_path.suffix.lower()

    def _save_mp4(path: Path) -> None:
        writer = _manim.FFMpegWriter(
            fps=fps, bitrate=2400,
            metadata={"title": fig_title, "artist": "pCCA ablation"},
        )
        anim_obj.save(str(path), writer=writer, dpi=dpi)
        print(f"  [animate] Saved MP4  ({path.stat().st_size / 1e6:.1f} MB): {path}")

    def _save_gif(path: Path) -> None:
        writer = _manim.PillowWriter(fps=fps)
        anim_obj.save(str(path.with_suffix('.gif')), writer=writer, dpi=dpi)
        print(f"  [animate] Saved GIF: {path.with_suffix('.gif')}")

    if ext == '.gif':
        _save_gif(output_path)
        return

    # Default: try mp4
    try:
        _save_mp4(output_path if ext == '.mp4' else output_path.with_suffix('.mp4'))
    except Exception as exc:
        warnings.warn(
            f"[animate] FFMpeg writer failed ({exc}); retrying as GIF."
        )
        _save_gif(output_path)


# =============================================================================
# Public API
# =============================================================================

def animate_step_panel(
        *,
        # ── identical to plot_step_panel ──────────────────────────────────
        X_i_z3d: np.ndarray,
        X_j_z3d: np.ndarray,
        X_i_flat: np.ndarray,
        X_j_flat: np.ndarray,
        X_i_res: np.ndarray,
        X_j_res: np.ndarray,
        sort_i: np.ndarray,
        sort_j: np.ndarray,
        Wx_pcca: np.ndarray,
        Wy_pcca: np.ndarray,
        rho_pcca: np.ndarray,
        Wx_cca: np.ndarray,
        Wy_cca: np.ndarray,
        rho_cca: np.ndarray,
        time_vec: np.ndarray,
        n_trials: int,
        T: int,
        fig_title: str,
        n_show: int = N_NEURONS_SHOW,
        Z_z3d: Optional[np.ndarray] = None,
        sort_z: Optional[np.ndarray] = None,
        Z_name: Optional[str] = None,
        nu_w_i: Optional[np.ndarray] = None,
        nu_w_j: Optional[np.ndarray] = None,
        psth_vmax: Optional[float] = None,
        # ── animation-specific ────────────────────────────────────────────
        window: int = 5,
        fps: float = 4.0,
        dpi: int = 120,
        output_path: Optional[Path] = None,
) -> None:
    """Animated 2-row × 8-column ablation panel with a rolling trial window.

    Produces a video (or GIF) in which the PSTH and latent projection panels
    evolve trial-by-trial while the three canonical-weight bar panels remain
    static.

    Column layout (mirrors ``plot_step_panel``)
    -------------------------------------------
    col 0  Raw PSTH              **animated**  rolling mean over ``window`` trials
    col 1  CCA weight            *static*
    col 2  CCA latent z(t)       **animated**  ``window`` trial traces + mean ± SEM
    col 3  Nuisance PSTH         **animated**
    col 4  Nuisance β·w₁         *static*
    col 5  pCCA weight           *static*
    col 6  Residual PSTH         **animated**
    col 7  pCCA latent z(t)      **animated**

    Each frame k (0-indexed) displays trials ``[k, k + window)``.
    Total number of frames = ``n_trials − window + 1``.

    PSTH colour limits are fixed globally (pre-computed from the full-trial
    mean of each region's z-scored data) so the colourmap does not flicker
    as the trial window advances.

    Parameters
    ----------
    (All parameters shared with ``plot_step_panel`` retain the same
    semantics.  New parameters:)

    window      : int, default 5
        Number of consecutive trials shown per frame.
    fps         : float, default 4.0
        Frames per second.  Values between 2–8 are typically comfortable.
    dpi         : int, default 120
        Resolution of the output video.
    output_path : Path or str, optional
        Destination file.  Extension determines format:
            ``.mp4`` → FFMpeg (default; falls back to GIF if unavailable)
            ``.gif`` → Pillow
        If None, saves to ``<fig_title_sanitised>_animation.mp4`` in the
        current directory.
    """
    if window < 1 or window > n_trials:
        raise ValueError(
            f"window must satisfy 1 ≤ window ≤ n_trials={n_trials}; "
            f"got window={window}."
        )
    n_frames = n_trials - window + 1
    print(
        f"  [animate] {n_frames} frames  "
        f"(n_trials={n_trials}, window={window}, fps={fps})"
    )

    # ── Preamble — identical to plot_step_panel ───────────────────────────────
    n_i = X_i_res.shape[1]
    n_j = X_j_res.shape[1]

    # Residual 3-D reconstruction: flat row = t * n_trials + trial
    X_i_res_z3d = _zscore_3d(
        X_i_res.reshape(T, n_trials, n_i).transpose(1, 2, 0)
    )  # (n_trials, n_i, T)
    X_j_res_z3d = _zscore_3d(
        X_j_res.reshape(T, n_trials, n_j).transpose(1, 2, 0)
    )  # (n_trials, n_j, T)

    z_i_p = latent_projections(X_i_res, Wx_pcca[:, 0], n_trials, T)
    z_j_p = latent_projections(X_j_res, Wy_pcca[:, 0], n_trials, T)
    z_i_c = latent_projections(X_i_flat, Wx_cca[:, 0], n_trials, T)
    z_j_c = latent_projections(X_j_flat, Wy_cca[:, 0], n_trials, T)

    z_i_p, z_j_p, flip_ip, flip_jp = apply_latent_sign_correction(
        z_i_p, z_j_p, time_vec
    )
    z_i_c, z_j_c, flip_ic, flip_jc = apply_latent_sign_correction(
        z_i_c, z_j_c, time_vec
    )

    w_pcca_i = Wx_pcca[:, 0] * (-1.0 if flip_ip else 1.0)
    w_pcca_j = Wy_pcca[:, 0] * (-1.0 if flip_jp else 1.0)
    w_cca_i = Wx_cca[:, 0] * (-1.0 if flip_ic else 1.0)
    w_cca_j = Wy_cca[:, 0] * (-1.0 if flip_jc else 1.0)

    rho0_p = float(rho_pcca[0]) if len(rho_pcca) > 0 else float('nan')
    rho0_c = float(rho_cca[0]) if len(rho_cca) > 0 else float('nan')
    z_label = Z_name if Z_name is not None else 'Z'

    # ── Pre-compute neuron selection indices ──────────────────────────────────
    # Matches the subsampling used by _draw_psth, applied once and reused
    # every frame to avoid redundant per-frame index arithmetic.
    def _sel(sort_idx: np.ndarray, n_neurons: int) -> np.ndarray:
        step = max(1, n_neurons // n_show)
        return sort_idx[::step][:n_show]

    sel_i = _sel(sort_i, X_i_z3d.shape[1])  # raw region I
    sel_j = _sel(sort_j, X_j_z3d.shape[1])  # raw region J
    sel_ri = _sel(sort_i, n_i)  # residual region I  (same order)
    sel_rj = _sel(sort_j, n_j)  # residual region J
    has_nuis = (Z_z3d is not None and sort_z is not None and Z_name is not None)
    sel_z = _sel(sort_z, Z_z3d.shape[1]) if has_nuis else None

    # ── Fixed colour limits (pre-computed from full-trial means) ─────────────
    # Using the full-trial mean gives the same vmax as the static figure;
    # fixing it across frames prevents colourmap flickering.
    if psth_vmax is None:
        psth_vmax = max(
            float(np.nanpercentile(np.abs(X_i_z3d.mean(0)), 99)), 0.5
        )

    # Residual PSTH: independent per-region limits (amplitude is smaller
    # than the raw PSTH after nuisance regression).
    _rvmax_i = max(
        float(np.nanpercentile(np.abs(X_i_res_z3d.mean(0)), 99)), 0.5
    )
    _rvmax_j = max(
        float(np.nanpercentile(np.abs(X_j_res_z3d.mean(0)), 99)), 0.5
    )
    _nvmax = (
        max(float(np.nanpercentile(np.abs(Z_z3d.mean(0)), 99)), 0.5)
        if has_nuis else psth_vmax
    )

    # ── Closure helpers ───────────────────────────────────────────────────────
    # Both helpers capture (time_vec, window, n_show) from the enclosing scope.

    def _windowed_psth(X_z3d_all: np.ndarray,
                       sel: np.ndarray,
                       k: int) -> np.ndarray:
        """Mean PSTH of trials [k, k+window).  Shape: (len(sel), T)."""
        return X_z3d_all[k: k + window].mean(axis=0)[sel]

    def _redraw_latent_ax(
            ax: plt.Axes,
            z_all: np.ndarray,
            k: int,
            color: str,
            base_title: str,
            add_ylabel: bool,
    ) -> None:
        """Clear *ax* and redraw a latent panel for the trial window at k."""
        ax.cla()
        _draw_latent(ax, z_all[k: k + window], time_vec, color, base_title)
        if add_ylabel:
            ax.set_ylabel('Latent projection', fontsize=9)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 8,
        figsize=(33.0, 8.0),
        gridspec_kw={
            'width_ratios': [3.5, 0.9, 2.8, 3.5, 0.9, 0.9, 3.5, 2.8],
            'hspace': 0.52,
            'wspace': 0.26,
        },
    )

    # ── Draw static panels (cols 1, 4, 5) ────────────────────────────────────
    _static_rows = [
        dict(name=TARGET_I, sort_idx=sort_i,
             w_pcca=w_pcca_i, w_cca=w_cca_i, nu_w=nu_w_i),
        dict(name=TARGET_J, sort_idx=sort_j,
             w_pcca=w_pcca_j, w_cca=w_cca_j, nu_w=nu_w_j),
    ]
    for row, rd in enumerate(_static_rows):

        # col 1 — CCA weight (amber / purple)
        _draw_weight_bar(
            axes[row, 1], rd['w_cca'], rd['sort_idx'], n_show,
            f"CCA weight\nρ₁ = {rho0_c:.3f}",
            pos_color=_C_CCA_POS, neg_color=_C_CCA_NEG,
        )

        # col 4 — nuisance β·w₁ (green / grey)
        if rd['nu_w'] is not None and sort_z is not None:
            _draw_weight_bar(
                axes[row, 4], rd['nu_w'], sort_z, n_show,
                f"{z_label}→{rd['name']}\nβ·w₁",
                pos_color=_C_BETA_POS, neg_color=_C_BETA_NEG,
            )
        else:
            axes[row, 4].text(
                0.5, 0.5, '—', ha='center', va='center',
                fontsize=12, color='#AAAAAA',
                transform=axes[row, 4].transAxes,
            )
            axes[row, 4].set_title(
                f"{z_label}→{rd['name']}\nβ·w₁",
                fontsize=7, color='#AAAAAA',
            )
            axes[row, 4].axis('off')

        # col 5 — pCCA weight (C3/C0)
        _draw_weight_bar(
            axes[row, 5], rd['w_pcca'], rd['sort_idx'], n_show,
            f"pCCA weight\nρ₁ = {rho0_p:.3f}",
            pos_color=_C_POS, neg_color=_C_NEG,
        )

    # ── Initialise animated PSTH panels (col 0, 3, 6) ────────────────────────
    # We create AxesImage objects directly (bypassing _draw_psth which returns
    # None) so we can call im.set_data() efficiently inside the update loop.

    def _init_psth_ax(
            ax: plt.Axes,
            X_z3d_all: np.ndarray,
            sel: np.ndarray,
            vmax: float,
            label: str,
            trial_lo: int = 1,
            trial_hi: int = None,
    ) -> plt.matplotlib.image.AxesImage:
        """Draw imshow for frame 0 and return the AxesImage for later updates."""
        if trial_hi is None:
            trial_hi = window
        psth0 = _windowed_psth(X_z3d_all, sel, 0)
        im = ax.imshow(
            psth0, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[time_vec[0], time_vec[-1], len(sel), 0],
            origin='upper',
        )
        ax.axvline(0.0, color='k', ls='--', lw=1.0, alpha=0.7)
        ax.set_xlabel('Time (s)', fontsize=7)
        ax.set_ylabel('Neurons (sorted)', fontsize=7)
        ax.set_title(
            f"{label}  [trials {trial_lo}–{trial_hi}]",
            fontsize=8, fontweight='bold',
        )
        ax.tick_params(labelsize=6)
        return im

    # col 0 — raw PSTH (rows 0 and 1)
    im_raw_i = _init_psth_ax(axes[0, 0], X_i_z3d, sel_i, psth_vmax, TARGET_I)
    im_raw_j = _init_psth_ax(axes[1, 0], X_j_z3d, sel_j, psth_vmax, TARGET_J)

    # col 3 — nuisance PSTH (same data in both rows)
    if has_nuis:
        im_nuis_0 = _init_psth_ax(
            axes[0, 3], Z_z3d, sel_z, _nvmax, f'Nuisance: {z_label}'
        )
        im_nuis_1 = _init_psth_ax(
            axes[1, 3], Z_z3d, sel_z, _nvmax, f'Nuisance: {z_label}'
        )
    else:
        for r in range(2):
            axes[r, 3].text(
                0.5, 0.5, 'No nuisance\nregion',
                ha='center', va='center', fontsize=9, color='gray',
                transform=axes[r, 3].transAxes,
            )
            axes[r, 3].axis('off')
        im_nuis_0 = im_nuis_1 = None

    # col 6 — residual PSTH (rows 0 and 1; independent per-region vmax)
    im_res_i = _init_psth_ax(
        axes[0, 6], X_i_res_z3d, sel_ri, _rvmax_i,
        f'{TARGET_I} resid | {z_label}',
    )
    im_res_j = _init_psth_ax(
        axes[1, 6], X_j_res_z3d, sel_rj, _rvmax_j,
        f'{TARGET_J} resid | {z_label}',
    )

    # ── Initialise animated latent panels (col 2 and 7) ──────────────────────
    _lat_rows = [
        dict(z_p=z_i_p, z_c=z_i_c, c_pcca=_CI_PCCA, c_cca=_CI_CCA),
        dict(z_p=z_j_p, z_c=z_j_c, c_pcca=_CJ_PCCA, c_cca=_CJ_CCA),
    ]
    for row, rd in enumerate(_lat_rows):
        _redraw_latent_ax(
            axes[row, 2], rd['z_c'], 0, rd['c_cca'],
            f"CCA  z(t)   ρ₁={rho0_c:.3f}",
            add_ylabel=(row == 0),
        )
        _redraw_latent_ax(
            axes[row, 7], rd['z_p'], 0, rd['c_pcca'],
            f"pCCA z(t)   ρ₁={rho0_p:.3f}",
            add_ylabel=(row == 0),
        )

    # ── Figure-level annotations ──────────────────────────────────────────────
    fig.suptitle(fig_title, fontsize=11, fontweight='bold', y=1.02)

    # Rolling-window counter in the top-right corner of the figure.
    # Updated each frame by the update() closure.
    frame_counter = fig.text(
        0.995, 0.995,
        f'Trials  1 – {window}  /  {n_trials}',
        ha='right', va='top',
        fontsize=9, color='#555555',
        transform=fig.transFigure,
    )

    # ── update() — called for every frame of FuncAnimation ───────────────────
    def update(frame: int):
        """Advance the rolling window by one trial and refresh animated panels."""
        k = frame  # 0-indexed first trial
        lo, hi = k + 1, k + window  # 1-indexed for display labels

        # ---- col 0  raw PSTH ------------------------------------------------
        im_raw_i.set_data(_windowed_psth(X_i_z3d, sel_i, k))
        im_raw_j.set_data(_windowed_psth(X_j_z3d, sel_j, k))
        axes[0, 0].set_title(
            f"{TARGET_I}  [trials {lo}–{hi}]", fontsize=8, fontweight='bold'
        )
        axes[1, 0].set_title(
            f"{TARGET_J}  [trials {lo}–{hi}]", fontsize=8, fontweight='bold'
        )

        # ---- col 3  nuisance PSTH -------------------------------------------
        if has_nuis:
            _npsth = _windowed_psth(Z_z3d, sel_z, k)
            im_nuis_0.set_data(_npsth)
            im_nuis_1.set_data(_npsth)
            axes[0, 3].set_title(
                f"Nuisance: {z_label}  [trials {lo}–{hi}]",
                fontsize=8, fontweight='bold',
            )
            axes[1, 3].set_title(
                f"Nuisance: {z_label}  [trials {lo}–{hi}]",
                fontsize=8, fontweight='bold',
            )

        # ---- col 6  residual PSTH -------------------------------------------
        im_res_i.set_data(_windowed_psth(X_i_res_z3d, sel_ri, k))
        im_res_j.set_data(_windowed_psth(X_j_res_z3d, sel_rj, k))
        axes[0, 6].set_title(
            f"{TARGET_I} resid | {z_label}  [trials {lo}–{hi}]",
            fontsize=8, fontweight='bold',
        )
        axes[1, 6].set_title(
            f"{TARGET_J} resid | {z_label}  [trials {lo}–{hi}]",
            fontsize=8, fontweight='bold',
        )

        # ---- col 2  CCA latent z(t)  &  col 7  pCCA latent z(t) ------------
        # ax.cla() + _draw_latent() is used because fill_between and multiple
        # per-trial LineCollection objects are easier to replace than to mutate
        # in-place.  With blit=False this is safe and straightforward.
        for row, rd in enumerate(_lat_rows):
            _redraw_latent_ax(
                axes[row, 2], rd['z_c'], k, rd['c_cca'],
                f"CCA  z(t)   trials {lo}–{hi}",
                add_ylabel=(row == 0),
            )
            _redraw_latent_ax(
                axes[row, 7], rd['z_p'], k, rd['c_pcca'],
                f"pCCA z(t)   trials {lo}–{hi}",
                add_ylabel=(row == 0),
            )

        # ---- frame counter --------------------------------------------------
        frame_counter.set_text(
            f'Trials  {lo} – {hi}  /  {n_trials}'
        )

        # blit=False: returning [] is fine; matplotlib redraws the whole canvas.
        return []

    #\Sigma^{\rm stim} = \frac{1}{\tau}\sum_s \mu_A \mu_B ── Assemble and save animation ───────────────────────────────────────────
    anim_obj = _manim.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=int(1000.0 / fps),  # milliseconds between frames
        blit=False,
        repeat=False,
    )

    if output_path is None:
        safe_title = (
            fig_title.split('\n')[0]
            .replace(' ', '_')
            .replace('/', '-')[:60]
        )
        output_path = Path(f"{safe_title}_animation.mp4")

    _try_save(anim_obj, Path(output_path), fps, dpi, fig_title)
    plt.close(fig)
# =============================================================================
# Public API  (continued)
# =============================================================================

def compile_steps_to_video(
        fig_paths: List[Path],
        output_path: Path,
        fps: float = 0.8,
        dpi: int = 150,
        title: str = "",
) -> None:
    """Compile saved per-step PNG figures into a single sequential video.

    Each PNG produced by ``run_sequential_removal`` (one per nuisance
    accumulation step) becomes one frame, so the video progresses from
    ``step00_baseline`` → ``step01_+r₁`` → … → ``step_M_+rₘ``.

    This is strictly a post-hoc compositing step: no data are re-read and
    no pCCA computations are repeated.  The function reads each PNG from
    disk with ``matplotlib.image.imread``, renders it into a single Axes
    via ``imshow``, and saves the resulting ``ArtistAnimation`` through the
    shared ``_try_save`` helper (FFMpeg → GIF fallback).

    Parameters
    ----------
    fig_paths   : ordered list of PNG paths, one per ablation step.
                  Must be non-empty; the first image determines the figure
                  aspect ratio.
    output_path : destination file; extension selects the writer
                  (.mp4 → FFMpeg, .gif → Pillow).
    fps         : frames per second.  Values between 0.5–2.0 are
                  comfortable for step-by-step inspection; default 0.8
                  gives ≈ 1.25 s per step.
    dpi         : output resolution.  150 is adequate for screen review;
                  increase to 200 for publication-quality exports.
    title       : figure suptitle (displayed in the top margin of every
                  frame; does not resize the image canvas).
    """
    from matplotlib.image import imread as _imread

    if not fig_paths:
        warnings.warn("[compile_steps_to_video] fig_paths is empty — nothing to compile.")
        return

    # ── Load all PNGs once; keep them in memory for the duration ─────────────
    images = [_imread(str(p)) for p in fig_paths]
    n_frames = len(images)

    # ── Infer figure size from the first image (preserve pixel aspect ratio) ──
    h_px, w_px = images[0].shape[:2]
    fig_w = min(w_px / dpi, 26.0)          # cap at 26 in so matplotlib is happy
    fig_h = fig_w * h_px / w_px

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.96)
    ax.axis('off')

    im = ax.imshow(images[0], interpolation='antialiased')

    # # ── Step label overlay (bottom-left corner of the Axes) ──────────────────
    # step_label = ax.text(
    #     0.005, 0.010,
    #     fig_paths[0].stem,
    #     transform=ax.transAxes,
    #     fontsize=7, va='bottom', ha='left',
    #     color='white',
    #     bbox=dict(facecolor='#111111', alpha=0.55, pad=2, linewidth=0),
    #     zorder=10,
    # )

    # # ── Frame counter (top-right corner of the figure) ────────────────────────
    # frame_counter = fig.text(
    #     0.995, 0.995,
    #     f'Step  1 / {n_frames}',
    #     ha='right', va='top',
    #     fontsize=8, color='#444444',
    #     transform=fig.transFigure,
    # )

    # if title:
    #     fig.suptitle(title, fontsize=8, fontweight='bold', y=1.00,
    #                  color='#222222')

    def update(k: int):
        im.set_data(images[k])
        #step_label.set_text(fig_paths[k].stem)
        #frame_counter.set_text(f'Step  {k + 1} / {n_frames}')
        return [im]

    anim = _manim.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=int(200.0 / fps),
        blit=True,
        repeat=False,
    )

    _try_save(anim, output_path, fps, dpi, title)
    plt.close(fig)
    print(f"  [compile_steps_to_video]  {n_frames} frames → {output_path}")



class StepResult:
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
    out_part1 = output_dir / "part1_sequential"
    out_part1.mkdir(parents=True, exist_ok=True)

    for t in (TARGET_I, TARGET_J):
        if t not in region_spikes:
            raise RuntimeError(
                f"Target region '{t}' not found in session.  "
                f"Available: {sorted(region_spikes)}"
            )

    # ── Nuisance queue (ordered, skipping targets) ────────────────────────
    nuisance_queue: List[str] = [
        r for r in ANATOMICAL_ORDER
        if r in region_spikes and r not in (TARGET_I, TARGET_J)
    ]
    print(
        f"  [Part 1] Nuisance queue ({len(nuisance_queue)} regions): "
        f"{nuisance_queue}"
    )

    # ── Global Rastermap: fit on targets + all nuisance regions ──────────
    all_regions_ordered = [TARGET_I, TARGET_J] + nuisance_queue
    global_sort = compute_global_neuron_order(region_spikes, all_regions_ordered)

    sort_i = global_sort[TARGET_I]
    sort_j = global_sort[TARGET_J]

    # ── Pre-compute flat and 3-D arrays ───────────────────────────────────
    X_i_raw  = region_spikes[TARGET_I]
    X_j_raw  = region_spikes[TARGET_J]

    X_i_flat = _zscore_flat(X_i_raw, subtract_psth=SUBTRACT_PSTH,
                            shuffle_trials=SHUFFLE_TRIALS, rng=_rng)
    X_j_flat = _zscore_flat(X_j_raw, subtract_psth=SUBTRACT_PSTH,rng=_rng2)
    # partner: never shuffled

    X_i_z3d = _zscore_3d(X_i_raw, subtract_psth=SUBTRACT_PSTH,
                         shuffle_trials=SHUFFLE_TRIALS, rng=_rng)
    X_j_z3d = _zscore_3d(X_j_raw, subtract_psth=SUBTRACT_PSTH,rng=_rng2)

    nuisance_flat: Dict[str, np.ndarray] = {
        r: _zscore_flat(region_spikes[r],subtract_psth=SUBTRACT_PSTH) for r in nuisance_queue
    }
    nuisance_z3d: Dict[str, np.ndarray] = {
        r: _zscore_3d(region_spikes[r],subtract_psth=SUBTRACT_PSTH) for r in nuisance_queue
    }
    # ── Global vmax: 99th percentile of |z-score| pooled across all regions ──
    _all_psth = np.concatenate(
        [X_i_z3d.mean(axis=0),
         X_j_z3d.mean(axis=0)]
        + [nuisance_z3d[r].mean(axis=0) for r in nuisance_queue],
        axis=0,  # stack along neuron axis
    )
    global_vmax = float(np.nanpercentile(np.abs(_all_psth), 99))

    # ── CCA baseline ─────────────────────────────────────────────────────
    Wx_cca, Wy_cca, rho_cca = ridge_cca(X_i_flat, X_j_flat)
    print(f"  [Part 1] CCA baseline  ρ₁ = {rho_cca[0]:.4f}")

    time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)

    step_results: List[StepResult] = []
    step_fig_paths: List[Path] = []
    accumulated_nuisance: List[str] = []

    steps_iter = [None] + nuisance_queue

    for step_idx, add_region in enumerate(steps_iter):
        if add_region is not None:
            accumulated_nuisance.append(add_region)

        # ── Build Z ──────────────────────────────────────────────────────
        if accumulated_nuisance:
            Z_flat = np.concatenate(
                [nuisance_flat[r] for r in accumulated_nuisance], axis=1
            )
        else:
            Z_flat = None

        Wx_p, Wy_p, rho_p, X_i_res, X_j_res = pcca(
            X_i_flat, X_j_flat, Z_flat,
        )

        # ── Nuisance weight vectors (col 6) ──────────────────────────────
        # Extract the slice of Beta that corresponds to the most recently
        # added (and displayed) nuisance region, so the weight vector has
        # length n_neurons(add_region) and aligns with sort_z.
        if add_region is not None and Z_flat is not None:
            offset_end = sum(
                nuisance_flat[r].shape[1] for r in accumulated_nuisance
            )
            offset_start = offset_end - nuisance_flat[add_region].shape[1]
            Z_last  = nuisance_flat[add_region]             # single-region Z
            # Re-use the same ridge formula on Z_last against each target.
            # (We deliberately solve for the last-added region alone to keep
            #  the weight-bar length = n_neurons(add_region), matching sort_z.)

            # nu_w_i  = _compute_nuisance_weight(Z_last, X_i_flat, Wx_p[:, 0])
            # nu_w_j  = _compute_nuisance_weight(Z_last, X_j_flat, Wy_p[:, 0])

            nu_w_i  = _compute_nuisance_weight(Z_flat, X_i_flat, Wx_p[:, 0])
            nu_w_j  = _compute_nuisance_weight(Z_flat, X_j_flat, Wy_p[:, 0])

            sort_z  = global_sort[add_region]
            Z_z3d   = nuisance_z3d[add_region]
        else:
            nu_w_i = nu_w_j = None
            sort_z  = None
            Z_z3d   = None

        z_i_mean = latent_projections(X_i_res, Wx_p[:, 0], n_trials, T).mean(axis=0)
        z_j_mean = latent_projections(X_j_res, Wy_p[:, 0], n_trials, T).mean(axis=0)

        label = (
            f"step{step_idx:02d}_+{add_region}"
            if accumulated_nuisance
            else "step00_CCA_baseline"
        )
        step_results.append(StepResult(
            label=label,
            nuisance_regions=list(accumulated_nuisance),
            rho_pcca=float(rho_p[0]),
            Wx=Wx_p,
            Wy=Wy_p,
            z_i_mean=z_i_mean,
            z_j_mean=z_j_mean,
        ))
        _z_ip_raw = latent_projections(X_i_res, Wx_p[:, 0], n_trials, T)
        _z_jp_raw = latent_projections(X_j_res, Wy_p[:, 0], n_trials, T)
        _z_ip_sc, _z_jp_sc, _, _ = apply_latent_sign_correction(
            _z_ip_raw, _z_jp_raw, time_vec
        )

        if not accumulated_nuisance:
            nuis_str = "Z = ∅  (plain CCA)"
        else:
            nuis_str = "Z = {" + ", ".join(accumulated_nuisance) + "}"

        fig_title = (
            f"{session_name}| {TARGET_I} ↔ {TARGET_J}  |  {nuis_str}\n"
            f"pCCA ρ₁ = {rho_p[0]:.4f}   |   CCA ρ₁ = {rho_cca[0]:.4f}"
        )

        fig = plot_step_panel(
            X_i_z3d=X_i_z3d,
            X_j_z3d=X_j_z3d,
            X_i_flat=X_i_flat,
            X_j_flat=X_j_flat,
            X_i_res=X_i_res,
            X_j_res=X_j_res,
            sort_i=sort_i,
            sort_j=sort_j,
            Wx_pcca=Wx_p,
            Wy_pcca=Wy_p,
            rho_pcca=rho_p,
            Wx_cca=Wx_cca,
            Wy_cca=Wy_cca,
            rho_cca=rho_cca,
            time_vec=time_vec,
            n_trials=n_trials,
            T=T,
            fig_title=fig_title,
            n_show=n_show,
            Z_z3d=Z_z3d,
            sort_z=sort_z,
            Z_name=add_region,
            nu_w_i=nu_w_i,
            nu_w_j=nu_w_j,
            psth_vmax=global_vmax,
        )

        fig_path = out_part1 / (
            f"{session_name}_part1_step{step_idx:02d}"
            f"_{'_'.join(accumulated_nuisance) if accumulated_nuisance else 'baseline'}"
            f"{SHUFFLE_TRIALS}_{SUBTRACT_PSTH}"
            ".png"
        )
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        step_fig_paths.append(fig_path)          # ← new line
        plt.close(fig)

        # # ── ← NEW: generate rolling-window animation for the same step ───────────────
        # vid_path = fig_path.with_suffix('.mp4')
        # animate_step_panel(
        #     X_i_z3d=X_i_z3d,
        #     X_j_z3d=X_j_z3d,
        #     X_i_flat=X_i_flat,
        #     X_j_flat=X_j_flat,
        #     X_i_res=X_i_res,
        #     X_j_res=X_j_res,
        #     sort_i=sort_i,
        #     sort_j=sort_j,
        #     Wx_pcca=Wx_p,
        #     Wy_pcca=Wy_p,
        #     rho_pcca=rho_p,
        #     Wx_cca=Wx_cca,
        #     Wy_cca=Wy_cca,
        #     rho_cca=rho_cca,
        #     time_vec=time_vec,
        #     n_trials=n_trials,
        #     T=T,
        #     fig_title=fig_title,
        #     n_show=n_show,
        #     Z_z3d=Z_z3d,
        #     sort_z=sort_z,
        #     Z_name=add_region,
        #     nu_w_i=nu_w_i,
        #     nu_w_j=nu_w_j,
        #     psth_vmax=global_vmax,
        #     window=5,
        #     fps=4.0,
        #     dpi=120,
        #     output_path=vid_path,
        # )
        # print(
        #     f"  [Part 1] step {step_idx:02d}  "
        #     f"nuisance={accumulated_nuisance}  "
        #     f"ρ₁={rho_p[0]:.4f}  saved: {fig_path.name}"
        # )


    ref = step_results[-1]

    # ── Compile all per-step PNGs into a single across-step video ────────────
    # compile_steps_to_video(
    #     fig_paths   = step_fig_paths,
    #     output_path = out_part1 / f"{session_name}_part1_steps_video.mp4",
    #     fps         = 0.8,
    #     dpi         = 150,
    #     title       = (
    #         f"Sequential cumulative removal  |  {session_name}\n"
    #         f"{TARGET_I} ↔ {TARGET_J}   "
    #         f"({len(step_fig_paths)} steps,  "
    #         f"reference ρ₁ = {ref.rho_pcca:.4f})"
    #     ),
    # )


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
    out_part2 = output_dir / "part2_ablation"
    out_part2.mkdir(parents=True, exist_ok=True)

    for t in (TARGET_I, TARGET_J):
        if t not in region_spikes:
            raise RuntimeError(f"Target region '{t}' not in session.")

    # ── Nuisance list ─────────────────────────────────────────────────────
    nuisance_all: List[str] = [
        r for r in ANATOMICAL_ORDER
        if r in region_spikes and r not in (TARGET_I, TARGET_J)
    ]


    all_regions_ordered: List[str] = [
        r for r in ANATOMICAL_ORDER
        if r in region_spikes
    ]

    if not nuisance_all:
        warnings.warn("No nuisance regions found; Part 2 has no ablation targets.")
        return

    # # ── Global Rastermap: fit on targets + all nuisance regions ──────────
    # all_regions_ordered = [TARGET_I, TARGET_J] + nuisance_all

    global_sort = compute_global_neuron_order(region_spikes, all_regions_ordered)

    sort_i = global_sort[TARGET_I]
    sort_j = global_sort[TARGET_J]

    # ── Pre-compute flat and 3-D arrays ───────────────────────────────────
    X_i_raw  = region_spikes[TARGET_I]
    X_j_raw  = region_spikes[TARGET_J]

    X_i_flat = _zscore_flat(X_i_raw, subtract_psth=SUBTRACT_PSTH,
                            shuffle_trials=SHUFFLE_TRIALS, rng=_rng)
    X_j_flat = _zscore_flat(X_j_raw, subtract_psth=SUBTRACT_PSTH,shuffle_trials=SHUFFLE_TRIALS,rng=_rng2)
    # partner: never shuffled

    X_i_z3d = _zscore_3d(X_i_raw, subtract_psth=SUBTRACT_PSTH,
                         shuffle_trials=SHUFFLE_TRIALS, rng=_rng)
    X_j_z3d = _zscore_3d(X_j_raw, subtract_psth=SUBTRACT_PSTH,shuffle_trials=SHUFFLE_TRIALS,rng=_rng2)

    nuisance_flat: Dict[str, np.ndarray] = {
        r: _zscore_flat(region_spikes[r],subtract_psth=SUBTRACT_PSTH) for r in nuisance_all
    }
    nuisance_z3d: Dict[str, np.ndarray] = {
        r: _zscore_3d(region_spikes[r],subtract_psth=SUBTRACT_PSTH) for r in nuisance_all
    }

    _all_psth = np.concatenate(
        [X_i_z3d.mean(axis=0),
         X_j_z3d.mean(axis=0)]
        + [nuisance_z3d[r].mean(axis=0) for r in nuisance_all],
        axis=0,
    )
    global_vmax = float(np.nanpercentile(np.abs(_all_psth), 99))

    time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)

    # ── CCA baseline ─────────────────────────────────────────────────────
    Wx_cca, Wy_cca, rho_cca = ridge_cca(X_i_flat, X_j_flat)
    print(f"  [Part 2] CCA baseline  ρ₁ = {rho_cca[0]:.4f}")

    # ── Full pCCA reference (Z = all nuisance regions) ───────────────────
    Z_full = np.concatenate([nuisance_flat[r] for r in nuisance_all], axis=1)
    Wx_ref, Wy_ref, rho_ref, X_i_ref_res, X_j_ref_res = pcca(
        X_i_flat, X_j_flat, Z_full,
    )
    zi_ref = latent_projections(X_i_ref_res, Wx_ref[:, 0], n_trials, T).mean(0)
    zj_ref = latent_projections(X_j_ref_res, Wy_ref[:, 0], n_trials, T).mean(0)

    full_pcca_ref = StepResult(
        label="full_pCCA_reference",
        nuisance_regions=nuisance_all,
        rho_pcca=float(rho_ref[0]),
        Wx=Wx_ref,
        Wy=Wy_ref,
        z_i_mean=zi_ref,
        z_j_mean=zj_ref,
    )
    print(f"  [Part 2] Full pCCA reference  ρ₁ = {rho_ref[0]:.4f}")

    # ── CCA baseline StepResult (for comparison tables) ───────────────────
    X_i_cca_res = X_i_flat.copy()
    X_j_cca_res = X_j_flat.copy()

    _z_ic_raw = latent_projections(X_i_cca_res, Wx_cca[:, 0], n_trials, T)
    _z_jc_raw = latent_projections(X_j_cca_res, Wy_cca[:, 0], n_trials, T)
    _z_ic_sc, _z_jc_sc, flip_ip, flip_jp = apply_latent_sign_correction(
        _z_ic_raw, _z_jc_raw, time_vec
    )

    Wx_c_f = Wx_cca[:, 0] * (-1.0 if flip_ip else 1.0)
    Wy_c_f = Wy_cca[:, 0] * (-1.0 if flip_jp else 1.0)

    zi_cca = latent_projections(X_i_cca_res, Wx_c_f, n_trials, T).mean(0)
    zj_cca = latent_projections(X_j_cca_res, Wy_c_f, n_trials, T).mean(0)


    cca_baseline_ref = StepResult(
        label="CCA_baseline",
        nuisance_regions=[],
        rho_pcca=float(rho_cca[0]),
        Wx=Wx_cca,
        Wy=Wy_cca,
        z_i_mean=zi_cca,
        z_j_mean=zj_cca,
    )

    # ── Single-region ablation loop ───────────────────────────────────────
    ablation_results: List[StepResult] = []

    for abl_idx, region in enumerate(nuisance_all):
        Z_single = nuisance_flat[region]

        Wx_p, Wy_p, rho_p, X_i_res, X_j_res = pcca(
            X_i_flat, X_j_flat, Z_single,
        )


        _z_ip_raw = latent_projections(X_i_res, Wx_p[:, 0], n_trials, T)
        _z_jp_raw = latent_projections(X_j_res, Wy_p[:, 0], n_trials, T)
        _z_ip_sc, _z_jp_sc, flip_ip, flip_jp = apply_latent_sign_correction(
            _z_ip_raw, _z_jp_raw, time_vec
        )

        Wx_p_f = Wx_p[:, 0] * (-1.0 if flip_ip else 1.0)
        Wy_p_f = Wy_p[:, 0] * (-1.0 if flip_jp else 1.0)


        z_i_mean = latent_projections(X_i_res, Wx_p_f, n_trials, T).mean(0)
        z_j_mean = latent_projections(X_j_res, Wy_p_f, n_trials, T).mean(0)


        # ── Nuisance weight vectors (col 6) ──────────────────────────────
        # Z is a single region, so Beta has exactly n_z rows and sort_z
        # (from the global order) has the same length → consistent indexing.
        nu_w_i = _compute_nuisance_weight(Z_single, X_i_flat, Wx_p[:, 0])
        nu_w_j = _compute_nuisance_weight(Z_single, X_j_flat, Wy_p[:, 0])
        sort_z = global_sort[region]

        label = f"abl{abl_idx:02d}_{region}"
        ablation_results.append(StepResult(
            label=label,
            nuisance_regions=[region],
            rho_pcca=float(rho_p[0]),
            Wx=Wx_p,
            Wy=Wy_p,
            z_i_mean=z_i_mean,
            z_j_mean=z_j_mean,
        ))


        fig_title = (
            f"Ablation {abl_idx:02d}]  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}  |  Z = {{{region}}}\n"
            f"pCCA ρ₁ = {rho_p[0]:.4f}   |   CCA ρ₁ = {rho_cca[0]:.4f}"
        )


        fig = plot_step_panel(
            X_i_z3d=X_i_z3d,
            X_j_z3d=X_j_z3d,
            X_i_flat=X_i_flat,
            X_j_flat=X_j_flat,
            X_i_res=X_i_res,
            X_j_res=X_j_res,
            sort_i=sort_i,
            sort_j=sort_j,
            Wx_pcca=Wx_p,
            Wy_pcca=Wy_p,
            rho_pcca=rho_p,
            Wx_cca=Wx_cca,
            Wy_cca=Wy_cca,
            rho_cca=rho_cca,
            time_vec=time_vec,
            n_trials=n_trials,
            T=T,
            fig_title=fig_title,
            n_show=n_show,
            Z_z3d=nuisance_z3d[region],
            sort_z=sort_z,
            Z_name=region,
            nu_w_i=nu_w_i,
            nu_w_j=nu_w_j,
            psth_vmax=global_vmax,
        )


        fig_path = out_part2 / f"{session_name}_abl{abl_idx:02d}_{region}_{SHUFFLE_TRIALS}_{SUBTRACT_PSTH}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # # # ── ← NEW: generate rolling-window animation for the same step ───────────────
        # vid_path = out_part2 / f"{session_name}_part2_abl{abl_idx:02d}_{region}_anim.mp4"
        # animate_step_panel(
        #     X_i_z3d=X_i_z3d,
        #     X_j_z3d=X_j_z3d,
        #     X_i_flat=X_i_flat,
        #     X_j_flat=X_j_flat,
        #     X_i_res=X_i_res,
        #     X_j_res=X_j_res,
        #     sort_i=sort_i,
        #     sort_j=sort_j,
        #     Wx_pcca=Wx_p,
        #     Wy_pcca=Wy_p,
        #     rho_pcca=rho_p,
        #     Wx_cca=Wx_cca,
        #     Wy_cca=Wy_cca,
        #     rho_cca=rho_cca,
        #     time_vec=time_vec,
        #     n_trials=n_trials,
        #     T=T,
        #     fig_title=fig_title,
        #     n_show=n_show,
        #     Z_z3d=nuisance_z3d[region],
        #     sort_z=sort_z,
        #     Z_name=region,
        #     nu_w_i=nu_w_i,
        #     nu_w_j=nu_w_j,
        #     psth_vmax=global_vmax,
        #     window=5,
        #     fps=4.0,
        #     dpi=120,
        #     output_path=vid_path,
        # )
        # print(
        #     f"  [Part 2] abl {abl_idx:02d}  Z = {{{region:<8}}}  "
        #     f"ρ₁={rho_p[0]:.4f}  saved: {fig_path.name}"
        # )

# =============================================================================
# 10.  Entry point
# =============================================================================

SESSIONS_TO_RUN = [
'yp020_220331'
    # MOs + VPMPO
    # 'yp020_220331', 'yp020_220401', 'yp021_220331', 'yp021_220402',
    # 'yp021_220403', 'yp021_220404', 'yp021_220405', 'yp021_220407',
    # # MOs + MOp
    # 'yp012_220208', 'yp013_220209', 'yp013_220211', 'yp013_220212',
    # 'yp020_220407'
    # # MOs + VALVM
]
# Sessions recording MOs + VPMPO + ORB :
#['yp020_220331', 'yp020_220401', 'yp021_220331',
# 'yp021_220402', 'yp021_220404', 'yp021_220405', 'yp021_220407']


def run_single_session(SESSION_NAME: str) -> None:
    """运行单个 Session 的完整分析流程"""
    BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    SESSION_FILE = (
            BASE_DIR
            / "pcca_sessions_cued_hit_long_results"
            / f"{SESSION_NAME}_analysis_results.mat"
    )

    # 1. 动态生成目标配对的文件夹名称，例如 "MOs_mPFC"
    target_pair_name = f"{TARGET_I} ↔ {TARGET_J}- subtract_psth={SUBTRACT_PSTH}-shuffle_trials={SHUFFLE_TRIALS}"

    # 2. 将新层级加入到 OUTPUT_DIR 的构建中
    OUTPUT_DIR = BASE_DIR / "Paper_output" / "pcca_ablation_8panel" / SESSION_NAME / target_pair_name

    # 确保新创建的带有两层子目录的路径可以成功建立
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"pCCA Sequential Ablation Analysis")
    print(f"  Session : {SESSION_NAME}")
    print(f"  Pair    : {TARGET_I} ↔ {TARGET_J}- subtract_psth={SUBTRACT_PSTH}-shuffle_trials={SHUFFLE_TRIALS}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 70)

    if not SESSION_FILE.exists():
        print(f"❌ [WARNING] File not found, skipping: {SESSION_FILE}")
        return

    region_spikes, n_trials, T = load_region_spikes(str(SESSION_FILE))

    # print("\n── Part 1: Sequential cumulative removal ──────────────────────────")
    # run_sequential_removal(
    #     region_spikes=region_spikes,
    #     n_trials=n_trials,
    #     T=T,
    #     output_dir=OUTPUT_DIR,
    #     session_name=SESSION_NAME,
    # )

    print("\n── Part 2: Single-region ablation ─────────────────────────────────")
    run_single_ablation(
        region_spikes=region_spikes,
        n_trials=n_trials,
        T=T,
        output_dir=OUTPUT_DIR,
        session_name=SESSION_NAME,
    )
    print(f"✨ Session {SESSION_NAME} Done.")


def main() -> None:
    total_sessions = len(SESSIONS_TO_RUN)
    for idx, session in enumerate(SESSIONS_TO_RUN, 1):
        print(f"\n🚀 [Processing {idx}/{total_sessions}] Starting {session}...")
        try:
            run_single_session(session)
        except Exception as e:
            print(f"💥 [ERROR] Failed to process {session}. Error: {e}")

    print("\n🎉 All sessions completed!")


if __name__ == "__main__":
    main()


# def main() -> None:
#     #/Users/shengyuancai/Downloads/Oxford_dataset/pcca_sessions_cued_hit_long_results/yp021_220407_analysis_results.mat
#     BASE_DIR     = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
#     SESSION_NAME = "yp020_220331"
#     SESSION_FILE = (
#             BASE_DIR
#             / "pcca_sessions_cued_hit_long_results"
#             / f"{SESSION_NAME}_analysis_results.mat"
#     )
#     OUTPUT_DIR = BASE_DIR / "Paper_output" / "pcca_ablation_8panel" / SESSION_NAME
#
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#
#     print("=" * 70)
#     print(f"pCCA Sequential Ablation Analysis")
#     print(f"  Session : {SESSION_NAME}")
#     print(f"  Pair    : {TARGET_I} ↔ {TARGET_J}")
#     print(f"  Output  : {OUTPUT_DIR}")
#     print("=" * 70)
#
#     region_spikes, n_trials, T = load_region_spikes(str(SESSION_FILE))
#
#     print("\n── Part 1: Sequential cumulative removal ──────────────────────────")
#     run_sequential_removal(
#         region_spikes=region_spikes,
#         n_trials=n_trials,
#         T=T,
#         output_dir=OUTPUT_DIR,
#         session_name=SESSION_NAME,
#     )
#
#     print("\n── Part 2: Single-region ablation ─────────────────────────────────")
#     run_single_ablation(
#         region_spikes=region_spikes,
#         n_trials=n_trials,
#         T=T,
#         output_dir=OUTPUT_DIR,
#         session_name=SESSION_NAME,
#     )
#
#     print("\nDone.")
#
#
# if __name__ == "__main__":
#     main()
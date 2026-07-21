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
SHUFFLE_TRIALS: bool = True


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


def compute_similarity(s: StepResult, ref: StepResult) -> Dict[str, float]:
    cos_i    = _cos_sim_abs(s.Wx[:, 0], ref.Wx[:, 0])
    cos_j    = _cos_sim_abs(s.Wy[:, 0], ref.Wy[:, 0])

    r_i      = float(pearsonr(s.z_i_mean, ref.z_i_mean)[0])
    r_j      = float(pearsonr(s.z_j_mean, ref.z_j_mean)[0])

    rho_diff = abs(s.rho_pcca - ref.rho_pcca)

    divergence = (
            (1 - cos_i)
            + (1 - cos_j)
            + (1 - max(r_i, -1.0))
            + (1 - max(r_j, -1.0))
            + rho_diff
    )

    return dict(
        cos_sim_i=cos_i,
        cos_sim_j=cos_j,
        latent_r_i=r_i,
        latent_r_j=r_j,
        rho_abs_diff=rho_diff,
        divergence=divergence,
    )


def identify_extremes(
        step_results: List[StepResult],
        ref: StepResult,
        exclude_ref: bool = True,
) -> Tuple[int, int, List[Dict[str, float]]]:
    sims          = [compute_similarity(s, ref) for s in step_results]
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
    labels = [s.label for s in step_results]
    rhos   = [s.rho_pcca for s in step_results]
    cos_i  = [d["cos_sim_i"]  for d in sims]
    cos_j  = [d["cos_sim_j"]  for d in sims]
    lat_i  = [d["latent_r_i"] for d in sims]
    lat_j  = [d["latent_r_j"] for d in sims]
    div    = [d["divergence"]  for d in sims]
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
    ax.plot(x, list(map(abs, lat_i)), "s-", color=_CI_PCCA, lw=1.8, ms=5, label=TARGET_I)
    ax.plot(x, list(map(abs, lat_j)), "^-", color=_CJ_PCCA, lw=1.8, ms=5, label=TARGET_J)
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
# 1b.  Supplementary diagnostic metrics
# =============================================================================

def _trial_rows(trial_idx: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """Row indices in the flat matrix for the given trial indices.

    Flat-matrix layout (from _zscore_flat):  row = t * n_trials + trial.
    All T time-points of trial k therefore occupy rows
        k,  k + n_trials,  k + 2·n_trials,  …,  k + (T-1)·n_trials.

    Parameters
    ----------
    trial_idx : 1-D integer array of trial numbers (0-indexed).
    n_trials  : total number of trials in the session.
    T         : number of time bins per trial.

    Returns
    -------
    1-D integer array of length len(trial_idx) * T, in ascending row order.
    """
    return (trial_idx[None, :] + np.arange(T)[:, None] * n_trials).ravel()


@dataclass
class SupplementaryMetrics:
    """
    Per-step supplementary diagnostic bundle, complementing StepResult.

    All angle quantities in degrees.  Variance fractions (r2_*) are
    normalised to the raw Frobenius norm² of their respective flat matrix,
    so r2_nuis + r2_comm + r2_priv ≈ 1.

    Fields
    ------
    theta_i_deg, theta_j_deg
        ∠(w_CCA, w_pCCA) in degrees for TARGET_I and TARGET_J.
        θ → 0° : pCCA barely rotates the CCA solution (nuisance had little
                  effect on the coupling direction).
        θ → 90°: nuisance removal drives the weight into an orthogonal
                  subspace — the strongest validation that partialling Z
                  genuinely separates two distinct coupling axes.

    kappa_i, kappa_j
        Cross-analysis weight collinearity:
            κ_I = |cos ∠(w_{IJ,I},  w_{IZ,I})|
        where w_{IJ,I} is the TARGET_I weight from pCCA(I, J | Z) (already
        computed) and w_{IZ,I} is the TARGET_I weight from the swapped
        analysis pCCA(I, Z | J) (computed here).

        κ → 0 : the two coupling axes in region I are orthogonal — desired;
                 indicates clean separation of partner-coupling vs.
                 nuisance-coupling directions.
        κ → 1 : Type-II collapse — both analyses converge on the same
                 direction in I (shared-noise amplification by CCA whitening).
        Set to NaN when Z_flat is None (no nuisance, step 0).

    rho1_cv_mean, rho1_cv_sem
        K-fold (default 5) cross-validated ρ₁.  Regression coefficients
        are estimated on training folds only; canonical weights are then
        obtained from the training residuals and applied to held-out test
        residuals.  Systematic deflation relative to in-sample ρ₁ signals
        overfitting to the nuisance-regression step.

    r2_nuis_i, r2_comm_i, r2_priv_i  (analogously _j)
        Three-way variance decomposition for each target region:
          r2_nuis  =  ||X − X̃||_F² / ||X||_F²
                      (variance removed by Z regression; measures nuisance
                       saturation — high values when Z accounts for most of
                       the target region's drive, as seen with ORB)
          r2_comm  =  SS(X̃ w₁^{pCCA}) / ||X||_F²
                      (variance of the residual on the first pCCA axis;
                       ideally comparable to r2_comm in the simulation
                       ground truth)
          r2_priv  =  1 − r2_nuis − r2_comm  (private / unexplained)

    lag_peak_ms
        Lag (ms) at which the normalised cross-correlation of the two
        sign-corrected mean pCCA latent traces is maximised, restricted to
        ±150 ms (physiologically plausible corticocortical / corticothalamic
        range).  Positive values indicate TARGET_I leads TARGET_J.

    lag_corr_at_peak
        Cross-correlation coefficient at lag_peak_ms.

    lag_axis_ms, xcorr_curve
        Full lag axis (ms) and normalised cross-correlation curve for
        plotting; both of length 2T − 1.
    """
    step_label:       str
    theta_i_deg:      float
    theta_j_deg:      float
    kappa_i:          float           # NaN if Z_flat is None
    kappa_j:          float
    rho1_cv_mean:     float
    rho1_cv_sem:      float
    r2_nuis_i:        float
    r2_comm_i:        float
    r2_priv_i:        float
    r2_nuis_j:        float
    r2_comm_j:        float
    r2_priv_j:        float
    lag_peak_ms:      float
    lag_corr_at_peak: float
    lag_axis_ms:      np.ndarray = _field(repr=False)
    xcorr_curve:      np.ndarray = _field(repr=False)


def compute_supplementary_metrics(
        *,
        X_i_flat:  np.ndarray,
        X_j_flat:  np.ndarray,
        Z_flat:    Optional[np.ndarray],
        X_i_res:   np.ndarray,
        X_j_res:   np.ndarray,
        Wx_pcca:   np.ndarray,
        Wy_pcca:   np.ndarray,
        Wx_cca:    np.ndarray,
        Wy_cca:    np.ndarray,
        z_i_p:     np.ndarray,          # (n_trials, T) sign-corrected pCCA latent
        z_j_p:     np.ndarray,
        n_trials:  int,
        T:         int,
        time_vec:  np.ndarray,
        step_label: str,
        n_cv_folds: int = 5,
        lam_cca:   float = LAMBDA_CCA,
        lam_hat:   float = LAMBDA_HAT,
) -> SupplementaryMetrics:
    """Compute the full supplementary diagnostic bundle for one ablation step.

    This function is a pure function of its inputs: it calls only
    ``residualize``, ``ridge_cca``, and ``_cos_sim_abs`` from this module
    plus standard numpy/scipy.  It can therefore be inserted directly after
    the existing ``pcca()`` + ``apply_latent_sign_correction()`` calls
    inside either ablation loop without restructuring the loop logic.

    Cross-validation implementation note
    -------------------------------------
    Trials are split into ``n_cv_folds`` contiguous blocks (no shuffling)
    to preserve temporal stationarity assumptions.  For each fold:

        1. Hat matrix  (Z'Z + λnI)⁻¹ Z'  is estimated on training trials only.
        2. Both training and held-out test residuals are formed by applying
           the TRAINING-fold regression coefficients — this prevents leakage
           of the nuisance-removal step across folds.
        3. CCA is fitted on training residuals; the resulting canonical weights
           are applied to test residuals; Pearson r of test-fold projections is
           recorded as the held-out ρ₁.

    κ computation note
    ------------------
    The swapped analyses pCCA(I, Z | J) and pCCA(J, Z | I) are computed on
    the full (non-CV) data.  When Z is high-dimensional, the cross-covariance
    in the CCA step is (n_i × n_z); ridge regularisation handles this
    gracefully.  κ is left as NaN when Z_flat is None or empty.
    """
    w_p_i = Wx_pcca[:, 0]
    w_p_j = Wy_pcca[:, 0]
    w_c_i = Wx_cca[:, 0]
    w_c_j = Wy_cca[:, 0]

    # ── 1.  CCA–pCCA rotation angle ───────────────────────────────────────
    theta_i_deg = float(np.degrees(
        np.arccos(np.clip(_cos_sim_abs(w_p_i, w_c_i), 0.0, 1.0))
    ))
    theta_j_deg = float(np.degrees(
        np.arccos(np.clip(_cos_sim_abs(w_p_j, w_c_j), 0.0, 1.0))
    ))

    # ── 2.  Cross-analysis collinearity κ ─────────────────────────────────
    # κ_I = |cos ∠(w_{IJ,I}, w_{IZ,I})|
    # Requires pCCA(I, Z | J) and pCCA(J, Z | I) — two additional analyses.
    if Z_flat is not None and Z_flat.shape[1] > 0:
        # pCCA(I, Z | J): partial J from both I and Z, then CCA
        Xi_res_wrtJ = residualize(X_i_flat, X_j_flat, lam_hat)
        Z_res_wrtJ  = residualize(Z_flat,   X_j_flat, lam_hat)
        Wx_IZ, _, _ = ridge_cca(Xi_res_wrtJ, Z_res_wrtJ,  lam_cca, 1)
        kappa_i     = float(_cos_sim_abs(w_p_i, Wx_IZ[:, 0]))

        # pCCA(J, Z | I): partial I from both J and Z, then CCA
        Xj_res_wrtI = residualize(X_j_flat, X_i_flat, lam_hat)
        Z_res_wrtI  = residualize(Z_flat,   X_i_flat, lam_hat)
        Wy_JZ, _, _ = ridge_cca(Xj_res_wrtI, Z_res_wrtI, lam_cca, 1)
        kappa_j     = float(_cos_sim_abs(w_p_j, Wy_JZ[:, 0]))
    else:
        kappa_i = kappa_j = float('nan')

    # ── 3.  Cross-validated ρ₁ ────────────────────────────────────────────
    fold_size  = n_trials // n_cv_folds
    trial_perm = np.arange(n_trials)          # contiguous folds, no shuffle
    rhos_cv: list = []

    for fold in range(n_cv_folds):
        te_trials = trial_perm[fold * fold_size: (fold + 1) * fold_size]
        tr_trials = np.concatenate([
            trial_perm[: fold * fold_size],
            trial_perm[(fold + 1) * fold_size:],
        ])
        tr_rows = _trial_rows(tr_trials, n_trials, T)
        te_rows = _trial_rows(te_trials, n_trials, T)

        Xi_tr, Xj_tr = X_i_flat[tr_rows], X_j_flat[tr_rows]
        Xi_te, Xj_te = X_i_flat[te_rows], X_j_flat[te_rows]

        if Z_flat is not None and Z_flat.shape[1] > 0:
            Z_tr = Z_flat[tr_rows]
            Z_te = Z_flat[te_rows]
            n_tr = len(tr_rows)
            # Regression coefficients estimated on training fold ONLY
            ZtZ_tr  = Z_tr.T @ Z_tr + lam_hat * n_tr * np.eye(Z_tr.shape[1])
            Beta_i  = np.linalg.solve(ZtZ_tr, Z_tr.T @ Xi_tr)   # (n_z, n_i)
            Beta_j  = np.linalg.solve(ZtZ_tr, Z_tr.T @ Xj_tr)   # (n_z, n_j)
            # Apply training coefficients to both folds (no leakage)
            Xi_tr_res = Xi_tr - Z_tr @ Beta_i
            Xj_tr_res = Xj_tr - Z_tr @ Beta_j
            Xi_te_res = Xi_te - Z_te @ Beta_i
            Xj_te_res = Xj_te - Z_te @ Beta_j
        else:
            Xi_tr_res, Xj_tr_res = Xi_tr, Xj_tr
            Xi_te_res, Xj_te_res = Xi_te, Xj_te

        Wx_cv, Wy_cv, _ = ridge_cca(Xi_tr_res, Xj_tr_res, lam_cca, 1)
        zi_te = Xi_te_res @ Wx_cv[:, 0]
        zj_te = Xj_te_res @ Wy_cv[:, 0]

        if np.std(zi_te) < 1e-9 or np.std(zj_te) < 1e-9:
            rhos_cv.append(0.0)
        else:
            rhos_cv.append(float(np.clip(pearsonr(zi_te, zj_te)[0], -1.0, 1.0)))

    rho1_cv_mean = float(np.mean(rhos_cv))
    rho1_cv_sem  = float(np.std(rhos_cv) / np.sqrt(max(n_cv_folds, 1)))

    # ── 4.  Variance partitioning ─────────────────────────────────────────
    # All three fractions are expressed relative to ||X||_F² (raw data),
    # so they sum to ≈ 1 and are comparable across regions and sessions.
    denom_i = float(np.sum(X_i_flat ** 2)) + 1e-12
    denom_j = float(np.sum(X_j_flat ** 2)) + 1e-12

    w_p_i_unit = w_p_i / (np.linalg.norm(w_p_i) + 1e-12)
    w_p_j_unit = w_p_j / (np.linalg.norm(w_p_j) + 1e-12)


    # (a) Nuisance: ||X_hat||_F^2 / ||X||_F^2
    X_i_hat = X_i_flat - X_i_res
    X_j_hat = X_j_flat - X_j_res

    r2_nuis_i = float(np.sum(X_i_hat ** 2)) / denom_i
    r2_nuis_j = float(np.sum(X_j_hat ** 2)) / denom_j

    # (b) Communication: ||X_res ŵ||^2 / ||X||_F^2  — NO mean centering
    #     This is the total SS of the scalar projection, including its DC component.
    proj_i = X_i_res @ w_p_i_unit  # shape (N,)
    proj_j = X_j_res @ w_p_j_unit
    r2_comm_i = float(np.sum(proj_i ** 2)) / denom_i
    r2_comm_j = float(np.sum(proj_j ** 2)) / denom_j

    # (c) Private: directly computed as the residual of X_res after projecting out ŵ
    #     NOT defined as a remainder — independently computable
    X_i_priv = X_i_res - np.outer(proj_i, w_p_i_unit)  # X_res (I - ŵŵ^T)
    X_j_priv = X_j_res - np.outer(proj_j, w_p_j_unit)
    r2_priv_i = float(np.sum(X_i_priv ** 2)) / denom_i
    r2_priv_j = float(np.sum(X_j_priv ** 2)) / denom_i

    # Diagnostic — under OLS the ridge cross-term is identically zero and
    # the three fractions sum exactly to 1.  With lam_hat = 1e-4 the gap is
    # negligible; you can verify it explicitly:
    #   gap_i = 1.0 - r2_nuis_i - r2_comm_i - r2_priv_i   # should be ≈ 0

    # ── 5.  Temporal lead–lag cross-correlation ───────────────────────────
    mean_i = z_i_p.mean(axis=0);  mean_i -= mean_i.mean()
    mean_j = z_j_p.mean(axis=0);  mean_j -= mean_j.mean()
    std_i  = float(np.std(mean_i)) + 1e-12
    std_j  = float(np.std(mean_j)) + 1e-12

    xcorr_full  = np.correlate(mean_i / std_i, mean_j / std_j, mode='full') / T
    lag_bins    = np.arange(-(T - 1), T)
    dt_ms       = float(time_vec[1] - time_vec[0]) * 1000.0
    lag_axis_ms = lag_bins.astype(float) * dt_ms

    # Restrict peak search to ±150 ms (corticocortical / corticothalamic range)
    max_lag_bins = min(int(150.0 / dt_ms), T - 1)
    search_mask  = np.abs(lag_bins) <= max_lag_bins
    peak_rel     = int(np.argmax(np.abs(xcorr_full[search_mask])))
    peak_abs     = int(np.where(search_mask)[0][peak_rel])

    return SupplementaryMetrics(
        step_label       = step_label,
        theta_i_deg      = theta_i_deg,
        theta_j_deg      = theta_j_deg,
        kappa_i          = kappa_i,
        kappa_j          = kappa_j,
        rho1_cv_mean     = rho1_cv_mean,
        rho1_cv_sem      = rho1_cv_sem,
        r2_nuis_i        = r2_nuis_i,
        r2_comm_i        = r2_comm_i,
        r2_priv_i        = r2_priv_i,
        r2_nuis_j        = r2_nuis_j,
        r2_comm_j        = r2_comm_j,
        r2_priv_j        = r2_priv_j,
        lag_peak_ms      = float(lag_axis_ms[peak_abs]),
        lag_corr_at_peak = float(xcorr_full[peak_abs]),
        lag_axis_ms      = lag_axis_ms,
        xcorr_curve      = xcorr_full,
    )

# =============================================================================
# 7b.  Supplementary diagnostic panel
# =============================================================================

def plot_supplementary_panel(
        supp_list:      List[SupplementaryMetrics],
        rho_pcca_list:  List[float],
        rho_cca:        float,
        title:          str,
        output_path:    Optional[Path] = None,
) -> plt.Figure:
    """Six-panel supplementary diagnostic figure for one ablation series.

    Row 0 — stability metrics (how trustworthy is the pCCA solution?):
        (A)  ρ₁ in-sample vs. cross-validated
        (B)  CCA–pCCA rotation angle θ  (both target regions)
        (C)  Cross-analysis collinearity κ  (both target regions)

    Row 1 — structural metrics (what does the solution actually encode?):
        (D)  Variance partition — TARGET_I  (stacked bar: nuisance/comm/private)
        (E)  Variance partition — TARGET_J
        (F)  pCCA latent cross-correlation  (all steps overlaid, viridis)

    Connection to the simulation compass plots
    ------------------------------------------
    Panel (C) is the real-data version of the simulation compass-plot θ:
    κ ≈ 0 maps to the ⊥ case (orthogonal hub vs. lateral axes in A),
    κ ≈ 1 maps to the Type-II collapse where both pCCA analyses converge
    on the shared-noise direction.  For the b-topology noise-free case
    shown in the uploaded simulation figure, the expected profile across
    ablation steps is: κ should remain near 0 so long as Region B (hub)
    is present in Z and Region Z (lateral) is the active partner.

    Parameters
    ----------
    supp_list      : one SupplementaryMetrics per step, in ablation order.
    rho_pcca_list  : matching in-sample ρ₁ values (from StepResult).
    rho_cca        : fixed CCA baseline ρ₁ (no nuisance).
    title          : suptitle string.
    output_path    : if provided, save PNG here.
    """
    n      = len(supp_list)
    x      = np.arange(n)
    labels = [s.step_label for s in supp_list]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(19, 9),
        gridspec_kw={'hspace': 0.58, 'wspace': 0.36},
    )

    def _xax(ax: plt.Axes) -> None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=7)
        ax.grid(alpha=0.22, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    theta_i      = [s.theta_i_deg      for s in supp_list]
    theta_j      = [s.theta_j_deg      for s in supp_list]
    kappa_i      = [s.kappa_i          for s in supp_list]
    kappa_j      = [s.kappa_j          for s in supp_list]
    rho_cv       = [s.rho1_cv_mean     for s in supp_list]
    rho_cv_sem   = [s.rho1_cv_sem      for s in supp_list]

    # ── Panel A: ρ₁ in-sample vs. CV ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, rho_pcca_list, 'o-', color='#922B21', lw=1.8, ms=5,
            label='ρ₁  in-sample')
    ax.errorbar(x, rho_cv, yerr=rho_cv_sem,
                fmt='s--', color='#2471A3', lw=1.5, ms=5, capsize=3,
                label='ρ₁  5-fold CV')
    ax.axhline(rho_cca, color='#7F8C8D', ls=':', lw=1.1, alpha=0.7,
               label='CCA  ρ₁  (Z = ∅)')
    ax.set_ylabel('Canonical correlation  ρ₁', fontsize=9)
    ax.set_title('(A)  In-sample vs. cross-validated  ρ₁\n'
                 'Large CV gap = overfitting to nuisance regression',
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(-0.05, 1.05)
    _xax(ax)

    # ── Panel B: CCA–pCCA rotation angle θ ────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, theta_i, 's-', color=_CI_PCCA, lw=1.8, ms=5, label=TARGET_I)
    ax.plot(x, theta_j, '^-', color=_CJ_PCCA, lw=1.8, ms=5, label=TARGET_J)
    ax.axhline(90, color='#AAB7B8', ls='--', lw=0.9, alpha=0.6,
               label='90°  (fully orthogonal)')
    ax.axhline(0,  color='#AAB7B8', ls=':',  lw=0.7, alpha=0.4)
    ax.set_ylabel('θ  CCA–pCCA weight angle  (°)', fontsize=9)
    ax.set_ylim(-3, 95)
    ax.set_title('(B)  CCA–pCCA rotation angle  θ\n'
                 r'θ → 90° : nuisance removal uncovered orthogonal axis',
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    _xax(ax)

    # ── Panel C: Cross-analysis collinearity κ ────────────────────────────
    # κ is the real-data analogue of the simulation compass-plot θ
    ax = axes[0, 2]
    ki_clean = [v if not np.isnan(v) else np.nan for v in kappa_i]
    kj_clean = [v if not np.isnan(v) else np.nan for v in kappa_j]
    ax.plot(x, ki_clean, 's-', color=_CI_PCCA, lw=1.8, ms=5,
            label=f'{TARGET_I}:  |cos∠(w_IJ, w_IZ)|')
    ax.plot(x, kj_clean, '^-', color=_CJ_PCCA, lw=1.8, ms=5,
            label=f'{TARGET_J}:  |cos∠(w_JI, w_JZ)|')
    ax.axhline(1.0, color='#E74C3C', ls='--', lw=0.9, alpha=0.55,
               label='κ = 1  (Type-II collapse)')
    ax.axhline(0.0, color='#27AE60', ls='--', lw=0.9, alpha=0.55,
               label='κ = 0  (orthogonal axes)')
    ax.fill_between(x, [0.8]*n, [1.0]*n, color='#FADBD8', alpha=0.35,
                    zorder=0)                          # danger zone shading
    ax.set_ylabel('κ  cross-analysis collinearity', fontsize=9)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title('(C)  Weight collinearity  κ\n'
                 r'κ → 1 : shared-noise collapse; κ → 0 : clean separation',
                 fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False)
    _xax(ax)

    # ── Panel D: Variance partition — TARGET_I ────────────────────────────
    ax = axes[1, 0]
    rn_i = [s.r2_nuis_i for s in supp_list]
    rc_i = [s.r2_comm_i for s in supp_list]
    rp_i = [s.r2_priv_i for s in supp_list]
    bottoms_dc = [n + c for n, c in zip(rn_i, rc_i)]
    ax.bar(x, rn_i,    color='#5D6D7E', alpha=0.88, label='Nuisance  r²')
    ax.bar(x, rc_i,    bottom=rn_i,       color=_CI_PCCA, alpha=0.88,
           label='Communication  r²  (pCCA axis)')
    # ax.bar(x, rp_i,    bottom=bottoms_dc, color='#AEB6BF', alpha=0.72,
    #        label='Private  r²')
    ax.set_ylabel('Fraction of total variance', fontsize=9)
    ax.set_title(f'(D)  Variance partition — {TARGET_I}',
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 0.15)
    _xax(ax)

    # ── Panel E: Variance partition — TARGET_J ────────────────────────────
    ax = axes[1, 1]
    rn_j = [s.r2_nuis_j for s in supp_list]
    rc_j = [s.r2_comm_j for s in supp_list]
    rp_j = [s.r2_priv_j for s in supp_list]
    bottoms_dcj = [n + c for n, c in zip(rn_j, rc_j)]
    ax.bar(x, rn_j,    color='#5D6D7E', alpha=0.88, label='Nuisance  r²')
    ax.bar(x, rc_j,    bottom=rn_j,       color=_CJ_PCCA, alpha=0.88,
           label='Communication  r²  (pCCA axis)')
    # ax.bar(x, rp_j,    bottom=bottoms_dcj, color='#AEB6BF', alpha=0.72,
    #        label='Private  r²')
    ax.set_ylabel('Fraction of total variance', fontsize=9)
    ax.set_title(f'(E)  Variance partition — {TARGET_J}',
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 0.15)
    _xax(ax)

    # ── Panel F: Temporal lead–lag cross-correlation (all steps) ─────────
    ax = axes[1, 2]
    mask_300 = np.abs(supp_list[0].lag_axis_ms) <= 300
    lag_ax   = supp_list[0].lag_axis_ms[mask_300]
    step_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    for k, (s, col) in enumerate(zip(supp_list, step_colors)):
        lw    = 0.50 if k in (1,2, n - 2) else 0.50
        alpha = 0.50 if k in (1,2, n - 2) else 0.50
        lbl   = s.step_label #if k in (1,2, n - 2) else None
        ax.plot(lag_ax, s.xcorr_curve[mask_300],
                color=col, lw=lw, alpha=alpha, label=lbl)
        ax.scatter([s.lag_peak_ms], [s.lag_corr_at_peak],
                   color=col, s=10, zorder=4)

    ax.axvline(0, color='k', ls=':', lw=0.8, alpha=0.45)
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.25)
    ax.set_xlim(-100, 100)
    ax.set_xlabel('Lag (ms)   [+ = TARGET_I leads TARGET_J]', fontsize=9)
    ax.set_ylabel('Normalised cross-correlation', fontsize=9)
    ax.set_title('(F)  pCCA latent cross-correlation across steps\n'
                 'viridis: step 0 (dark) → step N (bright);  dots = peak lag',
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight='bold')
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f'  [supp_panel] saved: {output_path}')
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
    supp_list: List[SupplementaryMetrics] = []
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


        supp_list.append(compute_supplementary_metrics(
            X_i_flat   = X_i_flat,
            X_j_flat   = X_j_flat,
            Z_flat     = Z_flat,
            X_i_res    = X_i_res,
            X_j_res    = X_j_res,
            Wx_pcca    = Wx_p,
            Wy_pcca    = Wy_p,
            Wx_cca     = Wx_cca,
            Wy_cca     = Wy_cca,
            z_i_p      = _z_ip_sc,
            z_j_p      = _z_jp_sc,
            n_trials   = n_trials,
            T          = T,
            time_vec   = time_vec,
            step_label = label,
        ))

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
    idx_closest, idx_furthest, sims = identify_extremes(step_results, ref)

    print("\n" + "=" * 90)
    print(f"[{TARGET_I} ↔ {TARGET_J}]  Summary: similarity to reference")
    print(f"Reference = {ref.label}  (ρ₁ = {ref.rho_pcca:.4f})")
    print_similarity_table(step_results, sims, ref.label, idx_closest, idx_furthest)
    print(
        f"  ▶  Most similar  : step {idx_closest:02d}  ({step_results[idx_closest].label})\n"
        f"  ▶  Most divergent: step {idx_furthest:02d}  ({step_results[idx_furthest].label})\n"
    )

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


    summary_fig = plot_summary_figure(
        step_results=step_results,
        sims=sims,
        idx_closest=idx_closest,
        idx_furthest=idx_furthest,
        ref_label=ref.label,
        title=(
            f"Part 1 — Sequential cumulative removal  |  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}   reference = {ref.label}"
        ),
        output_path=out_part1 / f"{session_name}_part1_summary.png",
    )
    plt.close(summary_fig)

    supp_fig = plot_supplementary_panel(
        supp_list     = supp_list,
        rho_pcca_list = [s.rho_pcca for s in step_results],
        rho_cca       = float(rho_cca[0]),
        title         = (
            f'Part 1 — Supplementary Diagnostics  |  {session_name}\n'
            f'{TARGET_I} ↔ {TARGET_J}'
        ),
        output_path   = out_part1 / f'{session_name}_part1_supplementary.png',
    )
    plt.close(supp_fig)

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
    supp_list_abl: List[SupplementaryMetrics] = []

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



        supp_list_abl.append(compute_supplementary_metrics(
            X_i_flat   = X_i_flat,
            X_j_flat   = X_j_flat,
            Z_flat     = Z_single,
            X_i_res    = X_i_res,
            X_j_res    = X_j_res,
            Wx_pcca    = Wx_p,
            Wy_pcca    = Wy_p,
            Wx_cca     = Wx_cca,
            Wy_cca     = Wy_cca,
            z_i_p      = _z_ip_sc,
            z_j_p      = _z_jp_sc,
            n_trials   = n_trials,
            T          = T,
            time_vec   = time_vec,
            step_label = label,
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

    # ── Summary tables ───────────────────────────────────────────────────
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

    fig_a = plot_summary_figure(
        step_results=ablation_results,
        sims=sims_vs_cca,
        idx_closest=idx_c_a,
        idx_furthest=idx_f_a,
        ref_label="CCA baseline  (Z = ∅)",
        title=(
            f"Part 2 — Single-region ablation  |  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}   reference: CCA baseline"
        ),
        output_path=out_part2 / f"{session_name}_{SHUFFLE_TRIALS}_{SUBTRACT_PSTH}_part2_summary_vs_CCA.png",
    )
    plt.close(fig_a)

    fig_b = plot_summary_figure(
        step_results=ablation_results,
        sims=sims_vs_full,
        idx_closest=idx_c_b,
        idx_furthest=idx_f_b,
        ref_label="full pCCA  (Z = all)",
        title=(
            f"Single-region ablation  |  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}   reference: full pCCA"
        ),
        output_path=out_part2 / f"{session_name}_{SHUFFLE_TRIALS}_{SUBTRACT_PSTH}_part2_summary_vs_fullpCCA.png",
    )
    plt.close(fig_b)

    supp_fig_abl = plot_supplementary_panel(
            supp_list     = supp_list_abl,
            rho_pcca_list = [s.rho_pcca for s in ablation_results],
            rho_cca       = float(rho_cca[0]),
            title         = (
                f'Part 2 — Supplementary Diagnostics  |  {session_name}\n'
                f'{TARGET_I} ↔ {TARGET_J}   (single-region ablation)'
            ),
            output_path   = out_part2 / f'{session_name}_{SHUFFLE_TRIALS}_{SUBTRACT_PSTH}_part2_supplementary.png',
        )
    plt.close(supp_fig_abl)
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
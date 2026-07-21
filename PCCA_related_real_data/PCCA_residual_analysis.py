"""
residual_cca_pcca.py
====================================================================================
Residual (noise-correlation) CCA / pCCA for the MOs ↔ VPMPO pair, with visualisations
that are *correct in the residual regime*.

This revision runs single-region ablation AUTOMATICALLY across every nuisance region
present in the session (one residual pCCA per region, Z = {r}), instead of a single
manually-specified region.  The residual CCA baseline (Z = ∅) is region-independent
and is therefore fit and rendered once.

────────────────────────────────────────────────────────────────────────────────────
MATHEMATICAL SETTING
────────────────────────────────────────────────────────────────────────────────────
Decompose single-trial activity into a stimulus-locked mean and a residual:

        x_A(t, n, s) = μ_A(n, s) + δ_A(t, n, s),     μ_A(n,s) = (1/T) Σ_t x_A(t,n,s)

The cross-region covariance splits additively,  Σ_AB = Σ_AB^stim + Σ_AB^noise , and:

    • SHUFFLE_TRIALS=True , SUBTRACT_PSTH=False  →  signal  Σ_AB^stim
    • SHUFFLE_TRIALS=False, SUBTRACT_PSTH=True   →  noise   Σ_AB^noise   ← THIS SCRIPT

By construction (1/T) Σ_t δ_A = 0, so any latent trial-mean vanishes:
        z̄_A(s) = ((1/T) Σ_t δ_A) w_A = 0 ,
and the same holds after nuisance removal (Z is PSTH-subtracted).  The coupling lives
in the joint per-(t,s) distribution, measured by ρ₁ = corr_{(t,s)}(z_A, z_B).  The
correct visual object is the CLOUD, not the CURVE — hence no trial-averaged latent or
residual PSTH is ever drawn, and `apply_latent_sign_correction` (mean-based) is omitted.

────────────────────────────────────────────────────────────────────────────────────
OUTPUTS  (per session, under  .../residual_cca_pcca/{session}/{MOs}_{VPMPO}/ )
────────────────────────────────────────────────────────────────────────────────────
  CCA_baseline/
      {session}_overview_CCA.png
      {session}_neurons_{MOs}_CCA.png
      {session}_neurons_{VPMPO}_CCA.png
  ablation_{region}/                         (one folder per nuisance region)
      {session}_overview_pCCA_{region}.png
      {session}_neurons_{MOs}_pCCA_{region}.png
      {session}_neurons_{VPMPO}_pCCA_{region}.png

Figures
  fig_residual_overview              — weight bars (top-|w| boxed) + per-trial latent
                                       heatmaps + joint (t,s) scatter with in-sample
                                       and held-out ρ₁.
  fig_highweight_neuron_residuals    — per top-|w| neuron: trial-averaged rate vs
                                       per-trial residuals (+ residual heatmap).

Core primitives (load, global-Rastermap order, z-score, ridge_cca, residualize, pcca,
_trial_rows) are copied VERBATIM from pcca_sequential_ablation.py so the data path,
the neuron ordering (fitted on RAW spike counts), and the CCA/pCCA mathematics are
byte-for-byte identical to the 8-panel pipeline.  Only ANATOMICAL_ORDER and safe_array
are imported from Useful_definition.

Dependencies: numpy, scipy, matplotlib, mat73, (optionally) rastermap.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr

from Useful_definition import ANATOMICAL_ORDER, safe_array

try:
    import mat73
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mat73 is required: pip install mat73") from exc

try:
    from rastermap import Rastermap
    _RASTERMAP_OK = True
except ImportError:
    _RASTERMAP_OK = False
    warnings.warn("rastermap not found; falling back to peak-time neuron ordering.")


# =============================================================================
# 0.  Configuration
# =============================================================================

TARGET_I = "MOs"
TARGET_J = "VPMPO"

# Residual / noise-correlation regime: subtract PSTH, do NOT shuffle.
SUBTRACT_PSTH:  bool = True
SHUFFLE_TRIALS: bool = False

# Single-region ablation is now run automatically across every nuisance region
# present in the session (Z = {r} for each r).  Set this to a list to restrict
# the set; leave None to use all present nuisance regions.
NUISANCE_REGIONS_OVERRIDE: Optional[List[str]] = None

LAMBDA_CCA   = 1e-4    # ridge added to Cxx, Cyy in whitening
LAMBDA_HAT   = 1e-4    # ridge on Z'Z in the hat matrix (scaled by n inside pcca)
N_COMPONENTS = 5       # canonical dimensions retained
TOP_K_NEURONS = 6      # high-|weight| neurons displayed per region
N_NEURONS_SHOW = 80    # neurons displayed in the weight-bar panels
TIME_RANGE_S = (-1.5, 3.0)

# ── Colour palette (mirrors the 8-panel script) ─────────────────────────────
_C_MOS  = "#C0392B"   # MOs
_C_VPM  = "#2471A3"   # VPMPO
_C_POS  = "#922B21"   # positive weight bar
_C_NEG  = "#1A5276"   # negative weight bar


# =============================================================================
# 1.  Core mathematics  (COPIED VERBATIM FROM pcca_sequential_ablation.py)
# =============================================================================
def _zscore_flat(
    X: np.ndarray,
    *,
    subtract_psth: bool = False,
    shuffle_trials: bool = False,
    rng: Optional[np.random.Generator] = None,
    perm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Z-score per neuron, optionally subtract PSTH and/or shuffle trials.

    Flat layout of the returned matrix:  row = t * n_trials + trial  (time outer).
    PSTH subtraction (axis=0 = trials) yields the residual δ with exact zero
    trial-mean at every (neuron, time).
    """
    n_trials, n, T = X.shape
    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
    flat = zscore(flat, axis=1, nan_policy="omit")
    np.nan_to_num(flat, nan=0.0, copy=False)

    X = flat.reshape(n, T, n_trials).transpose(2, 0, 1)

    if subtract_psth:
        X = X - X.mean(axis=0, keepdims=True)

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

    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
    return flat.T


def _zscore_3d(
    X: np.ndarray,
    *,
    subtract_psth: bool = False,
    shuffle_trials: bool = False,
    rng: Optional[np.random.Generator] = None,
    perm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Same transform as _zscore_flat but returned as (n_trials, n, T)."""
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
    q = Y.shape[1]
    k = min(n_components, p, q, n - 1)

    Cxx = X.T @ X / (n - 1)
    Cyy = Y.T @ Y / (n - 1)
    Cxy = X.T @ Y / (n - 1)

    A = _ridge_inv_sqrt(Cxx, lam)
    B = _ridge_inv_sqrt(Cyy, lam)

    U, S, Vt = np.linalg.svd(A @ Cxy @ B, full_matrices=False)
    k = min(k, len(S))

    Wx = A @ U[:, :k]
    Wy = B @ Vt[:k].T
    rho = np.clip(S[:k], 0.0, 1.0)
    return Wx, Wy, rho


def residualize(
    X_flat: np.ndarray,
    Z_flat: Optional[np.ndarray],
    lam_hat: float = LAMBDA_HAT,
) -> np.ndarray:
    """Ridge-regularised nuisance regression: X − Z (Z'Z + λnI)⁻¹ Z' X."""
    if Z_flat is None or Z_flat.ndim < 2 or Z_flat.shape[1] == 0:
        return X_flat.copy()
    n, m = Z_flat.shape
    ZtZ = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    Beta = np.linalg.solve(ZtZ, Z_flat.T)
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


def _trial_rows(trial_idx: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """Flat-matrix row indices for the given trials (row = t*n_trials + trial)."""
    return (trial_idx[None, :] + np.arange(T)[:, None] * n_trials).ravel()


# =============================================================================
# 2.  Data loading  (COPIED VERBATIM)
# =============================================================================

def load_region_spikes(
    session_path: str,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    data = mat73.loadmat(session_path)
    rd = data.get("region_data", {})
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
# 3.  Neuron ordering  (COPIED VERBATIM — global Rastermap on RAW spike counts)
# =============================================================================

def get_neuron_order(X: np.ndarray) -> np.ndarray:
    """Peak-time / single-region Rastermap fallback (used internally)."""
    n_neurons = X.shape[1]
    if _RASTERMAP_OK and n_neurons >= 5:
        try:
            mat = X.transpose(1, 2, 0).reshape(n_neurons, -1).astype(np.float64)
            mat = zscore(mat, axis=1, nan_policy="omit")
            np.nan_to_num(mat, nan=0.0, copy=False)
            mdl = Rastermap(n_PCs=min(50, n_neurons), locality=0.0, grid_upsample=5)
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
    """Fit one Rastermap on concatenated RAW (z-scored, not residualised) activity
    of all regions, then extract per-region sort indices from the global embedding."""
    flat_rows: List[np.ndarray] = []
    offsets: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for rname in all_region_names:
        X = region_spikes[rname]
        n_neurons = X.shape[1]
        flat_r = _zscore_flat(X).T          # raw z-scored (no PSTH subtraction)
        flat_rows.append(flat_r)
        offsets[rname] = (cursor, cursor + n_neurons)
        cursor += n_neurons

    combined = np.concatenate(flat_rows, axis=0)
    total_n = combined.shape[0]

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
            global_isort = mdl.isort
            print(
                f"  [global Rastermap]  fit on {total_n} neurons "
                f"({len(all_region_names)} regions)"
            )
        except Exception as exc:
            warnings.warn(
                f"Global Rastermap failed ({exc}); "
                "falling back to per-region peak-time ordering."
            )

    per_region_order: Dict[str, np.ndarray] = {}
    for rname in all_region_names:
        start, end = offsets[rname]
        if global_isort is not None:
            mask = (global_isort >= start) & (global_isort < end)
            per_region_order[rname] = global_isort[mask] - start
        else:
            per_region_order[rname] = get_neuron_order(region_spikes[rname])

    return per_region_order


# =============================================================================
# 4.  Residual-regime helpers
# =============================================================================

def _flat_to_3d(flat: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """Inverse of _zscore_flat's flattening: (T*n_trials, n) → (n_trials, n, T)."""
    n = flat.shape[1]
    return flat.reshape(T, n_trials, n).transpose(1, 2, 0)


def _heldout_rho(
    Xi: np.ndarray,
    Xj: np.ndarray,
    Z: Optional[np.ndarray],
    n_trials: int,
    T: int,
    lam_cca: float = LAMBDA_CCA,
    lam_hat: float = LAMBDA_HAT,
) -> float:
    """Split-half held-out ρ₁.

    Canonical weights (and the nuisance β) are estimated on the first half of
    TRIALS only and applied to the held-out half — preventing leakage of the
    nuisance-removal step and exposing the in-sample inflation that residual
    CCA/pCCA is prone to.
    """
    half = n_trials // 2
    if half < 2:
        return float("nan")
    tr = np.arange(half)
    te = np.arange(half, n_trials)
    rtr = _trial_rows(tr, n_trials, T)
    rte = _trial_rows(te, n_trials, T)

    Xi_tr, Xj_tr = Xi[rtr], Xj[rtr]
    Xi_te, Xj_te = Xi[rte], Xj[rte]

    if Z is not None and Z.shape[1] > 0:
        Z_tr, Z_te = Z[rtr], Z[rte]
        n_tr = len(rtr)
        ZtZ = Z_tr.T @ Z_tr + lam_hat * n_tr * np.eye(Z_tr.shape[1])
        Bi = np.linalg.solve(ZtZ, Z_tr.T @ Xi_tr)
        Bj = np.linalg.solve(ZtZ, Z_tr.T @ Xj_tr)
        Xi_tr = Xi_tr - Z_tr @ Bi
        Xj_tr = Xj_tr - Z_tr @ Bj
        Xi_te = Xi_te - Z_te @ Bi
        Xj_te = Xj_te - Z_te @ Bj

    Wx, Wy, _ = ridge_cca(Xi_tr, Xj_tr, lam_cca, n_components=1)
    zi = Xi_te @ Wx[:, 0]
    zj = Xj_te @ Wy[:, 0]
    if np.std(zi) < 1e-9 or np.std(zj) < 1e-9:
        return 0.0
    return float(np.clip(pearsonr(zi, zj)[0], -1.0, 1.0))


@dataclass
class ResidualFit:
    """Bundle of one residual CCA or pCCA solution (dominant component used)."""
    estimator: str                 # 'CCA' or 'pCCA'
    Wx: np.ndarray                 # (n_i, K)
    Wy: np.ndarray                 # (n_j, K)
    rho: np.ndarray                # (K,)  in-sample canonical correlations
    rho_heldout: float             # split-half held-out ρ₁
    z_i: np.ndarray                # (T*n_trials,) dominant latent, region I
    z_j: np.ndarray                # (T*n_trials,) dominant latent, region J
    Xi_res_3d: np.ndarray = field(repr=False)  # (n_trials, n_i, T) residual w acts on
    Xj_res_3d: np.ndarray = field(repr=False)  # (n_trials, n_j, T)


def fit_residual(
    Xi_flat: np.ndarray,
    Xj_flat: np.ndarray,
    Z_flat: Optional[np.ndarray],
    estimator: str,
    n_trials: int,
    T: int,
    n_components: int = N_COMPONENTS,
) -> ResidualFit:
    """Fit one residual estimator.

    Z_flat is None  → residual CCA (the weight acts on δ itself).
    Z_flat provided → residual pCCA (the weight acts on δ after nuisance removal,
                      i.e. the X_res returned by `pcca`).
    """
    if Z_flat is None or Z_flat.shape[1] == 0:
        Wx, Wy, rho = ridge_cca(Xi_flat, Xj_flat, n_components=n_components)
        Xi_res, Xj_res = Xi_flat, Xj_flat
    else:
        Wx, Wy, rho, Xi_res, Xj_res = pcca(
            Xi_flat, Xj_flat, Z_flat, n_components=n_components
        )
    return ResidualFit(
        estimator=estimator,
        Wx=Wx, Wy=Wy, rho=rho,
        rho_heldout=_heldout_rho(Xi_flat, Xj_flat, Z_flat, n_trials, T),
        z_i=Xi_res @ Wx[:, 0],
        z_j=Xj_res @ Wy[:, 0],
        Xi_res_3d=_flat_to_3d(Xi_res, n_trials, T),
        Xj_res_3d=_flat_to_3d(Xj_res, n_trials, T),
    )


# =============================================================================
# 5.  Figure primitives
# =============================================================================

def _weight_barh(
    ax: plt.Axes,
    w: np.ndarray,
    sort_idx: np.ndarray,
    top_k: int,
    title: str,
    pos_color: str = _C_POS,
    neg_color: str = _C_NEG,
    n_show: int = N_NEURONS_SHOW,
) -> None:
    """Signed canonical-weight bars in global-Rastermap order; top-|w| neurons boxed."""
    n = len(w)
    top = list(np.argsort(np.abs(w))[::-1][:top_k])

    if n <= n_show:
        sel = list(sort_idx)
    else:
        step = max(1, n // n_show)
        strided = list(sort_idx[::step][:n_show])
        sel_set = set(strided) | set(top)
        pos = {int(idx): r for r, idx in enumerate(sort_idx)}
        sel = sorted(sel_set, key=lambda i: pos.get(int(i), 1 << 30))

    ws = w[sel]
    ypos = np.arange(len(sel)) + 0.5
    colors = [pos_color if v >= 0 else neg_color for v in ws]
    ax.barh(ypos, ws, height=0.82, color=colors, alpha=0.83)

    top_set = set(int(t) for t in top)
    for i, idx in enumerate(sel):
        if int(idx) in top_set:
            ax.barh(ypos[i], ws[i], height=0.82, color="none",
                    edgecolor="k", lw=1.4, zorder=5)

    ax.axvline(0.0, color="k", lw=0.7, alpha=0.45)
    ax.set_ylim(len(sel), 0)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)
    plt.setp(ax.get_yticklabels(), visible=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def _latent_heatmap(
    ax: plt.Axes,
    Z_tt: np.ndarray,
    time_vec: np.ndarray,
    title: str,
    color_label: str,
    vmax: Optional[float] = None,
) -> None:
    """Per-trial latent z (trials × time): the object that survives in the residual
    regime — its trial-MEAN is ≈ 0, but its trial-by-trial structure is not."""
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(Z_tt), 99)) or 1.0
    im = ax.imshow(
        Z_tt, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=[time_vec[0], time_vec[-1], Z_tt.shape[0], 0], origin="upper",
    )
    ax.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Trial", fontsize=8)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(color_label, fontsize=6)
    cbar.ax.tick_params(labelsize=5)


# =============================================================================
# 6.  Overview figure (per estimator)
# =============================================================================

def fig_residual_overview(
    fit: ResidualFit,
    sort_i: np.ndarray,
    sort_j: np.ndarray,
    time_vec: np.ndarray,
    n_trials: int,
    T: int,
    session: str,
    z_name: str,
    top_k: int = TOP_K_NEURONS,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Weights + per-trial latent heatmaps + joint (t,s) scatter for one estimator."""
    wi, wj = fit.Wx[:, 0], fit.Wy[:, 0]
    zi, zj = fit.z_i, fit.z_j

    Zi_tt = zi.reshape(T, n_trials).T
    Zj_tt = zj.reshape(T, n_trials).T
    t_secs = time_vec[(np.arange(zi.size) // n_trials)]

    mosaic = [["wi", "hi", "sc"],
              ["wj", "hj", "sc"]]
    fig, axd = plt.subplot_mosaic(
        mosaic, figsize=(17.5, 8.5),
        gridspec_kw=dict(width_ratios=[1.0, 2.3, 2.7], hspace=0.42, wspace=0.34),
    )

    _weight_barh(axd["wi"], wi, sort_i, top_k,
                 f"{TARGET_I}  residual {fit.estimator} weight\n(top {top_k} boxed)")
    _weight_barh(axd["wj"], wj, sort_j, top_k,
                 f"{TARGET_J}  residual {fit.estimator} weight\n(top {top_k} boxed)")

    _latent_heatmap(axd["hi"], Zi_tt, time_vec,
                    f"{TARGET_I}  per-trial latent z(t,s)", "z (a.u.)")
    _latent_heatmap(axd["hj"], Zj_tt, time_vec,
                    f"{TARGET_J}  per-trial latent z(t,s)", "z (a.u.)")

    ax = axd["sc"]
    sc = ax.scatter(zi, zj, c=t_secs, cmap="viridis", s=4, alpha=0.30,
                    rasterized=True)
    b1, b0 = np.polyfit(zi, zj, 1)
    xs = np.array([zi.min(), zi.max()])
    ax.plot(xs, b0 + b1 * xs, "k-", lw=1.6, zorder=4)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlabel(f"$z_{{{TARGET_I}}}$  (residual latent)", fontsize=9)
    ax.set_ylabel(f"$z_{{{TARGET_J}}}$  (residual latent)", fontsize=9)
    ax.set_title(
        f"{fit.estimator} residual coupling — joint (trial × time) samples\n"
        f"ρ₁ in-sample = {fit.rho[0]:.3f}    |    "
        f"ρ₁ held-out = {fit.rho_heldout:.3f}",
        fontsize=9, fontweight="bold",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label("time in trial (s)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle(
        f"Residual {fit.estimator}  |  {session}  |  {TARGET_I} ↔ {TARGET_J}"
        f"   (Z = {{{z_name}}};  SUBTRACT_PSTH={SUBTRACT_PSTH}, "
        f"SHUFFLE_TRIALS={SHUFFLE_TRIALS})",
        fontsize=11, fontweight="bold", y=1.02,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [overview] saved: {output_path}")
    return fig


# =============================================================================
# 7.  KEY component — high-|weight| neuron residual vs trial-averaged rate
# =============================================================================

def fig_highweight_neuron_residuals(
    *,
    residual_3d: np.ndarray,        # (n_trials, n, T) — residual the weight acts on
    raw_z_3d: np.ndarray,           # (n_trials, n, T) — z-scored single-trial activity
    psth_z: np.ndarray,             # (n, T) — trial-averaged z-scored rate (μ_z)
    weights: np.ndarray,            # (n,) — dominant residual canonical weight
    region_name: str,
    estimator: str,
    color: str,
    time_vec: np.ndarray,
    residual_label: str = "residual δ",
    top_k: int = TOP_K_NEURONS,
    session: str = "",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """For each top-|w| neuron: trial-averaged rate beside its per-trial residuals.

    col 0  trial-averaged rate μ(n,s) (bold) over faint single-trial raw z(t,n,s).
    col 1  per-trial residual traces δ(t,n,s) (faint) + their trial-mean (≈ 0).
    col 2  residual heatmap (trials × time), RdBu_r.

    cols 0 & 1 share a symmetric y-scale; SNR = var_s(μ)/var(δ) is annotated (small
    SNR ⇒ noise-dominated ⇒ the canonical weight is carried by the residual).
    """
    n_trials = residual_3d.shape[0]
    order = np.argsort(np.abs(weights))[::-1][:top_k]

    rvmax = float(np.nanpercentile(np.abs(residual_3d), 98)) or 1.0

    fig, axes = plt.subplots(
        top_k, 3, figsize=(15.5, 2.6 * top_k),
        gridspec_kw=dict(hspace=0.62, wspace=0.26,
                         width_ratios=[1.45, 1.45, 1.25]),
    )
    if top_k == 1:
        axes = axes[None, :]

    for row, nidx in enumerate(order):
        w = float(weights[nidx])
        mu = psth_z[nidx]
        resid = residual_3d[:, nidx]
        raw = raw_z_3d[:, nidx]

        snr = float(np.var(mu)) / (float(np.var(resid)) + 1e-12)
        amp = max(
            float(np.nanpercentile(np.abs(raw), 99)),
            float(np.nanpercentile(np.abs(resid), 99)),
            float(np.nanmax(np.abs(mu))),
            0.5,
        )

        ax0 = axes[row, 0]
        for tr in range(n_trials):
            ax0.plot(time_vec, raw[tr], color=color, lw=0.25, alpha=0.05,
                     rasterized=True)
        ax0.plot(time_vec, mu, color=color, lw=2.6, zorder=4)
        ax0.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
        ax0.axhline(0.0, color="gray", lw=0.5, alpha=0.4)
        ax0.set_ylim(-amp, amp)
        ax0.set_ylabel("z-scored rate", fontsize=8)
        ax0.set_title(
            f"neuron {nidx}   w = {w:+.3f}   (rank {row + 1})\n"
            f"trial-avg μ(s)  [bold]  +  single trials   |   SNR = {snr:.2f}",
            fontsize=8,
        )

        ax1 = axes[row, 1]
        for tr in range(n_trials):
            ax1.plot(time_vec, resid[tr], color=color, lw=0.3, alpha=0.08,
                     rasterized=True)
        ax1.plot(time_vec, resid.mean(axis=0), color="k", ls="--", lw=1.7,
                 zorder=4, label="trial-mean ≈ 0")
        ax1.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
        ax1.axhline(0.0, color="gray", lw=0.5, alpha=0.4)
        ax1.set_ylim(-amp, amp)
        ax1.legend(fontsize=6, frameon=False, loc="upper right")
        ax1.set_title(
            f"per-trial {residual_label}  [faint]\nmean collapses to ≈ 0",
            fontsize=8,
        )

        ax2 = axes[row, 2]
        im = ax2.imshow(
            resid, aspect="auto", cmap="RdBu_r", vmin=-rvmax, vmax=rvmax,
            extent=[time_vec[0], time_vec[-1], n_trials, 0], origin="upper",
        )
        ax2.axvline(0.0, color="k", ls="--", lw=0.7, alpha=0.6)
        ax2.set_ylabel("Trial", fontsize=8)
        ax2.set_title(f"{residual_label}  (trials × time)", fontsize=8)
        cbar = fig.colorbar(im, ax=ax2, pad=0.02, shrink=0.85)
        cbar.ax.tick_params(labelsize=5)

        for a in (ax0, ax1):
            a.tick_params(labelsize=6)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        ax2.tick_params(labelsize=6)

        if row == top_k - 1:
            for a in (ax0, ax1, ax2):
                a.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(
        f"{region_name} — top {top_k} residual {estimator} neurons   |   {session}\n"
        f"trial-averaged rate (stimulus-locked)  vs  per-trial residuals "
        f"(trial-varying coupling substrate)",
        fontsize=11, fontweight="bold", y=1.005,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [neuron] saved: {output_path}")
    return fig


# =============================================================================
# 8.  Session driver — CCA once, then single-region ablation over ALL regions
# =============================================================================

def _emit_neuron_figs(
    fit: ResidualFit,
    Xi_z3d: np.ndarray,
    Xj_z3d: np.ndarray,
    psth_i: np.ndarray,
    psth_j: np.ndarray,
    time_vec: np.ndarray,
    out_dir: Path,
    session_name: str,
    tag: str,
    residual_label: str,
    top_k: int,
) -> None:
    """Emit the two high-weight-neuron figures (region I and region J) for `fit`."""
    plt.close(fig_highweight_neuron_residuals(
        residual_3d=fit.Xi_res_3d, raw_z_3d=Xi_z3d, psth_z=psth_i,
        weights=fit.Wx[:, 0], region_name=TARGET_I, estimator=fit.estimator,
        color=_C_MOS, time_vec=time_vec, residual_label=residual_label,
        top_k=top_k, session=session_name,
        output_path=out_dir / f"{session_name}_neurons_{TARGET_I}_{tag}.png",
    ))
    plt.close(fig_highweight_neuron_residuals(
        residual_3d=fit.Xj_res_3d, raw_z_3d=Xj_z3d, psth_z=psth_j,
        weights=fit.Wy[:, 0], region_name=TARGET_J, estimator=fit.estimator,
        color=_C_VPM, time_vec=time_vec, residual_label=residual_label,
        top_k=top_k, session=session_name,
        output_path=out_dir / f"{session_name}_neurons_{TARGET_J}_{tag}.png",
    ))


def run_session(
    session_name: str,
    base_dir: Path,
    top_k: int = TOP_K_NEURONS,
) -> None:
    session_file = (
        base_dir / "pcca_sessions_cued_hit_long_results"
        / f"{session_name}_analysis_results.mat"
    )
    if not session_file.exists():
        print(f"❌ [WARNING] File not found, skipping: {session_file}")
        return

    region_spikes, n_trials, T = load_region_spikes(str(session_file))
    for t in (TARGET_I, TARGET_J):
        if t not in region_spikes:
            raise RuntimeError(f"Target region '{t}' not in session {session_name}.")

    nuisance_all = [
        r for r in ANATOMICAL_ORDER
        if r in region_spikes and r not in (TARGET_I, TARGET_J)
    ]
    if NUISANCE_REGIONS_OVERRIDE is not None:
        nuisance_all = [r for r in NUISANCE_REGIONS_OVERRIDE if r in nuisance_all]
    if not nuisance_all:
        warnings.warn(f"[{session_name}] No nuisance regions — only CCA will be run.")

    all_regions = [r for r in ANATOMICAL_ORDER if r in region_spikes]
    global_sort = compute_global_neuron_order(region_spikes, all_regions)
    sort_i, sort_j = global_sort[TARGET_I], global_sort[TARGET_J]
    time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)

    Xi_raw, Xj_raw = region_spikes[TARGET_I], region_spikes[TARGET_J]

    # residual flats (PSTH-subtracted, NOT shuffled in this regime)
    Xi_flat = _zscore_flat(Xi_raw, subtract_psth=SUBTRACT_PSTH,
                           shuffle_trials=SHUFFLE_TRIALS)
    Xj_flat = _zscore_flat(Xj_raw, subtract_psth=SUBTRACT_PSTH,
                           shuffle_trials=SHUFFLE_TRIALS)

    # raw z-scored tensors for the per-neuron displays
    Xi_z3d = _zscore_3d(Xi_raw, subtract_psth=False, shuffle_trials=False)
    Xj_z3d = _zscore_3d(Xj_raw, subtract_psth=False, shuffle_trials=False)
    psth_i = Xi_z3d.mean(axis=0)
    psth_j = Xj_z3d.mean(axis=0)

    out_base = (
        base_dir / "Paper_output" / "residual_cca_pcca" / session_name
        / f"{TARGET_I}_{TARGET_J}"
    )
    out_base.mkdir(parents=True, exist_ok=True)

    # ── Residual CCA (Z = ∅) — region-independent, fit & rendered ONCE ──────
    cca = fit_residual(Xi_flat, Xj_flat, None, "CCA", n_trials, T)
    print(f"  [residual CCA ]  ρ₁ in-sample = {cca.rho[0]:.4f}   "
          f"held-out = {cca.rho_heldout:.4f}")
    cca_dir = out_base / "CCA_baseline"
    cca_dir.mkdir(parents=True, exist_ok=True)
    plt.close(fig_residual_overview(
        cca, sort_i, sort_j, time_vec, n_trials, T, session_name, "∅",
        top_k=top_k, output_path=cca_dir / f"{session_name}_overview_CCA.png",
    ))
    _emit_neuron_figs(cca, Xi_z3d, Xj_z3d, psth_i, psth_j, time_vec,
                      cca_dir, session_name, tag="CCA",
                      residual_label="residual δ", top_k=top_k)

    # ── Residual pCCA — single-region ablation across EVERY nuisance region ──
    print(f"  [residual pCCA]  ablating {len(nuisance_all)} regions: {nuisance_all}")
    rho_summary: List[Tuple[str, float, float]] = [
        ("CCA (Z=∅)", float(cca.rho[0]), cca.rho_heldout)
    ]
    for region in nuisance_all:
        Z_r = _zscore_flat(region_spikes[region], subtract_psth=SUBTRACT_PSTH)
        pf = fit_residual(Xi_flat, Xj_flat, Z_r, "pCCA", n_trials, T)
        print(f"    Z = {{{region:<8}}}  ρ₁ in-sample = {pf.rho[0]:.4f}   "
              f"held-out = {pf.rho_heldout:.4f}   "
              f"Δρ = {pf.rho[0] - cca.rho[0]:+.4f}")
        rho_summary.append((f"pCCA|{region}", float(pf.rho[0]), pf.rho_heldout))

        reg_dir = out_base / f"ablation_{region}"
        reg_dir.mkdir(parents=True, exist_ok=True)
        plt.close(fig_residual_overview(
            pf, sort_i, sort_j, time_vec, n_trials, T, session_name, region,
            top_k=top_k,
            output_path=reg_dir / f"{session_name}_overview_pCCA_{region}.png",
        ))
        _emit_neuron_figs(pf, Xi_z3d, Xj_z3d, psth_i, psth_j, time_vec,
                          reg_dir, session_name, tag=f"pCCA_{region}",
                          residual_label=f"residual δ | {region}", top_k=top_k)

    # ── Console summary of ρ₁ across the ablation series ─────────────────────
    print(f"\n  [{session_name}]  residual coupling ρ₁ summary "
          f"(in-sample | held-out):")
    for label, ri, rh in rho_summary:
        print(f"      {label:<16}  {ri:6.3f}  |  {rh:6.3f}")
    print(f"✨ Session {session_name} done → {out_base}")


# =============================================================================
# 9.  Entry point
# =============================================================================

SESSIONS_TO_RUN = [
    "yp020_220331",
    # 'yp020_220401', 'yp021_220331', 'yp021_220402',
    # 'yp021_220404', 'yp021_220405', 'yp021_220407',
]


def main() -> None:
    base_dir = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    total = len(SESSIONS_TO_RUN)
    for idx, session in enumerate(SESSIONS_TO_RUN, 1):
        print(f"\n🚀 [Processing {idx}/{total}] {session}  "
              f"(residual {TARGET_I} ↔ {TARGET_J}; single-region ablation over all "
              f"regions)")
        try:
            run_session(session, base_dir)
        except Exception as exc:                      # noqa: BLE001
            print(f"💥 [ERROR] {session}: {exc}")
    print("\n🎉 All sessions completed!")


if __name__ == "__main__":
    main()
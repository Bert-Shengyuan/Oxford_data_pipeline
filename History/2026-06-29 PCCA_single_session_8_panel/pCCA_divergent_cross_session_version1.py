"""
pcca_cross_session_ablation.py
==============================

Batch cross-session pCCA single-region ablation pipeline for the
MOs ↔ VPMPO target pair.

Pipeline overview
-----------------
For each session in SESSIONS_TO_RUN:
  1.  Load region spikes and z-score to flat matrices.
  2.  Compute the CCA baseline (Z = ∅) and, for each nuisance region r
      present in that session, pCCA(MOs, VPMPO | Z = {r}).
  3.  Compute StepResult and SupplementaryMetrics for every ablated region.
  4.  Save per-session plot_summary_figure  →  pcca_cross_session_summary/
  5.  Save per-session plot_supplementary_panel → pcca_cross_session_supplementary/
  6.  Cache all numeric metrics in a SessionAblationData object.

After all sessions:
  7.  Aggregate cached data by ablated region (union across sessions).
  8.  Cross-session summary figure (panels B, C, D reproduced as boxplots;
      panel A omitted) → pcca_cross_session_summary/cross_session_summary.png
  9.  Cross-session supplementary figure (panels A–E reproduced as boxplots /
      stacked bars; panel F omitted)
      → pcca_cross_session_supplementary/cross_session_supplementary.png

Output paths
------------
  BASE_DIR / "Paper_output" / "pcca_cross_session_summary"
      {session}_part1_summary_vs_CCA.png          (per-session)
      cross_session_summary_BCD.png               (cross-session)

  BASE_DIR / "Paper_output" / "pcca_cross_session_supplementary"
      {session}_part1_supplementary.png           (per-session)
      cross_session_supplementary_ABCDE.png       (cross-session)

Variance-partition visualisation (cross-session panels D & E)
-------------------------------------------------------------
For each ablated region r on the x-axis, a stacked horizontal bar encodes
the mean (across sessions that recorded r) of:

    ■ grey  (#5D6D7E)  r²_nuis  — variance of MOs (or VPMPO) removed by
                                   regressing out Z = {r}.
    ■ red / blue       r²_comm  — variance of the *residual* captured by
                                   the first pCCA communication axis w̃₁.

An error bar at the total stack height (r²_nuis + r²_comm) shows ±1 SEM
across sessions.  Individual session values are overlaid as jittered dots
at the total height so that between-session variability is transparent.

This is the natural cross-session generalisation of the per-session stacked
bar in plot_supplementary_panel panels D/E: it preserves the decomposition
logic while making variability across animals visually explicit.

Notes
-----
* r²_priv_j bug fix: in the original pcca_sequential_ablation.py the
  denominator for r²_priv_j mistakenly used denom_i; corrected here to
  denom_j throughout.
* All sign conventions, ridge parameters, and CV fold logic are identical
  to pcca_sequential_ablation.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import pearsonr, zscore

try:
    import mat73
except ImportError as exc:
    raise SystemExit("mat73 is required: pip install mat73") from exc

from Useful_definition import (
    ANATOMICAL_ORDER,
    apply_latent_sign_correction,
    safe_array,
)

# =============================================================================
# 0.  Global configuration
# =============================================================================
#
# TARGET_I     = "MOs"
# TARGET_J     = "MOp"
# SESSIONS_TO_RUN: List[str] = [
#     "yp012_220208",
#     "yp013_220209",
#     "yp013_220211",
#     "yp013_220212",
#     "yp020_220401",
#     "yp020_220407",
#     "yp021_220331",
#     "yp021_220401",
#     "yp021_220402",
#     "yp021_220403",
#     "yp021_220404",
#     "yp021_220407",
# ]

TARGET_I     = "MOs"
TARGET_J     = "VPMPO"

SESSIONS_TO_RUN: List[str] = [
    "yp020_220331",

    "yp020_220401",

    "yp021_220331",

    "yp021_220402",

    "yp021_220403",

    "yp021_220404",

    "yp021_220405",

    "yp021_220407",

]


LAMBDA_CCA   = 1e-4   # ridge on Cxx / Cyy whitening
LAMBDA_HAT   = 1e-4   # ridge on Z'Z in hat-matrix (scaled by n inside residualize)
N_COMPONENTS = 5
TIME_RANGE_S = (-1.5, 3.0)

# ── Colour palette (mirrors pcca_sequential_ablation.py) ──────────────────
_CI_PCCA    = "#C0392B"   # MOs  pCCA latent / weight bar
_CJ_PCCA    = "#2471A3"   # VPMPO pCCA latent / weight bar
_C_NUIS     = "#5D6D7E"   # nuisance variance component (stacked bar)
_C_DIV      = "#E74C3C"   # divergence bars — most-divergent highlight
_C_SIM      = "#27AE60"   # divergence bars — most-similar highlight
_C_NEUTRAL  = "#CACFD2"   # divergence bars — neutral


# =============================================================================
# 1.  Core mathematics
# =============================================================================

def _zscore_flat(X: np.ndarray) -> np.ndarray:
    """Flatten (n_trials, n, T) → (T·n_trials, n) with z-scoring across time."""
    n_trials, n, T = X.shape
    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
    flat = zscore(flat, axis=1, nan_policy="omit")
    np.nan_to_num(flat, nan=0.0, copy=False)
    return flat.T  # (T·n_trials, n)


def _ridge_inv_sqrt(C: np.ndarray, lam: float) -> np.ndarray:
    """Regularised inverse square-root of a symmetric PSD matrix."""
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
    X : (N, p)  centred data matrix.
    Y : (N, q)  centred data matrix.
    lam : ridge coefficient added to the diagonal of Cxx and Cyy.

    Returns
    -------
    Wx  : (p, k)  canonical weight matrix for X.
    Wy  : (q, k)  canonical weight matrix for Y.
    rho : (k,)    canonical correlations, clipped to [0, 1].
    """
    n, p = X.shape
    q    = Y.shape[1]
    k    = min(n_components, p, q, n - 1)
    Cxx  = X.T @ X / (n - 1)
    Cyy  = Y.T @ Y / (n - 1)
    Cxy  = X.T @ Y / (n - 1)
    A    = _ridge_inv_sqrt(Cxx, lam)
    B    = _ridge_inv_sqrt(Cyy, lam)
    U, S, Vt = np.linalg.svd(A @ Cxy @ B, full_matrices=False)
    k    = min(k, len(S))
    return A @ U[:, :k], B @ Vt[:k].T, np.clip(S[:k], 0.0, 1.0)


def residualize(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_hat: float = LAMBDA_HAT,
) -> np.ndarray:
    """
    Partial out Z from X via ridge-regularised OLS.

    The hat matrix is  H = Z (Z'Z + λn I)⁻¹ Z'.
    The residual is    X̃ = (I − H) X.

    Returns a copy of X_flat when Z_flat is None or empty.
    """
    if Z_flat is None or Z_flat.ndim < 2 or Z_flat.shape[1] == 0:
        return X_flat.copy()
    n, m  = Z_flat.shape
    ZtZ   = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    Beta  = np.linalg.solve(ZtZ, Z_flat.T)  # (m, N)
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
    Partial CCA: residualise X and Y with respect to Z, then CCA.

    Returns
    -------
    Wx_p, Wy_p : canonical weight matrices on the residualised data.
    rho_p      : canonical correlations.
    X_res, Y_res : residualised flat matrices (used for latent projections
                   and variance decomposition).
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
    """Project X onto w and reshape to (n_trials, T)."""
    return (X_flat @ w).reshape(T, n_trials).T


def _cos_sim_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute cosine similarity |cos ∠(a, b)|."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.abs(np.dot(a, b)) / (na * nb))


def _trial_rows(trial_idx: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """
    Row indices in the flat matrix  (layout: row = t · n_trials + trial)
    for all T time-points of the requested trial indices.
    """
    return (trial_idx[None, :] + np.arange(T)[:, None] * n_trials).ravel()


# =============================================================================
# 2.  Data loading
# =============================================================================

def load_region_spikes(
        session_path: str,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """
    Load per-region spike tensors from a MATLAB v7.3 results file.

    Returns
    -------
    region_spikes : {region_name: ndarray (n_trials, n_neurons, T)}
    n_trials      : number of trials (common across regions).
    T             : number of time bins per trial.
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
        f"  [load]  {len(region_spikes)} regions  "
        f"n_trials={n_trials_out}  T={T_out}"
    )
    return region_spikes, int(n_trials_out), int(T_out)


# =============================================================================
# 3.  Data containers
# =============================================================================

class StepResult:
    """Minimal per-ablation-step container (canonical weights + latent means)."""

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


@dataclass
class SupplementaryMetrics:
    """
    Seven-metric diagnostic bundle for one ablation step.

    Fields (all angles in degrees; variance fractions relative to ‖X‖²_F)
    -----------------------------------------------------------------------
    theta_{i,j}_deg
        ∠(w_CCA, w_pCCA) in degrees.  θ → 90° signals that nuisance removal
        uncovered a genuinely orthogonal communication subspace.

    kappa_{i,j}
        Cross-analysis collinearity |cos ∠(w_{IJ,I}, w_{IZ,I})|.
        κ → 0: clean axis separation; κ → 1: Type-II collapse.
        NaN when Z_flat is None (step 0 baseline).

    rho1_cv_mean, rho1_cv_sem
        5-fold cross-validated ρ₁: regression coefficients estimated on
        training folds only, applied to held-out test residuals.

    r2_nuis_{i,j}   ‖X̂‖²_F / ‖X‖²_F   (variance removed by Z regression)
    r2_comm_{i,j}   ‖X̃ ŵ₁‖² / ‖X‖²_F  (pCCA communication axis variance)
    r2_priv_{i,j}   ‖X̃ (I − ŵ₁ŵ₁ᵀ)‖²_F / ‖X‖²_F  (private/unexplained)

    lag_peak_ms, lag_corr_at_peak
        Peak of the normalised cross-correlation of the sign-corrected mean
        pCCA latents, restricted to ±150 ms.
    """
    step_label:       str
    theta_i_deg:      float
    theta_j_deg:      float
    kappa_i:          float
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


@dataclass
class SessionAblationData:
    """
    Per-session ablation cache consumed by the cross-session aggregator.

    All lists share a common index: element k corresponds to ablated
    region ``region_names[k]``.

    Fields
    ------
    session_name  : identifier string, e.g. ``'yp020_220331'``.
    rho_cca       : CCA baseline ρ₁ (Z = ∅) for this session.
    region_names  : ablated regions, sorted by ANATOMICAL_ORDER.
    step_results  : ``StepResult`` for pCCA(I, J | Z = {r}).
    sims_vs_cca   : ``compute_similarity(step, cca_baseline)`` per region.
    supp_metrics  : ``SupplementaryMetrics`` per region.
    """
    session_name:  str
    rho_cca:       float
    region_names:  List[str]
    step_results:  List[StepResult]
    sims_vs_cca:   List[Dict[str, float]]
    supp_metrics:  List[SupplementaryMetrics]


# =============================================================================
# 4.  Similarity metrics
# =============================================================================

def compute_similarity(s: StepResult, ref: StepResult) -> Dict[str, float]:
    """
    Compute pairwise similarity between step s and a reference solution.

    The composite divergence D ∈ [0, 4] sums four orthogonal penalty terms:

        D = (1 − |cos θ_i|) + (1 − |cos θ_j|)
            + (1 − max(r_i, −1)) + (1 − max(r_j, −1))

    where θ_k is the angle between the first canonical weight vectors and
    r_k is the Pearson correlation of the mean latent time-series.
    D = 0 iff the two solutions are identical; D = 4 iff they are
    anti-aligned in every metric simultaneously.
    """
    cos_i = _cos_sim_abs(s.Wx[:, 0], ref.Wx[:, 0])
    cos_j = _cos_sim_abs(s.Wy[:, 0], ref.Wy[:, 0])
    r_i   = float(pearsonr(s.z_i_mean, ref.z_i_mean)[0])
    r_j   = float(pearsonr(s.z_j_mean, ref.z_j_mean)[0])
    return dict(
        cos_sim_i    = cos_i,
        cos_sim_j    = cos_j,
        latent_r_i   = r_i,
        latent_r_j   = r_j,
        rho_abs_diff = abs(s.rho_pcca - ref.rho_pcca),
        divergence   = (
            (1 - cos_i) + (1 - cos_j)
            + (1 - max(r_i, -1.0)) + (1 - max(r_j, -1.0))
        ),
    )


def identify_extremes(
        step_results: List[StepResult],
        ref: StepResult,
        exclude_ref: bool = True,
) -> Tuple[int, int, List[Dict[str, float]]]:
    sims  = [compute_similarity(s, ref) for s in step_results]
    cands = list(
        range(len(step_results) - 1) if exclude_ref
        else range(len(step_results))
    )
    if not cands:
        return 0, 0, sims
    divs = [sims[i]["divergence"] for i in cands]
    return cands[int(np.argmin(divs))], cands[int(np.argmax(divs))], sims


def print_similarity_table(
        step_results: List[StepResult],
        sims: List[Dict[str, float]],
        ref_label: str,
        idx_closest: int,
        idx_furthest: int,
) -> None:
    hdr = (
        f"\n{'Step':<35}  {'ρ₁':>6}  {'|cosθ|_i':>8}  {'|cosθ|_j':>8}  "
        f"{'r_i':>6}  {'r_j':>6}  {'|Δρ|':>6}  {'D':>6}  Note"
    )
    print(hdr)
    print("─" * len(hdr))
    for k, (s, sim) in enumerate(zip(step_results, sims)):
        note = (
            "◀ MOST SIMILAR"   if k == idx_closest  else
            "◀ MOST DIVERGENT" if k == idx_furthest else ""
        )
        print(
            f"  {s.label:<33}  {s.rho_pcca:6.3f}  "
            f"{sim['cos_sim_i']:8.3f}  {sim['cos_sim_j']:8.3f}  "
            f"{sim['latent_r_i']:6.3f}  {sim['latent_r_j']:6.3f}  "
            f"{sim['rho_abs_diff']:6.3f}  {sim['divergence']:6.3f}  {note}"
        )
    print(f"\n  Reference: {ref_label}\n")


# =============================================================================
# 5.  Supplementary metrics computation
# =============================================================================

def compute_supplementary_metrics(
        *,
        X_i_flat:   np.ndarray,
        X_j_flat:   np.ndarray,
        Z_flat:     Optional[np.ndarray],
        X_i_res:    np.ndarray,
        X_j_res:    np.ndarray,
        Wx_pcca:    np.ndarray,
        Wy_pcca:    np.ndarray,
        Wx_cca:     np.ndarray,
        Wy_cca:     np.ndarray,
        z_i_p:      np.ndarray,   # (n_trials, T) sign-corrected pCCA latents
        z_j_p:      np.ndarray,
        n_trials:   int,
        T:          int,
        time_vec:   np.ndarray,
        step_label: str,
        n_cv_folds: int   = 5,
        lam_cca:    float = LAMBDA_CCA,
        lam_hat:    float = LAMBDA_HAT,
) -> SupplementaryMetrics:
    """
    Compute the full diagnostic bundle for one ablation step.

    This is a pure function of its inputs: it depends only on the math
    helpers in §1 and can be called directly after pcca() +
    apply_latent_sign_correction() in the ablation loop.

    Cross-validation note
    ---------------------
    Trials are split into n_cv_folds contiguous blocks (no shuffling) to
    preserve temporal stationarity.  For each fold, the ridge regression
    coefficients β are estimated on training rows *only* and applied to
    both training and held-out test rows, preventing leakage of the
    nuisance-removal step across folds.

    κ computation note
    ------------------
    The swapped analyses pCCA(I, Z | J) and pCCA(J, Z | I) are fitted on
    the full (non-CV) dataset.  When Z is absent (baseline CCA step),
    κ is left as NaN.
    """
    w_p_i = Wx_pcca[:, 0];  w_p_j = Wy_pcca[:, 0]
    w_c_i = Wx_cca[:, 0];   w_c_j = Wy_cca[:, 0]

    # ── 1.  CCA–pCCA rotation angle θ ────────────────────────────────────
    theta_i_deg = float(np.degrees(
        np.arccos(np.clip(_cos_sim_abs(w_p_i, w_c_i), 0.0, 1.0))
    ))
    theta_j_deg = float(np.degrees(
        np.arccos(np.clip(_cos_sim_abs(w_p_j, w_c_j), 0.0, 1.0))
    ))

    # ── 2.  Cross-analysis collinearity κ ─────────────────────────────────
    # κ_I = |cos ∠(w_{IJ,I},  w_{IZ,I})|  where w_{IZ,I} comes from
    # pCCA(I, Z | J).  κ_J analogously from pCCA(J, Z | I).
    if Z_flat is not None and Z_flat.shape[1] > 0:
        Wx_IZ, _, _ = ridge_cca(
            residualize(X_i_flat, X_j_flat, lam_hat),
            residualize(Z_flat,   X_j_flat, lam_hat),
            lam_cca, 1,
        )
        kappa_i = float(_cos_sim_abs(w_p_i, Wx_IZ[:, 0]))

        Wy_JZ, _, _ = ridge_cca(
            residualize(X_j_flat, X_i_flat, lam_hat),
            residualize(Z_flat,   X_i_flat, lam_hat),
            lam_cca, 1,
        )
        kappa_j = float(_cos_sim_abs(w_p_j, Wy_JZ[:, 0]))
    else:
        kappa_i = kappa_j = float("nan")

    # ── 3.  Cross-validated ρ₁ ────────────────────────────────────────────
    fold_size  = n_trials // n_cv_folds
    trial_perm = np.arange(n_trials)
    rhos_cv:   List[float] = []

    for fold in range(n_cv_folds):
        te = trial_perm[fold * fold_size : (fold + 1) * fold_size]
        tr = np.concatenate([
            trial_perm[: fold * fold_size],
            trial_perm[(fold + 1) * fold_size :],
        ])
        tr_r = _trial_rows(tr, n_trials, T)
        te_r = _trial_rows(te, n_trials, T)
        Xi_tr, Xj_tr = X_i_flat[tr_r], X_j_flat[tr_r]
        Xi_te, Xj_te = X_i_flat[te_r], X_j_flat[te_r]

        if Z_flat is not None and Z_flat.shape[1] > 0:
            Z_tr = Z_flat[tr_r];  Z_te = Z_flat[te_r]
            n_tr = len(tr_r)
            ZtZ  = Z_tr.T @ Z_tr + lam_hat * n_tr * np.eye(Z_tr.shape[1])
            Bi   = np.linalg.solve(ZtZ, Z_tr.T @ Xi_tr)
            Bj   = np.linalg.solve(ZtZ, Z_tr.T @ Xj_tr)
            Xi_tr = Xi_tr - Z_tr @ Bi;  Xi_te = Xi_te - Z_te @ Bi
            Xj_tr = Xj_tr - Z_tr @ Bj;  Xj_te = Xj_te - Z_te @ Bj

        Wx_cv, Wy_cv, _ = ridge_cca(Xi_tr, Xj_tr, lam_cca, 1)
        zi_te = Xi_te @ Wx_cv[:, 0]
        zj_te = Xj_te @ Wy_cv[:, 0]

        if np.std(zi_te) < 1e-9 or np.std(zj_te) < 1e-9:
            rhos_cv.append(0.0)
        else:
            rhos_cv.append(float(np.clip(pearsonr(zi_te, zj_te)[0], -1.0, 1.0)))

    rho1_cv_mean = float(np.mean(rhos_cv))
    rho1_cv_sem  = float(np.std(rhos_cv) / np.sqrt(max(n_cv_folds, 1)))

    # ── 4.  Variance partitioning ─────────────────────────────────────────
    # All three fractions are relative to ‖X‖²_F (raw data norm), so
    # r²_nuis + r²_comm + r²_priv ≈ 1 (small deviations from ridge / mean-
    # centring of projections).
    denom_i = float(np.sum(X_i_flat ** 2)) + 1e-12
    denom_j = float(np.sum(X_j_flat ** 2)) + 1e-12

    wu_i = w_p_i / (np.linalg.norm(w_p_i) + 1e-12)  # unit-normalised
    wu_j = w_p_j / (np.linalg.norm(w_p_j) + 1e-12)

    X_i_hat = X_i_flat - X_i_res;  X_j_hat = X_j_flat - X_j_res
    r2_nuis_i = float(np.sum(X_i_hat ** 2)) / denom_i
    r2_nuis_j = float(np.sum(X_j_hat ** 2)) / denom_j

    pi = X_i_res @ wu_i;  pj = X_j_res @ wu_j
    r2_comm_i = float(np.sum(pi ** 2)) / denom_i
    r2_comm_j = float(np.sum(pj ** 2)) / denom_j

    # Private: explicit Frobenius norm of the orthogonal complement
    r2_priv_i = float(np.sum((X_i_res - np.outer(pi, wu_i)) ** 2)) / denom_i
    r2_priv_j = float(np.sum((X_j_res - np.outer(pj, wu_j)) ** 2)) / denom_j  # fixed: was denom_i

    # ── 5.  Temporal lead–lag cross-correlation ───────────────────────────
    mi = z_i_p.mean(axis=0);  mi -= mi.mean()
    mj = z_j_p.mean(axis=0);  mj -= mj.mean()
    si  = float(np.std(mi)) + 1e-12
    sj  = float(np.std(mj)) + 1e-12
    xcf = np.correlate(mi / si, mj / sj, mode="full") / T

    lag_bins     = np.arange(-(T - 1), T)
    dt_ms        = float(time_vec[1] - time_vec[0]) * 1000.0
    lag_ms       = lag_bins.astype(float) * dt_ms
    max_lb       = min(int(150.0 / dt_ms), T - 1)
    mask         = np.abs(lag_bins) <= max_lb
    peak_abs     = int(np.where(mask)[0][int(np.argmax(np.abs(xcf[mask])))])

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
        lag_peak_ms      = float(lag_ms[peak_abs]),
        lag_corr_at_peak = float(xcf[peak_abs]),
        lag_axis_ms      = lag_ms,
        xcorr_curve      = xcf,
    )


# =============================================================================
# 6.  Per-session figure functions
#     (retained from pcca_sequential_ablation.py; called once per session)
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
    Four-panel per-session summary (panels A–D).

    (A) ρ₁ across ablated regions.
    (B) |cos θ| weight-vector similarity vs. CCA baseline.
    (C) Pearson r of mean latent time-series vs. CCA baseline.
    (D) Composite divergence D; green = closest, red = most divergent.
    """
    labels = [s.label for s in step_results]
    rhos   = [s.rho_pcca for s in step_results]
    cos_i  = [d["cos_sim_i"]  for d in sims]
    cos_j  = [d["cos_sim_j"]  for d in sims]
    lat_i  = [d["latent_r_i"] for d in sims]
    lat_j  = [d["latent_r_j"] for d in sims]
    div    = [d["divergence"]  for d in sims]
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8),
        gridspec_kw={"hspace": 0.52, "wspace": 0.35},
    )

    # Panel A — ρ₁
    ax = axes[0, 0]
    ax.plot(x, rhos, "o-", color=_CI_PCCA, lw=1.8, ms=5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel(r"pCCA  $\rho_1$", fontsize=9)
    ax.set_title("(A)  Dominant canonical correlation", fontsize=9)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # Panel B — cosine similarity
    ax = axes[0, 1]
    ax.plot(x, cos_i, "s-", color=_CI_PCCA, lw=1.8, ms=5, label=TARGET_I)
    ax.plot(x, cos_j, "^-", color=_CJ_PCCA, lw=1.8, ms=5, label=TARGET_J)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel(r"$|\cos\theta|$  vs CCA baseline", fontsize=9)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("(B)  Weight-vector cosine similarity to CCA baseline", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # Panel C — latent Pearson r
    ax = axes[1, 0]
    ax.plot(x, lat_i, "s-", color=_CI_PCCA, lw=1.8, ms=5, label=TARGET_I)
    ax.plot(x, lat_j, "^-", color=_CJ_PCCA, lw=1.8, ms=5, label=TARGET_J)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r  (mean latent vs. CCA baseline)", fontsize=9)
    ax.set_ylim(-1.08, 1.08)
    ax.set_title("(C)  Mean latent correlation to CCA baseline", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # Panel D — divergence
    ax = axes[1, 1]
    bar_colors = [_C_NEUTRAL] * len(x)
    if idx_closest  < len(bar_colors): bar_colors[idx_closest]  = _C_SIM
    if idx_furthest < len(bar_colors): bar_colors[idx_furthest] = _C_DIV
    ax.bar(x, div, color=bar_colors, alpha=0.88, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("Composite divergence  D  (max = 4)", fontsize=9)
    ax.set_title(
        "(D)  Divergence from CCA baseline\n"
        "  ■ green = most similar   ■ red = most divergent",
        fontsize=9,
    )
    ax.grid(alpha=0.2)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [summary]  saved: {output_path}")
    return fig


def plot_supplementary_panel(
        supp_list:      List[SupplementaryMetrics],
        rho_pcca_list:  List[float],
        rho_cca:        float,
        title:          str,
        output_path:    Optional[Path] = None,
) -> plt.Figure:
    """
    Six-panel per-session supplementary figure (panels A–F).

    Row 0 — solution stability:
        (A) ρ₁ in-sample vs. cross-validated
        (B) CCA–pCCA rotation angle θ
        (C) Cross-analysis collinearity κ

    Row 1 — structural encoding:
        (D) Variance partition — TARGET_I
        (E) Variance partition — TARGET_J
        (F) pCCA latent cross-correlation (all ablation steps, viridis)
    """
    n      = len(supp_list)
    x      = np.arange(n)
    labels = [s.step_label for s in supp_list]

    fig, axes = plt.subplots(
        2, 3, figsize=(19, 9),
        gridspec_kw={"hspace": 0.58, "wspace": 0.36},
    )

    def _xax(ax: plt.Axes) -> None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
        ax.grid(alpha=0.22, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    theta_i    = [s.theta_i_deg  for s in supp_list]
    theta_j    = [s.theta_j_deg  for s in supp_list]
    kappa_i    = [s.kappa_i      for s in supp_list]
    kappa_j    = [s.kappa_j      for s in supp_list]
    rho_cv     = [s.rho1_cv_mean for s in supp_list]
    rho_cv_sem = [s.rho1_cv_sem  for s in supp_list]

    # Panel A — ρ₁ in-sample vs. CV
    ax = axes[0, 0]
    ax.plot(x, rho_pcca_list, "o-", color=_CI_PCCA, lw=1.8, ms=5,
            label="ρ₁  in-sample")
    ax.errorbar(x, rho_cv, yerr=rho_cv_sem,
                fmt="s--", color=_CJ_PCCA, lw=1.5, ms=5, capsize=3,
                label="ρ₁  5-fold CV")
    ax.axhline(rho_cca, color="#7F8C8D", ls=":", lw=1.1, alpha=0.7,
               label="CCA  ρ₁  (Z = ∅)")
    ax.set_ylabel("Canonical correlation  ρ₁", fontsize=9)
    ax.set_title(
        "(A)  In-sample vs. cross-validated  ρ₁\n"
        "Large CV gap → overfitting to nuisance regression",
        fontsize=8.5,
    )
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(-0.05, 1.05)
    _xax(ax)

    # Panel B — θ
    ax = axes[0, 1]
    ax.plot(x, theta_i, "s-", color=_CI_PCCA, lw=1.8, ms=5, label=TARGET_I)
    ax.plot(x, theta_j, "^-", color=_CJ_PCCA, lw=1.8, ms=5, label=TARGET_J)
    ax.axhline(90, color="#AAB7B8", ls="--", lw=0.9, alpha=0.6,
               label="90°  (fully orthogonal)")
    ax.set_ylabel("θ  CCA–pCCA weight angle  (°)", fontsize=9)
    ax.set_ylim(-3, 95)
    ax.set_title(
        "(B)  CCA–pCCA rotation angle  θ\n"
        r"θ → 90°: nuisance removal uncovered an orthogonal axis",
        fontsize=8.5,
    )
    ax.legend(fontsize=7, frameon=False)
    _xax(ax)

    # Panel C — κ
    ax = axes[0, 2]
    ki_c = [v if not np.isnan(v) else np.nan for v in kappa_i]
    kj_c = [v if not np.isnan(v) else np.nan for v in kappa_j]
    ax.plot(x, ki_c, "s-", color=_CI_PCCA, lw=1.8, ms=5,
            label=f"{TARGET_I}: |cos∠(w_IJ, w_IZ)|")
    ax.plot(x, kj_c, "^-", color=_CJ_PCCA, lw=1.8, ms=5,
            label=f"{TARGET_J}: |cos∠(w_JI, w_JZ)|")
    ax.axhline(1.0, color=_C_DIV, ls="--", lw=0.9, alpha=0.55,
               label="κ = 1  (Type-II collapse)")
    ax.axhline(0.0, color=_C_SIM, ls="--", lw=0.9, alpha=0.55,
               label="κ = 0  (orthogonal axes)")
    ax.fill_between(x, [0.8] * n, [1.0] * n, color="#FADBD8", alpha=0.35, zorder=0)
    ax.set_ylabel("κ  cross-analysis collinearity", fontsize=9)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(
        "(C)  Weight collinearity  κ\n"
        r"κ → 1: shared-noise amplification;  κ → 0: clean separation",
        fontsize=8.5,
    )
    ax.legend(fontsize=6.5, frameon=False)
    _xax(ax)

    # Panel D — Variance partition TARGET_I
    ax = axes[1, 0]
    rn_i = [s.r2_nuis_i for s in supp_list]
    rc_i = [s.r2_comm_i for s in supp_list]
    ax.bar(x, rn_i,  color=_C_NUIS,  alpha=0.88, label="Nuisance  r²")
    ax.bar(x, rc_i,  bottom=rn_i, color=_CI_PCCA, alpha=0.88,
           label="Communication  r²  (pCCA axis)")
    ax.set_ylabel("Fraction of total variance", fontsize=9)
    ax.set_title(f"(D)  Variance partition — {TARGET_I}", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 0.15)
    _xax(ax)

    # Panel E — Variance partition TARGET_J
    ax = axes[1, 1]
    rn_j = [s.r2_nuis_j for s in supp_list]
    rc_j = [s.r2_comm_j for s in supp_list]
    ax.bar(x, rn_j,  color=_C_NUIS,  alpha=0.88, label="Nuisance  r²")
    ax.bar(x, rc_j,  bottom=rn_j, color=_CJ_PCCA, alpha=0.88,
           label="Communication  r²  (pCCA axis)")
    ax.set_ylabel("Fraction of total variance", fontsize=9)
    ax.set_title(f"(E)  Variance partition — {TARGET_J}", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 0.15)
    _xax(ax)

    # Panel F — Temporal lead–lag cross-correlation
    ax = axes[1, 2]
    mask_300 = np.abs(supp_list[0].lag_axis_ms) <= 300
    lag_ax   = supp_list[0].lag_axis_ms[mask_300]
    step_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))
    for s, col in zip(supp_list, step_colors):
        ax.plot(lag_ax, s.xcorr_curve[mask_300], color=col, lw=0.6, alpha=0.6,
                label=s.step_label)
        ax.scatter([s.lag_peak_ms], [s.lag_corr_at_peak], color=col, s=12, zorder=4)
    ax.axvline(0, color="k", ls=":", lw=0.8, alpha=0.45)
    ax.axhline(0, color="k", ls="-",  lw=0.5, alpha=0.25)
    ax.set_xlim(-100, 100)
    ax.set_xlabel("Lag (ms)   [+ = TARGET_I leads]", fontsize=9)
    ax.set_ylabel("Normalised cross-correlation", fontsize=9)
    ax.set_title(
        "(F)  pCCA latent cross-correlation\n"
        "viridis: region 0 (dark) → region N (bright);  dots = peak lag",
        fontsize=8.5,
    )
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [supp_panel]  saved: {output_path}")
    return fig


# =============================================================================
# 7.  Cross-session aggregation
# =============================================================================

def aggregate_by_region(
        all_data: List[SessionAblationData],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Aggregate ablation metrics across sessions, keyed by ablated region.

    For every session that recorded region r, the corresponding numeric
    values of every diagnostic metric are appended to the list at
    ``agg[r][metric_name]``.  Regions not recorded in a given session
    simply contribute no entry (lists are of variable length ≤ n_sessions).

    Returns
    -------
    agg : Dict[region_name, Dict[metric_name, List[float]]]
        Metric names:
          From compute_similarity:
            cos_sim_i, cos_sim_j, latent_r_i, latent_r_j,
            rho_abs_diff, divergence
          From StepResult:
            rho_pcca
          From SessionAblationData:
            rho_cca  (session-level baseline, repeated per region)
          From SupplementaryMetrics:
            theta_i_deg, theta_j_deg, kappa_i, kappa_j,
            rho1_cv_mean, rho1_cv_sem,
            r2_nuis_i, r2_comm_i, r2_priv_i,
            r2_nuis_j, r2_comm_j, r2_priv_j
    """
    _METRIC_KEYS = [
        "cos_sim_i", "cos_sim_j", "latent_r_i", "latent_r_j",
        "rho_abs_diff", "divergence",
        "rho_pcca", "rho_cca",
        "theta_i_deg", "theta_j_deg",
        "kappa_i", "kappa_j",
        "rho1_cv_mean", "rho1_cv_sem",
        "r2_nuis_i", "r2_comm_i", "r2_priv_i",
        "r2_nuis_j", "r2_comm_j", "r2_priv_j",
    ]
    agg: Dict[str, Dict[str, List[float]]] = {}

    for sess in all_data:
        for region, sr, sim, supp in zip(
                sess.region_names,
                sess.step_results,
                sess.sims_vs_cca,
                sess.supp_metrics,
        ):
            if region not in agg:
                agg[region] = {k: [] for k in _METRIC_KEYS}
            d = agg[region]
            # similarity metrics
            for k in ("cos_sim_i", "cos_sim_j", "latent_r_i",
                      "latent_r_j", "rho_abs_diff", "divergence"):
                d[k].append(sim[k])
            # StepResult scalar
            d["rho_pcca"].append(sr.rho_pcca)
            # session-level CCA baseline (repeated per region for alignment)
            d["rho_cca"].append(sess.rho_cca)
            # SupplementaryMetrics scalars
            d["theta_i_deg"].append(supp.theta_i_deg)
            d["theta_j_deg"].append(supp.theta_j_deg)
            d["kappa_i"].append(supp.kappa_i)
            d["kappa_j"].append(supp.kappa_j)
            d["rho1_cv_mean"].append(supp.rho1_cv_mean)
            d["rho1_cv_sem"].append(supp.rho1_cv_sem)
            d["r2_nuis_i"].append(supp.r2_nuis_i)
            d["r2_comm_i"].append(supp.r2_comm_i)
            d["r2_priv_i"].append(supp.r2_priv_i)
            d["r2_nuis_j"].append(supp.r2_nuis_j)
            d["r2_comm_j"].append(supp.r2_comm_j)
            d["r2_priv_j"].append(supp.r2_priv_j)

    return agg


def _sorted_region_order(agg: Dict[str, Dict[str, List[float]]]) -> List[str]:
    """
    Return ablated regions sorted by ANATOMICAL_ORDER, restricted to those
    present in agg (i.e. recorded in at least one session).
    """
    present = set(agg.keys())
    ordered = [r for r in ANATOMICAL_ORDER if r in present]
    # Append any regions not in ANATOMICAL_ORDER at the end (alphabetically)
    ordered += sorted(present - set(ordered))
    return ordered


# =============================================================================
# 8.  Cross-session visualisation helpers
# =============================================================================

def _annotate_n(
        ax: plt.Axes,
        positions: List[float],
        counts:    List[int],
        fontsize:  int  = 6,
        color:     str  = "#555555",
) -> None:
    """
    Annotate session counts just above the top of each boxplot group.

    Uses ``ax.get_xaxis_transform()`` so that x is in data coordinates
    and y = 1.01 is a fixed fraction above the top axis edge, making the
    annotation robust to whatever ylim is set afterwards.
    """
    tr = ax.get_xaxis_transform()
    for pos, n in zip(positions, counts):
        ax.text(pos, 1.01, f"n={n}", ha="center", va="bottom",
                fontsize=fontsize, color=color, transform=tr)


def _clean(vals: List[float]) -> List[float]:
    """Filter out NaN values from a list."""
    return [v for v in vals if not np.isnan(v)]


def _paired_boxplot(
        ax:          plt.Axes,
        region_order: List[str],
        agg:         Dict[str, Dict[str, List[float]]],
        key_a:       str,
        key_b:       str,
        label_a:     str,
        label_b:     str,
        color_a:     str,
        color_b:     str,
        ylabel:      str,
        title:       str,
        ylim:        Optional[Tuple[float, float]] = None,
        hline:       Optional[float]               = None,
        hline_label: Optional[str]                 = None,
) -> None:
    """
    Draw paired boxplots for two metrics across ablated regions.

    For n regions, series A occupies positions 3k + 1 and series B
    occupies positions 3k + 2, with x-tick labels centred at 3k + 1.5.
    This leaves a gap of width 1 between adjacent region pairs for visual
    separation.

    Parameters
    ----------
    key_a, key_b : keys into agg[region][...].
    hline        : optional horizontal reference line drawn across the full
                   x-range (e.g. the CCA baseline correlation).
    """
    n_reg   = len(region_order)
    pos_a   = [3 * k + 1   for k in range(n_reg)]
    pos_b   = [3 * k + 2   for k in range(n_reg)]
    tick_x  = [3 * k + 1.5 for k in range(n_reg)]

    data_a = [_clean(agg[r].get(key_a, [])) for r in region_order]
    data_b = [_clean(agg[r].get(key_b, [])) for r in region_order]

    _bp_kw = dict(
        widths=0.75, patch_artist=True,
        medianprops=dict(color="k", lw=1.5),
        manage_ticks=False,
    )
    ax.boxplot(
        data_a, positions=pos_a,
        boxprops=dict(facecolor=color_a, alpha=0.65),
        whiskerprops=dict(color=color_a), capprops=dict(color=color_a),
        flierprops=dict(marker="o", ms=3.5, color=color_a, alpha=0.55),
        **_bp_kw,
    )
    ax.boxplot(
        data_b, positions=pos_b,
        boxprops=dict(facecolor=color_b, alpha=0.65),
        whiskerprops=dict(color=color_b), capprops=dict(color=color_b),
        flierprops=dict(marker="s", ms=3.5, color=color_b, alpha=0.55),
        **_bp_kw,
    )

    if hline is not None:
        ax.axhline(hline, color="#7F8C8D", ls=":", lw=1.0, alpha=0.75,
                   label=hline_label or f"ref = {hline:.3f}")

    _annotate_n(ax, tick_x, [len(data_a[k]) for k in range(n_reg)])

    ax.set_xticks(tick_x)
    ax.set_xticklabels(region_order, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(0, 3 * n_reg)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(
        handles=[
            Patch(facecolor=color_a, alpha=0.65, label=label_a),
            Patch(facecolor=color_b, alpha=0.65, label=label_b),
        ],
        fontsize=7, frameon=False,
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.22, lw=0.6, axis="y")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _single_boxplot(
        ax:          plt.Axes,
        region_order: List[str],
        agg:         Dict[str, Dict[str, List[float]]],
        key:         str,
        color:       str,
        ylabel:      str,
        title:       str,
        ylim:        Optional[Tuple[float, float]] = None,
        hline:       Optional[float]               = None,
        hline_label: Optional[str]                 = None,
) -> None:
    """Draw a single-metric boxplot across ablated regions."""
    x    = list(range(len(region_order)))
    data = [_clean(agg[r].get(key, [])) for r in region_order]

    ax.boxplot(
        data, positions=x, widths=0.55, patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.65),
        medianprops=dict(color="k", lw=1.5),
        whiskerprops=dict(color=color), capprops=dict(color=color),
        flierprops=dict(marker="o", ms=3.5, color=color, alpha=0.55),
        manage_ticks=False,
    )
    if hline is not None:
        ax.axhline(hline, color="#7F8C8D", ls=":", lw=1.0, alpha=0.75,
                   label=hline_label)
        ax.legend(fontsize=7, frameon=False)

    _annotate_n(ax, x, [len(data[k]) for k in range(len(region_order))])

    ax.set_xticks(x)
    ax.set_xticklabels(region_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.22, lw=0.6, axis="y")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _variance_partition_bars(
        ax:           plt.Axes,
        region_order: List[str],
        agg:          Dict[str, Dict[str, List[float]]],
        key_nuis:     str,
        key_comm:     str,
        comm_color:   str,
        ylabel:       str,
        title:        str,
        ylim:         Optional[Tuple[float, float]] = None,
        seed:         int = 42,
) -> None:
    """
    Cross-session variance partition stacked bars with SEM and session dots.

    For each ablated region r:
        ■ grey  (_C_NUIS)   : mean r²_nuis across sessions
        ■ comm_color        : mean r²_comm stacked on top
        ┤ error bar at total stack top : ±1 SEM of (r²_nuis + r²_comm)
        ● jittered dots at total height : individual session values of
                                          (r²_nuis + r²_comm)

    Rationale for this scheme
    -------------------------
    The stacked bar preserves the additive decomposition visible in the
    per-session panels D/E.  Placing error bars on the *total* (rather
    than on each component separately) avoids the visual ambiguity of
    overlapping error whiskers on stacked segments.  The individual dots
    make between-session variability explicit at the animal level, which
    is essential for the reviewer's assessment of statistical robustness.
    """
    rng   = np.random.default_rng(seed)
    x_pos = np.arange(len(region_order), dtype=float)

    for k, r in enumerate(region_order):
        nuis_vals  = np.array(_clean(agg[r].get(key_nuis, [])))
        comm_vals  = np.array(_clean(agg[r].get(key_comm, [])))
        n_sess     = min(len(nuis_vals), len(comm_vals))
        if n_sess == 0:
            continue
        nuis_vals  = nuis_vals[:n_sess]
        comm_vals  = comm_vals[:n_sess]
        total_vals = nuis_vals + comm_vals

        mn = float(nuis_vals.mean())
        mc = float(comm_vals.mean())
        sem_tot = float(total_vals.std() / np.sqrt(max(n_sess, 1)))

        # Stacked bars
        ax.bar(x_pos[k], mn,  color=_C_NUIS,    alpha=0.85, width=0.6)
        ax.bar(x_pos[k], mc,  bottom=mn, color=comm_color, alpha=0.85, width=0.6)

        # SEM error bar at total stack top
        ax.errorbar(
            x_pos[k], mn + mc, yerr=sem_tot,
            fmt="none", color="#2C3E50", capsize=4, lw=1.2, zorder=5,
        )

        # Jittered individual session dots
        jitter = rng.uniform(-0.18, 0.18, n_sess)
        ax.scatter(
            x_pos[k] + jitter, total_vals,
            color="#2C3E50", s=14, alpha=0.70, zorder=6, linewidths=0,
        )

    _annotate_n(
        ax,
        list(x_pos),
        [min(len(_clean(agg[r].get(key_nuis, []))),
             len(_clean(agg[r].get(key_comm, []))))
         for r in region_order],
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(region_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(
        handles=[
            Patch(facecolor=_C_NUIS,    alpha=0.85, label="Nuisance  r²"),
            Patch(facecolor=comm_color,  alpha=0.85, label="Communication  r²"),
        ],
        fontsize=7, frameon=False,
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.22, lw=0.6, axis="y")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# =============================================================================
# 9.  Cross-session figure functions
# =============================================================================

def plot_cross_session_summary(
        agg:          Dict[str, Dict[str, List[float]]],
        region_order: List[str],
        all_rho_cca:  List[float],
        output_path:  Optional[Path] = None,
) -> plt.Figure:
    """
    Cross-session summary: panels B, C, D as boxplots (panel A omitted).

    Layout: 1 row × 3 columns
        (0,0)  Panel B: |cos θ| weight-vector similarity to CCA baseline
        (0,1)  Panel C: Pearson r of mean latent time-series vs. CCA baseline
        (0,2)  Panel D: Composite divergence D

    The x-axis enumerates ablated regions (sorted by ANATOMICAL_ORDER).
    Each box is a distribution across the sessions that recorded that
    region; session count is annotated above each group.

    Parameters
    ----------
    all_rho_cca : list of per-session CCA baseline ρ₁ values, used only to
                  display a mean reference line in a potential extended version;
                  not shown here since panel A is omitted.
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(20, 6),
        gridspec_kw={"wspace": 0.38},
    )

    # Panel B — |cos θ|
    _paired_boxplot(
        axes[0], region_order, agg,
        key_a="cos_sim_i", key_b="cos_sim_j",
        label_a=TARGET_I, label_b=TARGET_J,
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel=r"$|\cos\theta|$  vs CCA baseline",
        title=(
            "(B)  Weight-vector cosine similarity to CCA baseline\n"
            "per ablated region  ·  boxes = session distribution"
        ),
        ylim=(-0.05, 1.10),
        hline=1.0, hline_label="perfect match",
    )

    # Panel C — latent Pearson r
    _paired_boxplot(
        axes[1], region_order, agg,
        key_a="latent_r_i", key_b="latent_r_j",
        label_a=TARGET_I, label_b=TARGET_J,
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel="Pearson  r  (mean latent vs. CCA baseline)",
        title=(
            "(C)  Mean latent correlation to CCA baseline\n"
            "per ablated region  ·  boxes = session distribution"
        ),
        ylim=(-1.10, 1.10),
        hline=1.0, hline_label="perfect match",
    )

    # Panel D — divergence (single metric)
    _single_boxplot(
        axes[2], region_order, agg,
        key="divergence", color=_CI_PCCA,
        ylabel="Composite divergence  D  (max = 4)",
        title=(
            "(D)  Divergence D from CCA baseline\n"
            "per ablated region  ·  boxes = session distribution"
        ),
        ylim=(0.0, 4.10),
    )

    fig.suptitle(
        f"Cross-session ablation summary  |  {TARGET_I} ↔ {TARGET_J}\n"
        f"n = {len(SESSIONS_TO_RUN)} sessions  ·  reference: CCA baseline (Z = ∅)",
        fontsize=12, fontweight="bold",
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [cross-session summary]  saved: {output_path}")
    return fig


def plot_cross_session_supplementary(
        agg:          Dict[str, Dict[str, List[float]]],
        region_order: List[str],
        all_rho_cca:  List[float],
        output_path:  Optional[Path] = None,
) -> plt.Figure:
    """
    Cross-session supplementary: panels A–E (panel F omitted).

    Layout: 2 rows × 3 columns
        (0,0)  Panel A: ρ₁ in-sample and CV — paired boxplots per region,
                        with mean ± SEM CCA-baseline band.
        (0,1)  Panel B: θ (CCA–pCCA rotation angle) — paired boxplots.
        (0,2)  Panel C: κ (cross-analysis collinearity) — paired boxplots.
        (1,0)  Panel D: Variance partition — TARGET_I — stacked bars + SEM.
        (1,1)  Panel E: Variance partition — TARGET_J — stacked bars + SEM.
        (1,2)  [Reserved — legend + notes panel.]

    Variance partition scheme (panels D and E)
    ------------------------------------------
    See ``_variance_partition_bars`` docstring for the full rationale.
    In brief: stacked mean bars (nuisance grey + communication coloured)
    with ±1 SEM error bar on the total height and jittered individual
    session dots, preserving the per-session decomposition logic while
    making between-animal variability transparent.

    CCA baseline in panel A
    -----------------------
    Since ρ_CCA varies across sessions (different neuron counts, different
    signal-to-noise ratios), we represent it as a horizontal shaded band
    spanning mean(ρ_CCA) ± SEM(ρ_CCA) across sessions.  This is more
    honest than a fixed horizontal line.
    """
    fig, axes = plt.subplots(
        2, 3, figsize=(22, 12),
        gridspec_kw={"hspace": 0.55, "wspace": 0.38},
    )

    # ── Compute session-level CCA baseline band ──────────────────────────
    rho_cca_arr = np.array(all_rho_cca)
    rho_cca_mean = float(rho_cca_arr.mean())
    rho_cca_sem  = float(rho_cca_arr.std() / np.sqrt(max(len(rho_cca_arr), 1)))

    # ── Panel A — ρ₁ in-sample vs. CV ─────────────────────────────────────
    ax = axes[0, 0]
    _paired_boxplot(
        ax, region_order, agg,
        key_a="rho_pcca", key_b="rho1_cv_mean",
        label_a="ρ₁  in-sample", label_b="ρ₁  5-fold CV",
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel="Canonical correlation  ρ₁",
        title=(
            "(A)  In-sample vs. cross-validated  ρ₁\n"
            "Large CV gap → overfitting to nuisance regression"
        ),
        ylim=(-0.05, 1.05),
    )
    # CCA baseline shaded band
    n_reg = len(region_order)
    ax.axhspan(
        rho_cca_mean - rho_cca_sem,
        rho_cca_mean + rho_cca_sem,
        color="#7F8C8D", alpha=0.18,
        label=f"CCA ρ₁ = {rho_cca_mean:.3f} ± {rho_cca_sem:.3f}",
    )
    ax.axhline(rho_cca_mean, color="#7F8C8D", ls=":", lw=1.0, alpha=0.80)
    ax.legend(fontsize=6.5, frameon=False)

    # ── Panel B — θ ────────────────────────────────────────────────────────
    _paired_boxplot(
        axes[0, 1], region_order, agg,
        key_a="theta_i_deg", key_b="theta_j_deg",
        label_a=TARGET_I, label_b=TARGET_J,
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel="θ  CCA–pCCA weight angle  (°)",
        title=(
            "(B)  CCA–pCCA rotation angle  θ\n"
            r"θ → 90°: nuisance removal uncovered an orthogonal axis"
        ),
        ylim=(-3.0, 95.0),
        hline=90.0, hline_label="90°  (fully orthogonal)",
    )

    # ── Panel C — κ ────────────────────────────────────────────────────────
    ax = axes[0, 2]
    _paired_boxplot(
        ax, region_order, agg,
        key_a="kappa_i", key_b="kappa_j",
        label_a=f"{TARGET_I}  |cos∠(w_IJ, w_IZ)|",
        label_b=f"{TARGET_J}  |cos∠(w_JI, w_JZ)|",
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel="κ  cross-analysis collinearity",
        title=(
            "(C)  Weight collinearity  κ\n"
            r"κ → 1: shared-noise amplification;  κ → 0: clean separation"
        ),
        ylim=(-0.05, 1.10),
    )
    # Danger-zone shading (κ > 0.8)
    ax.fill_between(
        [-1, 3 * n_reg + 1], [0.8, 0.8], [1.05, 1.05],
        color="#FADBD8", alpha=0.35, zorder=0,
    )

    # ── Panel D — Variance partition TARGET_I ──────────────────────────────
    _variance_partition_bars(
        axes[1, 0], region_order, agg,
        key_nuis="r2_nuis_i", key_comm="r2_comm_i",
        comm_color=_CI_PCCA,
        ylabel="Fraction of total variance",
        title=(
            f"(D)  Variance partition — {TARGET_I}\n"
            "mean ± SEM across sessions  ·  dots = individual sessions"
        ),
        ylim=(0.0, 0.20),
    )

    # ── Panel E — Variance partition TARGET_J ──────────────────────────────
    _variance_partition_bars(
        axes[1, 1], region_order, agg,
        key_nuis="r2_nuis_j", key_comm="r2_comm_j",
        comm_color=_CJ_PCCA,
        ylabel="Fraction of total variance",
        title=(
            f"(E)  Variance partition — {TARGET_J}\n"
            "mean ± SEM across sessions  ·  dots = individual sessions"
        ),
        ylim=(0.0, 0.20),
    )

    # ── Panel (1,2) — Notes / legend ───────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    notes = (
        "Visualisation notes\n"
        "────────────────────────────────────\n"
        "Panels A–C: paired boxplots per ablated\n"
        "region.  Box = IQR; whiskers = 1.5×IQR;\n"
        "n = sessions recording that region.\n\n"
        "Panels D–E: stacked mean bars.\n"
        "  ■ grey   = mean r²_nuis\n"
        "  ■ colour = mean r²_comm\n"
        "  ┤        = ±1 SEM of total (nuis+comm)\n"
        "  ●        = individual session totals\n\n"
        f"Sessions: n = {len(SESSIONS_TO_RUN)}\n"
        f"Target pair: {TARGET_I} ↔ {TARGET_J}\n"
        f"Reference: CCA baseline (Z = ∅)\n"
        f"Mean CCA ρ₁: {rho_cca_mean:.3f} ± {rho_cca_sem:.3f} SEM"
    )
    ax.text(
        0.05, 0.95, notes,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=8.5, family="monospace",
        bbox=dict(facecolor="#F2F3F4", edgecolor="#BDC3C7",
                  boxstyle="round,pad=0.6"),
    )

    fig.suptitle(
        f"Cross-session supplementary diagnostics  |  {TARGET_I} ↔ {TARGET_J}\n"
        f"n = {len(SESSIONS_TO_RUN)} sessions  ·  single-region ablation",
        fontsize=12, fontweight="bold",
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [cross-session supp]  saved: {output_path}")
    return fig


# =============================================================================
# 10.  Per-session ablation runner
# =============================================================================

def run_single_ablation(
        region_spikes:    Dict[str, np.ndarray],
        n_trials:         int,
        T:                int,
        out_summary_dir:  Path,
        out_supp_dir:     Path,
        session_name:     str,
) -> Optional[SessionAblationData]:
    """
    Run single-region ablation for one session and save per-session figures.

    For every nuisance region r present in the session (excluding TARGET_I
    and TARGET_J), this function fits:

        CCA baseline    : ridge_cca(X_i, X_j)
        pCCA ablation r : pcca(X_i, X_j | Z = {r})

    and computes StepResult + SupplementaryMetrics per region.

    The CCA baseline StepResult is used as the reference for all divergence
    calculations (compare_similarity).  This mirrors the per-session
    ``plot_summary_figure`` semantics where panel D measures divergence
    from the no-nuisance solution.

    Per-session output files
    ------------------------
    ``{session_name}_part1_summary_vs_CCA.png``
        Saved to out_summary_dir.  Four panels A–D from plot_summary_figure
        (ablation steps on x-axis, CCA baseline as reference).

    ``{session_name}_part1_supplementary.png``
        Saved to out_supp_dir.  Six panels A–F from plot_supplementary_panel.

    Returns
    -------
    SessionAblationData for downstream aggregation, or None if either
    target region is missing from this session.
    """
    for t in (TARGET_I, TARGET_J):
        if t not in region_spikes:
            warnings.warn(
                f"  [{session_name}]  Target region '{t}' missing — skipping."
            )
            return None

    # ── Nuisance list (ANATOMICAL_ORDER, targets excluded) ────────────────
    nuisance_all: List[str] = [
        r for r in ANATOMICAL_ORDER
        if r in region_spikes and r not in (TARGET_I, TARGET_J)
    ]
    if not nuisance_all:
        warnings.warn(
            f"  [{session_name}]  No nuisance regions found — skipping."
        )
        return None

    print(
        f"  [{session_name}]  Ablating {len(nuisance_all)} nuisance regions: "
        f"{nuisance_all}"
    )

    # ── Pre-compute flat matrices ──────────────────────────────────────────
    X_i_flat = _zscore_flat(region_spikes[TARGET_I])
    X_j_flat = _zscore_flat(region_spikes[TARGET_J])
    nuisance_flat: Dict[str, np.ndarray] = {
        r: _zscore_flat(region_spikes[r]) for r in nuisance_all
    }

    time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)

    # ── CCA baseline ──────────────────────────────────────────────────────
    Wx_cca, Wy_cca, rho_cca = ridge_cca(X_i_flat, X_j_flat)
    rho_cca_val = float(rho_cca[0])
    print(f"  [{session_name}]  CCA baseline  ρ₁ = {rho_cca_val:.4f}")

    # CCA StepResult (reference for divergence computation)
    _zi_c_raw = latent_projections(X_i_flat, Wx_cca[:, 0], n_trials, T)
    _zj_c_raw = latent_projections(X_j_flat, Wy_cca[:, 0], n_trials, T)
    _zi_c_sc, _zj_c_sc, flip_ic, flip_jc = apply_latent_sign_correction(
        _zi_c_raw, _zj_c_raw, time_vec
    )
    Wx_c_f = Wx_cca[:, 0] * (-1.0 if flip_ic else 1.0)
    Wy_c_f = Wy_cca[:, 0] * (-1.0 if flip_jc else 1.0)

    cca_ref = StepResult(
        label="CCA_baseline",
        nuisance_regions=[],
        rho_pcca=rho_cca_val,
        Wx=Wx_cca,
        Wy=Wy_cca,
        z_i_mean=latent_projections(X_i_flat, Wx_c_f, n_trials, T).mean(0),
        z_j_mean=latent_projections(X_j_flat, Wy_c_f, n_trials, T).mean(0),
    )

    # ── Single-region ablation loop ────────────────────────────────────────
    ablation_results:  List[StepResult]          = []
    supp_list:         List[SupplementaryMetrics] = []

    for abl_idx, region in enumerate(nuisance_all):
        Z_single = nuisance_flat[region]
        Wx_p, Wy_p, rho_p, X_i_res, X_j_res = pcca(
            X_i_flat, X_j_flat, Z_single,
        )

        _zi_p_raw = latent_projections(X_i_res, Wx_p[:, 0], n_trials, T)
        _zj_p_raw = latent_projections(X_j_res, Wy_p[:, 0], n_trials, T)
        _zi_p_sc, _zj_p_sc, flip_ip, flip_jp = apply_latent_sign_correction(
            _zi_p_raw, _zj_p_raw, time_vec
        )

        Wx_p_f = Wx_p[:, 0] * (-1.0 if flip_ip else 1.0)
        Wy_p_f = Wy_p[:, 0] * (-1.0 if flip_jp else 1.0)
        z_i_mean = latent_projections(X_i_res, Wx_p_f, n_trials, T).mean(0)
        z_j_mean = latent_projections(X_j_res, Wy_p_f, n_trials, T).mean(0)

        label = f"abl{abl_idx:02d}_{region}"
        step  = StepResult(
            label=label,
            nuisance_regions=[region],
            rho_pcca=float(rho_p[0]),
            Wx=Wx_p,
            Wy=Wy_p,
            z_i_mean=z_i_mean,
            z_j_mean=z_j_mean,
        )
        ablation_results.append(step)

        supp = compute_supplementary_metrics(
            X_i_flat   = X_i_flat,
            X_j_flat   = X_j_flat,
            Z_flat     = Z_single,
            X_i_res    = X_i_res,
            X_j_res    = X_j_res,
            Wx_pcca    = Wx_p,
            Wy_pcca    = Wy_p,
            Wx_cca     = Wx_cca,
            Wy_cca     = Wy_cca,
            z_i_p      = _zi_p_sc,
            z_j_p      = _zj_p_sc,
            n_trials   = n_trials,
            T          = T,
            time_vec   = time_vec,
            step_label = label,
        )
        supp_list.append(supp)

        print(
            f"    [{session_name}]  abl {abl_idx:02d}  Z = {{{region:<8}}}  "
            f"ρ₁ = {rho_p[0]:.4f}  θ_i = {supp.theta_i_deg:.1f}°  "
            f"κ_i = {supp.kappa_i:.3f}"
        )

    # ── Similarity vs. CCA baseline ───────────────────────────────────────
    sims_vs_cca = [compute_similarity(s, cca_ref) for s in ablation_results]
    idx_c, idx_f, _ = identify_extremes(
        ablation_results, cca_ref, exclude_ref=False,
    )

    print(f"\n  [{session_name}]  Similarity to CCA baseline:")
    print_similarity_table(ablation_results, sims_vs_cca, "CCA_baseline", idx_c, idx_f)

    # ── Per-session summary figure ────────────────────────────────────────
    summary_fig = plot_summary_figure(
        step_results = ablation_results,
        sims         = sims_vs_cca,
        idx_closest  = idx_c,
        idx_furthest = idx_f,
        ref_label    = "CCA baseline  (Z = ∅)",
        title        = (
            f"Single-region ablation  |  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}   reference: CCA baseline"
        ),
        output_path  = out_summary_dir / f"{session_name}_part1_summary_vs_CCA.png",
    )
    plt.close(summary_fig)

    # ── Per-session supplementary figure ──────────────────────────────────
    supp_fig = plot_supplementary_panel(
        supp_list     = supp_list,
        rho_pcca_list = [s.rho_pcca for s in ablation_results],
        rho_cca       = rho_cca_val,
        title         = (
            f"Part 1 — Supplementary Diagnostics  |  {session_name}\n"
            f"{TARGET_I} ↔ {TARGET_J}   (single-region ablation)"
        ),
        output_path   = out_supp_dir / f"{session_name}_part1_supplementary.png",
    )
    plt.close(supp_fig)

    # ── Return cached data ────────────────────────────────────────────────
    return SessionAblationData(
        session_name = session_name,
        rho_cca      = rho_cca_val,
        region_names = nuisance_all,
        step_results = ablation_results,
        sims_vs_cca  = sims_vs_cca,
        supp_metrics = supp_list,
    )


# =============================================================================
# 11.  Entry point
# =============================================================================

def main() -> None:
    BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    SESSION_DIR = BASE_DIR / "pcca_sessions_cued_hit_long_results"

    out_summary = BASE_DIR / "Paper_output" / "pcca_cross_session_summary"/f"{TARGET_I}_{TARGET_J}"
    out_supp    = BASE_DIR / "Paper_output" / "pcca_cross_session_supplementary"/f"{TARGET_I}_{TARGET_J}"
    out_summary.mkdir(parents=True, exist_ok=True)
    out_supp.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"pCCA Cross-Session Ablation  |  {TARGET_I} ↔ {TARGET_J}")
    print(f"  Sessions  : {len(SESSIONS_TO_RUN)}")
    print(f"  Summary   : {out_summary}")
    print(f"  Supp      : {out_supp}")
    print("=" * 72)

    all_session_data: List[SessionAblationData] = []

    for idx, session_name in enumerate(SESSIONS_TO_RUN, 1):
        session_file = SESSION_DIR / f"{session_name}_analysis_results.mat"

        print(f"\n🔬 [{idx}/{len(SESSIONS_TO_RUN)}]  {session_name}")

        if not session_file.exists():
            print(f"  ❌  File not found, skipping: {session_file}")
            continue

        try:
            region_spikes, n_trials, T = load_region_spikes(str(session_file))
            result = run_single_ablation(
                region_spikes   = region_spikes,
                n_trials        = n_trials,
                T               = T,
                out_summary_dir = out_summary,
                out_supp_dir    = out_supp,
                session_name    = session_name,
            )
            if result is not None:
                all_session_data.append(result)
                print(f"  ✓  {session_name}  cached  "
                      f"({len(result.region_names)} ablated regions)")
        except Exception as exc:
            print(f"  💥  {session_name}  FAILED: {exc}")
            import traceback
            traceback.print_exc()

    if not all_session_data:
        print("\n⚠️  No sessions completed successfully — no cross-session figures.")
        return

    # ── Cross-session aggregation ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"Cross-session aggregation  ({len(all_session_data)} sessions)")
    print("=" * 72)

    agg          = aggregate_by_region(all_session_data)
    region_order = _sorted_region_order(agg)
    all_rho_cca  = [s.rho_cca for s in all_session_data]

    print(f"  Regions in aggregation  ({len(region_order)}): {region_order}")
    for r in region_order:
        n_sess = len(_clean(agg[r].get("rho_pcca", [])))
        print(f"    {r:<12}  n = {n_sess} sessions")

    # ── Cross-session summary figure (B, C, D) ─────────────────────────────
    cross_sum_fig = plot_cross_session_summary(
        agg          = agg,
        region_order = region_order,
        all_rho_cca  = all_rho_cca,
        output_path  = out_summary / "cross_session_summary_BCD.png",
    )
    plt.close(cross_sum_fig)

    # ── Cross-session supplementary figure (A–E) ──────────────────────────
    cross_supp_fig = plot_cross_session_supplementary(
        agg          = agg,
        region_order = region_order,
        all_rho_cca  = all_rho_cca,
        output_path  = out_supp / "cross_session_supplementary_ABCDE.png",
    )
    plt.close(cross_supp_fig)

    print("\n🎉  All done.")
    print(f"  Summary figures  → {out_summary}")
    print(f"  Supp figures     → {out_supp}")


if __name__ == "__main__":
    main()
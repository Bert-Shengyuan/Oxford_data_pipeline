"""
pcca_cross_session_ablation.py
==============================

Batch cross-session pCCA single-region ablation pipeline for the
MOs ↔ VPMPO target pair, now regime-aware (signal vs. residual).

────────────────────────────────────────────────────────────────────────────
REGIME SWITCHES
────────────────────────────────────────────────────────────────────────────
SUBTRACT_PSTH and SHUFFLE_TRIALS select which cross-region covariance the
ablation operates on (Σ_AB = Σ_AB^stim + Σ_AB^noise):

    SHUFFLE_TRIALS=True , SUBTRACT_PSTH=False  →  signal  Σ_AB^stim
    SHUFFLE_TRIALS=False, SUBTRACT_PSTH=True   →  noise   Σ_AB^noise  (residual)

The target tensors are independently trial-permuted (X_i with _rng, X_j with
_rng2) when SHUFFLE_TRIALS=True; the nuisance regions are never shuffled but
are PSTH-subtracted whenever SUBTRACT_PSTH=True, so the nuisance-removal step
operates on the same statistical object as the targets.

Why the latent similarity changed (residual regime)
---------------------------------------------------
Under PSTH subtraction the per-trial residual obeys (1/T)Σ_t δ = 0, so the
trial-AVERAGED latent z̄(s) = ((1/T)Σ_t δ) w = 0 identically.  Correlating two
≈0 vectors (the old metric) is pure noise.  We therefore correlate the FULL
(trial × time) latent matrices sample-by-sample, in absolute value:

        r_k = | corr_{(t,s)}( z_k^{step} , z_k^{ref} ) | ,     z_k ∈ ℝ^{n_tr × T}.

The absolute value absorbs the canonical pair's global sign freedom — exactly
the convention already used for |cos θ| on the weights.  Consequently every
retained metric is sign-invariant or relies only on the CCA-fixed *relative*
pair sign, and `apply_latent_sign_correction` (mean-based, undefined when the
mean is 0) is removed entirely.

Cross-correlation lag (per-session panel F)
-------------------------------------------
The lead–lag estimator is gated by ``paired_trials = not SHUFFLE_TRIALS``:
  • paired (residual)  → mean WITHIN-trial cross-correlogram (does not collapse);
  • shuffled (signal)  → cross-correlation of the trial-averaged latents.

Residual-coupling cross-session diagnostics (NEW)
-------------------------------------------------
``plot_cross_session_residual_coupling``:
  (A) per-session residual CCA coupling ρ₁ (in-sample and 5-fold CV);
  (B) per-region Δρ = ρ₁^{pCCA(Z={r})} − ρ₁^{CCA} (in-sample and CV), the signed
      change in residual coupling produced by partialling out each region, with a
      zero reference line.

Output paths (regime tag appended so signal/residual runs never overwrite)
--------------------------------------------------------------------------
  BASE_DIR/Paper_output/pcca_cross_session_summary/{I}_{J}_{REGIME_TAG}/
      {session}_part1_summary_vs_CCA.png
      cross_session_summary_BCD.png
      cross_session_residual_coupling.png        ← NEW
  BASE_DIR/Paper_output/pcca_cross_session_supplementary/{I}_{J}_{REGIME_TAG}/
      {session}_part1_supplementary.png
      cross_session_supplementary_ABCDE.png

Notes
-----
* r²_priv_j denominator fix (denom_j) retained.
* Ridge parameters and CV fold logic identical to pcca_sequential_ablation.py.
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
    safe_array,
)

# =============================================================================
# 0.  Global configuration
# =============================================================================

# TARGET_I     = "MOs"
# TARGET_J     = "VPMPO"
#
# SESSIONS_TO_RUN: List[str] = [
#     "yp020_220331",
#     "yp020_220401",
#     "yp021_220331",
#     "yp021_220402",
#     "yp021_220403",
#     "yp021_220404",
#     "yp021_220405",
#     "yp021_220407",
# ]

TARGET_I     = "MOs"
TARGET_J     = "VALVM"

SESSIONS_TO_RUN: List[str] = [
    "yp020_220331",
    "yp020_220407",
    "yp021_220401",
    "yp021_220403",
    "yp021_220404",
    "yp021_220407",
]


# ── Regime switches ──────────────────────────────────────────────────────────
# Residual / noise-correlation regime (default):
SUBTRACT_PSTH:  bool = False
SHUFFLE_TRIALS: bool = False
REGIME_TAG = f"psth{int(SUBTRACT_PSTH)}_shuf{int(SHUFFLE_TRIALS)}"

# Independent RNGs for the two targets (only used when SHUFFLE_TRIALS=True),
# matching the convention in pcca_sequential_ablation.py.
_rng  = np.random.default_rng(42)
_rng2 = np.random.default_rng(43)

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
_C_INSAMPLE = "#8E44AD"   # in-sample coupling (residual figure)
_C_CV       = "#16A085"   # cross-validated coupling (residual figure)


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
    """Flatten (n_trials, n, T) → (T·n_trials, n) with optional PSTH subtraction
    and/or trial shuffling.

    Flat layout: row = t · n_trials + trial (time outer).  When all flags are off
    this is identical to the original simple z-score-and-flatten.  PSTH subtraction
    (axis=0 = trials) enforces an exact zero trial-mean per (neuron, time).
    """
    n_trials, n, T = X.shape
    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
    flat = zscore(flat, axis=1, nan_policy="omit")
    np.nan_to_num(flat, nan=0.0, copy=False)

    if not subtract_psth and not shuffle_trials:
        return flat.T  # fast path, byte-identical to the legacy behaviour

    X = flat.reshape(n, T, n_trials).transpose(2, 0, 1)   # (n_trials, n, T)

    if subtract_psth:
        X = X - X.mean(axis=0, keepdims=True)

    if shuffle_trials:
        if perm is not None:
            if perm.shape != (n_trials,):
                raise ValueError(f"perm must have shape ({n_trials},); got {perm.shape}")
            X = X[perm]
        else:
            if rng is None:
                rng = np.random.default_rng()
            X = X[rng.permutation(n_trials)]

    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
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
    """Ridge-regularised CCA.  Returns Wx (p,k), Wy (q,k), rho (k,) in [0,1]."""
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
    """Partial out Z from X via ridge OLS: X̃ = (I − Z(Z'Z + λnI)⁻¹Z') X."""
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
    """Partial CCA: residualise X and Y wrt Z, then CCA.  Returns
    Wx_p, Wy_p, rho_p, X_res, Y_res."""
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


def _abs_pearson(A: np.ndarray, B: np.ndarray) -> float:
    """|Pearson r| between two equally-shaped arrays, computed on the raveled
    (trial × time) samples.  Sign-invariant — the canonical pair's global sign
    is arbitrary, and in the residual regime the trial-mean (the basis of the old
    sign-correction) is identically zero."""
    a = np.asarray(A).ravel()
    b = np.asarray(B).ravel()
    if a.size != b.size or a.size < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.abs(np.clip(pearsonr(a, b)[0], -1.0, 1.0)))


def _trial_rows(trial_idx: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """Flat-matrix row indices (layout row = t·n_trials + trial) for all T
    time-points of the requested trial indices."""
    return (trial_idx[None, :] + np.arange(T)[:, None] * n_trials).ravel()


def _cv_rho(
        Xi_flat: np.ndarray,
        Xj_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        n_trials: int,
        T: int,
        n_cv_folds: int = 5,
        lam_cca: float = LAMBDA_CCA,
        lam_hat: float = LAMBDA_HAT,
) -> float:
    """Mean 5-fold cross-validated ρ₁ (contiguous folds), replicating exactly the
    fold logic used inside compute_supplementary_metrics.  Used here for the CCA
    baseline (Z=None) so that Δρ can be evaluated on cross-validated footing."""
    fold_size  = n_trials // n_cv_folds
    if fold_size < 1:
        return float("nan")
    trial_perm = np.arange(n_trials)
    rhos: List[float] = []

    for fold in range(n_cv_folds):
        te = trial_perm[fold * fold_size : (fold + 1) * fold_size]
        tr = np.concatenate([
            trial_perm[: fold * fold_size],
            trial_perm[(fold + 1) * fold_size :],
        ])
        tr_r = _trial_rows(tr, n_trials, T)
        te_r = _trial_rows(te, n_trials, T)
        Xi_tr, Xj_tr = Xi_flat[tr_r], Xj_flat[tr_r]
        Xi_te, Xj_te = Xi_flat[te_r], Xj_flat[te_r]

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
            rhos.append(0.0)
        else:
            rhos.append(float(np.clip(pearsonr(zi_te, zj_te)[0], -1.0, 1.0)))

    return float(np.mean(rhos))


# =============================================================================
# 2.  Data loading
# =============================================================================

def load_region_spikes(
        session_path: str,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Load per-region spike tensors from a MATLAB v7.3 results file."""
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
    """Per-ablation-step container (canonical weights + FULL trial×time latents)."""

    __slots__ = [
        "label", "nuisance_regions", "rho_pcca",
        "Wx", "Wy", "z_i_lat", "z_j_lat",
    ]

    def __init__(
            self,
            label: str,
            nuisance_regions: List[str],
            rho_pcca: float,
            Wx: np.ndarray,
            Wy: np.ndarray,
            z_i_lat: np.ndarray,   # (n_trials, T)
            z_j_lat: np.ndarray,   # (n_trials, T)
    ) -> None:
        self.label            = label
        self.nuisance_regions = list(nuisance_regions)
        self.rho_pcca         = float(rho_pcca)
        self.Wx               = Wx
        self.Wy               = Wy
        self.z_i_lat          = z_i_lat
        self.z_j_lat          = z_j_lat


@dataclass
class SupplementaryMetrics:
    """Seven-metric diagnostic bundle for one ablation step (see field docs below)."""
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
    """Per-session ablation cache consumed by the cross-session aggregator.

    Lists share a common index: element k ↔ ablated region region_names[k].
    rho_cca / rho_cca_cv are session-level scalars (the Z=∅ baseline, in-sample
    and 5-fold CV respectively)."""
    session_name:  str
    rho_cca:       float
    rho_cca_cv:    float
    region_names:  List[str]
    step_results:  List[StepResult]
    sims_vs_cca:   List[Dict[str, float]]
    supp_metrics:  List[SupplementaryMetrics]


# =============================================================================
# 4.  Similarity metrics
# =============================================================================

def compute_similarity(s: StepResult, ref: StepResult) -> Dict[str, float]:
    """Pairwise similarity between step s and a reference solution.

    Composite divergence D ∈ [0, 4]:

        D = (1 − |cos θ_i|) + (1 − |cos θ_j|) + (1 − r_i) + (1 − r_j)

    where r_k = |corr_{(t,s)}(z_k^{step}, z_k^{ref})| is computed on the FULL
    (trial × time) latent matrices (NOT the trial average, which is ≈0 in the
    residual regime).  D = 0 ⇔ identical solutions in every term.
    """
    cos_i = _cos_sim_abs(s.Wx[:, 0], ref.Wx[:, 0])
    cos_j = _cos_sim_abs(s.Wy[:, 0], ref.Wy[:, 0])
    r_i   = _abs_pearson(s.z_i_lat, ref.z_i_lat)
    r_j   = _abs_pearson(s.z_j_lat, ref.z_j_lat)
    return dict(
        cos_sim_i    = cos_i,
        cos_sim_j    = cos_j,
        latent_r_i   = r_i,
        latent_r_j   = r_j,
        rho_abs_diff = abs(s.rho_pcca - ref.rho_pcca),
        divergence   = (1 - cos_i) + (1 - cos_j) + (1 - r_i) + (1 - r_j),
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
        f"{'|r|_i':>6}  {'|r|_j':>6}  {'|Δρ|':>6}  {'D':>6}  Note"
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
        X_i_flat:     np.ndarray,
        X_j_flat:     np.ndarray,
        Z_flat:       Optional[np.ndarray],
        X_i_res:      np.ndarray,
        X_j_res:      np.ndarray,
        Wx_pcca:      np.ndarray,
        Wy_pcca:      np.ndarray,
        Wx_cca:       np.ndarray,
        Wy_cca:       np.ndarray,
        z_i_p:        np.ndarray,   # (n_trials, T) pCCA latents (no sign correction)
        z_j_p:        np.ndarray,
        n_trials:     int,
        T:            int,
        time_vec:     np.ndarray,
        step_label:   str,
        paired_trials: bool = True,
        n_cv_folds:   int   = 5,
        lam_cca:      float = LAMBDA_CCA,
        lam_hat:      float = LAMBDA_HAT,
) -> SupplementaryMetrics:
    """Full diagnostic bundle for one ablation step.

    The latent inputs need not be sign-corrected: θ and κ use |cos|, the variance
    fractions are sign-invariant, and the lag estimator depends only on the
    CCA-fixed relative pair sign.

    paired_trials
    -------------
    True  → section 5 uses the mean WITHIN-trial cross-correlogram (correct when
            trials are paired, e.g. the residual regime; does not collapse).
    False → section 5 uses the cross-correlation of the trial-averaged latents
            (appropriate when trials were independently shuffled).
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

    # ── 3.  Cross-validated ρ₁ (contiguous folds; β estimated on train only) ─
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

    # ── 4.  Variance partitioning (fractions of ‖X‖²_F) ───────────────────
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

    r2_priv_i = float(np.sum((X_i_res - np.outer(pi, wu_i)) ** 2)) / denom_i
    r2_priv_j = float(np.sum((X_j_res - np.outer(pj, wu_j)) ** 2)) / denom_j  # denom_j

    # ── 5.  Temporal lead–lag cross-correlation (regime-gated) ────────────
    dt_ms    = float(time_vec[1] - time_vec[0]) * 1000.0
    lag_bins = np.arange(-(T - 1), T)
    lag_ms   = lag_bins.astype(float) * dt_ms

    if paired_trials:
        # Mean within-trial cross-correlogram: average of normalised per-trial
        # cross-correlations.  Survives PSTH subtraction (does not collapse).
        acc = np.zeros(2 * T - 1, dtype=float)
        cnt = 0
        for s in range(z_i_p.shape[0]):
            a = z_i_p[s] - z_i_p[s].mean()
            b = z_j_p[s] - z_j_p[s].mean()
            sa = float(np.std(a)); sb = float(np.std(b))
            if sa < 1e-9 or sb < 1e-9:
                continue
            acc += np.correlate(a / sa, b / sb, mode="full") / T
            cnt += 1
        xcf = acc / max(cnt, 1)
    else:
        # Cross-correlation of the trial-averaged latents (signal regime).
        mi = z_i_p.mean(axis=0);  mi -= mi.mean()
        mj = z_j_p.mean(axis=0);  mj -= mj.mean()
        si = float(np.std(mi)) + 1e-12
        sj = float(np.std(mj)) + 1e-12
        xcf = np.correlate(mi / si, mj / sj, mode="full") / T

    max_lb   = min(int(150.0 / dt_ms), T - 1)
    mask     = np.abs(lag_bins) <= max_lb
    peak_abs = int(np.where(mask)[0][int(np.argmax(np.abs(xcf[mask])))])

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
    """Four-panel per-session summary (A: ρ₁; B: |cos θ|; C: |latent r| on the full
    trial×time matrices; D: composite divergence)."""
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

    # Panel C — |latent r| on the FULL trial×time matrices
    ax = axes[1, 0]
    ax.plot(x, lat_i, "s-", color=_CI_PCCA, lw=1.8, ms=5, label=TARGET_I)
    ax.plot(x, lat_j, "^-", color=_CJ_PCCA, lw=1.8, ms=5, label=TARGET_J)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("|Pearson r|  (trial×time latent vs CCA)", fontsize=9)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("(C)  Latent correlation to CCA baseline",
                 fontsize=8.5)
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
    """Six-panel per-session supplementary figure (A: ρ₁ in-sample vs CV; B: θ;
    C: κ; D/E: variance partition; F: latent cross-correlation)."""
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
    ax.set_xlabel(f"Lag (ms)   [+ = {TARGET_I} leads]", fontsize=9)
    ax.set_ylabel("Normalised cross-correlation", fontsize=9)
    _lag_kind = ("mean within-trial xcorr" if not SHUFFLE_TRIALS
                 else "trial-averaged xcorr")
    ax.set_title(
        f"(F)  pCCA latent cross-correlation  [{_lag_kind}]\n"
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
    """Aggregate ablation metrics across sessions, keyed by ablated region.

    Within each region, all metric lists are appended in a common session order,
    so element k of every list corresponds to the same session — this alignment
    is what makes the per-region Δρ in _add_derived_metrics well-defined."""
    _METRIC_KEYS = [
        "cos_sim_i", "cos_sim_j", "latent_r_i", "latent_r_j",
        "rho_abs_diff", "divergence",
        "rho_pcca", "rho_cca", "rho_cca_cv",
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
            for k in ("cos_sim_i", "cos_sim_j", "latent_r_i",
                      "latent_r_j", "rho_abs_diff", "divergence"):
                d[k].append(sim[k])
            d["rho_pcca"].append(sr.rho_pcca)
            d["rho_cca"].append(sess.rho_cca)
            d["rho_cca_cv"].append(sess.rho_cca_cv)
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


def _add_derived_metrics(
        agg: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, List[float]]]:
    """Append per-region residual-coupling change Δρ (in-sample and CV).

        delta_rho_insample[k] = rho_pcca[k]     − rho_cca[k]
        delta_rho_cv[k]       = rho1_cv_mean[k] − rho_cca_cv[k]

    Lists are aligned by session within each region (see aggregate_by_region)."""
    for r, d in agg.items():
        rp  = d.get("rho_pcca", [])
        rc  = d.get("rho_cca", [])
        rpc = d.get("rho1_cv_mean", [])
        rcc = d.get("rho_cca_cv", [])
        d["delta_rho_insample"] = [p - c for p, c in zip(rp, rc)]
        d["delta_rho_cv"]       = [p - c for p, c in zip(rpc, rcc)]
    return agg


def _sorted_region_order(agg: Dict[str, Dict[str, List[float]]]) -> List[str]:
    """Ablated regions sorted by ANATOMICAL_ORDER, restricted to those present."""
    present = set(agg.keys())
    ordered = [r for r in ANATOMICAL_ORDER if r in present]
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
    """Annotate session counts just above the top of each boxplot group."""
    tr = ax.get_xaxis_transform()
    for pos, n in zip(positions, counts):
        ax.text(pos, 0.05, f"n={n}", ha="center", va="bottom",
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
    """Paired boxplots for two metrics across ablated regions (series A at 3k+1,
    series B at 3k+2, ticks centred at 3k+1.5)."""
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
    # Empty lists confuse ax.boxplot; substitute a sentinel and rely on n=0 label.
    _safe_a = [d if len(d) else [np.nan] for d in data_a]
    _safe_b = [d if len(d) else [np.nan] for d in data_b]
    ax.boxplot(
        _safe_a, positions=pos_a,
        boxprops=dict(facecolor=color_a, alpha=0.65),
        whiskerprops=dict(color=color_a), capprops=dict(color=color_a),
        flierprops=dict(marker="o", ms=3.5, color=color_a, alpha=0.55),
        **_bp_kw,
    )
    ax.boxplot(
        _safe_b, positions=pos_b,
        boxprops=dict(facecolor=color_b, alpha=0.65),
        whiskerprops=dict(color=color_b), capprops=dict(color=color_b),
        flierprops=dict(marker="s", ms=3.5, color=color_b, alpha=0.55),
        **_bp_kw,
    )

    if hline is not None:
        ax.axhline(hline, color="#7F8C8D", ls=":", lw=1.0, alpha=0.75)

    _annotate_n(ax, tick_x, [len(data_a[k]) for k in range(n_reg)])

    ax.set_xticks(tick_x)
    ax.set_xticklabels(region_order, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(0, 3 * n_reg)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    handles = [
        Patch(facecolor=color_a, alpha=0.65, label=label_a),
        Patch(facecolor=color_b, alpha=0.65, label=label_b),
    ]
    if hline is not None and hline_label is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#7F8C8D", ls=":", lw=1.0,
                              label=hline_label))
    ax.legend(handles=handles, fontsize=7, frameon=False)
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
    """Single-metric boxplot across ablated regions."""
    x    = list(range(len(region_order)))
    data = [_clean(agg[r].get(key, [])) for r in region_order]
    safe = [d if len(d) else [np.nan] for d in data]

    ax.boxplot(
        safe, positions=x, widths=0.55, patch_artist=True,
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
    """Cross-session variance partition stacked bars with SEM and session dots."""
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

        ax.bar(x_pos[k], mn,  color=_C_NUIS,    alpha=0.85, width=0.6)
        ax.bar(x_pos[k], mc,  bottom=mn, color=comm_color, alpha=0.85, width=0.6)
        ax.errorbar(
            x_pos[k], mn + mc, yerr=sem_tot,
            fmt="none", color="#2C3E50", capsize=4, lw=1.2, zorder=5,
        )
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


def _session_two_box(
        ax:        plt.Axes,
        data_a:    List[float],
        data_b:    List[float],
        label_a:   str,
        label_b:   str,
        color_a:   str,
        color_b:   str,
        ylabel:    str,
        title:     str,
        seed:      int = 7,
) -> None:
    """Two session-level boxplots (e.g. ρ_CCA in-sample vs CV) with jittered dots."""
    rng = np.random.default_rng(seed)
    da  = _clean(data_a)
    db  = _clean(data_b)
    ax.boxplot(
        [da if da else [np.nan], db if db else [np.nan]],
        positions=[0, 1], widths=0.5, patch_artist=True,
        medianprops=dict(color="k", lw=1.5), manage_ticks=False,
        boxprops=dict(alpha=0.6),
    )
    # Colour the two boxes individually.
    for patch, col in zip(ax.patches, (color_a, color_b)):
        patch.set_facecolor(col)
    for xpos, vals, col in ((0, da, color_a), (1, db, color_b)):
        if vals:
            ax.scatter(xpos + rng.uniform(-0.12, 0.12, len(vals)), vals,
                       color=col, s=20, alpha=0.8, zorder=5, linewidths=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{label_a}\n(n={len(da)})", f"{label_b}\n(n={len(db)})"],
                       fontsize=8)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
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
    """Cross-session summary: panels B (|cos θ|), C (|latent r|), D (divergence)."""
    fig, axes = plt.subplots(
        1, 3, figsize=(20, 6),
        gridspec_kw={"wspace": 0.38},
    )

    _paired_boxplot(
        axes[0], region_order, agg,
        key_a="cos_sim_i", key_b="cos_sim_j",
        label_a=TARGET_I, label_b=TARGET_J,
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel=r"$|\cos\theta|$  vs CCA baseline",
        title=("(A)  Weight-vector cosine similarity to CCA baseline\n"
               "per ablated region  ·  boxes = session distribution"),
        ylim=(-0.05, 1.10),
        hline=1.0, hline_label="perfect match",
    )

    _paired_boxplot(
        axes[1], region_order, agg,
        key_a="latent_r_i", key_b="latent_r_j",
        label_a=TARGET_I, label_b=TARGET_J,
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel="|Pearson r|  (trial×time latent vs CCA)",
        title=("(B)  Latent correlation to CCA baseline\n"
               "full trial×time matrices  ·  boxes = session distribution"),
        ylim=(-0.05, 1.10),
        hline=1.0, hline_label="perfect match",
    )

    _single_boxplot(
        axes[2], region_order, agg,
        key="divergence", color=_CI_PCCA,
        ylabel="Composite divergence  D  (max = 4)",
        title=("(C)  Divergence D from CCA baseline\n"
               "per ablated region  ·  boxes = session distribution"),
        ylim=(0.0, 4.10),
    )

    fig.suptitle(
        f"Cross-session ablation summary  |  {TARGET_I} ↔ {TARGET_J}  "
        f"[{REGIME_TAG}]\n"
        f"n = {len(SESSIONS_TO_RUN)} sessions  ·  reference: CCA baseline (Z = ∅)",
        fontsize=12, fontweight="bold",
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [cross-session summary]  saved: {output_path}")
    return fig


def plot_cross_session_coupling(
        agg:             Dict[str, Dict[str, List[float]]],
        region_order:    List[str],
        all_rho_cca:     List[float],
        all_rho_cca_cv:  List[float],
        output_path:     Optional[Path] = None,
) -> plt.Figure:
    """Residual-coupling cross-session diagnostics.

    (A) per-session residual CCA coupling ρ₁ — in-sample vs 5-fold CV.
    (B) per-region Δρ = ρ₁^{pCCA(Z={r})} − ρ₁^{CCA}, in-sample and CV, with a
        zero reference line.  Δρ < 0 ⇒ region r carried shared variance whose
        removal weakens the residual coupling; Δρ > 0 ⇒ removal de-confounded an
        apparent suppression.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(19, 6),
        gridspec_kw={"wspace": 0.26, "width_ratios": [1.0, 2.6]},
    )

    # Panel A — session-level residual CCA coupling
    _session_two_box(
        axes[0],
        data_a=all_rho_cca, data_b=all_rho_cca_cv,
        label_a="in-sample", label_b="5-fold CV",
        color_a=_C_INSAMPLE, color_b=_C_CV,
        ylabel="CCA coupling  ρ₁  (Z = ∅)",
        title=("(A)  CCA coupling across sessions\n"
               "boxes = session distribution  ·  dots = sessions"),
    )

    # Panel B — per-region Δρ (in-sample & CV)
    _paired_boxplot(
        axes[1], region_order, agg,
        key_a="delta_rho_insample", key_b="delta_rho_cv",
        label_a="Δρ  in-sample", label_b="Δρ  5-fold CV",
        color_a=_C_INSAMPLE, color_b=_C_CV,
        ylabel=r"$\Delta\rho = \rho_1^{\mathrm{pCCA}(Z=\{r\})} - \rho_1^{\mathrm{CCA}}$",
        title=("(B)  Change in residual coupling from single-region ablation\n"),
               # "per ablated region  ·  boxes = session distribution"),
        hline=0.0, hline_label="no change (Δρ = 0)",
    )

    fig.suptitle(
        f"Cross-session residual coupling  |  {TARGET_I} ↔ {TARGET_J}  "
        f"[{REGIME_TAG}]   "
        f"n = {len(SESSIONS_TO_RUN)} sessions  ·  ρ₁ = corr_(t,s)(z_I, z_J)",
        fontsize=12, fontweight="bold",
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [cross-session residual coupling]  saved: {output_path}")
    return fig


def plot_cross_session_supplementary(
        agg:          Dict[str, Dict[str, List[float]]],
        region_order: List[str],
        all_rho_cca:  List[float],
        output_path:  Optional[Path] = None,
) -> plt.Figure:
    """Cross-session supplementary: panels A–E (panel F omitted)."""
    fig, axes = plt.subplots(
        2, 3, figsize=(22, 12),
        gridspec_kw={"hspace": 0.55, "wspace": 0.38},
    )

    rho_cca_arr  = np.array(all_rho_cca)
    rho_cca_mean = float(rho_cca_arr.mean())
    rho_cca_sem  = float(rho_cca_arr.std() / np.sqrt(max(len(rho_cca_arr), 1)))

    # Panel A — ρ₁ in-sample vs. CV
    ax = axes[0, 0]
    _paired_boxplot(
        ax, region_order, agg,
        key_a="rho_pcca", key_b="rho1_cv_mean",
        label_a="ρ₁  in-sample", label_b="ρ₁  5-fold CV",
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel="Canonical correlation  ρ₁",
        title=("(A)  In-sample vs. cross-validated  ρ₁\n"
               "Large CV gap → overfitting to nuisance regression"),
        ylim=(-0.05, 1.05),
    )
    n_reg = len(region_order)
    ax.axhspan(
        rho_cca_mean - rho_cca_sem, rho_cca_mean + rho_cca_sem,
        color="#7F8C8D", alpha=0.18,
        label=f"CCA ρ₁ = {rho_cca_mean:.3f} ± {rho_cca_sem:.3f}",
    )
    ax.axhline(rho_cca_mean, color="#7F8C8D", ls=":", lw=1.0, alpha=0.80)
    ax.legend(fontsize=6.5, frameon=False)

    # Panel B — θ
    _paired_boxplot(
        axes[0, 1], region_order, agg,
        key_a="theta_i_deg", key_b="theta_j_deg",
        label_a=TARGET_I, label_b=TARGET_J,
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel="θ  CCA–pCCA weight angle  (°)",
        title=("(B)  CCA–pCCA rotation angle  θ\n"
               r"θ → 90°: nuisance removal uncovered an orthogonal axis"),
        ylim=(-3.0, 95.0),
        hline=90.0, hline_label="90°  (fully orthogonal)",
    )

    # Panel C — κ
    ax = axes[0, 2]
    _paired_boxplot(
        ax, region_order, agg,
        key_a="kappa_i", key_b="kappa_j",
        label_a=f"{TARGET_I}  |cos∠(w_IJ, w_IZ)|",
        label_b=f"{TARGET_J}  |cos∠(w_JI, w_JZ)|",
        color_a=_CI_PCCA, color_b=_CJ_PCCA,
        ylabel="κ  cross-analysis collinearity",
        title=("(C)  Weight collinearity  κ\n"
               r"κ → 1: shared-noise amplification;  κ → 0: clean separation"),
        ylim=(-0.05, 1.10),
    )
    ax.fill_between(
        [-1, 3 * n_reg + 1], [0.8, 0.8], [1.05, 1.05],
        color="#FADBD8", alpha=0.35, zorder=0,
    )

    # Panel D — Variance partition TARGET_I
    _variance_partition_bars(
        axes[1, 0], region_order, agg,
        key_nuis="r2_nuis_i", key_comm="r2_comm_i",
        comm_color=_CI_PCCA,
        ylabel="Fraction of total variance",
        title=(f"(D)  Variance partition — {TARGET_I}\n"
               "mean ± SEM across sessions  ·  dots = individual sessions"),
        ylim=(0.0, 0.20),
    )

    # Panel E — Variance partition TARGET_J
    _variance_partition_bars(
        axes[1, 1], region_order, agg,
        key_nuis="r2_nuis_j", key_comm="r2_comm_j",
        comm_color=_CJ_PCCA,
        ylabel="Fraction of total variance",
        title=(f"(E)  Variance partition — {TARGET_J}\n"
               "mean ± SEM across sessions  ·  dots = individual sessions"),
        ylim=(0.0, 0.20),
    )

    # Panel (1,2) — Notes
    ax = axes[1, 2]
    ax.axis("off")
    notes = (
        "Visualisation notes\n"
        "────────────────────────────────────\n"
        "Regime: "
        f"SUBTRACT_PSTH={SUBTRACT_PSTH}, SHUFFLE_TRIALS={SHUFFLE_TRIALS}\n\n"
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
        f"Cross-session supplementary diagnostics  |  {TARGET_I} ↔ {TARGET_J}  "
        f"[{REGIME_TAG}]\n"
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
    """Run single-region ablation for one session, save per-session figures, and
    return a SessionAblationData cache for cross-session aggregation."""
    for t in (TARGET_I, TARGET_J):
        if t not in region_spikes:
            warnings.warn(f"  [{session_name}]  Target region '{t}' missing — skipping.")
            return None

    nuisance_all: List[str] = [
        r for r in ANATOMICAL_ORDER
        if r in region_spikes and r not in (TARGET_I, TARGET_J)
    ]
    if not nuisance_all:
        warnings.warn(f"  [{session_name}]  No nuisance regions found — skipping.")
        return None

    print(f"  [{session_name}]  Ablating {len(nuisance_all)} nuisance regions: "
          f"{nuisance_all}")

    paired_trials = not SHUFFLE_TRIALS

    # ── Flat matrices (regime-aware) ──────────────────────────────────────
    # Targets: PSTH-subtracted and (optionally) independently trial-shuffled.
    X_i_flat = _zscore_flat(region_spikes[TARGET_I], subtract_psth=SUBTRACT_PSTH,
                            shuffle_trials=SHUFFLE_TRIALS, rng=_rng)
    X_j_flat = _zscore_flat(region_spikes[TARGET_J], subtract_psth=SUBTRACT_PSTH,
                            shuffle_trials=SHUFFLE_TRIALS, rng=_rng2)
    # Nuisance: PSTH-subtracted, never shuffled.
    nuisance_flat: Dict[str, np.ndarray] = {
        r: _zscore_flat(region_spikes[r], subtract_psth=SUBTRACT_PSTH)
        for r in nuisance_all
    }

    time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)

    # ── CCA baseline (in-sample + 5-fold CV) ──────────────────────────────
    Wx_cca, Wy_cca, rho_cca = ridge_cca(X_i_flat, X_j_flat)
    rho_cca_val    = float(rho_cca[0])
    rho_cca_cv_val = _cv_rho(X_i_flat, X_j_flat, None, n_trials, T)
    print(f"  [{session_name}]  CCA baseline  ρ₁ = {rho_cca_val:.4f}  "
          f"(CV {rho_cca_cv_val:.4f})")

    cca_ref = StepResult(
        label="CCA_baseline",
        nuisance_regions=[],
        rho_pcca=rho_cca_val,
        Wx=Wx_cca,
        Wy=Wy_cca,
        z_i_lat=latent_projections(X_i_flat, Wx_cca[:, 0], n_trials, T),
        z_j_lat=latent_projections(X_j_flat, Wy_cca[:, 0], n_trials, T),
    )

    # ── Single-region ablation loop ───────────────────────────────────────
    ablation_results:  List[StepResult]          = []
    supp_list:         List[SupplementaryMetrics] = []

    for abl_idx, region in enumerate(nuisance_all):
        Z_single = nuisance_flat[region]
        Wx_p, Wy_p, rho_p, X_i_res, X_j_res = pcca(X_i_flat, X_j_flat, Z_single)

        z_i_lat = latent_projections(X_i_res, Wx_p[:, 0], n_trials, T)  # (n_tr, T)
        z_j_lat = latent_projections(X_j_res, Wy_p[:, 0], n_trials, T)

        label = f"abl{abl_idx:02d}_{region}"
        step  = StepResult(
            label=label,
            nuisance_regions=[region],
            rho_pcca=float(rho_p[0]),
            Wx=Wx_p,
            Wy=Wy_p,
            z_i_lat=z_i_lat,
            z_j_lat=z_j_lat,
        )
        ablation_results.append(step)

        supp = compute_supplementary_metrics(
            X_i_flat      = X_i_flat,
            X_j_flat      = X_j_flat,
            Z_flat        = Z_single,
            X_i_res       = X_i_res,
            X_j_res       = X_j_res,
            Wx_pcca       = Wx_p,
            Wy_pcca       = Wy_p,
            Wx_cca        = Wx_cca,
            Wy_cca        = Wy_cca,
            z_i_p         = z_i_lat,
            z_j_p         = z_j_lat,
            n_trials      = n_trials,
            T             = T,
            time_vec      = time_vec,
            step_label    = label,
            paired_trials = paired_trials,
        )
        supp_list.append(supp)

        print(
            f"    [{session_name}]  abl {abl_idx:02d}  Z = {{{region:<8}}}  "
            f"ρ₁ = {rho_p[0]:.4f}  Δρ = {rho_p[0] - rho_cca_val:+.4f}  "
            f"θ_i = {supp.theta_i_deg:.1f}°  κ_i = {supp.kappa_i:.3f}"
        )

    # ── Similarity vs. CCA baseline ───────────────────────────────────────
    sims_vs_cca = [compute_similarity(s, cca_ref) for s in ablation_results]
    idx_c, idx_f, _ = identify_extremes(ablation_results, cca_ref, exclude_ref=False)

    print(f"\n  [{session_name}]  Similarity to CCA baseline:")
    print_similarity_table(ablation_results, sims_vs_cca, "CCA_baseline", idx_c, idx_f)

    # ── Per-session figures ───────────────────────────────────────────────
    plt.close(plot_summary_figure(
        step_results = ablation_results,
        sims         = sims_vs_cca,
        idx_closest  = idx_c,
        idx_furthest = idx_f,
        ref_label    = "CCA baseline  (Z = ∅)",
        title        = (f"Single-region ablation  |  {session_name}  [{REGIME_TAG}]\n"
                        f"{TARGET_I} ↔ {TARGET_J}   reference: CCA baseline"),
        output_path  = out_summary_dir / f"{session_name}_part1_summary_vs_CCA.png",
    ))

    plt.close(plot_supplementary_panel(
        supp_list     = supp_list,
        rho_pcca_list = [s.rho_pcca for s in ablation_results],
        rho_cca       = rho_cca_val,
        title         = (f"Part 1 — Supplementary Diagnostics  |  {session_name}  "
                         f"[{REGIME_TAG}]\n"
                         f"{TARGET_I} ↔ {TARGET_J}   (single-region ablation)"),
        output_path   = out_supp_dir / f"{session_name}_part1_supplementary.png",
    ))

    return SessionAblationData(
        session_name = session_name,
        rho_cca      = rho_cca_val,
        rho_cca_cv   = rho_cca_cv_val,
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

    out_summary = (BASE_DIR / "Paper_output" / "pcca_cross_session_summary"
                   / f"{TARGET_I}_{TARGET_J}_{REGIME_TAG}")
    out_supp    = (BASE_DIR / "Paper_output" / "pcca_cross_session_supplementary"
                   / f"{TARGET_I}_{TARGET_J}_{REGIME_TAG}")
    out_summary.mkdir(parents=True, exist_ok=True)
    out_supp.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"pCCA Cross-Session Ablation  |  {TARGET_I} ↔ {TARGET_J}  [{REGIME_TAG}]")
    print(f"  SUBTRACT_PSTH = {SUBTRACT_PSTH}   SHUFFLE_TRIALS = {SHUFFLE_TRIALS}")
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

    agg            = aggregate_by_region(all_session_data)
    agg            = _add_derived_metrics(agg)
    region_order   = _sorted_region_order(agg)
    all_rho_cca    = [s.rho_cca    for s in all_session_data]
    all_rho_cca_cv = [s.rho_cca_cv for s in all_session_data]

    print(f"  Regions in aggregation  ({len(region_order)}): {region_order}")
    for r in region_order:
        n_sess = len(_clean(agg[r].get("rho_pcca", [])))
        print(f"    {r:<12}  n = {n_sess} sessions")

    # ── Cross-session summary figure (B, C, D) ─────────────────────────────
    plt.close(plot_cross_session_summary(
        agg          = agg,
        region_order = region_order,
        all_rho_cca  = all_rho_cca,
        output_path  = out_summary / "cross_session_summary_ABC.png",
    ))

    # ── Cross-session residual coupling figure (NEW) ──────────────────────
    plt.close(plot_cross_session_coupling(
        agg             = agg,
        region_order    = region_order,
        all_rho_cca     = all_rho_cca,
        all_rho_cca_cv  = all_rho_cca_cv,
        output_path     = out_summary / "cross_session_coupling.png",
    ))

    # ── Cross-session supplementary figure (A–E) ──────────────────────────
    plt.close(plot_cross_session_supplementary(
        agg          = agg,
        region_order = region_order,
        all_rho_cca  = all_rho_cca,
        output_path  = out_supp / "cross_session_supplementary_ABCDE.png",
    ))

    print("\n🎉  All done.")
    print(f"  Summary figures  → {out_summary}")
    print(f"  Supp figures     → {out_supp}")


if __name__ == "__main__":
    main()
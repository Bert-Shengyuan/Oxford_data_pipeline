"""
pcca_cross_session_ablation.py
==============================

Batch cross-session pCCA single-region ablation pipeline for the
MOs ↔ VPMPO target pair, regime-aware (signal vs. residual), now extended
with:

  (A) Single-session multi-state overlay figures (mirrors the per-session
      2x2 summary / 2x3 supplementary layout used in this file, overlaid
      across raw / shuffled / psth-subtracted states, mode="all3"|"first_two").

  (B) Cross-session two-regime comparison figures (psth0_shuf0 vs psth0_shuf1):
        - divergence boxplots per region, side by side
        - residual coupling: 5-fold CV CCA coupling (panel A) and per-region
          5-fold CV Δρ (panel B), side by side

(... original module docstring retained verbatim above this point in your
     actual file — kept here only abbreviated for message length ...)
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
#


TARGET_I     = "MOs"
TARGET_J     = "MOp"
SESSIONS_TO_RUN: List[str] = [
    "yp012_220208",
    "yp013_220209",
    "yp013_220211",
    "yp013_220212",
    "yp020_220401",
    "yp020_220407",
    "yp021_220331",
    "yp021_220401",
    "yp021_220402",
    "yp021_220403",
    "yp021_220404",
    "yp021_220407",
]



# TARGET_I     = "MOs"
# TARGET_J     = "VALVM"
#
# SESSIONS_TO_RUN: List[str] = [
#     "yp020_220331",
#     "yp020_220407",
#     "yp021_220401",
#     "yp021_220403",
#     "yp021_220404",
#     "yp021_220407",
# ]

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


# ── Regime switches (default single-run behaviour; unchanged) ──────────────
SUBTRACT_PSTH:  bool = False
SHUFFLE_TRIALS: bool = False
REGIME_TAG = f"psth{int(SUBTRACT_PSTH)}_shuf{int(SHUFFLE_TRIALS)}"

# NOTE: _rng / _rng2 are kept for any external code that imports them, but
# run_single_ablation() no longer reads them directly — see section 10b.
_rng  = np.random.default_rng(42)
_rng2 = np.random.default_rng(43)

LAMBDA_CCA   = 1e-4
LAMBDA_HAT   = 1e-4
N_COMPONENTS = 5
TIME_RANGE_S = (-1.5, 3.0)

# ── Colour palette ───────────────────────────────────────────────────────
_CI_PCCA    = "#C0392B"
_CJ_PCCA    = "#2471A3"
_C_NUIS     = "#5D6D7E"
_C_DIV      = "#E74C3C"
_C_SIM      = "#27AE60"
_C_NEUTRAL  = "#CACFD2"
_C_INSAMPLE = "#8E44AD"
_C_CV       = "#16A085"


# =============================================================================
# 0b.  Multi-state / multi-regime configuration  (NEW)
# =============================================================================

# ── Task A: per-session 3-state overlay (raw / shuffled / psth-subtracted) ──
STATE_ORDER: List[str] = ["raw", "shuffled", "psth_subtracted"]

STATE_CONFIG: Dict[str, Dict[str, bool]] = {
    "raw":             dict(subtract_psth=False, shuffle_trials=False),
    "shuffled":        dict(subtract_psth=False, shuffle_trials=True),
    "psth_subtracted": dict(subtract_psth=True,  shuffle_trials=False),
}

STATE_STYLE: Dict[str, Dict] = {
    "raw":             dict(ls='-',  marker='o', alpha=1.00, display="Raw"),
    "shuffled":        dict(ls='--', marker='s', alpha=0.85, display="Shuffled"),
    "psth_subtracted": dict(ls=':',  marker='^', alpha=0.85, display="PSTH-subtracted"),
}

_STATE_BAR_COLOR: Dict[str, str] = {
    "raw":             "#5DADE2",
    "shuffled":        "#F5B041",
    "psth_subtracted": "#AF7AC5",
}

MODE_STATES: Dict[str, List[str]] = {
    "all3":      ["raw", "shuffled", "psth_subtracted"],
    "first_two": ["raw", "shuffled"],
}

# ── Task B: cross-session 2-regime comparison (psth0_shuf0 vs psth0_shuf1) ──
_C_REGIME_RAW  = "#5DADE2"   # psth0_shuf0
_C_REGIME_SHUF = "#F5B041"   # psth0_shuf1


# =============================================================================
# 1.  Core mathematics   (unchanged)
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

    if not subtract_psth and not shuffle_trials:
        return flat.T

    X = flat.reshape(n, T, n_trials).transpose(2, 0, 1)

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
    return flat.T


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


def latent_projections(
        X_flat: np.ndarray,
        w: np.ndarray,
        n_trials: int,
        T: int,
) -> np.ndarray:
    return (X_flat @ w).reshape(T, n_trials).T


def _cos_sim_abs(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.abs(np.dot(a, b)) / (na * nb))


def _abs_pearson(A: np.ndarray, B: np.ndarray) -> float:
    a = np.asarray(A).ravel()
    b = np.asarray(B).ravel()
    if a.size != b.size or a.size < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.abs(np.clip(pearsonr(a, b)[0], -1.0, 1.0)))


def _trial_rows(trial_idx: np.ndarray, n_trials: int, T: int) -> np.ndarray:
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
# 2.  Data loading   (unchanged)
# =============================================================================

def load_region_spikes(
        session_path: str,
) -> Tuple[Dict[str, np.ndarray], int, int]:
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
# 3.  Data containers   (SessionAblationData gains optional bookkeeping fields)
# =============================================================================

class StepResult:
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
            z_i_lat: np.ndarray,
            z_j_lat: np.ndarray,
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
    """Per-session ablation cache consumed by the cross-session aggregator
    AND (NEW) by the per-session multi-state overlay figures."""
    session_name:  str
    rho_cca:       float
    rho_cca_cv:    float
    region_names:  List[str]
    step_results:  List[StepResult]
    sims_vs_cca:   List[Dict[str, float]]
    supp_metrics:  List[SupplementaryMetrics]
    # ── NEW bookkeeping (optional; defaults keep old call sites valid) ─────
    state_key:      Optional[str]  = None
    subtract_psth:  Optional[bool] = None
    shuffle_trials: Optional[bool] = None
    regime_tag:     Optional[str]  = None


# =============================================================================
# 4.  Similarity metrics   (unchanged)
# =============================================================================

def compute_similarity(s: StepResult, ref: StepResult) -> Dict[str, float]:
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
# 5.  Supplementary metrics computation   (unchanged)
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
        z_i_p:        np.ndarray,
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
    w_p_i = Wx_pcca[:, 0];  w_p_j = Wy_pcca[:, 0]
    w_c_i = Wx_cca[:, 0];   w_c_j = Wy_cca[:, 0]

    theta_i_deg = float(np.degrees(
        np.arccos(np.clip(_cos_sim_abs(w_p_i, w_c_i), 0.0, 1.0))
    ))
    theta_j_deg = float(np.degrees(
        np.arccos(np.clip(_cos_sim_abs(w_p_j, w_c_j), 0.0, 1.0))
    ))

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

    denom_i = float(np.sum(X_i_flat ** 2)) + 1e-12
    denom_j = float(np.sum(X_j_flat ** 2)) + 1e-12

    wu_i = w_p_i / (np.linalg.norm(w_p_i) + 1e-12)
    wu_j = w_p_j / (np.linalg.norm(w_p_j) + 1e-12)

    X_i_hat = X_i_flat - X_i_res;  X_j_hat = X_j_flat - X_j_res
    r2_nuis_i = float(np.sum(X_i_hat ** 2)) / denom_i
    r2_nuis_j = float(np.sum(X_j_hat ** 2)) / denom_j

    pi = X_i_res @ wu_i;  pj = X_j_res @ wu_j
    r2_comm_i = float(np.sum(pi ** 2)) / denom_i
    r2_comm_j = float(np.sum(pj ** 2)) / denom_j

    r2_priv_i = float(np.sum((X_i_res - np.outer(pi, wu_i)) ** 2)) / denom_i
    r2_priv_j = float(np.sum((X_j_res - np.outer(pj, wu_j)) ** 2)) / denom_j

    dt_ms    = float(time_vec[1] - time_vec[0]) * 1000.0
    lag_bins = np.arange(-(T - 1), T)
    lag_ms   = lag_bins.astype(float) * dt_ms

    if paired_trials:
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
#     (plot_summary_figure unchanged; plot_supplementary_panel gains an
#      explicit `shuffle_trials` param instead of reading the module global)
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
        2, 2, figsize=(14, 8),
        gridspec_kw={"hspace": 0.52, "wspace": 0.35},
    )

    ax = axes[0, 0]
    ax.plot(x, rhos, "o-", color=_CI_PCCA, lw=1.8, ms=5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel(r"pCCA  $\rho_1$", fontsize=9)
    ax.set_title("(A)  Dominant canonical correlation", fontsize=9)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

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

    ax = axes[1, 0]
    ax.plot(x, lat_i, "s-", color=_CI_PCCA, lw=1.8, ms=5, label=TARGET_I)
    ax.plot(x, lat_j, "^-", color=_CJ_PCCA, lw=1.8, ms=5, label=TARGET_J)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("|Pearson r|  (trial×time latent vs CCA)", fontsize=9)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("(C)  Latent correlation to CCA baseline", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

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
        shuffle_trials: bool = SHUFFLE_TRIALS,   # ← NEW (was a global read)
        output_path:    Optional[Path] = None,
) -> plt.Figure:
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
    _lag_kind = ("mean within-trial xcorr" if not shuffle_trials
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
# 6b.  Single-session multi-state overlay figures  (NEW — Task A)
# =============================================================================

def plot_summary_figure_multistate(
        bundles: Dict[str, SessionAblationData],
        mode: str = "all3",
        title: str = "",
        output_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlay of plot_summary_figure() (2x2: ρ₁ / cos / latent-r / divergence)
    across 2 or 3 preprocessing states for ONE session. State is encoded by
    linestyle (STATE_STYLE); region identity keeps its usual colour."""
    if mode not in MODE_STATES:
        raise ValueError(f"mode must be one of {list(MODE_STATES)}; got {mode!r}")
    states = [s for s in MODE_STATES[mode] if s in bundles]
    if len(states) < 2:
        raise ValueError(f"Need ≥2 states present for mode={mode!r}; found {states}.")

    ref0   = bundles[states[0]]
    labels = [s.label for s in ref0.step_results]
    x      = np.arange(len(labels))
    n_states = len(states)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.32})

    # Panel A — ρ₁
    ax = axes[0, 0]
    for st in states:
        b, sty = bundles[st], STATE_STYLE[st]
        rhos = [s.rho_pcca for s in b.step_results]
        ax.plot(x, rhos, ls=sty['ls'], marker=sty['marker'], color=_CI_PCCA,
                lw=1.8, ms=6, alpha=sty['alpha'], label=sty['display'])
        ax.axhline(b.rho_cca, color='#7F8C8D', ls=sty['ls'], lw=1.0,
                   alpha=0.5 * sty['alpha'])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8)
    ax.set_ylabel(r"pCCA  $\rho_1$", fontsize=10)
    ax.set_title("(A)  Dominant canonical correlation", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)

    # Panel B — cosine similarity
    ax = axes[0, 1]
    for st in states:
        b, sty = bundles[st], STATE_STYLE[st]
        cos_i = [d['cos_sim_i'] for d in b.sims_vs_cca]
        cos_j = [d['cos_sim_j'] for d in b.sims_vs_cca]
        ax.plot(x, cos_i, ls=sty['ls'], marker='s', color=_CI_PCCA, lw=1.8,
                ms=6, alpha=sty['alpha'], label=f"{TARGET_I} ({sty['display']})")
        ax.plot(x, cos_j, ls=sty['ls'], marker='^', color=_CJ_PCCA, lw=1.8,
                ms=6, alpha=sty['alpha'], label=f"{TARGET_J} ({sty['display']})")
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8)
    ax.set_ylabel(r"$|\cos\theta|$  vs CCA baseline", fontsize=10)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("(B)  Weight-vector cosine similarity to CCA baseline", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)

    # Panel C — latent r
    ax = axes[1, 0]
    for st in states:
        b, sty = bundles[st], STATE_STYLE[st]
        lat_i = [d['latent_r_i'] for d in b.sims_vs_cca]
        lat_j = [d['latent_r_j'] for d in b.sims_vs_cca]
        ax.plot(x, lat_i, ls=sty['ls'], marker='s', color=_CI_PCCA, lw=1.8,
                ms=6, alpha=sty['alpha'], label=f"{TARGET_I} ({sty['display']})")
        ax.plot(x, lat_j, ls=sty['ls'], marker='^', color=_CJ_PCCA, lw=1.8,
                ms=6, alpha=sty['alpha'], label=f"{TARGET_J} ({sty['display']})")
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8)
    ax.set_ylabel("|Pearson r|  (trial×time latent vs CCA)", fontsize=10)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("(C)  Latent correlation to CCA baseline", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)

    # Panel D — divergence, clustered bars (one cluster per step, one bar/state)
    ax = axes[1, 1]
    width = 0.8 / n_states
    for k, st in enumerate(states):
        b = bundles[st]
        div = [d['divergence'] for d in b.sims_vs_cca]
        offset = (k - (n_states - 1) / 2) * width
        ax.bar(x + offset, div, width=width * 0.92, color=_STATE_BAR_COLOR[st],
               alpha=0.88, label=STATE_STYLE[st]['display'])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8)
    ax.set_ylabel("Composite divergence  D  (max = 4)", fontsize=10)
    ax.set_title("(D)  Divergence from CCA baseline", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)

    fig.suptitle(
        f"{title}   |   states: " + " vs. ".join(STATE_STYLE[s]['display'] for s in states),
        fontsize=13, fontweight='bold',
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  [multistate summary]  saved: {output_path}")
    return fig


def plot_supplementary_panel_multistate(
        bundles: Dict[str, SessionAblationData],
        mode: str = "all3",
        title: str = "",
        output_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlay of plot_supplementary_panel() (2x3) across 2 or 3 states.
    A/B/C overlay per-step lines by state. D/E collapse to state-level means
    (full per-step detail already lives in the single-state figure). F shows
    one step-averaged lag curve per state."""
    if mode not in MODE_STATES:
        raise ValueError(f"mode must be one of {list(MODE_STATES)}; got {mode!r}")
    states = [s for s in MODE_STATES[mode] if s in bundles]
    if len(states) < 2:
        raise ValueError(f"Need ≥2 states present for mode={mode!r}; found {states}.")

    ref0   = bundles[states[0]]
    labels = [s.step_label for s in ref0.supp_metrics]
    x      = np.arange(len(labels))
    n_states = len(states)

    fig, axes = plt.subplots(2, 3, figsize=(20, 9.6),
                             gridspec_kw={'hspace': 0.62, 'wspace': 0.38})

    def _xax(ax: plt.Axes) -> None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=7)
        ax.grid(alpha=0.22, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    # Panel A — ρ₁ in-sample & CV
    ax = axes[0, 0]
    for st in states:
        b, sty = bundles[st], STATE_STYLE[st]
        rho_in = [r.rho_pcca for r in b.step_results]
        rho_cv = [s.rho1_cv_mean for s in b.supp_metrics]
        ax.plot(x, rho_in, ls=sty['ls'], marker='o', color=_CI_PCCA, lw=1.6,
                ms=4, alpha=sty['alpha'], label=f"ρ₁ in-sample ({sty['display']})")
        ax.plot(x, rho_cv, ls=sty['ls'], marker='s', color=_CJ_PCCA, lw=1.6,
                ms=4, alpha=sty['alpha'], label=f"ρ₁ 5-fold CV ({sty['display']})")
        ax.axhline(b.rho_cca, color='#7F8C8D', ls=sty['ls'], lw=1.0,
                   alpha=0.5 * sty['alpha'])
    ax.set_ylabel('Canonical correlation  ρ₁', fontsize=9)
    ax.set_title('(A)  ρ₁ in-sample vs. cross-validated', fontsize=8.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, frameon=False)
    _xax(ax)

    # Panel B — θ
    ax = axes[0, 1]
    for st in states:
        b, sty = bundles[st], STATE_STYLE[st]
        th_i = [s.theta_i_deg for s in b.supp_metrics]
        th_j = [s.theta_j_deg for s in b.supp_metrics]
        ax.plot(x, th_i, ls=sty['ls'], marker='s', color=_CI_PCCA, lw=1.6,
                ms=4, alpha=sty['alpha'], label=f"{TARGET_I} ({sty['display']})")
        ax.plot(x, th_j, ls=sty['ls'], marker='^', color=_CJ_PCCA, lw=1.6,
                ms=4, alpha=sty['alpha'], label=f"{TARGET_J} ({sty['display']})")
    ax.axhline(90, color='#AAB7B8', ls='--', lw=0.9, alpha=0.6)
    ax.set_ylabel('θ  CCA–pCCA weight angle  (°)', fontsize=9)
    ax.set_ylim(-3, 95)
    ax.set_title('(B)  CCA–pCCA rotation angle  θ', fontsize=8.5)
    ax.legend(fontsize=6, frameon=False)
    _xax(ax)

    # Panel C — κ
    ax = axes[0, 2]
    for st in states:
        b, sty = bundles[st], STATE_STYLE[st]
        ki = [s.kappa_i for s in b.supp_metrics]
        kj = [s.kappa_j for s in b.supp_metrics]
        ax.plot(x, ki, ls=sty['ls'], marker='s', color=_CI_PCCA, lw=1.6,
                ms=4, alpha=sty['alpha'], label=f"{TARGET_I} ({sty['display']})")
        ax.plot(x, kj, ls=sty['ls'], marker='^', color=_CJ_PCCA, lw=1.6,
                ms=4, alpha=sty['alpha'], label=f"{TARGET_J} ({sty['display']})")
    ax.axhline(1.0, color=_C_DIV, ls='--', lw=0.9, alpha=0.55)
    ax.axhline(0.0, color=_C_SIM, ls='--', lw=0.9, alpha=0.55)
    ax.set_ylabel('κ  cross-analysis collinearity', fontsize=9)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title('(C)  Weight collinearity  κ', fontsize=8.5)
    ax.legend(fontsize=6, frameon=False)
    _xax(ax)

    # Panels D & E — state-level mean variance partition
    for ax, region_label, panel_tag, get_rn, get_rc in (
        (axes[1, 0], TARGET_I, 'D', lambda s: s.r2_nuis_i, lambda s: s.r2_comm_i),
        (axes[1, 1], TARGET_J, 'E', lambda s: s.r2_nuis_j, lambda s: s.r2_comm_j),
    ):
        xs = np.arange(n_states)
        rn_mean = [np.mean([get_rn(s) for s in bundles[st].supp_metrics]) for st in states]
        rc_mean = [np.mean([get_rc(s) for s in bundles[st].supp_metrics]) for st in states]
        region_color = _CI_PCCA if region_label == TARGET_I else _CJ_PCCA
        ax.bar(xs, rn_mean, color=_C_NUIS, alpha=0.88, label='Nuisance  r²  (mean)')
        ax.bar(xs, rc_mean, bottom=rn_mean, color=region_color, alpha=0.88,
               label='Communication  r²  (mean)')
        ax.set_xticks(xs)
        ax.set_xticklabels([STATE_STYLE[st]['display'] for st in states],
                           rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Mean fraction of total variance', fontsize=9)
        ax.set_title(f'({panel_tag})  Mean variance partition — {region_label}', fontsize=8.5)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    # Panel F — step-averaged lead–lag, one curve per state
    ax = axes[1, 2]
    mask_300 = np.abs(ref0.supp_metrics[0].lag_axis_ms) <= 300
    lag_ax = ref0.supp_metrics[0].lag_axis_ms[mask_300]
    for st in states:
        b, sty = bundles[st], STATE_STYLE[st]
        mean_curve = np.mean([s.xcorr_curve for s in b.supp_metrics], axis=0)
        ax.plot(lag_ax, mean_curve[mask_300], ls=sty['ls'], color=_STATE_BAR_COLOR[st],
                lw=2.0, alpha=0.9, label=f"{sty['display']}  (mean over steps)")
        peak_rel = int(np.argmax(np.abs(mean_curve[mask_300])))
        ax.scatter([lag_ax[peak_rel]], [mean_curve[mask_300][peak_rel]],
                   color=_STATE_BAR_COLOR[st], s=28, zorder=4)
    ax.axvline(0, color='k', ls=':', lw=0.8, alpha=0.45)
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.25)
    ax.set_xlim(-100, 100)
    ax.set_xlabel(f'Lag (ms)   [+ = {TARGET_I} leads {TARGET_J}]', fontsize=9)
    ax.set_ylabel('Normalised cross-correlation', fontsize=9)
    ax.set_title('(F)  Step-averaged pCCA latent cross-correlation', fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    fig.suptitle(
        f"{title}   |   states: " + " vs. ".join(STATE_STYLE[s]['display'] for s in states),
        fontsize=12, fontweight='bold',
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f'  [multistate supp_panel]  saved: {output_path}')
    return fig


# =============================================================================
# 7.  Cross-session aggregation   (unchanged)
# =============================================================================

def aggregate_by_region(
        all_data: List[SessionAblationData],
) -> Dict[str, Dict[str, List[float]]]:
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
    for r, d in agg.items():
        rp  = d.get("rho_pcca", [])
        rc  = d.get("rho_cca", [])
        rpc = d.get("rho1_cv_mean", [])
        rcc = d.get("rho_cca_cv", [])
        d["delta_rho_insample"] = [p - c for p, c in zip(rp, rc)]
        d["delta_rho_cv"]       = [p - c for p, c in zip(rpc, rcc)]
    return agg


def _sorted_region_order(agg: Dict[str, Dict[str, List[float]]]) -> List[str]:
    present = set(agg.keys())
    ordered = [r for r in ANATOMICAL_ORDER if r in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _merged_region_order(
        agg_a: Dict[str, Dict[str, List[float]]],
        agg_b: Dict[str, Dict[str, List[float]]],
) -> List[str]:
    """NEW: region order for two-regime comparisons — union of regions
    present in either agg, in ANATOMICAL_ORDER."""
    present = set(agg_a.keys()) | set(agg_b.keys())
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
    tr = ax.get_xaxis_transform()
    for pos, n in zip(positions, counts):
        ax.text(pos, 0.05, f"n={n}", ha="center", va="bottom",
                fontsize=fontsize, color=color, transform=tr)


def _clean(vals: List[float]) -> List[float]:
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


def _paired_boxplot_dual_agg(
        ax:           plt.Axes,
        region_order: List[str],
        agg_a:        Dict[str, Dict[str, List[float]]],
        agg_b:        Dict[str, Dict[str, List[float]]],
        key:          str,
        label_a:      str,
        label_b:      str,
        color_a:      str,
        color_b:      str,
        ylabel:       str,
        title:        str,
        ylim:         Optional[Tuple[float, float]] = None,
        hline:        Optional[float]               = None,
        hline_label:  Optional[str]                 = None,
) -> None:
    """NEW: paired boxplots of the SAME metric `key`, drawn from TWO DIFFERENT
    aggregation dicts (e.g. two preprocessing regimes), per ablated region.
    Same geometry as _paired_boxplot, but the two series come from agg_a /
    agg_b instead of two keys within one agg dict."""
    n_reg  = len(region_order)
    pos_a  = [3 * k + 1   for k in range(n_reg)]
    pos_b  = [3 * k + 2   for k in range(n_reg)]
    tick_x = [3 * k + 1.5 for k in range(n_reg)]

    data_a = [_clean(agg_a.get(r, {}).get(key, [])) for r in region_order]
    data_b = [_clean(agg_b.get(r, {}).get(key, [])) for r in region_order]

    _bp_kw = dict(widths=0.75, patch_artist=True,
                  medianprops=dict(color="k", lw=1.5), manage_ticks=False)
    _safe_a = [d if len(d) else [np.nan] for d in data_a]
    _safe_b = [d if len(d) else [np.nan] for d in data_b]
    ax.boxplot(_safe_a, positions=pos_a,
               boxprops=dict(facecolor=color_a, alpha=0.65),
               whiskerprops=dict(color=color_a), capprops=dict(color=color_a),
               flierprops=dict(marker="o", ms=3.5, color=color_a, alpha=0.55),
               **_bp_kw)
    ax.boxplot(_safe_b, positions=pos_b,
               boxprops=dict(facecolor=color_b, alpha=0.65),
               whiskerprops=dict(color=color_b), capprops=dict(color=color_b),
               flierprops=dict(marker="s", ms=3.5, color=color_b, alpha=0.55),
               **_bp_kw)

    if hline is not None:
        ax.axhline(hline, color="#7F8C8D", ls=":", lw=1.0, alpha=0.75)

    _annotate_n(ax, tick_x, [len(data_a[k]) for k in range(n_reg)])

    ax.set_xticks(tick_x)
    ax.set_xticklabels(region_order, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(0, 3 * n_reg)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    handles = [Patch(facecolor=color_a, alpha=0.65, label=label_a),
               Patch(facecolor=color_b, alpha=0.65, label=label_b)]
    if hline is not None and hline_label is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#7F8C8D", ls=":", lw=1.0, label=hline_label))
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
    rng = np.random.default_rng(seed)
    da  = _clean(data_a)
    db  = _clean(data_b)
    ax.boxplot(
        [da if da else [np.nan], db if db else [np.nan]],
        positions=[0, 1], widths=0.5, patch_artist=True,
        medianprops=dict(color="k", lw=1.5), manage_ticks=False,
        boxprops=dict(alpha=0.6),
    )
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
# 9.  Cross-session figure functions (EXISTING — now take regime_tag /
#     n_sessions / subtract_psth / shuffle_trials as parameters so they
#     label correctly when run_cross_session_pipeline() is called twice
#     in one process; default = module globals, so old call style still works)
# =============================================================================

def plot_cross_session_summary(
        agg:          Dict[str, Dict[str, List[float]]],
        region_order: List[str],
        all_rho_cca:  List[float],
        regime_tag:   Optional[str] = None,
        n_sessions:   Optional[int] = None,
        output_path:  Optional[Path] = None,
) -> plt.Figure:
    rt = regime_tag if regime_tag is not None else REGIME_TAG
    ns = n_sessions if n_sessions is not None else len(SESSIONS_TO_RUN)

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
        f"[{rt}]\n"
        f"n = {ns} sessions  ·  reference: CCA baseline (Z = ∅)",
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
        regime_tag:      Optional[str] = None,
        n_sessions:      Optional[int] = None,
        output_path:     Optional[Path] = None,
) -> plt.Figure:
    rt = regime_tag if regime_tag is not None else REGIME_TAG
    ns = n_sessions if n_sessions is not None else len(SESSIONS_TO_RUN)

    fig, axes = plt.subplots(
        1, 2, figsize=(19, 6),
        gridspec_kw={"wspace": 0.26, "width_ratios": [1.0, 2.6]},
    )

    _session_two_box(
        axes[0],
        data_a=all_rho_cca, data_b=all_rho_cca_cv,
        label_a="in-sample", label_b="5-fold CV",
        color_a=_C_INSAMPLE, color_b=_C_CV,
        ylabel="CCA coupling  ρ₁  (Z = ∅)",
        title=("(A)  CCA coupling across sessions\n"
               "boxes = session distribution  ·  dots = sessions"),
    )

    _paired_boxplot(
        axes[1], region_order, agg,
        key_a="delta_rho_insample", key_b="delta_rho_cv",
        label_a="Δρ  in-sample", label_b="Δρ  5-fold CV",
        color_a=_C_INSAMPLE, color_b=_C_CV,
        ylabel=r"$\Delta\rho = \rho_1^{\mathrm{pCCA}(Z=\{r\})} - \rho_1^{\mathrm{CCA}}$",
        title=("(B)  Change in residual coupling from single-region ablation\n"),
        hline=0.0, hline_label="no change (Δρ = 0)",
    )

    fig.suptitle(
        f"Cross-session residual coupling  |  {TARGET_I} ↔ {TARGET_J}  "
        f"[{rt}]   "
        f"n = {ns} sessions  ·  ρ₁ = corr_(t,s)(z_I, z_J)",
        fontsize=12, fontweight="bold",
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [cross-session residual coupling]  saved: {output_path}")
    return fig


def plot_cross_session_supplementary(
        agg:            Dict[str, Dict[str, List[float]]],
        region_order:   List[str],
        all_rho_cca:    List[float],
        regime_tag:     Optional[str] = None,
        n_sessions:     Optional[int] = None,
        subtract_psth:  Optional[bool] = None,
        shuffle_trials: Optional[bool] = None,
        output_path:    Optional[Path] = None,
) -> plt.Figure:
    """UNCHANGED in content/layout, per your instruction — only gains
    parameters so the title/notes are correct when this is called once
    per regime within the same process."""
    rt = regime_tag    if regime_tag    is not None else REGIME_TAG
    ns = n_sessions     if n_sessions     is not None else len(SESSIONS_TO_RUN)
    sp_flag = subtract_psth  if subtract_psth  is not None else SUBTRACT_PSTH
    sh_flag = shuffle_trials if shuffle_trials is not None else SHUFFLE_TRIALS

    fig, axes = plt.subplots(
        2, 3, figsize=(22, 12),
        gridspec_kw={"hspace": 0.55, "wspace": 0.38},
    )

    rho_cca_arr  = np.array(all_rho_cca)
    rho_cca_mean = float(rho_cca_arr.mean())
    rho_cca_sem  = float(rho_cca_arr.std() / np.sqrt(max(len(rho_cca_arr), 1)))

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

    _variance_partition_bars(
        axes[1, 0], region_order, agg,
        key_nuis="r2_nuis_i", key_comm="r2_comm_i",
        comm_color=_CI_PCCA,
        ylabel="Fraction of total variance",
        title=(f"(D)  Variance partition — {TARGET_I}\n"
               "mean ± SEM across sessions  ·  dots = individual sessions"),
        ylim=(0.0, 0.20),
    )

    _variance_partition_bars(
        axes[1, 1], region_order, agg,
        key_nuis="r2_nuis_j", key_comm="r2_comm_j",
        comm_color=_CJ_PCCA,
        ylabel="Fraction of total variance",
        title=(f"(E)  Variance partition — {TARGET_J}\n"
               "mean ± SEM across sessions  ·  dots = individual sessions"),
        ylim=(0.0, 0.20),
    )

    ax = axes[1, 2]
    ax.axis("off")
    notes = (
        "Visualisation notes\n"
        "────────────────────────────────────\n"
        "Regime: "
        f"SUBTRACT_PSTH={sp_flag}, SHUFFLE_TRIALS={sh_flag}\n\n"
        "Panels A–C: paired boxplots per ablated\n"
        "region.  Box = IQR; whiskers = 1.5×IQR;\n"
        "n = sessions recording that region.\n\n"
        "Panels D–E: stacked mean bars.\n"
        "  ■ grey   = mean r²_nuis\n"
        "  ■ colour = mean r²_comm\n"
        "  ┤        = ±1 SEM of total (nuis+comm)\n"
        "  ●        = individual session totals\n\n"
        f"Sessions: n = {ns}\n"
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
        f"[{rt}]\n"
        f"n = {ns} sessions  ·  single-region ablation",
        fontsize=12, fontweight="bold",
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [cross-session supp]  saved: {output_path}")
    return fig


# =============================================================================
# 9b.  Cross-session TWO-REGIME comparison figures  (NEW — Task B)
# =============================================================================
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
import dabest

# =============================================================================
# A0-poster font scale (841 x 1189 mm, viewed from ~1.5-2 m).
# =============================================================================
POSTER_FONT = dict(
    base=24, axis_label=24, tick=24, title=36, annot=24, legend=22,
)
SCREEN_FONT = dict(base=11, axis_label=12, tick=9, title=13, annot=10, legend=9)

# Default regime colours -- match _C_REGIME_RAW / _C_REGIME_SHUF already
# defined in pcca_cross_session_ablation.py. Pass your own through the
# color_a / color_b arguments if you'd rather not duplicate them here.
_C_REGIME_RAW = "#5DADE2"
_C_REGIME_SHUF = "#F5B041"


def _clean(vals: List[float]) -> List[float]:
    """Drop NaNs. Duplicate of the helper already in your main script --
    delete this copy if you're pasting the function in alongside it."""
    return [v for v in vals if not np.isnan(v)]


# =============================================================================
# Statistics: vectorised bootstrap CI + permutation test for the unpaired
# mean difference. Operates on plain 1-D arrays -- nothing dabest-specific.
# =============================================================================

def _bootstrap_mean_diff(
        control: np.ndarray,
        test: np.ndarray,
        n_boot: int = 5000,
        ci: float = 95.0,
        rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float, np.ndarray]:
    """Percentile bootstrap for Delta = mean(test) - mean(control).

    Returns (point_estimate, ci_lo, ci_hi, bootstrap_samples). A full
    bias-corrected-and-accelerated (BCa) interval -- what DABEST itself
    reports -- is a modest extension of this (correcting for skew and
    median bias in the bootstrap distribution) if you want it later;
    the percentile interval is transparent and adequate for a first pass.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n_c, n_t = len(control), len(test)
    point = float(np.mean(test) - np.mean(control)) if (n_c and n_t) else np.nan
    if n_c == 0 or n_t == 0:
        return point, np.nan, np.nan, np.array([])

    idx_c = rng.integers(0, n_c, size=(n_boot, n_c))
    idx_t = rng.integers(0, n_t, size=(n_boot, n_t))
    boot = test[idx_t].mean(axis=1) - control[idx_c].mean(axis=1)

    lo = float(np.percentile(boot, (100 - ci) / 2))
    hi = float(np.percentile(boot, 100 - (100 - ci) / 2))
    return point, lo, hi, boot


def _permutation_pvalue(
        control: np.ndarray,
        test: np.ndarray,
        n_perm: int = 5000,
        rng: Optional[np.random.Generator] = None,
) -> float:
    """Two-sided permutation p-value for Delta = mean(test) - mean(control):
    P(|Delta_perm| >= |Delta_obs|) under the null of exchangeable labels."""
    rng = rng if rng is not None else np.random.default_rng(1)
    pooled = np.concatenate([control, test])
    n_c, n = len(control), len(pooled)
    if n_c == 0 or len(test) == 0:
        return np.nan
    obs = abs(np.mean(test) - np.mean(control))

    order = np.argsort(rng.random((n_perm, n)), axis=1)  # vectorised shuffle
    perm = pooled[order]
    diffs = np.abs(perm[:, n_c:].mean(axis=1) - perm[:, :n_c].mean(axis=1))
    return float((np.sum(diffs >= obs) + 1) / (n_perm + 1))  # +1/+1: avoid p=0


def _sig_stars(p: float) -> str:
    """Empty string if not significant -- we only annotate significant
    comparisons, per your instruction."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# =============================================================================
# Layout geometry: tight within-region spacing, wide between-region spacing.
# =============================================================================

def _region_positions(
        n_reg: int,
        w_within: float = 0.85,  # raw<->shuffled spacing WITHIN a region
        w_between: float = 2.4,  # spacing BETWEEN region pairs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos_raw, pos_shuf, pos_mid), one entry per region."""
    pos_raw = np.arange(n_reg) * (w_within + w_between)
    pos_shuf = pos_raw + w_within
    pos_mid = (pos_raw + pos_shuf) / 2.0
    return pos_raw, pos_shuf, pos_mid


def _half_violin(
        ax, x0: float, samples: np.ndarray, color: str,
        width: float = 0.55, side: str = "right", alpha: float = 0.55,
) -> None:
    """KDE-based half-violin of the bootstrap distribution, DABEST-style
    'cloud' attached to one side of the point estimate."""
    if samples.size < 5 or np.std(samples) < 1e-12:
        return
    kde = gaussian_kde(samples)
    y_grid = np.linspace(samples.min(), samples.max(), 200)
    dens = kde(y_grid)
    dens = dens / dens.max() * width
    sign = 1.0 if side == "right" else -1.0
    ax.fill_betweenx(y_grid, x0, x0 + sign * dens, color=color, alpha=alpha,
                     lw=0, zorder=2)


# =============================================================================
# Main plotting function
# =============================================================================

def plot_cross_session_divergence_comparison(
        agg_a: Dict[str, Dict[str, List[float]]],
        agg_b: Dict[str, Dict[str, List[float]]],
        region_order: List[str],
        label_a: str,
        label_b: str,
        key: str = "divergence",
        color_a: str = _C_REGIME_RAW,
        color_b: str = _C_REGIME_SHUF,
        output_path: Optional[str] = None,
        font: Dict[str, int] = POSTER_FONT,
        dot_size: float = 110,
        w_within: float = 0.85,
        w_between: float = 2.4,
        n_boot: int = 5000,
        n_perm: int = 5000,
        seed: int = 0,
) -> plt.Figure:
    """Two-row estimation plot.

    Row 1 (raw data): per-session divergence D for each region x regime,
    plotted as an enlarged jittered scatter with a shaded bar behind each
    column running from 0 to the group median. No x-tick labels -- the
    identity of each column is carried by row 2 instead, since the two
    rows share an x-axis.

    Row 2 (effect size): bootstrap distribution of
    Delta_r = mean(D^shuffled_r) - mean(D^raw_r), one per region, centred
    between that region's two row-1 columns. x-ticks are placed at the SAME
    positions as row 1's columns and labelled '{region} raw' /
    '{region} shuffled'. A permutation-test significance marker (*/**/***)
    is placed immediately to the left of each region's distribution when
    the comparison is significant.
    """
    region_order = [
        r for r in region_order
        if len(_clean(agg_a.get(r, {}).get(key, []))) > 0
           or len(_clean(agg_b.get(r, {}).get(key, []))) > 0
    ]
    n_reg = len(region_order)
    if n_reg == 0:
        raise ValueError(f"No data found for key={key!r} across {region_order}")

    pos_raw, pos_shuf, pos_mid = _region_positions(n_reg, w_within, w_between)
    rng = np.random.default_rng(seed)

    with plt.rc_context({
        "font.size": font["base"],
        "xtick.labelsize": font["tick"],
        "ytick.labelsize": font["tick"],
        "legend.fontsize": font["legend"],
        "axes.linewidth": 1.4,
    }):
        fig = plt.figure(figsize=(max(14, 3.0 * n_reg), 15))
        gs = GridSpec(2, 1, height_ratios=[1.5, 1.0], hspace=0.08, figure=fig)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

        # ---------------- Row 1: raw data ----------------
        for k, r in enumerate(region_order):
            vals_a = np.array(_clean(agg_a.get(r, {}).get(key, [])))
            vals_b = np.array(_clean(agg_b.get(r, {}).get(key, [])))

            for x0, vals, col in ((pos_raw[k], vals_a, color_a),
                                  (pos_shuf[k], vals_b, color_b)):
                if vals.size == 0:
                    continue
                med = float(np.median(vals))

                # Shaded bar extending to the median, drawn behind the points.
                ax_top.bar(x0, med, width=w_within * 0.72, color=col,
                           alpha=0.20, edgecolor="none", zorder=0)
                ax_top.hlines(med, x0 - w_within * 0.36, x0 + w_within * 0.36,
                              color="k", lw=2.2, zorder=4)

                # Enlarged jittered scatter.
                jitter = rng.uniform(-w_within * 0.17, w_within * 0.17, size=vals.size)
                ax_top.scatter(x0 + jitter, vals, s=dot_size, color=col,
                               alpha=0.85, zorder=3, linewidths=0.4,
                               edgecolors="white")

        ax_top.set_xlim(pos_raw[0] - w_between * 0.4, pos_shuf[-1] + w_between * 0.4)
        ax_top.set_xticks([])  # Row 1: no x labels
        ax_top.tick_params(bottom=False)
        ax_top.set_ylabel("Composite divergence  D  (max = 4)",
                          fontsize=font["axis_label"])
        for sp in ("top", "right"):
            ax_top.spines[sp].set_visible(False)
        ax_top.grid(alpha=0.2, axis="y")

        # ---------------- Row 2: effect size ----------------
        ax_bot.axhline(0.0, color="#7F8C8D", ls="--", lw=1.4, alpha=0.8, zorder=1)

        for k, r in enumerate(region_order):
            vals_a = np.array(_clean(agg_a.get(r, {}).get(key, [])))
            vals_b = np.array(_clean(agg_b.get(r, {}).get(key, [])))
            if vals_a.size == 0 or vals_b.size == 0:
                continue

            point, lo, hi, boot = _bootstrap_mean_diff(
                vals_a, vals_b, n_boot=n_boot,
                rng=np.random.default_rng(seed + k),
            )
            p_val = _permutation_pvalue(
                vals_a, vals_b, n_perm=n_perm,
                rng=np.random.default_rng(seed + 1000 + k),
            )
            stars = _sig_stars(p_val)

            x0 = pos_mid[k]
            _half_violin(ax_bot, x0, boot, color=color_b, width=w_within * 0.55)
            ax_bot.plot([x0, x0], [lo, hi], color="k", lw=2.4, zorder=4)
            ax_bot.scatter([x0], [point], color="k", s=dot_size * 0.85, zorder=5)

            # Significance asterisks, to the LEFT of the distribution --
            # only drawn when the comparison is significant.
            if stars:
                ax_bot.text(
                    x0 - w_within * 0.7, point, stars,
                    ha="right", va="center", fontsize=font["annot"],
                    fontweight="bold", color="#C0392B",
                )

        tick_pos = np.concatenate([pos_raw, pos_shuf])
        tick_lab = [f"{r} raw" for r in region_order] + \
                   [f"{r} shuffled" for r in region_order]
        ax_bot.set_xticks(tick_pos)
        ax_bot.set_xticklabels(tick_lab, rotation=45, ha="right",
                               fontsize=font["tick"])
        ax_bot.set_ylabel(
            r"$\Delta = \overline{D}_{\mathrm{shuffled}} - \overline{D}_{\mathrm{raw}}$",
            fontsize=font["axis_label"],
        )
        for sp in ("top", "right"):
            ax_bot.spines[sp].set_visible(False)
        ax_bot.grid(alpha=0.2, axis="y")

        # IMPORTANT: sharex=True means ax_top and ax_bot share one Locator/
        # Formatter -- setting ax_bot's ticks above silently re-populates
        # ax_top's tick labels too (undoing the earlier `set_xticks([])`).
        # Suppress row 1's tick marks/labels explicitly, after row 2's
        # ticks are finalised, not before.
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # fig.suptitle(
        #     f"Cross-session divergence comparison  |  {label_a}  vs.  {label_b}\n"
        #     r"$*\,p<0.05\;\;**\,p<0.01\;\;***\,p<0.001$  (two-sided permutation test, "
        #     f"{n_perm} permutations)",
        #     fontsize=font["title"], fontweight="bold", y=1.02,
        # )

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  [estimation plot]  saved: {output_path}")

    return fig
# def plot_cross_session_divergence_comparison(
#         agg_a:        Dict[str, Dict[str, List[float]]],
#         agg_b:        Dict[str, Dict[str, List[float]]],
#         region_order: List[str],
#         label_a:      str,
#         label_b:      str,
#         output_path:  Optional[Path] = None,
# ) -> plt.Figure:
#     """Task B.1: per-region boxplots of the composite divergence metric, side
#     by side for two regimes (e.g. psth0_shuf1 vs. psth0_shuf0)."""
#     fig, ax = plt.subplots(1, 1, figsize=(max(10, 1.0 * len(region_order)), 6.5))
#
#     _paired_boxplot_dual_agg(
#         ax, region_order, agg_a, agg_b,
#         key="divergence",
#         label_a=label_a, label_b=label_b,
#         color_a=_C_REGIME_RAW, color_b=_C_REGIME_SHUF,
#         ylabel="Composite divergence  D  (max = 4)",
#         title=(f"Divergence from CCA baseline — regime comparison\n"
#                f"{label_a}  vs.  {label_b}   ·  boxes = session distribution"),
#         ylim=(0.0, 4.10),
#     )
#
#     # fig.suptitle(
#     #     f"Cross-session divergence comparison  |  {TARGET_I} ↔ {TARGET_J}\n"
#     #     f"{label_a}  vs.  {label_b}",
#     #     fontsize=13, fontweight="bold",
#     # )
#     if output_path is not None:
#         fig.savefig(output_path, dpi=200, bbox_inches="tight")
#         print(f"  [divergence comparison]  saved: {output_path}")
#     return fig


def plot_cross_session_coupling_comparison(
        agg_a:         Dict[str, Dict[str, List[float]]],
        agg_b:         Dict[str, Dict[str, List[float]]],
        rho_cca_cv_a:  List[float],
        rho_cca_cv_b:  List[float],
        region_order:  List[str],
        label_a:       str,
        label_b:       str,
        output_path:   Optional[Path] = None,
        font: Dict[str, int] = POSTER_FONT,
) -> plt.Figure:
    """Task B.2: regime-comparison residual coupling diagnostics.

    (A) session-level 5-fold CV CCA coupling ρ₁, regime A vs. regime B
        (in-sample excluded, per instruction).
    (B) per-region 5-fold CV Δρ = ρ₁_CV^{pCCA(Z={r})} − ρ₁_CV^{CCA},
        regime A vs. regime B.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(19, 6),
        gridspec_kw={"wspace": 0.26, "width_ratios": [1.0, 2.6]},
    )

    _session_two_box(
        axes[0],
        data_a=rho_cca_cv_a, data_b=rho_cca_cv_b,
        label_a=label_a, label_b=label_b,
        color_a=_C_REGIME_RAW, color_b=_C_REGIME_SHUF,
        ylabel="CCA coupling  ρ₁  (5-fold CV, Z = ∅)",
        title=("(A)  5-fold CV CCA coupling across sessions\n"
               "boxes = session distribution  ·  dots = sessions"),
    )
    # return _dabest_multi_group_estimation_plot(
    #     agg_a, agg_b, region_order,
    #     key="divergence",
    #     label_a=label_a, label_b=label_b,
    #     ylabel="Composite divergence  D  (max = 4)",
    #     sup_title=(f"Cross-session divergence comparison  |  "
    #                f"{TARGET_I} \u2194 {TARGET_J}\n{label_a}  vs.  {label_b}"),
    #     color_a=_C_REGIME_RAW, color_b=_C_REGIME_SHUF,
    #     output_path=output_path,
    #     font=font,
    # )

    _paired_boxplot_dual_agg(
        axes[1], region_order, agg_a, agg_b,
        key="delta_rho_cv",
        label_a=f"Δρ CV  ({label_a})", label_b=f"Δρ CV  ({label_b})",
        color_a=_C_REGIME_RAW, color_b=_C_REGIME_SHUF,
        ylabel=r"$\Delta\rho_{CV} = \rho_{1,CV}^{\mathrm{pCCA}(Z=\{r\})} - \rho_{1,CV}^{\mathrm{CCA}}$",
        title="(B)  Change in 5-fold CV residual coupling from single-region ablation",
        hline=0.0, hline_label="no change (Δρ_CV = 0)",
    )

    fig.suptitle(
        f"Cross-session residual coupling — regime comparison  |  {TARGET_I} ↔ {TARGET_J}\n"
        f"{label_a}  vs.  {label_b}   ·  5-fold CV only",
        fontsize=13, fontweight="bold",
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  [coupling comparison]  saved: {output_path}")
    return fig


# =============================================================================
# 10.  Per-session ablation runner   (MODIFIED — parameterised regime,
#       fresh local RNGs, returns enriched SessionAblationData)
# =============================================================================

def run_single_ablation(
        region_spikes:    Dict[str, np.ndarray],
        n_trials:         int,
        T:                int,
        out_summary_dir:  Path,
        out_supp_dir:     Path,
        session_name:     str,
        subtract_psth:    bool = SUBTRACT_PSTH,    # ← NEW (was a global read)
        shuffle_trials:   bool = SHUFFLE_TRIALS,   # ← NEW (was a global read)
        state_key:        Optional[str] = None,    # ← NEW (cosmetic label)
        regime_tag:       Optional[str] = None,    # ← NEW (cosmetic label)
) -> Optional[SessionAblationData]:
    """Run single-region ablation for one session under one preprocessing
    regime, save per-session figures, and return a SessionAblationData cache
    for cross-session aggregation AND/OR per-session multi-state overlays."""
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

    regime_tag = regime_tag or f"psth{int(subtract_psth)}_shuf{int(shuffle_trials)}"
    tag_str    = state_key if state_key is not None else regime_tag

    print(f"  [{session_name} | {tag_str}]  Ablating {len(nuisance_all)} nuisance "
          f"regions: {nuisance_all}")

    paired_trials = not shuffle_trials

    # ── Fresh, locally-scoped RNGs (NOT the module-level _rng / _rng2) ────
    # Necessary because this function may now be called repeatedly within
    # one process (once per state, or once per regime) — reusing the module
    # globals would silently consume RNG state across calls and make the
    # shuffle non-reproducible run-to-run.
    rng_i = np.random.default_rng(42)
    rng_j = np.random.default_rng(43)

    X_i_flat = _zscore_flat(region_spikes[TARGET_I], subtract_psth=subtract_psth,
                            shuffle_trials=shuffle_trials, rng=rng_i)
    X_j_flat = _zscore_flat(region_spikes[TARGET_J], subtract_psth=subtract_psth,
                            shuffle_trials=shuffle_trials, rng=rng_j)
    nuisance_flat: Dict[str, np.ndarray] = {
        r: _zscore_flat(region_spikes[r], subtract_psth=subtract_psth)
        for r in nuisance_all
    }

    time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)

    Wx_cca, Wy_cca, rho_cca = ridge_cca(X_i_flat, X_j_flat)
    rho_cca_val    = float(rho_cca[0])
    rho_cca_cv_val = _cv_rho(X_i_flat, X_j_flat, None, n_trials, T)
    print(f"  [{session_name} | {tag_str}]  CCA baseline  ρ₁ = {rho_cca_val:.4f}  "
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

    ablation_results:  List[StepResult]          = []
    supp_list:         List[SupplementaryMetrics] = []

    for abl_idx, region in enumerate(nuisance_all):
        Z_single = nuisance_flat[region]
        Wx_p, Wy_p, rho_p, X_i_res, X_j_res = pcca(X_i_flat, X_j_flat, Z_single)

        z_i_lat = latent_projections(X_i_res, Wx_p[:, 0], n_trials, T)
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
            f"    [{session_name} | {tag_str}]  abl {abl_idx:02d}  Z = {{{region:<8}}}  "
            f"ρ₁ = {rho_p[0]:.4f}  Δρ = {rho_p[0] - rho_cca_val:+.4f}  "
            f"θ_i = {supp.theta_i_deg:.1f}°  κ_i = {supp.kappa_i:.3f}"
        )

    sims_vs_cca = [compute_similarity(s, cca_ref) for s in ablation_results]
    idx_c, idx_f, _ = identify_extremes(ablation_results, cca_ref, exclude_ref=False)

    print(f"\n  [{session_name} | {tag_str}]  Similarity to CCA baseline:")
    print_similarity_table(ablation_results, sims_vs_cca, "CCA_baseline", idx_c, idx_f)

    # plt.close(plot_summary_figure(
    #     step_results = ablation_results,
    #     sims         = sims_vs_cca,
    #     idx_closest  = idx_c,
    #     idx_furthest = idx_f,
    #     ref_label    = "CCA baseline  (Z = ∅)",
    #     title        = (f"Single-region ablation  |  {session_name}  [{regime_tag}]\n"
    #                     f"{TARGET_I} ↔ {TARGET_J}   reference: CCA baseline"),
    #     output_path  = out_summary_dir / f"{session_name}_part1_summary_vs_CCA.png",
    # ))
    #
    # plt.close(plot_supplementary_panel(
    #     supp_list      = supp_list,
    #     rho_pcca_list  = [s.rho_pcca for s in ablation_results],
    #     rho_cca        = rho_cca_val,
    #     title          = (f"Part 1 — Supplementary Diagnostics  |  {session_name}  "
    #                       f"[{regime_tag}]\n"
    #                       f"{TARGET_I} ↔ {TARGET_J}   (single-region ablation)"),
    #     shuffle_trials = shuffle_trials,
    #     output_path    = out_supp_dir / f"{session_name}_part1_supplementary.png",
    # ))

    return SessionAblationData(
        session_name   = session_name,
        rho_cca        = rho_cca_val,
        rho_cca_cv     = rho_cca_cv_val,
        region_names   = nuisance_all,
        step_results   = ablation_results,
        sims_vs_cca    = sims_vs_cca,
        supp_metrics   = supp_list,
        state_key      = state_key,
        subtract_psth  = subtract_psth,
        shuffle_trials = shuffle_trials,
        regime_tag     = regime_tag,
    )


# =============================================================================
# 10b.  Single-session multi-state orchestrator  (NEW — Task A)
# =============================================================================

def run_three_state_single_session(session_name: str) -> None:
    """Run the single-region ablation three times for ONE session (raw /
    shuffled / psth-subtracted) and produce the two multi-state overlay
    figures, in both modes ('all3' and 'first_two')."""
    BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    SESSION_FILE = (BASE_DIR / "pcca_sessions_cued_hit_long_results"
                    / f"{session_name}_analysis_results.mat")
    if not SESSION_FILE.exists():
        print(f"❌ [WARNING] File not found, skipping: {SESSION_FILE}")
        return

    region_spikes, n_trials, T = load_region_spikes(str(SESSION_FILE))

    OUTPUT_ROOT = (BASE_DIR / "Paper_output" / "pcca_single_session_multistate"
                   / f"{TARGET_I}_{TARGET_J}" / session_name)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    bundles: Dict[str, SessionAblationData] = {}
    for state_key in STATE_ORDER:
        cfg = STATE_CONFIG[state_key]
        state_dir = OUTPUT_ROOT / state_key
        state_dir.mkdir(parents=True, exist_ok=True)
        result = run_single_ablation(
            region_spikes=region_spikes, n_trials=n_trials, T=T,
            out_summary_dir=state_dir, out_supp_dir=state_dir,
            session_name=session_name,
            subtract_psth=cfg["subtract_psth"], shuffle_trials=cfg["shuffle_trials"],
            state_key=state_key,
        )
        if result is not None:
            bundles[state_key] = result

    for mode in ("all3", "first_two"):
        states_avail = [s for s in MODE_STATES[mode] if s in bundles]
        if len(states_avail) < 2:
            print(f"  [skip] mode={mode!r}: only {states_avail} available.")
            continue
        plot_summary_figure_multistate(
            bundles, mode=mode,
            title=f"{TARGET_I} ↔ {TARGET_J}  |  {session_name}",
            output_path=OUTPUT_ROOT / f"{session_name}_multistate_{mode}_summary.png",
        )
        plt.close('all')
        plot_supplementary_panel_multistate(
            bundles, mode=mode,
            title=f"Multi-state Supplementary  |  {session_name}\n{TARGET_I} ↔ {TARGET_J}",
            output_path=OUTPUT_ROOT / f"{session_name}_multistate_{mode}_supplementary.png",
        )
        plt.close('all')

    print(f"✨ Multi-state single-session comparison for {session_name} done.")


# =============================================================================
# 11.  Cross-session pipeline  (REFACTORED out of main(); MODIFIED to take
#       subtract_psth / shuffle_trials as arguments so it can run twice in
#       one process for the Task B regime comparison)
# =============================================================================

def run_cross_session_pipeline(
        subtract_psth:  bool,
        shuffle_trials: bool,
        sessions:       Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, List[float]]], List[str], List[float], List[float]]:
    """Run the full per-session ablation + cross-session aggregation pipeline
    for ONE (subtract_psth, shuffle_trials) regime, save the existing
    per-session and cross-session figures (unchanged content/layout), and
    return (agg, region_order, all_rho_cca, all_rho_cca_cv) so callers can
    build further comparisons across regimes."""
    regime_tag = f"psth{int(subtract_psth)}_shuf{int(shuffle_trials)}"
    sessions   = sessions if sessions is not None else SESSIONS_TO_RUN

    BASE_DIR    = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    SESSION_DIR = BASE_DIR / "pcca_sessions_cued_hit_long_results"

    out_summary = (BASE_DIR / "Paper_output" / "pcca_cross_session_summary"
                   / f"{TARGET_I}_{TARGET_J}_{regime_tag}")
    out_supp    = (BASE_DIR / "Paper_output" / "pcca_cross_session_supplementary"
                   / f"{TARGET_I}_{TARGET_J}_{regime_tag}")
    out_summary.mkdir(parents=True, exist_ok=True)
    out_supp.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"pCCA Cross-Session Ablation  |  {TARGET_I} ↔ {TARGET_J}  [{regime_tag}]")
    print(f"  SUBTRACT_PSTH = {subtract_psth}   SHUFFLE_TRIALS = {shuffle_trials}")
    print(f"  Sessions  : {len(sessions)}")
    print(f"  Summary   : {out_summary}")
    print(f"  Supp      : {out_supp}")
    print("=" * 72)

    all_session_data: List[SessionAblationData] = []

    for idx, session_name in enumerate(sessions, 1):
        session_file = SESSION_DIR / f"{session_name}_analysis_results.mat"
        print(f"\n🔬 [{idx}/{len(sessions)}]  {session_name}  [{regime_tag}]")

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
                subtract_psth   = subtract_psth,
                shuffle_trials  = shuffle_trials,
                regime_tag      = regime_tag,
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
        print(f"\n⚠️  [{regime_tag}]  No sessions completed successfully — "
              "no cross-session figures.")
        return {}, [], [], []

    print("\n" + "=" * 72)
    print(f"Cross-session aggregation  [{regime_tag}]  ({len(all_session_data)} sessions)")
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

    # plt.close(plot_cross_session_summary(
    #     agg, region_order, all_rho_cca,
    #     regime_tag=regime_tag, n_sessions=len(sessions),
    #     output_path=out_summary / "cross_session_summary_ABC.png",
    # ))
    #
    # plt.close(plot_cross_session_coupling(
    #     agg, region_order, all_rho_cca, all_rho_cca_cv,
    #     regime_tag=regime_tag, n_sessions=len(sessions),
    #     output_path=out_summary / "cross_session_coupling.png",
    # ))
    #
    # plt.close(plot_cross_session_supplementary(
    #     agg, region_order, all_rho_cca,
    #     regime_tag=regime_tag, n_sessions=len(sessions),
    #     subtract_psth=subtract_psth, shuffle_trials=shuffle_trials,
    #     output_path=out_supp / "cross_session_supplementary_ABCDE.png",
    # ))
    #
    # print(f"\n✓ [{regime_tag}]  Cross-session figures saved.")
    # print(f"  Summary  → {out_summary}")
    # print(f"  Supp     → {out_supp}")

    return agg, region_order, all_rho_cca, all_rho_cca_cv


# =============================================================================
# 11b.  Two-regime cross-session comparison orchestrator  (NEW — Task B)
# =============================================================================

def run_two_regime_cross_session_comparison(
        sessions: Optional[List[str]] = None,
) -> None:
    """Run the cross-session pipeline twice — psth0_shuf0 and psth0_shuf1 —
    then build the two NEW comparison figures (divergence; residual
    coupling, CV-only) requested for Task B. plot_cross_session_supplementary
    is intentionally NOT touched/duplicated here."""
    sessions = sessions if sessions is not None else SESSIONS_TO_RUN

    print("\n" + "#" * 72)
    print("Two-regime cross-session comparison:  psth0_shuf0  vs.  psth0_shuf1")
    print("#" * 72)

    agg_raw, region_order_raw, _rho_cca_raw, rho_cca_cv_raw = run_cross_session_pipeline(
        subtract_psth=False, shuffle_trials=False, sessions=sessions,
    )
    agg_shuf, region_order_shuf, _rho_cca_shuf, rho_cca_cv_shuf = run_cross_session_pipeline(
        subtract_psth=False, shuffle_trials=True, sessions=sessions,
    )

    if not agg_raw or not agg_shuf:
        print("⚠️  One or both regimes produced no data — skipping comparison figures.")
        return

    region_order = _merged_region_order(agg_raw, agg_shuf)

    BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    out_compare = (BASE_DIR / "Paper_output" / "pcca_cross_session_regime_comparison"
                   / f"{TARGET_I}_{TARGET_J}")
    out_compare.mkdir(parents=True, exist_ok=True)

    label_raw  = "Raw "
    label_shuf = "Shuffled"

    plt.close(plot_cross_session_divergence_comparison(
        agg_a=agg_raw, agg_b=agg_shuf,
        region_order=region_order,
        label_a=label_raw, label_b=label_shuf,
        output_path=out_compare / "cross_session_divergence_psth0_shuf0_vs_shuf1.png",
    ))

    plt.close(plot_cross_session_coupling_comparison(
        agg_a=agg_raw, agg_b=agg_shuf,
        rho_cca_cv_a=rho_cca_cv_raw, rho_cca_cv_b=rho_cca_cv_shuf,
        region_order=region_order,
        label_a=label_raw, label_b=label_shuf,
        output_path=out_compare / "cross_session_coupling_psth0_shuf0_vs_shuf1.png",
    ))

    print(f"\n🎉 Two-regime comparison figures saved → {out_compare}")


# =============================================================================
# 12.  Entry point
# =============================================================================

# ── Toggles ──────────────────────────────────────────────────────────────
RUN_SINGLE_REGIME_PIPELINE:    bool = False   # original single-regime cross-session run
RUN_SINGLE_SESSION_MULTISTATE: bool = False # NEW — Task A
RUN_TWO_REGIME_COMPARISON:     bool = True   # NEW — Task B


def main() -> None:
    if RUN_SINGLE_REGIME_PIPELINE:
        run_cross_session_pipeline(subtract_psth=SUBTRACT_PSTH, shuffle_trials=SHUFFLE_TRIALS)

    if RUN_SINGLE_SESSION_MULTISTATE:
        for session in SESSIONS_TO_RUN:
            run_three_state_single_session(session)

    if RUN_TWO_REGIME_COMPARISON:
        run_two_regime_cross_session_comparison()

    print("\n🎉  All requested pipelines complete.")


if __name__ == "__main__":
    main()
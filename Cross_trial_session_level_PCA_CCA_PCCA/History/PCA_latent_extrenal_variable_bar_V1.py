#!/usr/bin/env python3
r"""
pca_cross_session_peak_delta_variance.py
================================================================================

Cross-session PCA-latent PEAK, DELTA, and BEHAVIOURAL-VARIANCE analysis.

This is the PCA-input sibling of ``cross_trial_type_peak_delta_reward_
variance.py`` (the pCCA/CCA version of this pipeline). Everything about the
five downstream analyses -- peak comparison, delta comparison, behavioural
variance for the reference and projected conditions, and per-region latent
traces -- is unchanged in spirit. What changes is the unit of analysis:

  * pCCA/CCA version : one canonical/partial-canonical variate pair per
                        REGION PAIR (region_i = "u", region_j = "v"), drawn
                        from ``pcca_sessions_*_results`` / ``sessions_*_
                        results`` (CCA) .mat files.
  * THIS version      : one principal-component score trace per INDIVIDUAL
                        REGION, drawn from ``sessions_*_results`` .mat files'
                        ``pca_results.<REGION>.coefficients`` field, following
                        the loading convention of ``cross_trial_type_pca_
                        analysis.py`` (``CrossTrialTypePCAAnalyzer`` /
                        ``CrossSessionPCAAnalyzer``).

Because PCA does not require two co-recorded regions, every bar chart below
collapses the pCCA/CCA script's "region_i / region_j" role split into a
single bar per region, and the region-pair/category grouping collapses into
a flat, anatomically ordered region list (grouped into broad anatomical
bands purely for the background shading, matching this project's visual
grammar).

Mathematical framework
-----------------------------------------------------------------------------
**Projection.** For region $r$, session $s$, let $W_{r,s}\in\mathbb R^{n\times
k}$ be the PCA loading matrix fit on the reference condition (read directly
from that session's reference-condition ``.mat`` file, exactly as
``CrossTrialTypePCAAnalyzer.extract_pca_weights`` does). For ANY trial type
$c$ (including the reference itself -- projected here uniformly, not read
from the file's own pre-computed mean/std fields, so that per-trial latents
are available for the variance-explained tasks below):
$$
z_{r,s,c}(t) = \tilde X_{r,s,c}(t)\, W_{r,s}[:, :k], \qquad
\tilde X = \text{column-wise } z\text{-score of } X
$$
This is literally the paper's own Fig. 3e / Fig. 5 procedure (the paper's
"training PCA subspace" is exactly $W_{r,s}$), so this script -- unlike its
pCCA/CCA sibling -- needs no analogy-by-substitution for tasks 1-2.

**Sign alignment.** Delegated verbatim to ``CrossSessionPCAAnalyzer.
aggregate_projections`` (imported, not copied): Z2 spectral synchronisation
(leading eigenvector of the inter-session correlation matrix) per region,
followed by an epoch-mean and peak-polarity convention, with the resulting
per-session flip decision from the REFERENCE trial type re-applied to every
other trial type of that session -- the direct region-level analogue of the
pCCA/CCA script's pair-level ``_align_signs_spectral``.

**Peak (task 1), Delta (task 2), Behavioural variance (tasks 3-4).**
Formulas identical to the pCCA/CCA sibling script (see its docstring for the
full derivation); reproduced here only insofar as the code implementing them
is copied primitive-for-primitive, per this project's "primitive copying
over importing" convention. The one difference worth flagging explicitly:
``NORMALIZE_PEAK_BY_REFERENCE`` (Task 1) is *more* literal here than in the
pCCA/CCA sibling, since the paper's Methods describe this within-session
peak-amplitude scaling specifically in terms of the "training PCA subspace".

Flexibility
-----------------------------------------------------------------------------
Everything Shengyuan is likely to want to change lives in the single
"USER-CONFIGURABLE PARAMETERS" block below:
  * ``REFERENCE_TYPE``     : which trial type the PCA subspace is trained on
  * ``ACTIVE_TRIAL_TYPES`` : trim to 2 trial types or extend back to 3
  * ``COMPONENT_INDICES``  : which PCA dimension(s) to analyze
  * ``VARIANCE_METHOD``    : 'marginal' | 'leave_one_out'
  * ``DELTA_METRIC``       : 'auc_of_difference' | 'peak_of_difference'
  * ``NORMALIZE_PEAK_BY_REFERENCE`` : True | False
  * ``REGIONS_OF_INTEREST`` : which individual regions to analyze/plot

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import zscore, wilcoxon
from scipy.interpolate import BSpline

warnings.filterwarnings('ignore')

# =============================================================================
# 0.  Import the existing PCA pipeline module (high-level classes imported
#     directly, per this project's convention -- cross_trial_type_pca_
#     analysis.py is treated as a stable library module here, exactly as
#     cross_trial_type_cca_analysis.py is treated by the pCCA/CCA sibling
#     script).
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_trial_type_pca_analysis import (   # noqa: E402
    CrossTrialTypePCAAnalyzer,
    CrossSessionPCAAnalyzer,
    TRIAL_TYPE_COLORS,
    ANATOMICAL_ORDER,
    MIN_SESSIONS_THRESHOLD,
    get_anatomical_index,
)

try:
    import mat73  # noqa: F401  (transitively required by cross_trial_type_pca_analysis)
    _MAT73_OK = True
except Exception:
    _MAT73_OK = False
    warnings.warn("mat73 not importable -- the analysis loop will fail; "
                  "install with `pip install mat73`.")


# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS
# =============================================================================

# ---- Reference / trial-type selection --------------------------------------
REFERENCE_TYPE: str = 'cued_hit_long'
ACTIVE_TRIAL_TYPES: List[str] = [
    'cued_hit_long',
    'spont_miss_long',
    # 'spont_hit_long',
]

# ---- Paths -------------------------------------------------------------
BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
BEHAVIOR_DIR = BASE_DIR / "Paper_output" / "tapproach_sessions"
OUTPUT_DIR = BASE_DIR / "Paper_output" / f"cross_trial_type_pca_{REFERENCE_TYPE}_peak_delta_variance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Sessions ------------------------------------------------------------
SESSIONS: List[str] = [
    'yp010_220209', 'yp010_220210', 'yp010_220211', 'yp010_220212',
    'yp012_220208', 'yp012_220209', 'yp012_220210', 'yp012_220211', 'yp012_220212',
    'yp013_220209', 'yp013_220210', 'yp013_220211', 'yp013_220212',
    'yp014_220208', 'yp014_220209', 'yp014_220210', 'yp014_220211', 'yp014_220212',
    'yp020_220331', 'yp020_220401', 'yp020_220402', 'yp020_220403', 'yp020_220404',
    'yp020_220405', 'yp020_220407',
    'yp021_220331', 'yp021_220401', 'yp021_220402', 'yp021_220403', 'yp021_220404',
    'yp021_220405', 'yp021_220407',
]

# ---- PCA dimensionality --------------------------------------------------
N_COMPONENTS: int = 5             # how many components are read out of the .mat file
COMPONENT_INDICES: List[int] = [0]  # which of those to actually analyze/plot here

MIN_SESSIONS: int = MIN_SESSIONS_THRESHOLD  # reuse the pipeline's own threshold

# ---- Task 1 / 2: peak & delta -------------------------------------------
PEAK_WINDOW_S: Tuple[float, float] = (0.0, 1)
DELTA_WINDOW_S: Tuple[float, float] = PEAK_WINDOW_S
NORMALIZE_PEAK_BY_REFERENCE: bool = True
DELTA_METRIC: str = 'auc_of_difference'    # 'auc_of_difference' | 'peak_of_difference'

# ---- Task 3 / 4: behavioural variance ------------------------------------
BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = (-1.0, 2.0)
BEHAVIOR_FS: float = 50.0
BEHAVIOR_T_OFFSET: float = -1.0
LAMBDA_R2: float = 1e-4
VARIANCE_METHOD: str = 'marginal'  # 'marginal' | 'leave_one_out'
EXTERNAL_VARIABLES: List[str] = ['position', 'speed', 'reward_presence', 'reward_consumption']

# Reward-kernel construction (see docstring "Reward kernel" note in the
# pCCA/CCA sibling script -- same FIXED-WINDOW ASSUMPTION applies here)
REWARD_PRESENCE_WINDOW_S: Tuple[float, float] = (0.0, 0.5)
REWARD_CONSUMPTION_WINDOW_S: Tuple[float, float] = (0.5, 1.5)
N_REWARD_CONSUMPTION_BASIS: int = 7
REWARD_SPLINE_DEGREE: int = 2

# ---- Caching / output -----------------------------------------------------
USE_CACHED_DATA: bool = True
SAVE_DPI: int = 400


# =============================================================================
# 2.  Individual regions & anatomical grouping (replaces the pCCA/CCA
#     sibling's 21-pair / 7-category PAIR_CATEGORIES; same 7 regions this
#     project's pCCA/CCA pipeline already analyzes as pairs, now analyzed
#     individually, arranged in anatomical order).
# =============================================================================
REGIONS_OF_INTEREST: List[str] = sorted(
    ['ORB', 'MOp', 'MOs', 'STR', 'VALVM', 'VPMPO', 'HY'],
    key=get_anatomical_index,
)

# Broad anatomical bands, purely for background shading -- the individual
# regions within each band are already anatomically ordered above.
REGION_CATEGORIES: List[Tuple[str, List[str]]] = [
    ("cortical",     [r for r in REGIONS_OF_INTEREST if r in ('ORB', 'MOp', 'MOs')]),
    ("striatal",     [r for r in REGIONS_OF_INTEREST if r in ('STR',)]),
    ("thalamic",     [r for r in REGIONS_OF_INTEREST if r in ('VALVM', 'VPMPO')]),
    ("hypothalamic", [r for r in REGIONS_OF_INTEREST if r in ('HY',)]),
]
REGION_CATEGORIES = [(cat, regs) for cat, regs in REGION_CATEGORIES if regs]

DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
}


def _display_name(region: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(region, region)


# Category background colours -- same palette keys/hues as pcca_cross_
# session_mi_bar.py's CATEGORY_COLORS, remapped onto these 4 anatomical bands
CATEGORY_COLORS: Dict[str, str] = {
    "cortical":     "#DD8452",  # was "cortico-cortical"
    "striatal":     "#937860",  # was "to STR"
    "thalamic":     "#4C72B0",  # was "thalamic-thalamic"
    "hypothalamic": "#8172B2",  # was "to HY"
}

EXTERNAL_VAR_COLORS: Dict[str, str] = {
    'position':           "#DE6E4B",
    'speed':              "#4B7DDE",
    'reward_presence':    "#55A868",
    'reward_consumption': "#B07AA1",
}

DOT_COLOR = "#262626"
BAR_HEIGHT = 0.62
GROUP_GAP = 1.35
CLUSTER_GAP = 0.55     # extra spacing between trial-type clusters within one region (task 4 only)
REGION_GAP = 0.15      # extra spacing between different regions within one category
DOT_JITTER_FRAC = 0.35
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 15
CLUSTER_HATCH_CYCLE = [None, "///", "xxx"]


# =============================================================================
# 3.  Low-level primitives -- copied verbatim from the pCCA/CCA sibling
#     script (project convention: primitives are copied, not imported).
# =============================================================================

# ---- 3a. Behavioural loading -----------------------------------------------
def load_behavior_regressors(
        session_name: str,
        behavior_dir: Path = BEHAVIOR_DIR,
        trial_label: str = "cued hit long",
        fs: float = BEHAVIOR_FS,
        t_offset: float = BEHAVIOR_T_OFFSET,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-trial position (x, y, z) and speed for one session, filtered to
    trials matching `trial_label`.  Returns (pos (n,3,T), speed (n,1,T), t (T,))."""
    pos_path = behavior_dir / f"{session_name}_pos.npy"
    speed_path = behavior_dir / f"{session_name}_speed.npy"
    label_path = behavior_dir / f"{session_name}_task_label.npy"
    for p in (pos_path, speed_path, label_path):
        if not p.exists():
            raise FileNotFoundError(f"Behaviour file not found: {p}")
    pos = np.load(pos_path)
    speed = np.load(speed_path)
    labels = np.load(label_path, allow_pickle=True)
    if speed.ndim == 2:
        speed = speed[:, None, :]
    sel = (labels == trial_label)
    if not np.any(sel):
        available = sorted(set(labels.tolist()))
        raise ValueError(f"No behaviour trials labelled '{trial_label}' "
                         f"(available: {available}).")
    pos_sel = pos[sel].astype(np.float32)
    speed_sel = speed[sel].astype(np.float32)
    T_behav = pos_sel.shape[-1]
    t_behav = np.arange(T_behav, dtype=np.float64) / fs + t_offset
    return pos_sel, speed_sel, t_behav


def _trial_type_to_behavior_label(trial_type: str) -> str:
    return trial_type.replace('_', ' ')


def _load_behavior_safe(
        session_name: str, trial_type: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    label = _trial_type_to_behavior_label(trial_type)
    try:
        return load_behavior_regressors(session_name, trial_label=label)
    except (FileNotFoundError, ValueError) as exc:
        warnings.warn(f"[{session_name}/{trial_type}] behaviour unavailable ({exc}).")
        return None


# ---- 3b. Ridge R^2 ---------------------------------------------------------
def variance_explained(latent_2d: np.ndarray, design_3d: np.ndarray,
                       lam: float = LAMBDA_R2) -> float:
    r"""Ridge R^2 for a latent (n_trials, T) explained by a behavioural design
    (n_trials, C, T). Flattened over (trial, time), finite-masked, mean-centred
    (intercept), design columns z-scored for conditioning. R^2 in [0, 1],
    clipped; invariant to a global sign flip of the latent (so no flip-
    alignment step is required upstream of this function)."""
    n_tr, T = latent_2d.shape
    ell = latent_2d.reshape(-1).astype(np.float64)
    Z = np.transpose(design_3d, (0, 2, 1)).reshape(n_tr * T, -1).astype(np.float64)
    finite = np.isfinite(ell) & np.all(np.isfinite(Z), axis=1)
    if finite.sum() < (Z.shape[1] + 2):
        return 0.0
    ell = ell[finite]
    Z = Z[finite]
    zsd = Z.std(axis=0)
    zsd[zsd < 1e-12] = 1.0
    Z = (Z - Z.mean(axis=0, keepdims=True)) / zsd
    ell_c = ell - ell.mean()
    ss_tot = float(ell_c @ ell_c)
    if ss_tot < 1e-12:
        return 0.0
    n, m = Z.shape
    ZtZ = Z.T @ Z + lam * n * np.eye(m)
    beta = np.linalg.solve(ZtZ, Z.T @ ell_c)
    resid = ell_c - Z @ beta
    return float(np.clip(1.0 - float(resid @ resid) / ss_tot, 0.0, 1.0))


def variance_explained_unique_loo(
        latent_2d: np.ndarray, design_dict: Dict[str, np.ndarray],
        lam: float = LAMBDA_R2,
) -> Dict[str, float]:
    """Unique (leave-one-out, 'no-refit') R^2 per predictor block."""
    n_tr, T = latent_2d.shape
    ell = latent_2d.reshape(-1).astype(np.float64)

    names = list(design_dict.keys())
    blocks, slices, col = [], {}, 0
    for name in names:
        Z_i = np.transpose(design_dict[name], (0, 2, 1)).reshape(n_tr * T, -1).astype(np.float64)
        blocks.append(Z_i)
        slices[name] = slice(col, col + Z_i.shape[1])
        col += Z_i.shape[1]
    Z = np.concatenate(blocks, axis=1)

    finite = np.isfinite(ell) & np.all(np.isfinite(Z), axis=1)
    out = {name: 0.0 for name in names}
    if finite.sum() < (Z.shape[1] + 2):
        return out
    ell_f = ell[finite]
    Z_f = Z[finite]
    zsd = Z_f.std(axis=0)
    zsd[zsd < 1e-12] = 1.0
    Z_f = (Z_f - Z_f.mean(axis=0, keepdims=True)) / zsd
    ell_c = ell_f - ell_f.mean()
    ss_tot = float(ell_c @ ell_c)
    if ss_tot < 1e-12:
        return out

    n, m = Z_f.shape
    ZtZ = Z_f.T @ Z_f + lam * n * np.eye(m)
    beta_full = np.linalg.solve(ZtZ, Z_f.T @ ell_c)
    resid_full = ell_c - Z_f @ beta_full
    r2_full = float(np.clip(1.0 - float(resid_full @ resid_full) / ss_tot, 0.0, 1.0))

    for name in names:
        beta_drop = beta_full.copy()
        beta_drop[slices[name]] = 0.0
        resid_drop = ell_c - Z_f @ beta_drop
        r2_drop = float(np.clip(1.0 - float(resid_drop @ resid_drop) / ss_tot, 0.0, 1.0))
        out[name] = max(0.0, r2_full - r2_drop)
    return out


# ---- 3c. Per-trial PCA projection -- the region-level analogue of the
#          pCCA/CCA sibling's _project_trials_single_region, applied here
#          UNIFORMLY to every trial type (including the reference) so tasks
#          1-4 all have trial-resolved, freshly z-scored-and-projected
#          latents, rather than relying on the file's own pre-computed
#          mean/std-only projections for the reference condition. -----------
def _project_trials_single_region(X_raw: np.ndarray, W: np.ndarray, n_components: int) -> np.ndarray:
    """
    X_raw : (n_trials, n_neurons, n_time) raw spike-rate tensor for one region
    W     : (n_neurons, >=n_components) PCA coefficient matrix, extracted
            from the reference-condition fit.

    Returns (n_trials, n_time, n_components).
    """
    n_trials, n_neurons, n_time = X_raw.shape
    sampled = np.transpose(X_raw, (1, 2, 0))                     # (n_neurons, n_time, n_trials)
    flat = sampled.reshape(n_neurons, n_time * n_trials).T       # (n_time*n_trials, n_neurons)
    flat = zscore(flat, axis=0, nan_policy='omit')
    flat = np.nan_to_num(flat, nan=0.0)
    proj = flat @ W[:, :n_components]                            # (n_time*n_trials, n_components)
    proj = proj.T.reshape(n_components, n_time, n_trials)        # (k, T, n_trials)
    return np.transpose(proj, (2, 1, 0))                         # (n_trials, T, k)


# ---- 3d. Reward-kernel design matrix ---------------------------------------
def _bspline_basis_matrix(t: np.ndarray, n_basis: int, degree: int = REWARD_SPLINE_DEGREE) -> np.ndarray:
    """Cubic B-spline basis (matching R's bs()) evaluated at t, spanning
    [t.min(), t.max()], with n_basis basis functions and a clamped, evenly
    spaced interior knot sequence."""
    t_min, t_max = float(t.min()), float(t.max())
    n_basis = n_basis + 2
    n_interior = max(n_basis - degree - 1, 0)
    interior_knots = (np.linspace(t_min, t_max, n_interior + 2)[1:-1]
                      if n_interior > 0 else np.array([]))
    knots = np.concatenate([np.full(degree + 1, t_min), interior_knots, np.full(degree + 1, t_max)])
    n_coef = len(knots) - degree - 1
    basis = np.zeros((n_coef, t.size))
    for i in range(n_coef):
        c = np.zeros(n_coef)
        c[i] = 1.4
        spline = BSpline(knots, c, degree, extrapolate=False)
        basis[i] = np.nan_to_num(spline(t), nan=0.0)
    basis_end = basis[1:-1, :]
    return basis_end  # (n_basis, T)


def build_reward_presence_design(
        t_behav: np.ndarray,
        n_trials: int,
        presence_window: Tuple[float, float] = REWARD_PRESENCE_WINDOW_S,
) -> np.ndarray:
    """(n_trials, 1, T) step-function 'reward presence' regressor."""
    presence = ((t_behav >= presence_window[0]) & (t_behav <= presence_window[1])).astype(float)
    return np.tile(presence[None, None, :], (n_trials, 1, 1))


def build_reward_consumption_design(
        t_behav: np.ndarray,
        n_trials: int,
        consumption_window: Tuple[float, float] = REWARD_CONSUMPTION_WINDOW_S,
        n_basis: int = N_REWARD_CONSUMPTION_BASIS,
) -> np.ndarray:
    """(n_trials, n_basis, T) cubic B-spline 'reward consumption' kernel."""
    lo, hi = consumption_window
    mask = (t_behav >= lo) & (t_behav <= hi)
    consumption = np.zeros((n_basis, t_behav.size))
    if mask.sum() >= (n_basis + REWARD_SPLINE_DEGREE + 1):
        consumption[:, mask] = _bspline_basis_matrix(t_behav[mask], n_basis)
    return np.tile(consumption[None, :, :], (n_trials, 1, 1))


def _build_predictor_designs(
        pos: np.ndarray, speed: np.ndarray, t_behav: np.ndarray, n_trials: int,
) -> Dict[str, np.ndarray]:
    """Map EXTERNAL_VARIABLES names to (n_trials, C, T) design blocks."""
    out: Dict[str, np.ndarray] = {}
    if 'position' in EXTERNAL_VARIABLES:
        out['position'] = pos
    if 'speed' in EXTERNAL_VARIABLES:
        out['speed'] = speed
    if 'position+speed' in EXTERNAL_VARIABLES:
        out['position+speed'] = np.concatenate([pos, speed], axis=1)
    if 'reward_presence' in EXTERNAL_VARIABLES:
        out['reward_presence'] = build_reward_presence_design(t_behav, n_trials)
    if 'reward_consumption' in EXTERNAL_VARIABLES:
        out['reward_consumption'] = build_reward_consumption_design(t_behav, n_trials)
    return out


# ---- 3e. Small numeric helpers ---------------------------------------------
def _sem(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _signed_peak_in_window(trace: np.ndarray, time_vec: np.ndarray,
                           window: Tuple[float, float]) -> float:
    lo, hi = window
    mask = (time_vec >= lo - 1e-6) & (time_vec <= hi + 1e-6)
    windowed = trace[mask]
    if windowed.size == 0:
        return float('nan')
    idx = np.argmax(np.abs(windowed))
    return float(windowed[idx])


def _compute_delta(ref_trace: np.ndarray, comp_trace: np.ndarray, time_vec: np.ndarray,
                   window: Tuple[float, float], metric: str) -> float:
    lo, hi = window
    mask = (time_vec >= lo - 1e-6) & (time_vec <= hi + 1e-6)
    d = comp_trace[mask] - ref_trace[mask]
    if d.size == 0:
        return float('nan')
    if metric == 'peak_of_difference':
        idx = np.argmax(np.abs(d))
        return float(d[idx])
    elif metric == 'auc_of_difference':
        return float(np.mean(d))
    raise ValueError(f"Unknown DELTA_METRIC: {metric!r}")


def _paired_wilcoxon(a: np.ndarray, b: Optional[np.ndarray] = None) -> float:
    x = a if b is None else (a - b)
    if x.size < 5 or np.allclose(x, x[0]):
        return float('nan')
    try:
        _, p = wilcoxon(x)
        return float(p)
    except ValueError:
        return float('nan')


def _sessions_for_trial_type(cs: CrossSessionPCAAnalyzer, trial_type: str) -> List[str]:
    """Recover, in the SAME order CrossSessionPCAAnalyzer.aggregate_projections
    used internally, which sessions contributed to `trial_type`'s stacked
    z_sessions array. Python dicts preserve insertion order, so this is
    exact, not a heuristic -- the direct region-level analogue of the
    pCCA/CCA sibling's _sessions_for_trial_type."""
    return [s for s, proj in cs.session_projections.items() if trial_type in proj]


def _prepare_regression_inputs(
        latent: np.ndarray, time_bins: np.ndarray,
        pos: np.ndarray, speed: np.ndarray, t_behav: np.ndarray,
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
    """Crop the per-trial neural latent (n_trials, T_neural) and the
    behavioural tensors to BEHAVIOR_TIME_RANGE_S, match trial/time counts
    (shorter-of-the-two truncation), and build the predictor design dict.
    Returns None if there isn't enough overlap to regress."""
    lo, hi = BEHAVIOR_TIME_RANGE_S
    neural_mask = (time_bins >= lo - 1e-6) & (time_bins <= hi + 1e-6)
    behav_mask = (t_behav >= lo - 1e-6) & (t_behav <= hi + 1e-6)
    if neural_mask.sum() < 2 or behav_mask.sum() < 2:
        return None

    L = latent[:, neural_mask]
    P = pos[:, :, behav_mask]
    S = speed[:, :, behav_mask]
    t_win = t_behav[behav_mask]

    n = min(L.shape[0], P.shape[0])
    T = min(L.shape[1], P.shape[2], t_win.size)
    if n < 3 or T < 2:
        return None
    L = L[:n, :T]
    P = P[:n, :, :T]
    S = S[:n, :, :T]
    t_win = t_win[:T]

    design_dict = _build_predictor_designs(P, S, t_win, n_trials=n)
    return L, design_dict, t_win


# =============================================================================
# 4.  Main data-gathering pass: ONE loop over (session, region) feeds BOTH
#     the peak/delta pipeline (tasks 1-2, via CrossSessionPCAAnalyzer) and
#     the behavioural-variance pipeline (tasks 3-4, via fresh per-trial
#     projections), so every .mat file is only loaded once per session.
# =============================================================================
def run_full_analysis(
        sessions: List[str] = SESSIONS,
        region_list: List[str] = REGIONS_OF_INTEREST,
        base_dir: Path = BASE_DIR,
        reference_type: str = REFERENCE_TYPE,
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        n_components: int = N_COMPONENTS,
        component_indices: List[int] = COMPONENT_INDICES,
        min_sessions: int = MIN_SESSIONS,
) -> Tuple[Dict[str, CrossSessionPCAAnalyzer], List[dict]]:
    region_analyzers: Dict[str, CrossSessionPCAAnalyzer] = {}
    behavior_records: List[dict] = []

    for s_idx, session_name in enumerate(sessions, 1):
        print("\n" + "=" * 70)
        print(f"SESSION {s_idx}/{len(sessions)}: {session_name}")
        print("=" * 70)

        analyzer = CrossTrialTypePCAAnalyzer(
            base_dir=str(base_dir), session_name=session_name,
            reference_type=reference_type, n_components=n_components,
        )
        if not analyzer.load_all_trial_types():
            print(f"  [skip session] could not load trial types for {session_name}")
            continue

        behavior_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

        for region in region_list:
            if region not in analyzer.available_regions:
                continue
            if not analyzer.extract_pca_weights(region):
                continue
            if not analyzer.extract_neural_data_for_region(region):
                continue

            W = analyzer.pca_weights[region]['coefficients']
            n_comp_use = min(n_components, W.shape[1] if W.ndim > 1 else 1)
            if n_comp_use < 1 or max(component_indices) >= n_comp_use:
                continue

            if region not in region_analyzers:
                region_analyzers[region] = CrossSessionPCAAnalyzer(
                    base_dir=str(base_dir), region=region,
                    reference_type=reference_type, n_components=n_components,
                    min_sessions=min_sessions,
                )

            session_projections_for_region: Dict[str, dict] = {}

            for trial_type in analyzer.available_trial_types:
                if trial_type not in active_trial_types:
                    continue
                if trial_type not in analyzer.neural_data or region not in analyzer.neural_data[trial_type]:
                    continue

                X_raw = analyzer.neural_data[trial_type][region]
                z_trials = _project_trials_single_region(X_raw, W, n_comp_use)  # (n_trials, T, k)
                z_mean = np.mean(z_trials, axis=0)                             # (T, k)

                # ---- tasks 1 & 2 data path (feeds CrossSessionPCAAnalyzer) --
                session_projections_for_region[trial_type] = dict(
                    z_mean=z_mean, n_trials=z_trials.shape[0],
                )

                # ---- tasks 3 & 4 data path (fresh per-trial regression) -----
                if trial_type not in behavior_cache:
                    behavior_cache[trial_type] = _load_behavior_safe(session_name, trial_type)
                behav = behavior_cache[trial_type]
                if behav is None:
                    continue
                pos_full, speed_full, t_behav_full = behav

                for comp_idx in component_indices:
                    latent = z_trials[:, :, comp_idx]
                    prep = _prepare_regression_inputs(
                        latent, analyzer.time_bins, pos_full, speed_full, t_behav_full
                    )
                    if prep is None:
                        continue
                    latent_c, design_dict, t_win = prep

                    if VARIANCE_METHOD == 'marginal':
                        r2_by_var = {name: variance_explained(latent_c, d)
                                    for name, d in design_dict.items()}
                    elif VARIANCE_METHOD == 'leave_one_out':
                        r2_by_var = variance_explained_unique_loo(latent_c, design_dict)
                    else:
                        raise ValueError(f"Unknown VARIANCE_METHOD: {VARIANCE_METHOD!r}")

                    for var_name, r2_val in r2_by_var.items():
                        behavior_records.append(dict(
                            session=session_name, region=region, trial_type=trial_type,
                            component=comp_idx, predictor=var_name, r2=r2_val,
                        ))

            if session_projections_for_region:
                region_analyzers[region].add_session_result(
                    session_name, session_projections_for_region, {}, analyzer.time_bins
                )

        del analyzer  # drop this session's raw spike tensors before moving on

    # cross-session aggregation (sign alignment + mean/SEM across sessions),
    # delegated entirely to CrossSessionPCAAnalyzer.aggregate_projections.
    print("\n" + "=" * 70)
    print("CROSS-SESSION AGGREGATION (sign alignment + mean/SEM across sessions)")
    print("=" * 70)
    for region, cs in region_analyzers.items():
        n_sess = len(cs.session_projections)
        if n_sess < min_sessions:
            print(f"  {region}: {n_sess} sessions (skipping, < {min_sessions})")
            continue
        cs.aggregate_projections()

    print("\n" + "=" * 70)
    print("DATA-GATHERING COMPLETE")
    print(f"  regions with >=1 session : {len(region_analyzers)}")
    print(f"  regions successfully aggregated : "
          f"{sum(1 for cs in region_analyzers.values() if cs.aggregated_projections)}")
    print(f"  behavioural-variance records  : {len(behavior_records)}")
    print("=" * 70)
    return region_analyzers, behavior_records


# =============================================================================
# 5.  Task 1 -- peak comparison (Fig. 3e style)
# =============================================================================
def aggregate_peak_records(
        region_analyzers: Dict[str, CrossSessionPCAAnalyzer],
        component_idx: int = COMPONENT_INDICES[0],
        window: Tuple[float, float] = PEAK_WINDOW_S,
        normalize: bool = NORMALIZE_PEAK_BY_REFERENCE,
        reference_type: str = REFERENCE_TYPE,
) -> Tuple[Dict[str, Dict[str, dict]], List[dict]]:
    """
    Returns
    -------
    summary : {region: {trial_type: {mean, sem, values, n, wilcoxon_p_vs_ref}}}
    records : flat per-session rows, for the CSV.
    """
    summary: Dict[str, Dict[str, dict]] = {}
    records: List[dict] = []

    for region, cs in region_analyzers.items():
        if reference_type not in cs.aggregated_projections:
            continue
        ref_agg = cs.aggregated_projections[reference_type]
        sess_names_ref = _sessions_for_trial_type(cs, reference_type)
        ref_sessions = ref_agg['z_sessions']
        ref_peak_by_session = {
            sname: _signed_peak_in_window(ref_sessions[i, :, component_idx], cs.time_bins, window)
            for i, sname in enumerate(sess_names_ref)
        }

        for trial_type in cs.available_trial_types:
            if trial_type not in cs.aggregated_projections:
                continue
            agg = cs.aggregated_projections[trial_type]
            sessions = agg['z_sessions']
            sess_names_this = _sessions_for_trial_type(cs, trial_type)

            values, matched_ref = [], []
            for i, sname in enumerate(sess_names_this):
                raw_peak = _signed_peak_in_window(sessions[i, :, component_idx], cs.time_bins, window)
                ref_peak = ref_peak_by_session.get(sname)
                if normalize:
                    if ref_peak is None or abs(ref_peak) < 1e-12:
                        continue
                    peak_val = raw_peak / abs(ref_peak)
                    ref_val = ref_peak / abs(ref_peak)
                else:
                    if ref_peak is None:
                        continue
                    peak_val = raw_peak
                    ref_val = ref_peak
                values.append(peak_val)
                matched_ref.append(ref_val)
                records.append(dict(
                    session=sname, region=region,
                    trial_type=trial_type, component=component_idx, peak=peak_val,
                ))

            if not values:
                continue
            values_arr = np.asarray(values, dtype=float)
            ref_arr = np.asarray(matched_ref, dtype=float)
            p_vs_ref = (float('nan') if trial_type == reference_type
                       else _paired_wilcoxon(values_arr, ref_arr))
            summary.setdefault(region, {})[trial_type] = dict(
                mean=float(values_arr.mean()), sem=_sem(values_arr),
                values=values_arr, n=values_arr.size, wilcoxon_p_vs_ref=p_vs_ref,
            )
    return summary, records


def plot_task1_peak_bars(summary: dict, save_path: Path,
                         active_trial_types: List[str] = ACTIVE_TRIAL_TYPES) -> Optional[plt.Figure]:
    region_clusters: Dict[str, List[list]] = {}
    for region, per_type in summary.items():
        ordered_types = [t for t in ([REFERENCE_TYPE] + sorted(set(active_trial_types) - {REFERENCE_TYPE}))
                         if t in per_type]
        cluster = [dict(mean=per_type[t]['mean'], sem=per_type[t]['sem'], values=per_type[t]['values'],
                        color=TRIAL_TYPE_COLORS.get(t, 'gray'), alpha=0.9, hatch=None)
                  for t in ordered_types]
        if cluster:
            region_clusters[region] = [cluster]  # single cluster per region

    legend = [(t.replace('_', ' '), TRIAL_TYPE_COLORS.get(t, 'gray'), None) for t in active_trial_types]
    ylabel = ("Peak amplitude, normalized to reference |peak| (a.u.)" if NORMALIZE_PEAK_BY_REFERENCE
             else "Peak amplitude (a.u.)")
    return plot_grouped_region_bars(
        REGION_CATEGORIES, region_clusters, save_path, xlabel=ylabel,
        legend_entries=legend, vline_zero=not NORMALIZE_PEAK_BY_REFERENCE,
    )


# =============================================================================
# 6.  Task 2 -- delta comparison (Fig. 5 style)
# =============================================================================
def aggregate_delta_records(
        region_analyzers: Dict[str, CrossSessionPCAAnalyzer],
        component_idx: int = COMPONENT_INDICES[0],
        window: Tuple[float, float] = DELTA_WINDOW_S,
        normalize: bool = NORMALIZE_PEAK_BY_REFERENCE,
        reference_type: str = REFERENCE_TYPE,
        metric: str = DELTA_METRIC,
) -> Tuple[Dict[str, Dict[str, dict]], List[dict]]:
    summary: Dict[str, Dict[str, dict]] = {}
    records: List[dict] = []

    for region, cs in region_analyzers.items():
        if reference_type not in cs.aggregated_projections:
            continue
        ref_agg = cs.aggregated_projections[reference_type]
        sess_names_ref = _sessions_for_trial_type(cs, reference_type)
        ref_sessions = ref_agg['z_sessions']
        ref_trace_by_session, ref_peak_by_session = {}, {}
        for i, sname in enumerate(sess_names_ref):
            ref_trace_by_session[sname] = ref_sessions[i, :, component_idx]
            ref_peak_by_session[sname] = _signed_peak_in_window(
                ref_sessions[i, :, component_idx], cs.time_bins, window)

        for trial_type in cs.available_trial_types:
            if trial_type == reference_type or trial_type not in cs.aggregated_projections:
                continue
            agg = cs.aggregated_projections[trial_type]
            sessions = agg['z_sessions']
            sess_names_this = _sessions_for_trial_type(cs, trial_type)

            values = []
            for i, sname in enumerate(sess_names_this):
                if sname not in ref_trace_by_session:
                    continue
                ref_trace = ref_trace_by_session[sname]
                comp_trace = sessions[i, :, component_idx]
                if normalize:
                    denom = abs(ref_peak_by_session[sname])
                    if denom < 1e-12:
                        continue
                    ref_trace = ref_trace / denom
                    comp_trace = comp_trace / denom
                delta = _compute_delta(ref_trace, comp_trace, cs.time_bins, window, metric)
                values.append(delta)
                records.append(dict(
                    session=sname, region=region,
                    trial_type=trial_type, component=component_idx, delta=delta,
                ))

            if not values:
                continue
            values_arr = np.asarray(values, dtype=float)
            summary.setdefault(region, {})[trial_type] = dict(
                mean=float(values_arr.mean()), sem=_sem(values_arr),
                values=values_arr, n=values_arr.size,
                wilcoxon_p_vs_zero=_paired_wilcoxon(values_arr),
            )
    return summary, records


def plot_task2_delta_bars(summary: dict, save_path: Path,
                          active_trial_types: List[str] = ACTIVE_TRIAL_TYPES) -> Optional[plt.Figure]:
    non_ref = [t for t in active_trial_types if t != REFERENCE_TYPE]
    region_clusters: Dict[str, List[list]] = {}
    for region, per_type in summary.items():
        cluster = [dict(mean=per_type[t]['mean'], sem=per_type[t]['sem'], values=per_type[t]['values'],
                        color=TRIAL_TYPE_COLORS.get(t, 'gray'), alpha=0.9, hatch=None)
                  for t in non_ref if t in per_type]
        if cluster:
            region_clusters[region] = [cluster]

    legend = [(t.replace('_', ' '), TRIAL_TYPE_COLORS.get(t, 'gray'), None) for t in non_ref]
    metric_label = "AUC of (reference - comparison)" if DELTA_METRIC == 'auc_of_difference' \
        else "Peak of (reference - comparison)"
    xlabel = f"{metric_label} [{'normalized' if NORMALIZE_PEAK_BY_REFERENCE else 'raw'}]"
    return plot_grouped_region_bars(
        REGION_CATEGORIES, region_clusters, save_path, xlabel=xlabel,
        legend_entries=legend, vline_zero=True,
    )


# =============================================================================
# 7.  Tasks 3 & 4 -- behavioural variance explained
# =============================================================================
def aggregate_behavior_variance(
        behavior_records: List[dict],
) -> Dict[Tuple[str, str], Dict[str, dict]]:
    """Returns {(region, trial_type): {predictor: {mean, sem, values, n}}},
    pooled across sessions (and, if COMPONENT_INDICES has >1 entry, across
    components too)."""
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for rec in behavior_records:
        key = (rec['region'], rec['trial_type'], rec['predictor'])
        grouped.setdefault(key, []).append(rec['r2'])

    out: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for (region, trial_type, predictor), vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size < MIN_SESSIONS:
            continue
        out.setdefault((region, trial_type), {})[predictor] = dict(
            mean=float(arr.mean()), sem=_sem(arr), values=arr, n=arr.size,
        )
    return out


def plot_task3_variance_bars(summary: dict, save_path: Path,
                             reference_type: str = REFERENCE_TYPE) -> Optional[plt.Figure]:
    region_clusters_by_variable: Dict[str, Dict[str, List[list]]] = {
        v: {} for v in EXTERNAL_VARIABLES
    }
    for (region, trial_type), per_pred in summary.items():
        if trial_type != reference_type:
            continue
        for v in EXTERNAL_VARIABLES:
            if v not in per_pred:
                continue
            stats = per_pred[v]
            bar = dict(mean=stats['mean'], sem=stats['sem'], values=stats['values'],
                      color=EXTERNAL_VAR_COLORS.get(v, 'gray'), alpha=0.9, hatch=None)
            region_clusters_by_variable[v][region] = [[bar]]

    return plot_grouped_region_bars_multipanel(
        REGION_CATEGORIES, region_clusters_by_variable, EXTERNAL_VARIABLES, save_path,
        common_xlabel=f"Variance explained, $R^2$ ({reference_type.replace('_', ' ')})",
        xlim=(0, 0.5), legend_entries=None,
    )


def plot_task4_variance_bars(summary: dict, save_path: Path,
                             active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
                             reference_type: str = REFERENCE_TYPE) -> Optional[plt.Figure]:
    non_ref = [t for t in active_trial_types if t != reference_type]
    region_clusters_by_variable: Dict[str, Dict[str, List[list]]] = {
        v: {} for v in EXTERNAL_VARIABLES
    }
    all_regions = {k[0] for k in summary.keys()}
    for v in EXTERNAL_VARIABLES:
        for region in all_regions:
            clusters: List[list] = []
            for ti, trial_type in enumerate(non_ref):
                hatch = CLUSTER_HATCH_CYCLE[ti % len(CLUSTER_HATCH_CYCLE)]
                per_pred = summary.get((region, trial_type))
                if per_pred and v in per_pred:
                    stats = per_pred[v]
                    bar = dict(mean=stats['mean'], sem=stats['sem'], values=stats['values'],
                              color=EXTERNAL_VAR_COLORS.get(v, 'gray'), alpha=0.9, hatch=hatch)
                    clusters.append([bar])
            if clusters:
                region_clusters_by_variable[v][region] = clusters

    legend = [(t.replace('_', ' '), '#bbbbbb', CLUSTER_HATCH_CYCLE[ti % len(CLUSTER_HATCH_CYCLE)])
              for ti, t in enumerate(non_ref)]
    return plot_grouped_region_bars_multipanel(
        REGION_CATEGORIES, region_clusters_by_variable, EXTERNAL_VARIABLES, save_path,
        common_xlabel="Variance explained, $R^2$ (projected conditions)",
        xlim=(0, 0.5), legend_entries=legend,
    )


# =============================================================================
# 8.  Task 5 -- per-region latent traces across sessions
# =============================================================================
def plot_task5_latent_traces(
        region_analyzers: Dict[str, CrossSessionPCAAnalyzer],
        save_path: Path,
        component_idx: int = COMPONENT_INDICES[0],
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        row_height: float = 1.5,
        fig_width: float = 4.0,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """One row per region (the pCCA/CCA sibling has one row per (pair,
    region) since it has two regions per pair; PCA has just one). Trace
    styling (individual sessions thin/low-alpha, mean bold, SEM shading,
    dashed line at t=0) is identical to the sibling script. Y-axis is left
    to auto-scale rather than fixed, since PCA score magnitude (raw
    z-scored activity projected onto a unit-norm loading vector) is not on
    the same bounded scale as a canonical correlation variate."""
    rows: List[str] = []
    row_category: Dict[int, str] = {}
    for category, regions in REGION_CATEGORIES:
        for region in regions:
            cs = region_analyzers.get(region)
            if cs is None or not cs.aggregated_projections:
                continue
            row_category[len(rows)] = category
            rows.append(region)

    if not rows:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    n_rows = (len(rows) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width * 2, row_height * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for r, region in enumerate(rows):
        ax = axes[r]
        cs = region_analyzers[region]
        ax.set_facecolor(CATEGORY_COLORS.get(row_category[r], '#888888'))
        ax.patch.set_alpha(0.07)

        for trial_type in active_trial_types:
            if trial_type not in cs.aggregated_projections:
                continue
            agg = cs.aggregated_projections[trial_type]
            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')

            session_traces = agg['z_sessions'][:, :, component_idx]
            for sess_trace in session_traces:
                ax.plot(cs.time_bins, sess_trace, color=color, linewidth=0.5, alpha=0.2, zorder=1)

            mean_trace = agg['z_mean'][:, component_idx]
            sem_trace = agg['z_sem'][:, component_idx]
            is_ref = (trial_type == REFERENCE_TYPE)
            ax.plot(cs.time_bins, mean_trace, color=color, linewidth=2.0 if is_ref else 1.4,
                    alpha=0.85 if is_ref else 0.75,
                    label=f"{trial_type.replace('_', ' ')} (n={agg['n_sessions']})", zorder=3)
            ax.fill_between(cs.time_bins, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=color, alpha=0.15, zorder=2)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
        ax.set_xlim(cs.time_bins[0], cs.time_bins[-1])
        ax.text(0.01, 0.90, f"{_display_name(region)}",
                transform=ax.transAxes, fontsize=TICK_FONTSIZE - 6, va='top', ha='left')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)
        if r == 0:
            ax.legend(fontsize=LEGEND_FONTSIZE - 3, loc='upper right', frameon=False)

    for i in range(len(rows), len(axes)):
        axes[i].axis('off')

    bottom_left_idx = (n_rows - 1) * 2
    if bottom_left_idx < len(rows):
        axes[bottom_left_idx].set_xlabel("Time from reach (s)", fontsize=TICK_FONTSIZE)
        axes[bottom_left_idx].tick_params(axis='x', labelsize=TICK_FONTSIZE - 2, labelbottom=True)

    bottom_right_idx = len(rows) - 1 if len(rows) % 2 != 0 else (n_rows - 1) * 2 + 1
    if bottom_right_idx < len(rows) and bottom_right_idx != bottom_left_idx:
        axes[bottom_right_idx].set_xlabel("Time from reach (s)", fontsize=TICK_FONTSIZE)
        axes[bottom_right_idx].tick_params(axis='x', labelsize=TICK_FONTSIZE - 2, labelbottom=True)

    fig.tight_layout(h_pad=0.15)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 9.  Shared plotting engine -- direct region-level analogue of the pCCA/CCA
#     sibling's plot_grouped_pair_bars / plot_grouped_pair_bars_multipanel.
#     Every style constant (category background bands, jitter/error-bar/dot
#     styling, font sizes, dpi, spine removal, inverted y-axis) is unchanged;
#     the only structural difference is that a "cluster" here holds bars
#     that are each already region-specific (no region_i/region_j role
#     split, hence no colour-lightening step).
# =============================================================================
def plot_grouped_region_bars(
        region_categories: List[Tuple[str, List[str]]],
        region_clusters: Dict[str, List[List[dict]]],
        save_path: Path,
        xlabel: str,
        xlim: Optional[Tuple[float, float]] = None,
        legend_entries: Optional[List[Tuple[str, str, Optional[str]]]] = None,
        vline_zero: bool = False,
        fig_width: float = 11.0,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """
    region_clusters[region] : list of clusters; each cluster is a list of
    bar dicts {mean, sem, values, color, alpha, hatch}. One cluster per
    region for tasks 1/2 (a single row of 1-3 bars).
    """
    groups = []
    for category, regions in region_categories:
        present = [r for r in regions if r in region_clusters and region_clusters[r]]
        if present:
            groups.append((category, present))
    if not groups:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    y = 0.0
    flat_bars: List[Tuple[float, dict]] = []
    dot_x: List[float] = []
    dot_y: List[float] = []
    region_label_pos: Dict[str, float] = {}
    group_spans: List[Tuple[str, float, float]] = []
    rng = np.random.default_rng(0)

    for category, regions in groups:
        y_start = y
        for region in regions:
            clusters = region_clusters[region]
            region_y0 = y
            for ci, cluster in enumerate(clusters):
                for bar in cluster:
                    flat_bars.append((y, bar))
                    vals = np.asarray(bar['values'], dtype=float)
                    jitter = rng.uniform(-BAR_HEIGHT / 2 * DOT_JITTER_FRAC,
                                         BAR_HEIGHT / 2 * DOT_JITTER_FRAC, size=vals.size)
                    dot_x.extend(vals.tolist())
                    dot_y.extend((y + jitter).tolist())
                    y += 1.0
                if ci < len(clusters) - 1:
                    y += CLUSTER_GAP
            region_label_pos[region] = (region_y0 + y - 1.0) / 2.0
            y += REGION_GAP
        group_spans.append((category, y_start, y - REGION_GAP - 1.0))
        y += GROUP_GAP - 1.0

    n_bars = len(flat_bars)
    fig_h = max(5.0, 0.42 * n_bars + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_h))

    for category, y_lo, y_hi in group_spans:
        ax.axhspan(y_lo - BAR_HEIGHT / 2 - 0.25, y_hi + BAR_HEIGHT / 2 + 0.25,
                   color=CATEGORY_COLORS.get(category, '#888888'), alpha=0.07, zorder=0)

    means, sems, ys = [], [], []
    for yy, bar in flat_bars:
        ax.barh(yy, bar['mean'], height=BAR_HEIGHT, color=bar['color'],
                edgecolor='white', linewidth=0.6, alpha=bar.get('alpha', 0.9),
                hatch=bar.get('hatch'), zorder=2)
        means.append(bar['mean']); sems.append(bar['sem']); ys.append(yy)

    ax.errorbar(means, ys, xerr=sems, fmt='none', ecolor='black',
                elinewidth=1.8, capsize=4, capthick=1.8, zorder=3)
    ax.scatter(dot_x, dot_y, s=22, color=DOT_COLOR, alpha=0.55, linewidths=0, zorder=4)

    ax.set_yticks(list(region_label_pos.values()))
    ax.set_yticklabels([_display_name(r) for r in region_label_pos.keys()], fontsize=TICK_FONTSIZE)
    ax.margins(y=0.015)
    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if vline_zero:
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    ax.set_xlabel(xlabel, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if legend_entries:
        handles = [Patch(facecolor=c, label=l, alpha=0.9, hatch=h) for l, c, h in legend_entries]
        ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=False, loc='lower right')

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


def plot_grouped_region_bars_multipanel(
        region_categories: List[Tuple[str, List[str]]],
        region_clusters_by_variable: Dict[str, Dict[str, List[List[dict]]]],
        variable_names: List[str],
        save_path: Path,
        common_xlabel: str,
        xlim: Optional[Tuple[float, float]] = None,
        legend_entries: Optional[List[Tuple[str, str, Optional[str]]]] = None,
        panel_width: float = 4.6,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """One subplot per entry in `variable_names`, all sharing ONE y-axis
    layout (regions grouped by anatomical category) computed from the union
    of regions across all variables, so a region sits on the same row in
    every panel."""
    def _footprint(clusters: List[List[dict]]) -> float:
        if not clusters:
            return 1.0
        return sum(len(cl) for cl in clusters) + max(0, len(clusters) - 1) * CLUSTER_GAP

    all_regions = set()
    for v in variable_names:
        all_regions |= {r for r, c in region_clusters_by_variable.get(v, {}).items() if c}

    groups = []
    for category, regions in region_categories:
        present = [r for r in regions if r in all_regions]
        if present:
            groups.append((category, present))
    if not groups:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    y = 0.0
    region_y0: Dict[str, float] = {}
    region_label_pos: Dict[str, float] = {}
    group_spans: List[Tuple[str, float, float]] = []
    for category, regions in groups:
        y_start = y
        for region in regions:
            footprint = max(_footprint(region_clusters_by_variable.get(v, {}).get(region, []))
                            for v in variable_names)
            region_y0[region] = y
            region_label_pos[region] = y + footprint / 2.0 - 0.5
            y += footprint + REGION_GAP
        group_spans.append((category, y_start, y - REGION_GAP))
        y += GROUP_GAP - 1.0

    n_panels = len(variable_names)
    fig_h = max(5.0, 0.42 * y + 2.2)
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_width * n_panels, fig_h), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)

    for panel_idx, (variable_name, ax) in enumerate(zip(variable_names, axes)):
        for category, y_lo, y_hi in group_spans:
            ax.axhspan(y_lo - BAR_HEIGHT / 2 - 0.25, y_hi + BAR_HEIGHT / 2 + 0.25,
                       color=CATEGORY_COLORS.get(category, '#888888'), alpha=0.07, zorder=0)

        means, sems, ys, dot_x, dot_y = [], [], [], [], []
        for category, regions in groups:
            for region in regions:
                yy = region_y0[region]
                clusters = region_clusters_by_variable.get(variable_name, {}).get(region, [])
                for ci, cluster in enumerate(clusters):
                    for bar in cluster:
                        ax.barh(yy, bar['mean'], height=BAR_HEIGHT, color=bar['color'],
                                edgecolor='white', linewidth=0.6, alpha=bar.get('alpha', 0.9),
                                hatch=bar.get('hatch'), zorder=2)
                        means.append(bar['mean']); sems.append(bar['sem']); ys.append(yy)
                        vals = np.asarray(bar['values'], dtype=float)
                        jitter = rng.uniform(-BAR_HEIGHT / 2 * DOT_JITTER_FRAC,
                                             BAR_HEIGHT / 2 * DOT_JITTER_FRAC, size=vals.size)
                        dot_x.extend(vals.tolist())
                        dot_y.extend((yy + jitter).tolist())
                        yy += 1.0
                    if ci < len(clusters) - 1:
                        yy += CLUSTER_GAP

        ax.errorbar(means, ys, xerr=sems, fmt='none', ecolor='black',
                    elinewidth=1.8, capsize=4, capthick=1.8, zorder=3)
        ax.scatter(dot_x, dot_y, s=22, color=DOT_COLOR, alpha=0.55, linewidths=0, zorder=4)
        ax.set_title(variable_name.replace('_', ' '), fontsize=TICK_FONTSIZE)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.tick_params(axis='x', labelsize=TICK_FONTSIZE - 2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if panel_idx > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)
        if legend_entries and panel_idx == n_panels - 1:
            handles = [Patch(facecolor=c, label=l, alpha=0.9, hatch=h) for l, c, h in legend_entries]
            ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=False, loc='lower right')

    axes[0].set_yticks(list(region_label_pos.values()))
    axes[0].set_yticklabels([_display_name(r) for r in region_label_pos.keys()], fontsize=TICK_FONTSIZE)
    axes[0].margins(y=0.015)
    axes[0].invert_yaxis()
    fig.supxlabel(common_xlabel, fontsize=TICK_FONTSIZE)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 10.  CSV I/O -- simple existence-check caching, matching this project's
#      established convention (no config-hash keying).
# =============================================================================
def _write_records_csv(records: List[dict], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"  [csv] {len(records)} rows -> {path}")


def _write_peak_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region', 'trial_type', 'n_sessions', 'mean_peak', 'sem_peak', 'wilcoxon_p_vs_ref'])
        for region, per_type in sorted(summary.items(), key=lambda kv: get_anatomical_index(kv[0])):
            for trial_type, stats in per_type.items():
                w.writerow([region, trial_type, stats['n'], f"{stats['mean']:.6f}",
                           f"{stats['sem']:.6f}", stats['wilcoxon_p_vs_ref']])
    print(f"  [csv] -> {path}")


def _write_delta_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region', 'trial_type', 'n_sessions', 'mean_delta', 'sem_delta', 'wilcoxon_p_vs_zero'])
        for region, per_type in sorted(summary.items(), key=lambda kv: get_anatomical_index(kv[0])):
            for trial_type, stats in per_type.items():
                w.writerow([region, trial_type, stats['n'], f"{stats['mean']:.6f}",
                           f"{stats['sem']:.6f}", stats['wilcoxon_p_vs_zero']])
    print(f"  [csv] -> {path}")


def _write_variance_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region', 'trial_type', 'predictor', 'n_sessions', 'mean_r2', 'sem_r2'])
        for (region, trial_type), per_pred in sorted(summary.items(), key=lambda kv: get_anatomical_index(kv[0][0])):
            for predictor, stats in per_pred.items():
                w.writerow([region, trial_type, predictor, stats['n'],
                           f"{stats['mean']:.6f}", f"{stats['sem']:.6f}"])
    print(f"  [csv] -> {path}")


# =============================================================================
# 11.  Driver
# =============================================================================
def main() -> None:
    print("=" * 70)
    print("CROSS-SESSION PCA PEAK / DELTA / BEHAVIOURAL-VARIANCE ANALYSIS")
    print("=" * 70)
    print(f"  reference type     : {REFERENCE_TYPE}")
    print(f"  active trial types : {ACTIVE_TRIAL_TYPES}")
    print(f"  regions of interest: {REGIONS_OF_INTEREST}")
    print(f"  component indices  : {COMPONENT_INDICES}")
    print(f"  peak window (s)    : {PEAK_WINDOW_S}")
    print(f"  delta window (s)   : {DELTA_WINDOW_S}  metric={DELTA_METRIC}")
    print(f"  normalize by ref   : {NORMALIZE_PEAK_BY_REFERENCE}")
    print(f"  behaviour window   : {BEHAVIOR_TIME_RANGE_S}")
    print(f"  variance method    : {VARIANCE_METHOD}")
    print(f"  external variables : {EXTERNAL_VARIABLES}")
    print(f"  output directory   : {OUTPUT_DIR}")
    print("=" * 70)

    behav_records_path = OUTPUT_DIR / "behavior_variance_records.csv"
    peak_records_path = OUTPUT_DIR / "peak_records.csv"

    if USE_CACHED_DATA and behav_records_path.exists() and peak_records_path.exists():
        print(f"\n[cache] found existing per-session CSVs in {OUTPUT_DIR}; "
              f"NOTE: cached CSVs feed the summary/plotting stage only -- the "
              f"region_analyzers needed for tasks 1-2's flip-aligned trace "
              f"data are NOT cached, so a full re-run is still required "
              f"unless you additionally cache/reload aggregated_projections "
              f"yourself. Set USE_CACHED_DATA = False to force a clean run.")

    region_analyzers, behavior_records = run_full_analysis()

    # ---- Task 1: peak -----------------------------------------------------
    for comp_idx in COMPONENT_INDICES:
        print(f"\n--- Task 1: peak comparison (component {comp_idx}) ---")
        peak_summary, peak_records = aggregate_peak_records(
            region_analyzers, component_idx=comp_idx)
        _write_records_csv(peak_records, OUTPUT_DIR / f"task1_peak_records_comp{comp_idx}.csv")
        _write_peak_summary_csv(peak_summary, OUTPUT_DIR / f"task1_peak_summary_comp{comp_idx}.csv")
        plot_task1_peak_bars(peak_summary, OUTPUT_DIR / f"task1_peak_comp{comp_idx}.png")

        # ---- Task 2: delta --------------------------------------------------
        print(f"\n--- Task 2: delta comparison (component {comp_idx}) ---")
        delta_summary, delta_records = aggregate_delta_records(
            region_analyzers, component_idx=comp_idx)
        _write_records_csv(delta_records, OUTPUT_DIR / f"task2_delta_records_comp{comp_idx}.csv")
        _write_delta_summary_csv(delta_summary, OUTPUT_DIR / f"task2_delta_summary_comp{comp_idx}.csv")
        plot_task2_delta_bars(delta_summary, OUTPUT_DIR / f"task2_delta_comp{comp_idx}.png")

    # ---- Tasks 3 & 4: behavioural variance ---------------------------------
    print("\n--- Tasks 3-4: behavioural variance explained ---")
    _write_records_csv(behavior_records, behav_records_path)
    variance_summary = aggregate_behavior_variance(behavior_records)
    _write_variance_summary_csv(variance_summary, OUTPUT_DIR / "task3_4_variance_summary.csv")
    plot_task3_variance_bars(variance_summary, OUTPUT_DIR / "task3_variance_reference.png")
    plot_task4_variance_bars(variance_summary, OUTPUT_DIR / "task4_variance_projected.png")

    # ---- Task 5: latent traces across sessions -----------------------------
    print("\n--- Task 5: latent traces across sessions ---")
    for comp_idx in COMPONENT_INDICES:
        plot_task5_latent_traces(
            region_analyzers, OUTPUT_DIR / f"task5_latent_traces_comp{comp_idx}.png",
            component_idx=comp_idx)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Figures and CSVs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
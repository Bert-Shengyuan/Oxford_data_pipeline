#!/usr/bin/env python3
r"""
pCCA_all_regions_out_behaviour_v2.py
================================================================================

Version 2.0 of `pCCA_all_regions_out_behaviour.py` ("v1"): same six
quantities (Parts 1a/1b/2a/2b/2c/2c'), same core PCA/ridge/pCCA math, same
21-pair / 7-category `REGION_PAIRS`, same per-session .pkl-per-session
caching convention -- but Parts 2a/2b/2c/2c' (every PAIRED computation) are
now fit on `N_SAMPLE_DRAWS` (10) independently RESAMPLED neuron subsets per
region per session, instead of once on a fixed, previously-stored
`selected_neurons` set. Parts 1a/1b are untouched: they still fit on
whichever neurons `selected_neurons` names, via the exact same
`load_region_spikes` / `sel = safe_array(info.get("selected_neurons"))`
call site as v1.

--------------------------------------------------------------------------------
Why two neuron pools
--------------------------------------------------------------------------------
v1 loads each region's spikes ONCE, subset to `selected_neurons` at load
time (`load_region_spikes`), and both Part 1 (region-level PCA) and Part 2
(paired PCA/pCCA) fit on that same, single, fixed neuron set for the whole
session.

This version keeps that pool -- `region_flat_sel` below -- for Part 1 ONLY,
untouched. Part 2 instead draws its neurons from a SECOND, independently
loaded pool, `region_flat_full`: every recorded neuron, unfiltered by
`selected_neurons` (`load_region_spikes_full`, a new loader that mirrors
`load_region_spikes` minus the `sel` filter). Part 2's job becomes,
per pair, per session: sample `TARGET_SAMPLE_SIZE` (e.g. 50) neurons from
each of region_i, region_j, and every eligible nuisance region, `
N_SAMPLE_DRAWS` (10) separate times, and rerun the SAME Part-2a/2b/2c/2c'
computation on each draw -- see `sample_paired_draws` / `sample_nuisance_
draws` for exactly how the 10 draws are constructed, and
`_compute_pair_sampled` for the per-pair, per-draw loop.

--------------------------------------------------------------------------------
The sampling scheme (item 2 of the request)
--------------------------------------------------------------------------------
Two different rules apply depending on the ROLE a region is playing for a
given pair's computation:

  * region_i / region_j themselves ("paired" role, `sample_paired_draws`):
    the first `m = min(N_SAMPLE_DRAWS, n_full // TARGET_SAMPLE_SIZE)` draws
    are cut, ZERO-overlapping, from ONE random permutation of that region's
    `n_full` recorded neurons -- so together they sweep roughly
    `m * TARGET_SAMPLE_SIZE` of the `n_full` population while each
    individual draw is still a uniformly random subset (the permutation
    itself is random). The remaining `N_SAMPLE_DRAWS - m` draws are
    independent uniform-random draws from the FULL population, with no
    overlap constraint. Worked example (from the request): region_i has
    210 neurons, region_j 300, target 50 -> region_i's first 4 draws sweep
    ~200 of its 210 neurons (draws 5-10 fully random), region_j's first 6
    draws sweep its full 300 (draws 7-10 fully random).

  * every OTHER recorded region entering a pair's nuisance design Z
    ("nuisance" role, `sample_nuisance_draws`): simply `N_SAMPLE_DRAWS`
    independent uniform-random draws, no sweep/coverage requirement.

A region's draws depend only on ITS OWN `n_full` and role -- not on which
partner/pair is currently being computed -- so both draw sets are computed
ONCE per region per session (`paired_draws`, `nuisance_draws` in
`compute_private_latents_for_session`) and reused across every pair that
region participates in, whether as region_i/region_j (paired role) or as
nuisance for some OTHER pair (nuisance role). The same physical region can
therefore carry two different, independently-seeded draw sets in one
session (e.g. MOp is "paired"-sampled for the six pairs it anchors, and
separately "nuisance"-sampled for pairs like VALVM-VPMPO that treat MOp as
part of the rest of the network).

Every draw's sampled neuron indices (into that region's FULL, unfiltered
neuron axis -- the same axis `region_subregion_labels_full` is aligned to)
are stored alongside its result (item 3): `PrivateLatentPairDrawResult.
neuron_idx_i/neuron_idx_j/nuisance_neuron_idx`, `HubOrientationPCADrawResult
.neuron_idx_hub/nuisance_neuron_idx`.

Sampling is deterministic and reproducible across runs: each (session,
region, role) key gets its own `np.random.Generator`, seeded from
`SAMPLE_RNG_SEED` plus a CRC32 hash of that key (`_make_region_rng`) --
not from an in-process shared RNG whose draws would depend on iteration
order.

--------------------------------------------------------------------------------
Top-pCCA-weight neuron selection + saved residualized activity
--------------------------------------------------------------------------------
Once a pair's `N_SAMPLE_DRAWS` draws are all computed (for BOTH 2c and
2c' -- these are fit and selected independently, since their nuisance Z,
and therefore their residuals and weights, differ), `_select_top_weight_
neurons` pools every (draw, component, neuron) |pCCA weight| entry for
region_i (from `Wx`) and, separately, region_j (from `Wy`) across all 10
draws, and keeps whichever entries fall at or above the pooled
`TOP_WEIGHT_FRACTION` (default 0.2 -> top 20%) percentile cutoff. The same
physical neuron can be sampled -- and qualify -- in more than one draw
(the 10 draws are not required to be disjoint, see above); such a neuron
is collapsed to exactly ONE entry, keyed by its FULL-population neuron
index, not one entry per qualifying draw.

For each such deduplicated neuron, its residualized, cross-trial activity
is also saved: the relevant per-neuron column of `pcca`'s own `X_i_res`/
`X_j_res` (the SAME residual `z_i_lat`/`z_j_lat` are themselves projected
from -- nothing is recomputed), reshaped to (n_trials, T), averaged
elementwise across every draw in which that neuron qualified. Averaging
(rather than, say, keeping only its single highest-weight draw) reflects
that this selection is explicitly pooled/robust ACROSS draws -- a neuron
recurring across multiple different random nuisance-regression draws
should be represented by its typical residualized activity under that
treatment, not by whichever one draw happened to rank it highest.

Both the selection and the saved residuals live on `PrivateLatentPairResult
.selected_neurons_i` / `.selected_neurons_j` (a `SelectedNeuronSet` each)
-- since `.pairs` (2c) and `.pairs_regions_only` (2c') are each a dict of
this SAME dataclass, this one field pair automatically covers "both 2c and
2c'" without any separate plumbing. See `PrivateLatentAnalyzer.
get_selected_neurons` for the read-side accessor.

--------------------------------------------------------------------------------
Everything NOT covered above is unchanged from v1
--------------------------------------------------------------------------------
`_zscore_flat`, `_ridge_inv_sqrt`, `ridge_cca`, `_fit_ridge_beta`,
`_apply_ridge_beta`, `residualize`, `residualize_with_explained`,
`_trial_kfold_indices`, `pcca`, `latent_projections`, `pca_fit_and_project`,
anatomical canonicalisation, `REGION_PAIRS`/`PAIR_CATEGORIES`/`HUB_REGIONS`,
the laminar/subregion Part-3 weight-ratio metrics, and `load_region_spikes`
(Part 1's loader, byte-for-byte including the `sel = safe_array(info.get
("selected_neurons"))` line -- item 1) are all copied verbatim from v1.
Part 2's ridge-regression math (`residualize_with_explained`, `pcca`) is
IDENTICAL to v1's -- only which columns of which matrices it is handed has
changed. Note one algebraic simplification this version relies on: because
a ridge hat-matrix regression is column-independent (each neuron's Beta
column depends only on the shared nuisance Z, never on any OTHER neuron's
column), residualizing a region's FULL neuron set against behaviour ONCE
and then slicing out a draw's sampled columns (`behav_res_by_region_full`)
is numerically IDENTICAL to slicing first and residualizing each draw's
smaller subset separately -- so that regression is done once per region
per session, not once per region per draw.

Output goes to a SEPARATE directory from v1
(`pcca_all_regions_out_behaviour_v2_sampled_sessions_{trial_type}_
{align_mode}_results/`), so this script can be run alongside v1 without
overwriting its cache.

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import pickle
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import zscore

try:
    import mat73
except ImportError as exc:
    raise SystemExit("mat73 is required: pip install mat73") from exc

from Useful_definition import ANATOMICAL_ORDER, safe_array


# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS -- identical to v1, plus the new
#     Part-2 sampling knobs at the bottom of this section.
# =============================================================================

TRIAL_TYPE: str = "cued_hit_long"          # "cued_hit_long" | "spont_hit_long" | "spont_miss_long"

ALIGN_MODES: Dict[str, str] = {
    "default_move_onset": "align to movement onset using t_approach.start_time directly",
    "cue_onset":           "align to cue onset: start_time - t_approach.cue",
    "bar_off_onset":       "align to bar-off onset: start_time + t_approach.bar_off",
    "reward_onset":        "align to reward onset: start_time + t_approach.drop_time",
}

ALIGNMENT_WINDOWS_S: Dict[str, Tuple[float, float]] = {
    "default_move_onset": (-1.0, 2.0),
    "cue_onset":           (-0.8, 2.2),
    "bar_off_onset":       (-2.0, 1.0),
    "reward_onset":        (-1.2, 1.8),
}

ALIGN: str = "default_move_onset"
if ALIGN not in ALIGN_MODES:
    raise ValueError(
        f"ALIGN={ALIGN!r} is not a supported alignment mode; "
        f"choose one of {sorted(ALIGN_MODES)}."
    )

BASE_DIR: Path = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
BEHAVIOR_DIR: Path = BASE_DIR / "Paper_output" / f"tapproach_sessions_{ALIGN}"

# ---- CCA / pCCA dimensionality & regularisation (matches v1) ------------
N_COMPONENTS: int = 1
LAMBDA_CCA: float = 1e-4
LAMBDA_HAT: float = 1e-4

# ---- Part 2c/2c' cross-validation (matches v1 / oxford_session_pipeline_mdl.m)
CV_FOLDS: int = 10
CV_RNG_SEED: int = 12345

# ---- Nuisance-region eligibility -- a region needs at least this many
#      RECORDED (full-population) neurons to be allowed into a pair's
#      nuisance design at all, independent of how many of those neurons a
#      given draw happens to sample (matches v1's
#      `perform_session_pcca.m`-derived threshold).
MIN_NEURONS_PER_REGION: int = 50

# ---- PCA dimensionality for Parts 1a/1b/2a/2b (matches v1) ---------------
N_PCA_COMPONENTS: int = 5

# ---- Time windows (matches v1) -------------------------------------------
TIME_RANGE_S: Tuple[float, float] = ALIGNMENT_WINDOWS_S[ALIGN]
BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = ALIGNMENT_WINDOWS_S[ALIGN]

# ---- Regime toggles (matches v1) -----------------------------------------
SUBTRACT_PSTH: bool = False
SHUFFLE_TRIALS: bool = False
REQUIRE_BEHAVIOR: bool = True

EXCLUDED_REGIONS: List[str] = []
SESSIONS: Optional[List[str]] = None

# ---- NEW v2 knobs: Part 2's neuron-sampling scheme (item 2 of the
#      request) ------------------------------------------------------------
# Number of neurons drawn per region, per draw, for every paired
# computation (Parts 2a/2b/2c/2c') -- the "target sample size" in the
# request's own worked example.
TARGET_SAMPLE_SIZE: int = 50

# Number of independent neuron-sampling draws generated per region, per
# session, and therefore per pair (item 3: "store results for all 10
# draws per session").
N_SAMPLE_DRAWS: int = 10

# Base seed combined with a per-(session, region, role) hash
# (`_make_region_rng`) so sampling is deterministic and reproducible
# across runs, yet independent across regions/sessions/roles.
SAMPLE_RNG_SEED: int = 20260916

# ---- NEW: top-pCCA-weight neuron selection -------------------------------
# For each pair, each region side (i/j), each condition (2c/2c'), the
# fraction of pooled |pCCA weight| magnitudes (pooled across ALL
# N_SAMPLE_DRAWS draws) counted as "top" -- e.g. 0.2 = top 20%. See
# `_select_top_weight_neurons`.
TOP_WEIGHT_FRACTION: float = 0.2


def mat_subdir_name(trial_type: str, align_mode: str = ALIGN) -> str:
    """MATLAB-pipeline session-region-data source folder (unchanged from v1
    -- v2 reads the exact same neural .mat inputs, it only changes how
    Part 2's neurons are chosen from them)."""
    return f"{trial_type}_{align_mode}_results"


def out_subdir_name(trial_type: str, align_mode: str = ALIGN) -> str:
    """This script's own output folder -- deliberately distinct from v1's
    (`pcca_all_regions_out_behaviour_sessions_...`) so the two scripts'
    caches never collide."""
    return f"pcca_all_regions_out_behaviour_v2_sampled_sessions_{trial_type}_{align_mode}_results"


def behavior_label_for(trial_type: str) -> str:
    """'cued_hit_long' -> 'cued hit long' (matches v1)."""
    return trial_type.replace("_", " ")


# =============================================================================
# 2.  Anatomical canonicalisation -- copied verbatim from v1 /
#     cross_trial_type_cca_analysis.py.
# =============================================================================

def get_anatomical_index(region: str) -> int:
    """Get anatomical ordering index for a region."""
    try:
        return ANATOMICAL_ORDER.index(region)
    except ValueError:
        return len(ANATOMICAL_ORDER)


def sort_pair_by_anatomy(region_i: str, region_j: str) -> Tuple[str, str]:
    """Sort a region pair by anatomical order (region_i = earlier, region_j = later)."""
    idx_i = get_anatomical_index(region_i)
    idx_j = get_anatomical_index(region_j)
    if idx_i <= idx_j:
        return (region_i, region_j)
    else:
        return (region_j, region_i)


# =============================================================================
# 3.  Region-pair categories -- copied verbatim from v1.
# =============================================================================

PAIR_CATEGORIES: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("thalamic-thalamic", [
        ("VPMPO", "VALVM"),
    ]),
    ("cortico-cortical", [
        ("MOp", "MOs"),
        ("MOp", "ORB"),
        ("MOs", "ORB"),
    ]),
    ("cortico-motor thalamic", [
        ("MOp", "VALVM"),
        ("ORB", "VALVM"),
        ("MOs", "VALVM"),
    ]),
    ("cortico-sensory thalamic", [
        ("MOp", "VPMPO"),
        ("ORB", "VPMPO"),
        ("MOs", "VPMPO"),
    ]),
    ("to HY", [
        ("VALVM", "HY"),
        ("VPMPO", "HY"),
        ("MOp", "HY"),
        ("MOs", "HY"),
        ("ORB", "HY"),
    ]),
    ("to STR", [
        ("VALVM", "STR"),
        ("VPMPO", "STR"),
        ("MOp", "STR"),
        ("MOs", "STR"),
        ("ORB", "STR"),
    ]),
    ("other", [
        ("HY", "STR"),
    ]),
]

PAIR_CATEGORIES = [
    (category, [sort_pair_by_anatomy(*pair) for pair in pairs])
    for category, pairs in PAIR_CATEGORIES
]

REGION_PAIRS: List[Tuple[str, str]] = [
    (ri, rj) for _, pairs in PAIR_CATEGORIES for (ri, rj) in pairs
]

HUB_REGIONS: List[str] = sorted(
    {region for pair in REGION_PAIRS for region in pair},
    key=get_anatomical_index,
)


# =============================================================================
# 3b. Subregion / laminar-depth classification for Part 3's weight-ratio
#     metrics -- copied verbatim from v1.
# =============================================================================

EXCLUDED_SUBREGION_LABELS: List[str] = ["out"]

LAMINAR_DEPTH_MAP: Dict[str, str] = {
    "ACAd5":   "layer-deep",
    "ACAd6a":  "layer-deep",
    "PL6a":    "layer-deep",
    "ILA6a":   "layer-deep",
    "FRP6a":   "layer-deep",

    "ORBl23":  "layer-shallow",
    "ORBvl23": "layer-shallow",
    "ORBl5":   "layer-deep",
    "ORBvl5":  "layer-deep",
    "ORBl6a":  "layer-deep",
    "ORBm6a":  "layer-deep",
    "ORBvl6a": "layer-deep",
    "ORBl6b":  "layer-deep",
    "ORBvl6b": "layer-deep",

    "MOp1":    "layer-shallow",
    "MOp23":   "layer-shallow",
    "MOp5":    "layer-deep",
    "MOp6a":   "layer-deep",
    "MOp6b":   "layer-deep",

    "MOs1":    "layer-shallow",
    "MOs23":   "layer-shallow",
    "MOs5":    "layer-deep",
    "MOs6a":   "layer-deep",
    "MOs6b":   "layer-deep",
}


def get_laminar_depth(
        subregion_label: str,
        depth_map: Dict[str, str] = LAMINAR_DEPTH_MAP,
        strict: bool = False,
) -> Optional[str]:
    """Map a cortical subregion label (e.g. 'MOp5') to 'layer-shallow' /
    'layer-deep'. `strict=False` returns None for an unmapped label."""
    if subregion_label in depth_map:
        return depth_map[subregion_label]
    if strict:
        raise KeyError(
            f"'{subregion_label}' has no laminar depth assignment "
            f"(not a laminar cortical subregion, or an unresolved/"
            f"subcortical label)."
        )
    return None


CORTICAL_REGIONS: List[str] = ["MOp", "MOs", "ORB"]


# =============================================================================
# 4.  Core PCA / pCCA / CCA primitives -- copied verbatim from v1
#     (`pCCA_all_regions_out_behaviour.py`), which itself copied them from
#     `pCCA_sensitive_realsingle_Session_8panel.py`. Nothing in this
#     section changes between v1 and v2 -- only WHICH columns of WHICH
#     matrices these functions are called on differs (see Section 7).
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
    return flat.T   # (T * n_trials, n)


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


def _fit_ridge_beta(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_hat: float = LAMBDA_HAT,
) -> Optional[np.ndarray]:
    """Ridge hat-matrix regression coefficients of X_flat onto Z_flat:

        Beta = (Z^T Z + lam_hat * n * I)^(-1) Z^T X     (m, n_features)

    None if there is no nuisance to regress out."""
    if Z_flat is None or Z_flat.ndim < 2 or Z_flat.shape[1] == 0:
        return None
    n, m = Z_flat.shape
    ZtZ  = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    return np.linalg.solve(ZtZ, Z_flat.T @ X_flat)


def _apply_ridge_beta(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        beta: Optional[np.ndarray],
) -> np.ndarray:
    """Apply a Beta already fit by `_fit_ridge_beta`. `beta=None` returns
    X_flat unchanged."""
    if beta is None:
        return X_flat.copy()
    return X_flat - Z_flat @ beta


def residualize(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_hat: float = LAMBDA_HAT,
) -> np.ndarray:
    """Single ridge fit-and-apply on the same rows."""
    beta = _fit_ridge_beta(X_flat, Z_flat, lam_hat)
    return _apply_ridge_beta(X_flat, Z_flat, beta)


def residualize_with_explained(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_hat: float = LAMBDA_HAT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same joint ridge hat-matrix regression as `residualize`, returning
    BOTH the explained and residual halves (Part 2a / Part 2b)."""
    beta = _fit_ridge_beta(X_flat, Z_flat, lam_hat)
    if beta is None:
        return np.zeros_like(X_flat), X_flat.copy()
    explained = Z_flat @ beta
    residual  = X_flat - explained
    return explained, residual


def _trial_kfold_indices(
        n_trials: int,
        T: int,
        n_samples: int,
        n_folds: int = CV_FOLDS,
        seed: int = CV_RNG_SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition trials into `n_folds` folds, trial-boundary-aware (see v1's
    docstring for the full rationale -- unchanged here)."""
    n_folds = max(2, min(int(n_folds), n_trials))
    rng = np.random.default_rng(seed)
    trial_perm = rng.permutation(n_trials)
    trial_of_row = np.tile(np.arange(n_trials), T)
    if trial_of_row.size != n_samples:
        raise ValueError(
            f"_trial_kfold_indices: n_trials*T ({trial_of_row.size}) != "
            f"n_samples ({n_samples}) -- caller passed a flattened matrix "
            f"whose row count does not match n_trials/T."
        )

    fold_bounds = np.linspace(0, n_trials, n_folds + 1).astype(int)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_folds):
        test_trials = trial_perm[fold_bounds[f]:fold_bounds[f + 1]]
        if test_trials.size == 0:
            continue
        test_mask = np.isin(trial_of_row, test_trials)
        train_idx = np.flatnonzero(~test_mask)
        test_idx  = np.flatnonzero(test_mask)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def pcca(
        X_flat: np.ndarray,
        Y_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        n_trials: int,
        T: int,
        lam_cca: float = LAMBDA_CCA,
        lam_hat: float = LAMBDA_HAT,
        n_components: int = N_COMPONENTS,
        n_folds: int = CV_FOLDS,
        seed: int = CV_RNG_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K-fold cross-validated private pCCA between X_flat and Y_flat,
    conditioning on nuisance Z_flat. Copied verbatim from v1 -- see that
    file's docstring for the full within-fold/held-out rationale."""
    N = X_flat.shape[0]
    p, q = X_flat.shape[1], Y_flat.shape[1]
    k = max(1, min(n_components, p, q))

    def _full_data_residuals() -> Tuple[np.ndarray, np.ndarray]:
        beta_X = _fit_ridge_beta(X_flat, Z_flat, lam_hat)
        beta_Y = _fit_ridge_beta(Y_flat, Z_flat, lam_hat)
        X_res = _apply_ridge_beta(X_flat, Z_flat, beta_X)
        Y_res = _apply_ridge_beta(Y_flat, Z_flat, beta_Y)
        return X_res, Y_res

    def _full_data_fit() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_res, Y_res = _full_data_residuals()
        Wx, Wy, rho = ridge_cca(X_res, Y_res, lam_cca, k)
        return Wx, Wy, rho, X_res, Y_res

    folds = _trial_kfold_indices(n_trials, T, N, n_folds, seed)

    Wx_sum = np.zeros((p, k))
    Wy_sum = np.zeros((q, k))
    Wx_ref: Optional[np.ndarray] = None
    rho_folds: List[np.ndarray] = []
    n_fit = 0

    for train_idx, test_idx in folds:
        X_tr, Y_tr = X_flat[train_idx], Y_flat[train_idx]
        X_te, Y_te = X_flat[test_idx], Y_flat[test_idx]

        if Z_flat is not None:
            Z_tr, Z_te = Z_flat[train_idx], Z_flat[test_idx]
        else:
            Z_tr = Z_te = None

        beta_X = _fit_ridge_beta(X_tr, Z_tr, lam_hat)
        beta_Y = _fit_ridge_beta(Y_tr, Z_tr, lam_hat)
        X_tr_res = _apply_ridge_beta(X_tr, Z_tr, beta_X)
        Y_tr_res = _apply_ridge_beta(Y_tr, Z_tr, beta_Y)
        X_te_res = _apply_ridge_beta(X_te, Z_te, beta_X)
        Y_te_res = _apply_ridge_beta(Y_te, Z_te, beta_Y)

        Wx_f, Wy_f, _ = ridge_cca(X_tr_res, Y_tr_res, lam_cca, k)
        if Wx_f.shape[1] < k or Wy_f.shape[1] < k:
            continue

        if Wx_ref is None:
            Wx_ref = Wx_f.copy()
        else:
            for c in range(k):
                if np.dot(Wx_f[:, c], Wx_ref[:, c]) < 0.0:
                    Wx_f[:, c] *= -1.0
                    Wy_f[:, c] *= -1.0

        Wx_sum += Wx_f
        Wy_sum += Wy_f
        n_fit += 1

        u_te = X_te_res @ Wx_f
        v_te = Y_te_res @ Wy_f
        fold_rho = np.full(k, np.nan)
        for c in range(k):
            if u_te[:, c].std() > 0 and v_te[:, c].std() > 0:
                fold_rho[c] = np.corrcoef(u_te[:, c], v_te[:, c])[0, 1]
        rho_folds.append(fold_rho)

    if n_fit == 0:
        warnings.warn(
            "pcca: every CV fold degenerated (too few trials / "
            "rank-deficient residual) -- falling back to a single "
            "in-sample fit for this pair."
        )
        return _full_data_fit()

    Wx_mean = Wx_sum / n_fit
    Wy_mean = Wy_sum / n_fit
    rho = np.nanmean(np.stack(rho_folds, axis=0), axis=0)
    rho = np.clip(np.nan_to_num(rho, nan=0.0), 0.0, 1.0)

    X_res, Y_res = _full_data_residuals()
    return Wx_mean.astype(np.float64), Wy_mean.astype(np.float64), rho, X_res, Y_res


def latent_projections(X_flat: np.ndarray, W: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """Project flattened, residualized activity onto every column of a
    canonical weight matrix at once -> (n_trials, T, K)."""
    K = W.shape[1]
    proj = X_flat @ W
    return proj.reshape(T, n_trials, K).transpose(1, 0, 2)


def pca_fit_and_project(
        X_flat: np.ndarray,
        n_trials: int,
        T: int,
        n_components: int = N_PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit standard (SVD-based, column-centred) PCA on a flattened
    (T * n_trials, n_neurons) activity matrix and project it back through
    its own loadings. Copied verbatim from v1."""
    mean = X_flat.mean(axis=0, keepdims=True)
    Xc = X_flat - mean
    n = Xc.shape[0]
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    W = Vt[:k].T
    var = (S ** 2) / max(n - 1, 1)
    total_var = var.sum()
    explained_var_ratio = (
        var[:k] / total_var if total_var > 0 else np.zeros(k, dtype=np.float64)
    )
    latent = latent_projections(Xc, W, n_trials, T)
    return W, explained_var_ratio, mean, latent


# =============================================================================
# 4b. NEW -- Part 2's neuron-sampling primitives (item 2 of the request).
#     See the module docstring, "The sampling scheme", for the full
#     rationale; these two functions implement it.
# =============================================================================

def _make_region_rng(session_name: str, region: str, role: str) -> np.random.Generator:
    """Deterministic RNG for one (session, region, role) key -- `role` is
    "paired" (this region is region_i/region_j of some pair) or "nuisance"
    (this region enters some OTHER pair's nuisance design). The same
    physical region gets two INDEPENDENT draw sets, one per role, since a
    region like MOp is both a "paired" hub for its own six pairs and a
    "nuisance" contributor to pairs it isn't part of."""
    key = f"{session_name}|{region}|{role}".encode("utf-8")
    digest = zlib.crc32(key) & 0xFFFFFFFF
    return np.random.default_rng(np.random.SeedSequence([SAMPLE_RNG_SEED, digest]))


def sample_paired_draws(
        n_full: int,
        rng: np.random.Generator,
        target: int = TARGET_SAMPLE_SIZE,
        n_draws: int = N_SAMPLE_DRAWS,
) -> List[np.ndarray]:
    """`n_draws` neuron-index draws (each length `min(target, n_full)`,
    values in `[0, n_full)`) for a region playing the "paired" role
    (region_i or region_j of some pair): the first
    `m = min(n_draws, n_full // target)` draws are cut, non-overlapping,
    from ONE random permutation of the `n_full` population -- so together
    they sweep roughly `m * target` of it with the minimum possible
    overlap (zero) while each individual draw stays a uniformly random
    subset. The remaining `n_draws - m` draws are independent uniform
    random samples of `target` neurons from the FULL population, with no
    overlap constraint against each other or against the sweep draws.

    If `n_full < target` there is no way to draw `target` neurons without
    replacement even once; every draw then uses the full (capped)
    population instead, and a warning is printed once.
    """
    if n_full <= 0:
        return [np.array([], dtype=np.int64) for _ in range(n_draws)]

    if n_full < target:
        warnings.warn(
            f"sample_paired_draws: n_full={n_full} < target={target}; "
            f"every one of the {n_draws} draws will use all {n_full} "
            f"available neurons instead."
        )
        return [np.arange(n_full, dtype=np.int64) for _ in range(n_draws)]

    perm = rng.permutation(n_full)
    m = min(n_draws, n_full // target)

    draws: List[np.ndarray] = []
    for b in range(m):
        draws.append(np.sort(perm[b * target:(b + 1) * target]).astype(np.int64))
    for _ in range(n_draws - m):
        draws.append(np.sort(rng.choice(n_full, size=target, replace=False)).astype(np.int64))
    return draws


def sample_nuisance_draws(
        n_full: int,
        rng: np.random.Generator,
        target: int = TARGET_SAMPLE_SIZE,
        n_draws: int = N_SAMPLE_DRAWS,
) -> List[np.ndarray]:
    """`n_draws` independent random neuron-index draws (each length
    `min(target, n_full)`) for a region playing the "nuisance" role -- no
    sweep/coverage requirement, just `n_draws` separate random samples
    (item 2: "simply generate 10 distinct random draws"). A short
    bounded retry avoids an exact-duplicate draw where the population is
    large enough that duplicates are avoidable at all; it is not a
    guarantee when `n_full` is only slightly larger than `target`."""
    if n_full <= 0:
        return [np.array([], dtype=np.int64) for _ in range(n_draws)]

    if n_full < target:
        warnings.warn(
            f"sample_nuisance_draws: n_full={n_full} < target={target}; "
            f"every one of the {n_draws} draws will use all {n_full} "
            f"available neurons instead."
        )
        return [np.arange(n_full, dtype=np.int64) for _ in range(n_draws)]

    seen: List[frozenset] = []
    draws: List[np.ndarray] = []
    for _ in range(n_draws):
        idx = rng.choice(n_full, size=target, replace=False)
        for _attempt in range(4):
            key = frozenset(idx.tolist())
            if key not in seen:
                break
            idx = rng.choice(n_full, size=target, replace=False)
        seen.append(frozenset(idx.tolist()))
        draws.append(np.sort(idx).astype(np.int64))
    return draws


# =============================================================================
# 5.  Data loading. `load_region_spikes` (Part 1's loader) is copied
#     VERBATIM from v1 -- item 1's requirement -- including the
#     `sel = safe_array(info.get("selected_neurons"))` line.
#     `load_region_spikes_full` is NEW: structurally identical, minus the
#     `sel` filter, so Part 2 can sample from every recorded neuron.
#     `crop_time_window` / `load_behavior_regressors` are copied verbatim.
# =============================================================================

def load_region_spikes(
        session_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], int, int]:
    """Load per-region spike tensors AND per-neuron subregion labels for one
    session, subset to `selected_neurons` -- Part 1's (1a/1b) input pool,
    unchanged from v1. See v1's own docstring for the full rationale
    behind not dropping `EXCLUDED_SUBREGION_LABELS` neurons here."""
    data = mat73.loadmat(session_path)
    rd   = data.get("region_data", {})
    regs = rd.get("regions", {})

    region_spikes: Dict[str, np.ndarray] = {}
    region_subregion_labels: Dict[str, List[str]] = {}
    n_trials_out = T_out = None

    for rname, info in regs.items():
        if not isinstance(info, dict):
            continue
        sd = safe_array(info.get("spike_data"))
        if sd is None or sd.ndim != 3:
            continue
        n_full = sd.shape[1]

        labels_full = info.get("subregion_labels")
        if labels_full is None:
            labels_full = ["unknown"] * n_full
        elif not isinstance(labels_full, list):
            labels_full = [str(v) for v in np.asarray(labels_full).ravel()]
        else:
            labels_full = [str(v) for v in labels_full]
        if len(labels_full) != n_full:
            warnings.warn(
                f"[load_region_spikes] {rname}: subregion_labels length "
                f"({len(labels_full)}) != n_neurons ({n_full}); ignoring labels."
            )
            labels_full = ["unknown"] * n_full

        sel = safe_array(info.get("selected_neurons"))
        if sel is not None and sel.size > 0:
            idx0 = sel.ravel().astype(int) - 1
            sd = sd[:, idx0, :]
            labels = [labels_full[i] for i in idx0]
        else:
            labels = labels_full

        region_spikes[rname] = sd.astype(np.float32)
        region_subregion_labels[rname] = labels
        if n_trials_out is None:
            n_trials_out, _, T_out = sd.shape

    print(
        f"    [load_region_spikes]  {len(region_spikes)} regions loaded  "
        f"| n_trials={n_trials_out}  T={T_out}"
    )
    return region_spikes, region_subregion_labels, int(n_trials_out), int(T_out)


def load_region_spikes_full(
        session_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], int, int]:
    """Load per-region spike tensors AND per-neuron subregion labels for
    EVERY recorded neuron of one session -- Part 2's sampling pool,
    deliberately bypassing `selected_neurons` entirely (item 2: "...
    replacing the previously stored selected_neurons parameter
    entirely"). Structurally identical to `load_region_spikes` (same
    `region_data.regions` tree of the same file) minus that one filtering
    step; kept as its own function, rather than a flag on
    `load_region_spikes`, so item 1's "keep the PCA1a/1b section exactly
    as it is" has one single, untouched call site elsewhere in this file."""
    data = mat73.loadmat(session_path)
    rd   = data.get("region_data", {})
    regs = rd.get("regions", {})

    region_spikes: Dict[str, np.ndarray] = {}
    region_subregion_labels: Dict[str, List[str]] = {}
    n_trials_out = T_out = None

    for rname, info in regs.items():
        if not isinstance(info, dict):
            continue
        sd = safe_array(info.get("spike_data"))
        if sd is None or sd.ndim != 3:
            continue
        n_full = sd.shape[1]

        labels_full = info.get("subregion_labels")
        if labels_full is None:
            labels_full = ["unknown"] * n_full
        elif not isinstance(labels_full, list):
            labels_full = [str(v) for v in np.asarray(labels_full).ravel()]
        else:
            labels_full = [str(v) for v in labels_full]
        if len(labels_full) != n_full:
            warnings.warn(
                f"[load_region_spikes_full] {rname}: subregion_labels length "
                f"({len(labels_full)}) != n_neurons ({n_full}); ignoring labels."
            )
            labels_full = ["unknown"] * n_full

        region_spikes[rname] = sd.astype(np.float32)
        region_subregion_labels[rname] = labels_full
        if n_trials_out is None:
            n_trials_out, _, T_out = sd.shape

    print(
        f"    [load_region_spikes_full]  {len(region_spikes)} regions loaded  "
        f"| n_trials={n_trials_out}  T={T_out}  (full population, no "
        f"selected_neurons filter)"
    )
    return region_spikes, region_subregion_labels, int(n_trials_out), int(T_out)


def crop_time_window(
        region_spikes: Dict[str, np.ndarray],
        time_vec_full: np.ndarray,
        window: Tuple[float, float],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Crop the trailing time axis of every region's (n_trials, n, T) tensor
    to the closed interval `window` (copied verbatim from v1)."""
    lo, hi = window
    mask = (time_vec_full >= lo - 1e-6) & (time_vec_full <= hi + 1e-6)
    if mask.sum() < 2:
        raise ValueError(
            f"Requested window {window} has < 2 overlapping samples with "
            f"time_vec_full range [{time_vec_full[0]:.3f}, "
            f"{time_vec_full[-1]:.3f}]."
        )
    cropped = {r: X[:, :, mask] for r, X in region_spikes.items()}
    return cropped, time_vec_full[mask]


def load_behavior_regressors(
        session_name: str,
        behavior_dir: Path = BEHAVIOR_DIR,
        trial_label: str = "cued hit long",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-trial position (x, y, z) and speed traces for one session,
    filtered to trials matching `trial_label` (copied verbatim from v1)."""
    pkl_path = behavior_dir / f"{session_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Behaviour file not found: {pkl_path}")

    with open(pkl_path, "rb") as fh:
        session_data = pickle.load(fh)

    pos    = np.asarray(session_data["pos"])
    speed  = np.asarray(session_data["speed"])
    labels = np.asarray(session_data["task_label"], dtype=object)
    t_behav = np.asarray(session_data["time"], dtype=np.float64)

    if speed.ndim == 2:
        speed = speed[:, None, :]

    sel = (labels == trial_label)
    if not np.any(sel):
        available = sorted(set(labels.tolist()))
        raise ValueError(
            f"No behaviour trials with label '{trial_label}' found in "
            f"{pkl_path.name} (available labels: {available})."
        )

    pos_sel   = pos[sel].astype(np.float32)
    speed_sel = speed[sel].astype(np.float32)
    return pos_sel, speed_sel, t_behav


# =============================================================================
# 6.  Result containers. `SubregionWeightMetrics` and `RegionPCAResult`
#     (Part 1) are copied verbatim from v1. Part 2's containers are NEW:
#     each pair/hub-orientation result is now `N_SAMPLE_DRAWS` independent
#     `*DrawResult` entries instead of one fixed-neuron-set result, each
#     carrying its own sampled neuron indices (item 3).
# =============================================================================

@dataclass
class SubregionWeightMetrics:
    """Part 3's weight-ratio measure for ONE PCA/CCA component of ONE
    region's weight vector (copied verbatim from v1 -- see that file for
    the full derivation of every field below)."""
    region: str
    is_cortical: bool
    n_neurons_total: int
    n_neurons_resolved: int
    weight_mass_fraction: Dict[str, float] = field(default_factory=dict)
    neuron_count_fraction: Dict[str, float] = field(default_factory=dict)
    enrichment_ratio: Dict[str, float] = field(default_factory=dict)
    dominant_group: Optional[str] = None
    dominant_ratio: Optional[float] = None
    dominant_ratio_label: Optional[str] = None


@dataclass
class RegionPCAResult:
    """Part 1a (`region=raw`) / Part 1b (`out_behaviour=True`) region-level
    PCA, one hub region, one session -- unchanged from v1, still fit on
    `selected_neurons`."""
    region: str
    W: np.ndarray
    explained_variance_ratio: np.ndarray
    latent: np.ndarray
    mean: np.ndarray
    n_neurons: int
    subregion_weight_metrics: List[SubregionWeightMetrics] = field(default_factory=list)


@dataclass
class PrivateLatentPairDrawResult:
    """ONE of `N_SAMPLE_DRAWS` independent neuron-sampling draws behind a
    pair's Part 2c ('.pairs') / 2c' ('.pairs_regions_only') private pCCA
    result. `neuron_idx_i` / `neuron_idx_j` / `nuisance_neuron_idx` index
    into each region's FULL (unfiltered) neuron axis -- the same axis
    `PrivateLatentSessionResult.region_subregion_labels_full` is aligned
    to (item 3: "store the sampled neuron indices for each draw")."""
    draw_idx: int
    rho: np.ndarray                      # (K,)
    Wx: np.ndarray                       # (n_sampled_i, K)
    Wy: np.ndarray                       # (n_sampled_j, K)
    z_i_lat: np.ndarray                  # (n_trials, T, K)
    z_j_lat: np.ndarray                  # (n_trials, T, K)
    n_neurons_i: int
    n_neurons_j: int
    neuron_idx_i: np.ndarray
    neuron_idx_j: np.ndarray
    nuisance_regions: List[str] = field(default_factory=list)
    nuisance_neuron_idx: Dict[str, np.ndarray] = field(default_factory=dict)
    z_dim_total: int = 0
    subregion_weight_metrics_i: List[SubregionWeightMetrics] = field(default_factory=list)
    subregion_weight_metrics_j: List[SubregionWeightMetrics] = field(default_factory=list)


@dataclass
class SelectedNeuronResidual:
    """ONE neuron (identified by its index into the region's FULL,
    unfiltered neuron axis) that fell in the top `TOP_WEIGHT_FRACTION` of
    pooled |pCCA weight| magnitudes across this pair's `N_SAMPLE_DRAWS`
    draws, for one region side (i or j) of one condition (2c or 2c').
    `residual` is that neuron's residualized, cross-trial (n_trials, T)
    activity -- the SAME per-neuron column of `pcca`'s own `X_i_res`/
    `X_j_res` used to build `z_i_lat`/`z_j_lat`, reshaped to (n_trials, T)
    -- averaged across every draw in which the neuron BOTH was sampled AND
    qualified for the top-weight cut, so a neuron sampled (and
    qualifying) in more than one draw collapses to exactly one entry here
    (the request's "keep it only once")."""
    neuron_idx: int
    residual: np.ndarray                 # (n_trials, T) -- averaged across qualifying draws
    n_draws_qualified: int
    qualifying_draw_idx: List[int] = field(default_factory=list)
    qualifying_abs_weight: List[float] = field(default_factory=list)   # per qualifying draw, max |weight| across components


@dataclass
class SelectedNeuronSet:
    """Top-pCCA-weight neuron selection + saved residualized activity for
    ONE region side of ONE pair, ONE condition (2c/2c'), one session --
    `region`'s deduplicated top-`top_fraction` neurons, pooled across all
    `N_SAMPLE_DRAWS` draws' pCCA weights. See `_select_top_weight_neurons`
    for exactly how `weight_threshold` and `neurons` are derived."""
    region: str
    top_fraction: float
    weight_threshold: float              # pooled-magnitude cutoff actually applied (NaN if no draws)
    n_total_weight_occurrences: int      # how many (draw, component, neuron) |weight| entries were pooled
    neurons: List[SelectedNeuronResidual] = field(default_factory=list)


@dataclass
class PrivateLatentPairResult:
    """Part 2c ('.pairs') / 2c' ('.pairs_regions_only') for one
    canonicalized pair, one session: `N_SAMPLE_DRAWS` independent draws
    (`draws[k].draw_idx == k`), each an otherwise-complete v1-style
    private-pCCA result on that draw's sampled neurons, PLUS
    `selected_neurons_i`/`selected_neurons_j` -- the top-pCCA-weight
    neuron selection (and saved residualized data) pooled across those
    draws for region_i / region_j respectively. Since `.pairs` and
    `.pairs_regions_only` are each a dict of this SAME dataclass, one
    instance's `selected_neurons_i/j` is always specific to whichever
    condition (2c or 2c') that instance represents -- there is no
    cross-condition pooling."""
    region_i: str
    region_j: str
    draws: List[PrivateLatentPairDrawResult] = field(default_factory=list)
    selected_neurons_i: Optional[SelectedNeuronSet] = None
    selected_neurons_j: Optional[SelectedNeuronSet] = None


@dataclass
class HubOrientationPCADrawResult:
    """ONE of `N_SAMPLE_DRAWS` draws behind Parts 2a ('_network') / 2b
    ('_residual') for ONE hub orientation of one pair."""
    draw_idx: int
    hub: str
    partner: str
    W_network: np.ndarray
    explained_variance_ratio_network: np.ndarray
    latent_network: np.ndarray
    W_residual: np.ndarray
    explained_variance_ratio_residual: np.ndarray
    latent_residual: np.ndarray
    n_neurons_hub: int
    neuron_idx_hub: np.ndarray
    nuisance_regions: List[str] = field(default_factory=list)
    nuisance_neuron_idx: Dict[str, np.ndarray] = field(default_factory=dict)
    z_dim_total: int = 0
    subregion_weight_metrics_network: List[SubregionWeightMetrics] = field(default_factory=list)
    subregion_weight_metrics_residual: List[SubregionWeightMetrics] = field(default_factory=list)


@dataclass
class HubPairPCAResult:
    """Parts 2a + 2b for one canonicalized pair, one session -- both hub
    orientations, each a list of `N_SAMPLE_DRAWS` draws, keyed IDENTICALLY
    to `.pairs` (same (region_i, region_j) key space)."""
    region_i: str
    region_j: str
    region_i_as_hub_draws: List[HubOrientationPCADrawResult] = field(default_factory=list)
    region_j_as_hub_draws: List[HubOrientationPCADrawResult] = field(default_factory=list)


@dataclass
class PrivateLatentSessionResult:
    """One session's v2 results, spanning Parts 1a/1b/2a/2b/2c/2c' -- the
    unit pickled to {session}_analysis_results.pkl. `region_pca_raw` /
    `region_pca_out_behaviour` (Part 1) are unchanged in shape from v1.
    `pairs` / `pairs_regions_only` / `hub_pca_pairs` (Part 2) now hold
    `N_SAMPLE_DRAWS`-draw result objects instead of a single fixed result.
    `region_subregion_labels_full` is Part 2's neuron-index reference
    frame -- the labels list `neuron_idx_i`/`neuron_idx_hub`/etc. above
    index into, aligned to `region_flat_full`'s (unfiltered) neuron axis;
    kept distinct from `region_subregion_labels` (Part 1's, `
    selected_neurons`-filtered axis) so a caller is never tempted to index
    one pool's labels with the other pool's indices."""
    session: str
    trial_type: str
    time_vec: np.ndarray
    n_trials: int
    T: int
    behavior_available: bool
    behavior_channel_labels: List[str] = field(default_factory=list)
    excluded_regions: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    pairs: Dict[Tuple[str, str], PrivateLatentPairResult] = field(default_factory=dict)
    pairs_regions_only: Dict[Tuple[str, str], PrivateLatentPairResult] = field(default_factory=dict)
    region_pca_raw: Dict[str, RegionPCAResult] = field(default_factory=dict)
    region_pca_out_behaviour: Dict[str, RegionPCAResult] = field(default_factory=dict)
    hub_pca_pairs: Dict[Tuple[str, str], HubPairPCAResult] = field(default_factory=dict)
    align_mode: str = ALIGN
    region_subregion_labels: Dict[str, List[str]] = field(default_factory=dict)       # Part 1 pool (selected_neurons)
    region_subregion_labels_full: Dict[str, List[str]] = field(default_factory=dict)  # Part 2 pool (full population)


def config_fingerprint() -> Dict[str, Any]:
    """Snapshot of every parameter that changes the numerical result,
    stored inside each session's pickle so a stale cache is detected, not
    silently reused (matches v1's own caching contract)."""
    return dict(
        align_mode=ALIGN,
        n_components=N_COMPONENTS,
        n_pca_components=N_PCA_COMPONENTS,
        lambda_cca=LAMBDA_CCA,
        lambda_hat=LAMBDA_HAT,
        excluded_regions=tuple(EXCLUDED_REGIONS),
        anatomical_order=tuple(ANATOMICAL_ORDER),
        region_pairs=tuple(REGION_PAIRS),
        time_range_s=tuple(TIME_RANGE_S),
        behavior_time_range_s=tuple(BEHAVIOR_TIME_RANGE_S),
        subtract_psth=SUBTRACT_PSTH,
        shuffle_trials=SHUFFLE_TRIALS,
        require_behavior=REQUIRE_BEHAVIOR,
        min_neurons_per_region=MIN_NEURONS_PER_REGION,
        cv_folds=CV_FOLDS,
        cv_rng_seed=CV_RNG_SEED,
        # v2-specific: Part 2's neuron-sampling scheme.
        target_sample_size=TARGET_SAMPLE_SIZE,
        n_sample_draws=N_SAMPLE_DRAWS,
        sample_rng_seed=SAMPLE_RNG_SEED,
        # v2-specific: top-pCCA-weight neuron selection + saved residuals.
        top_weight_fraction=TOP_WEIGHT_FRACTION,
        result_schema_version=2,   # v2 pipeline's own schema counter --
        # bumped: added selected_neurons_i/j (top-weight neuron selection
        # + averaged residualized activity) to PrivateLatentPairResult.
    )


# =============================================================================
# 7.  Per-region / per-pair / per-draw / per-session computation.
#     Parts 1a/1b use `_compute_region_pca` (unchanged from v1). Parts
#     2a/2b/2c/2c' are now computed per-draw by `_compute_hub_orientation_
#     draw` / `_compute_pair_result_draw`, looped over `N_SAMPLE_DRAWS`
#     draws per pair by `_compute_pair_sampled`.
# =============================================================================

def _other_region_nuisance_list_full(
        region_i: str,
        region_j: str,
        region_flat_full: Dict[str, np.ndarray],
        min_neurons_per_region: int = MIN_NEURONS_PER_REGION,
) -> List[str]:
    """Every anatomically-ordered region recorded this session, other than
    {region_i, region_j}, not in EXCLUDED_REGIONS, and with AT LEAST
    `min_neurons_per_region` neurons IN ITS FULL (unfiltered) POPULATION
    -- eligibility is about how much this region was actually recorded,
    not about how many of its neurons a given draw later samples."""
    return [
        r for r in ANATOMICAL_ORDER
        if r in region_flat_full
        and r not in (region_i, region_j)
        and r not in EXCLUDED_REGIONS
        and region_flat_full[r].shape[1] >= min_neurons_per_region
    ]


def _subregion_group_for(region: str, label: str) -> Optional[str]:
    """Collapse one neuron's raw `subregion_labels` string to Part 3's
    group label (copied verbatim from v1)."""
    if not label or label in EXCLUDED_SUBREGION_LABELS or label == "unknown":
        return None
    if region in CORTICAL_REGIONS:
        return get_laminar_depth(label, strict=False)
    return label


def compute_subregion_weight_metrics(
        region: str,
        W: np.ndarray,
        labels: Optional[List[str]],
) -> List[SubregionWeightMetrics]:
    """Part 3's weight-ratio measure, one `SubregionWeightMetrics` per
    column of `W` (copied verbatim from v1; `labels` here is whichever
    subset of subregion labels matches `W`'s neuron axis for THIS call --
    Part 1 passes the `selected_neurons`-filtered labels, Part 2 passes
    the sampled-draw's subset of the full-population labels)."""
    n_neurons, K = W.shape
    is_cortical = region in CORTICAL_REGIONS

    if not labels or len(labels) != n_neurons:
        return [
            SubregionWeightMetrics(
                region=region, is_cortical=is_cortical,
                n_neurons_total=n_neurons, n_neurons_resolved=0,
            )
            for _ in range(K)
        ]

    groups = [_subregion_group_for(region, lab) for lab in labels]
    resolved_mask = np.array([g is not None for g in groups])
    n_resolved = int(resolved_mask.sum())
    groups_res = [g for g, keep in zip(groups, resolved_mask) if keep]
    unique_groups = sorted(set(groups_res))

    results: List[SubregionWeightMetrics] = []
    for k in range(K):
        metrics = SubregionWeightMetrics(
            region=region, is_cortical=is_cortical,
            n_neurons_total=n_neurons, n_neurons_resolved=n_resolved,
        )
        if n_resolved > 0:
            w_res = np.abs(W[resolved_mask, k]).astype(np.float64)
            total_mass = float(w_res.sum())

            mass_frac: Dict[str, float] = {}
            count_frac: Dict[str, float] = {}
            for g in unique_groups:
                g_mask = np.array([gg == g for gg in groups_res])
                mass_frac[g] = float(w_res[g_mask].sum() / total_mass) if total_mass > 0 else 0.0
                count_frac[g] = float(g_mask.sum()) / n_resolved
            metrics.weight_mass_fraction = mass_frac
            metrics.neuron_count_fraction = count_frac
            metrics.enrichment_ratio = {
                g: (mass_frac[g] / count_frac[g]) if count_frac[g] > 0 else float("nan")
                for g in unique_groups
            }

            if is_cortical:
                shallow = mass_frac.get("layer-shallow", 0.0)
                deep = mass_frac.get("layer-deep", 0.0)
                metrics.dominant_group = "layer-shallow" if shallow >= deep else "layer-deep"
                if deep > 0:
                    metrics.dominant_ratio = shallow / deep
                elif shallow > 0:
                    metrics.dominant_ratio = float("inf")
                else:
                    metrics.dominant_ratio = float("nan")
                metrics.dominant_ratio_label = "layer-shallow:layer-deep"
            elif unique_groups:
                top_group = max(unique_groups, key=lambda g: mass_frac[g])
                rest_frac = 1.0 - mass_frac[top_group]
                metrics.dominant_group = top_group
                metrics.dominant_ratio = (
                    mass_frac[top_group] / rest_frac if rest_frac > 0 else float("inf")
                )
                metrics.dominant_ratio_label = f"{top_group}:rest"

        results.append(metrics)
    return results


def _compute_region_pca(
        region: str,
        X_flat: np.ndarray,
        n_trials: int,
        T: int,
        n_components: int = N_PCA_COMPONENTS,
        labels: Optional[List[str]] = None,
) -> RegionPCAResult:
    """Part 1a/1b building block -- unchanged from v1."""
    W, explained_variance_ratio, mean, latent = pca_fit_and_project(
        X_flat, n_trials, T, n_components)
    return RegionPCAResult(
        region=region,
        W=W.astype(np.float32),
        explained_variance_ratio=explained_variance_ratio.astype(np.float64),
        latent=latent.astype(np.float32),
        mean=mean.astype(np.float32),
        n_neurons=int(X_flat.shape[1]),
        subregion_weight_metrics=compute_subregion_weight_metrics(region, W, labels),
    )


def _compute_pair_result_draw(
        draw_idx: int,
        region_i: str,
        region_j: str,
        X_i_sub: np.ndarray,
        X_j_sub: np.ndarray,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
        Z_flat: Optional[np.ndarray],
        nuisance_all: List[str],
        nuisance_idx_k: Dict[str, np.ndarray],
        n_trials: int,
        T: int,
        labels_i: Optional[List[str]] = None,
        labels_j: Optional[List[str]] = None,
) -> Tuple[PrivateLatentPairDrawResult, np.ndarray, np.ndarray]:
    """Part 2c (`Z_flat` = nuisance + behaviour) / 2c' (`Z_flat` = nuisance
    only) for ONE draw of one pair -- same `pcca` call v1's
    `_compute_pair_result` made, just on this draw's sampled columns.
    Returns `(draw_result, X_i_res, X_j_res)`: the latter two are the SAME
    residualized, cross-trial matrices `z_i_lat`/`z_j_lat` were projected
    from, handed back so the caller can pool them across draws for
    `_select_top_weight_neurons` without refitting `pcca`."""
    Wx, Wy, rho, X_i_res, X_j_res = pcca(
        X_i_sub, X_j_sub, Z_flat,
        n_trials=n_trials, T=T,
        lam_cca=LAMBDA_CCA, lam_hat=LAMBDA_HAT, n_components=N_COMPONENTS,
        n_folds=CV_FOLDS, seed=CV_RNG_SEED,
    )
    z_i_lat = latent_projections(X_i_res, Wx, n_trials, T)
    z_j_lat = latent_projections(X_j_res, Wy, n_trials, T)

    draw_result = PrivateLatentPairDrawResult(
        draw_idx=draw_idx,
        rho=np.asarray(rho, dtype=np.float64),
        Wx=Wx.astype(np.float32),
        Wy=Wy.astype(np.float32),
        z_i_lat=z_i_lat.astype(np.float32),
        z_j_lat=z_j_lat.astype(np.float32),
        n_neurons_i=int(X_i_sub.shape[1]),
        n_neurons_j=int(X_j_sub.shape[1]),
        neuron_idx_i=idx_i.astype(np.int64),
        neuron_idx_j=idx_j.astype(np.int64),
        nuisance_regions=list(nuisance_all),
        nuisance_neuron_idx={r: nuisance_idx_k[r].astype(np.int64) for r in nuisance_all},
        z_dim_total=int(Z_flat.shape[1]) if Z_flat is not None else 0,
        subregion_weight_metrics_i=compute_subregion_weight_metrics(region_i, Wx, labels_i),
        subregion_weight_metrics_j=compute_subregion_weight_metrics(region_j, Wy, labels_j),
    )
    return draw_result, X_i_res, X_j_res


def _select_top_weight_neurons(
        region: str,
        draw_Wx: List[np.ndarray],           # per draw: (n_sampled, K) -- Wx (region_i) or Wy (region_j)
        draw_neuron_idx: List[np.ndarray],   # per draw: (n_sampled,) -- neuron_idx_i or neuron_idx_j (FULL axis)
        draw_X_res: List[np.ndarray],        # per draw: (T*n_trials, n_sampled) -- X_i_res or X_j_res
        n_trials: int,
        T: int,
        top_fraction: float = TOP_WEIGHT_FRACTION,
) -> SelectedNeuronSet:
    """Pool every (draw, component, neuron) |weight| entry for one region
    side of one pair/condition across all `N_SAMPLE_DRAWS` draws, take the
    top `top_fraction` by pooled magnitude (a single percentile cutoff
    over the pooled distribution -- "the top 20% of pCCA weights across
    the 10 draws"), and collapse the qualifying entries to ONE result per
    UNIQUE neuron (deduplicated across draws, per "if a neuron appears
    more than once across the 10 draws, keep it only once"). Each
    neuron's saved residualized, cross-trial activity is the elementwise
    average, over every draw in which it qualified, of that draw's
    `X_res` column for this neuron -- the same residualized matrix
    `_compute_pair_result_draw` already computed via `pcca`, not a fresh
    refit.
    """
    all_abs: List[float] = []
    entries: List[Tuple[int, int, float]] = []   # (draw_idx, neuron_idx, abs_weight)
    for d, (Wx, nidx) in enumerate(zip(draw_Wx, draw_neuron_idx)):
        if Wx.size == 0 or nidx.size == 0:
            continue
        K = Wx.shape[1]
        for k in range(K):
            abs_w = np.abs(Wx[:, k]).astype(np.float64)
            for pos in range(nidx.shape[0]):
                w = float(abs_w[pos])
                all_abs.append(w)
                entries.append((d, int(nidx[pos]), w))

    if not all_abs:
        return SelectedNeuronSet(
            region=region, top_fraction=top_fraction,
            weight_threshold=float("nan"), n_total_weight_occurrences=0, neurons=[],
        )

    threshold = float(np.percentile(np.asarray(all_abs, dtype=np.float64),
                                     100.0 * (1.0 - top_fraction)))

    qualifying: Dict[int, List[Tuple[int, float]]] = {}
    for d, nu, w in entries:
        if w >= threshold:
            qualifying.setdefault(nu, []).append((d, w))

    neurons: List[SelectedNeuronResidual] = []
    for nu, occurrences in qualifying.items():
        draw_ids = sorted({d for d, _ in occurrences})
        max_w_by_draw: Dict[int, float] = {}
        for d, w in occurrences:
            max_w_by_draw[d] = max(max_w_by_draw.get(d, 0.0), w)

        traces: List[np.ndarray] = []
        for d in draw_ids:
            pos_arr = np.flatnonzero(draw_neuron_idx[d] == nu)
            if pos_arr.size == 0:
                continue   # defensive -- should not happen, nu came from this draw's own entries
            col = draw_X_res[d][:, int(pos_arr[0])]
            traces.append(col.reshape(T, n_trials).T)   # (n_trials, T)
        if not traces:
            continue

        neurons.append(SelectedNeuronResidual(
            neuron_idx=nu,
            residual=np.mean(np.stack(traces, axis=0), axis=0).astype(np.float32),
            n_draws_qualified=len(draw_ids),
            qualifying_draw_idx=draw_ids,
            qualifying_abs_weight=[max_w_by_draw[d] for d in draw_ids],
        ))

    neurons.sort(key=lambda r: r.neuron_idx)
    return SelectedNeuronSet(
        region=region, top_fraction=top_fraction, weight_threshold=threshold,
        n_total_weight_occurrences=len(entries), neurons=neurons,
    )


def _compute_hub_orientation_draw(
        draw_idx: int,
        hub: str,
        partner: str,
        X_hub_behav_res_sub: np.ndarray,
        idx_hub: np.ndarray,
        Z_nuisance: Optional[np.ndarray],
        nuisance_all: List[str],
        nuisance_idx_k: Dict[str, np.ndarray],
        n_trials: int,
        T: int,
        labels_hub: Optional[List[str]] = None,
) -> HubOrientationPCADrawResult:
    """Parts 2a + 2b for ONE draw of one hub orientation -- same
    `residualize_with_explained` call v1's `_compute_hub_orientation_pca`
    made, just on this draw's sampled columns. `X_hub_behav_res_sub` is
    already this draw's hub columns of the FULL-population, behaviour-
    residualized pool (`behav_res_by_region_full`, sliced once per region
    per session -- see the module docstring's "column-independent ridge"
    note for why slicing after residualizing is exact)."""
    explained, residual = residualize_with_explained(X_hub_behav_res_sub, Z_nuisance, LAMBDA_HAT)

    W_net, evr_net, _mean_net, lat_net = pca_fit_and_project(
        explained, n_trials, T, N_PCA_COMPONENTS)
    W_res, evr_res, _mean_res, lat_res = pca_fit_and_project(
        residual, n_trials, T, N_PCA_COMPONENTS)

    return HubOrientationPCADrawResult(
        draw_idx=draw_idx,
        hub=hub,
        partner=partner,
        W_network=W_net.astype(np.float32),
        explained_variance_ratio_network=evr_net.astype(np.float64),
        latent_network=lat_net.astype(np.float32),
        W_residual=W_res.astype(np.float32),
        explained_variance_ratio_residual=evr_res.astype(np.float64),
        latent_residual=lat_res.astype(np.float32),
        n_neurons_hub=int(X_hub_behav_res_sub.shape[1]),
        neuron_idx_hub=idx_hub.astype(np.int64),
        nuisance_regions=list(nuisance_all),
        nuisance_neuron_idx={r: nuisance_idx_k[r].astype(np.int64) for r in nuisance_all},
        z_dim_total=int(Z_nuisance.shape[1]) if Z_nuisance is not None else 0,
        subregion_weight_metrics_network=compute_subregion_weight_metrics(hub, W_net, labels_hub),
        subregion_weight_metrics_residual=compute_subregion_weight_metrics(hub, W_res, labels_hub),
    )


def _compute_pair_sampled(
        region_i: str,
        region_j: str,
        region_flat_full: Dict[str, np.ndarray],
        behav_res_by_region_full: Dict[str, np.ndarray],
        region_subregion_labels_full: Dict[str, List[str]],
        paired_draws: Dict[str, List[np.ndarray]],
        nuisance_draws: Dict[str, List[np.ndarray]],
        behavior_Z_flat: Optional[np.ndarray],
        n_trials: int,
        T: int,
) -> Tuple[
    Optional[PrivateLatentPairResult],
    Optional[PrivateLatentPairResult],
    Optional[HubPairPCAResult],
]:
    """Parts 2a/2b/2c/2c' for one canonicalized pair, one session --
    `N_SAMPLE_DRAWS` independent neuron-sampling draws (item 3), each
    otherwise following v1's own `_compute_pair_result` (2c/2c') /
    `_compute_hub_pair_pca_result` (2a/2b) computation. Both hub
    orientations and both pCCA nuisance variants of a given draw `k` share
    the SAME sampled neuron subset for region_i/region_j/every nuisance
    region -- only the model each is fit under (joint pCCA vs. sequential
    PCA, with vs. without behaviour) differs -- exactly mirroring how v1's
    2a/2b/2c/2c' all shared one fixed neuron set per pair.
    """
    if region_i not in region_flat_full or region_j not in region_flat_full:
        return None, None, None
    if region_i not in paired_draws or region_j not in paired_draws:
        return None, None, None

    nuisance_all = _other_region_nuisance_list_full(
        region_i, region_j, region_flat_full, MIN_NEURONS_PER_REGION)

    labels_i_full = region_subregion_labels_full.get(region_i)
    labels_j_full = region_subregion_labels_full.get(region_j)

    pair_draws: List[PrivateLatentPairDrawResult] = []
    pair_draws_regions_only: List[PrivateLatentPairDrawResult] = []
    hub_i_draws: List[HubOrientationPCADrawResult] = []
    hub_j_draws: List[HubOrientationPCADrawResult] = []

    # Per-draw neuron-index axes (shared by 2c/2c', since both conditions
    # sample the SAME region_i/region_j neurons for a given draw k -- only
    # the nuisance Z differs) and per-draw residualized matrices for each
    # condition, pooled after the loop by `_select_top_weight_neurons`.
    draw_idx_i_list: List[np.ndarray] = []
    draw_idx_j_list: List[np.ndarray] = []
    draw_Xi_res_2c: List[np.ndarray] = []
    draw_Xj_res_2c: List[np.ndarray] = []
    draw_Xi_res_2cp: List[np.ndarray] = []
    draw_Xj_res_2cp: List[np.ndarray] = []

    for k in range(N_SAMPLE_DRAWS):
        idx_i = paired_draws[region_i][k]
        idx_j = paired_draws[region_j][k]
        nuisance_idx_k = {r: nuisance_draws[r][k] for r in nuisance_all}

        X_i_sub = region_flat_full[region_i][:, idx_i]
        X_j_sub = region_flat_full[region_j][:, idx_j]
        Z_nuisance_parts = [
            region_flat_full[r][:, nuisance_idx_k[r]] for r in nuisance_all
        ]
        Z_nuisance = np.concatenate(Z_nuisance_parts, axis=1) if Z_nuisance_parts else None

        labels_i_sub = [labels_i_full[i] for i in idx_i] if labels_i_full else None
        labels_j_sub = [labels_j_full[j] for j in idx_j] if labels_j_full else None

        # ---- 2c: AllRegions + Behaviour ------------------------------
        z_parts = list(Z_nuisance_parts)
        if behavior_Z_flat is not None:
            z_parts.append(behavior_Z_flat)
        Z_full = np.concatenate(z_parts, axis=1) if z_parts else None
        draw_result_2c, Xi_res_2c, Xj_res_2c = _compute_pair_result_draw(
            k, region_i, region_j, X_i_sub, X_j_sub, idx_i, idx_j, Z_full,
            nuisance_all, nuisance_idx_k, n_trials, T, labels_i_sub, labels_j_sub)
        pair_draws.append(draw_result_2c)

        # ---- 2c': AllRegions only -------------------------------------
        draw_result_2cp, Xi_res_2cp, Xj_res_2cp = _compute_pair_result_draw(
            k, region_i, region_j, X_i_sub, X_j_sub, idx_i, idx_j, Z_nuisance,
            nuisance_all, nuisance_idx_k, n_trials, T, labels_i_sub, labels_j_sub)
        pair_draws_regions_only.append(draw_result_2cp)

        draw_idx_i_list.append(idx_i)
        draw_idx_j_list.append(idx_j)
        draw_Xi_res_2c.append(Xi_res_2c)
        draw_Xj_res_2c.append(Xj_res_2c)
        draw_Xi_res_2cp.append(Xi_res_2cp)
        draw_Xj_res_2cp.append(Xj_res_2cp)

        # ---- 2a/2b: region_i as hub, region_j as partner ---------------
        X_i_behav_res_sub = behav_res_by_region_full[region_i][:, idx_i]
        hub_i_draws.append(_compute_hub_orientation_draw(
            k, region_i, region_j, X_i_behav_res_sub, idx_i, Z_nuisance,
            nuisance_all, nuisance_idx_k, n_trials, T, labels_i_sub))

        # ---- 2a/2b: region_j as hub, region_i as partner ---------------
        X_j_behav_res_sub = behav_res_by_region_full[region_j][:, idx_j]
        hub_j_draws.append(_compute_hub_orientation_draw(
            k, region_j, region_i, X_j_behav_res_sub, idx_j, Z_nuisance,
            nuisance_all, nuisance_idx_k, n_trials, T, labels_j_sub))

    # ---- Top-pCCA-weight neuron selection + saved residualized activity,
    #      pooled across all N_SAMPLE_DRAWS draws, separately for region_i
    #      and region_j and separately for 2c / 2c' (4 selections total). --
    selected_i_2c = _select_top_weight_neurons(
        region_i, [d.Wx for d in pair_draws], draw_idx_i_list, draw_Xi_res_2c, n_trials, T)
    selected_j_2c = _select_top_weight_neurons(
        region_j, [d.Wy for d in pair_draws], draw_idx_j_list, draw_Xj_res_2c, n_trials, T)
    selected_i_2cp = _select_top_weight_neurons(
        region_i, [d.Wx for d in pair_draws_regions_only], draw_idx_i_list, draw_Xi_res_2cp, n_trials, T)
    selected_j_2cp = _select_top_weight_neurons(
        region_j, [d.Wy for d in pair_draws_regions_only], draw_idx_j_list, draw_Xj_res_2cp, n_trials, T)

    pair_result = PrivateLatentPairResult(
        region_i=region_i, region_j=region_j, draws=pair_draws,
        selected_neurons_i=selected_i_2c, selected_neurons_j=selected_j_2c)
    pair_result_regions_only = PrivateLatentPairResult(
        region_i=region_i, region_j=region_j, draws=pair_draws_regions_only,
        selected_neurons_i=selected_i_2cp, selected_neurons_j=selected_j_2cp)
    hub_pair_result = HubPairPCAResult(
        region_i=region_i, region_j=region_j,
        region_i_as_hub_draws=hub_i_draws, region_j_as_hub_draws=hub_j_draws,
    )
    return pair_result, pair_result_regions_only, hub_pair_result


def compute_private_latents_for_session(
        session_name: str,
        trial_type: str = TRIAL_TYPE,
        mat_dir: Optional[Path] = None,
        align_mode: str = ALIGN,
) -> Optional[PrivateLatentSessionResult]:
    """Compute Parts 1a/1b/2a/2b/2c/2c' for one session of one trial type /
    alignment mode. Returns None if the session cannot be processed at
    all (missing source .mat file, a crop window with < 2 overlapping
    samples, or -- when REQUIRE_BEHAVIOR=True, the default -- missing
    behavioural tracking) -- same failure contract as v1.
    """
    mat_dir = mat_dir if mat_dir is not None else (BASE_DIR / mat_subdir_name(trial_type, align_mode))
    session_file = mat_dir / f"{session_name}_analysis_results.mat"
    if not session_file.exists():
        print(f"  [skip] {session_name}: source file not found -> {session_file}")
        return None

    # ---- Part 1 pool: selected_neurons-filtered (item 1, unchanged) -----
    region_spikes_sel, region_subregion_labels_sel, n_trials, T = load_region_spikes(
        str(session_file))
    if not region_spikes_sel:
        print(f"  [skip] {session_name}: no regions loaded")
        return None

    # ---- Part 2 pool: full/unfiltered population (item 2) ---------------
    region_spikes_full, region_subregion_labels_full, n_trials_full, T_full = (
        load_region_spikes_full(str(session_file)))
    if not region_spikes_full:
        print(f"  [skip] {session_name}: no regions loaded (full pool)")
        return None
    if n_trials_full != n_trials or T_full != T:
        warnings.warn(
            f"[{session_name}] Part-1/Part-2 pools disagree on n_trials/T "
            f"({n_trials}/{T} vs {n_trials_full}/{T_full}) from the same "
            f"source file; proceeding with the Part-1 (selected_neurons) "
            f"pool's n_trials/T for both."
        )

    # ---- Crop to the behavioural-tracking window, applied identically to
    #      BOTH pools via the SAME pristine (pre-crop) time vector --------
    time_vec_raw = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)
    try:
        region_spikes_sel, time_vec = crop_time_window(
            region_spikes_sel, time_vec_raw, BEHAVIOR_TIME_RANGE_S)
    except ValueError as exc:
        print(f"  [skip] {session_name}: {exc}")
        return None
    region_spikes_full, _ = crop_time_window(
        region_spikes_full, time_vec_raw, BEHAVIOR_TIME_RANGE_S)
    T = time_vec.shape[0]

    # ---- Behaviour (position + speed), filtered to this trial type ------
    behav_combined_raw: Optional[np.ndarray] = None
    behavior_channel_labels: List[str] = []
    label = behavior_label_for(trial_type)
    try:
        pos_sel, speed_sel, _t_behav = load_behavior_regressors(
            session_name, trial_label=label)
    except (FileNotFoundError, ValueError) as exc:
        if REQUIRE_BEHAVIOR:
            print(f"  [skip] {session_name}: behaviour unavailable ({exc})")
            return None
        warnings.warn(
            f"[{session_name}] behaviour unavailable ({exc}); proceeding "
            f"with AllRegions-only nuisance (no behaviour term) since "
            f"REQUIRE_BEHAVIOR=False."
        )
    else:
        n_trials_behav, T_behav = pos_sel.shape[0], pos_sel.shape[-1]
        n_common = min(n_trials, n_trials_behav)
        T_common = min(T, T_behav)
        if n_common != n_trials or n_common != n_trials_behav:
            warnings.warn(
                f"[{session_name}] trial-count mismatch (neural={n_trials}, "
                f"behaviour={n_trials_behav}); truncating both to the first "
                f"{n_common} trials (assumes matching trial order)."
            )
        if T_common != T or T_common != T_behav:
            warnings.warn(
                f"[{session_name}] time-axis length mismatch (neural={T}, "
                f"behaviour={T_behav}); truncating both to the first "
                f"{T_common} samples."
            )
        region_spikes_sel = {r: X[:n_common, :, :T_common] for r, X in region_spikes_sel.items()}
        region_spikes_full = {r: X[:n_common, :, :T_common] for r, X in region_spikes_full.items()}
        time_vec = time_vec[:T_common]
        n_trials, T = n_common, T_common

        pos_raw   = pos_sel[:n_common, :, :T_common].astype(np.float32)
        speed_raw = speed_sel[:n_common, :, :T_common].astype(np.float32)
        behav_combined_raw = np.concatenate([pos_raw, speed_raw], axis=1)
        behavior_channel_labels = ["x", "y", "z", "speed"]

    # ---- Flatten + z-score both pools once, reused across every pair ----
    region_flat_sel: Dict[str, np.ndarray] = {
        r: _zscore_flat(X, subtract_psth=SUBTRACT_PSTH, shuffle_trials=SHUFFLE_TRIALS)
        for r, X in region_spikes_sel.items()
    }
    region_flat_full: Dict[str, np.ndarray] = {
        r: _zscore_flat(X, subtract_psth=SUBTRACT_PSTH, shuffle_trials=SHUFFLE_TRIALS)
        for r, X in region_spikes_full.items()
    }
    behavior_Z_flat: Optional[np.ndarray] = None
    if behav_combined_raw is not None:
        behavior_Z_flat = _zscore_flat(behav_combined_raw, subtract_psth=SUBTRACT_PSTH)

    # ---- Part 1 (1a/1b): unchanged from v1, fit on region_flat_sel ------
    region_pca_raw: Dict[str, RegionPCAResult] = {}
    region_pca_out_behaviour: Dict[str, RegionPCAResult] = {}
    for region in region_flat_sel.keys():
        region_pca_raw[region] = _compute_region_pca(
            region, region_flat_sel[region], n_trials, T,
            labels=region_subregion_labels_sel.get(region))

        X_behav_res = residualize(region_flat_sel[region], behavior_Z_flat, LAMBDA_HAT)
        region_pca_out_behaviour[region] = _compute_region_pca(
            region, X_behav_res, n_trials, T,
            labels=region_subregion_labels_sel.get(region))

    # ---- Part 2 prep: behaviour-residualize the FULL pool ONCE per
    #      region (item 2's "replacing selected_neurons entirely" --
    #      this residual is Part 2's own, on the full population, not a
    #      reuse of Part 1's). Column-independent ridge regression means
    #      slicing a draw's sampled columns out of this AFTER residualizing
    #      is numerically identical to residualizing that smaller subset
    #      directly (see module docstring) -- so this is done once per
    #      region per session, not once per region per draw. -------------
    behav_res_by_region_full: Dict[str, np.ndarray] = {
        r: residualize(region_flat_full[r], behavior_Z_flat, LAMBDA_HAT)
        for r in region_flat_full
    }

    # ---- Part 2 sampling: precompute each region's draws ONCE per
    #      session (item 2: "... 10 separate times per region") -- a
    #      "paired" draw set for every HUB_REGION present (used whenever
    #      that region is region_i/region_j of some pair) and a
    #      "nuisance" draw set for every recorded region (used whenever
    #      that region enters some OTHER pair's nuisance design). --------
    paired_draws: Dict[str, List[np.ndarray]] = {}
    for region in HUB_REGIONS:
        if region not in region_flat_full:
            continue
        n_full = region_flat_full[region].shape[1]
        rng = _make_region_rng(session_name, region, "paired")
        paired_draws[region] = sample_paired_draws(n_full, rng)

    nuisance_draws: Dict[str, List[np.ndarray]] = {}
    for region in region_flat_full:
        n_full = region_flat_full[region].shape[1]
        rng = _make_region_rng(session_name, region, "nuisance")
        nuisance_draws[region] = sample_nuisance_draws(n_full, rng)

    # ---- Part 2: every pair in REGION_PAIRS, N_SAMPLE_DRAWS draws each --
    pairs: Dict[Tuple[str, str], PrivateLatentPairResult] = {}
    pairs_regions_only: Dict[Tuple[str, str], PrivateLatentPairResult] = {}
    hub_pca_pairs: Dict[Tuple[str, str], HubPairPCAResult] = {}
    for ri_raw, rj_raw in REGION_PAIRS:
        region_i, region_j = sort_pair_by_anatomy(ri_raw, rj_raw)

        pair_result, pair_result_ro, hub_result = _compute_pair_sampled(
            region_i, region_j, region_flat_full, behav_res_by_region_full,
            region_subregion_labels_full, paired_draws, nuisance_draws,
            behavior_Z_flat, n_trials, T,
        )
        if pair_result is not None:
            pairs[(region_i, region_j)] = pair_result
        if pair_result_ro is not None:
            pairs_regions_only[(region_i, region_j)] = pair_result_ro
        if hub_result is not None:
            hub_pca_pairs[(region_i, region_j)] = hub_result

    print(
        f"  [{session_name}] 2c pCCA {len(pairs)}/{len(REGION_PAIRS)} pairs "
        f"(AllRegions+Behaviour), 2c' pCCA {len(pairs_regions_only)}/{len(REGION_PAIRS)} "
        f"pairs (AllRegions only), "
        f"2a/2b hub-PCA {len(hub_pca_pairs)}/{len(REGION_PAIRS)} pairs "
        f"(each x{N_SAMPLE_DRAWS} sampling draws), "
        f"1a/1b region-PCA {len(region_pca_raw)}/{len(HUB_REGIONS)} regions  "
        f"(n_trials={n_trials}, T={T}, behaviour="
        f"{'yes' if behav_combined_raw is not None else 'no'})"
    )

    return PrivateLatentSessionResult(
        session=session_name,
        trial_type=trial_type,
        time_vec=time_vec.astype(np.float64),
        n_trials=n_trials,
        T=T,
        behavior_available=behav_combined_raw is not None,
        behavior_channel_labels=behavior_channel_labels,
        excluded_regions=list(EXCLUDED_REGIONS),
        config=config_fingerprint(),
        pairs=pairs,
        pairs_regions_only=pairs_regions_only,
        region_pca_raw=region_pca_raw,
        region_pca_out_behaviour=region_pca_out_behaviour,
        hub_pca_pairs=hub_pca_pairs,
        align_mode=align_mode,
        region_subregion_labels=region_subregion_labels_sel,
        region_subregion_labels_full=region_subregion_labels_full,
    )


# =============================================================================
# 8.  Orchestration -- same shape as v1, writing to this script's own
#     output directory (`out_subdir_name`).
# =============================================================================

def run_all_sessions(
        trial_type: str = TRIAL_TYPE,
        sessions: Optional[List[str]] = None,
        overwrite: bool = False,
        align_mode: str = ALIGN,
) -> None:
    """Compute and pickle every session's PrivateLatentSessionResult for
    `trial_type` / `align_mode`. Caching contract matches v1: an existing
    {session}_analysis_results.pkl is reused only if its stored `config`
    matches the CURRENT `config_fingerprint()`."""
    sessions = sessions if sessions is not None else SESSIONS
    mat_dir = BASE_DIR / mat_subdir_name(trial_type, align_mode)
    out_dir = BASE_DIR / out_subdir_name(trial_type, align_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sessions is None:
        session_files = sorted(mat_dir.glob("*_analysis_results.mat"))
        sessions = [f.stem.replace("_analysis_results", "") for f in session_files]

    print("=" * 70)
    print("v2.0  |  Region/hub PCA (1a/1b/2a/2b) + private pCCA (2c/2c')  |  "
          "resampled neurons")
    print(f"  trial_type : {trial_type}   (behaviour label = '{behavior_label_for(trial_type)}')")
    print(f"  align_mode : {align_mode}")
    print(f"  source dir : {mat_dir}")
    print(f"  output dir : {out_dir}")
    print(f"  sessions   : {len(sessions)}")
    print(f"  pairs      : {len(REGION_PAIRS)}  (across {len(PAIR_CATEGORIES)} categories)")
    print(f"  hub regions: {len(HUB_REGIONS)}  {HUB_REGIONS}")
    print(f"  pca comps  : {N_PCA_COMPONENTS}  (Parts 1a/1b/2a/2b; N_COMPONENTS={N_COMPONENTS} for 2c/2c's pCCA)")
    print(f"  sampling   : target={TARGET_SAMPLE_SIZE} neurons/region, "
          f"{N_SAMPLE_DRAWS} draws/pair/session (Parts 2a/2b/2c/2c' only)")
    print("=" * 70)

    current_config = config_fingerprint()
    n_written = n_cached = n_skipped = 0

    for idx, session_name in enumerate(sessions, 1):
        print(f"\n\U0001F680 [{idx}/{len(sessions)}] {session_name}")
        out_path = out_dir / f"{session_name}_analysis_results.pkl"

        if out_path.exists() and not overwrite:
            try:
                with open(out_path, "rb") as fh:
                    cached: PrivateLatentSessionResult = pickle.load(fh)
                if cached.config == current_config:
                    print(f"  cached, config unchanged -> skip")
                    n_cached += 1
                    continue
                print(f"  cached copy has a stale config -> recomputing")
            except Exception as exc:
                print(f"  cached copy unreadable ({exc}) -> recomputing")

        try:
            result = compute_private_latents_for_session(
                session_name, trial_type, mat_dir, align_mode=align_mode)
        except Exception as exc:
            print(f"  \U0001F4A5 [ERROR] {session_name}: {exc}")
            n_skipped += 1
            continue

        has_any_result = result is not None and (
            result.pairs or result.pairs_regions_only or result.hub_pca_pairs
            or result.region_pca_raw
        )
        if not has_any_result:
            n_skipped += 1
            continue

        with open(out_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✨ saved -> {out_path}")
        n_written += 1

    print("\n" + "=" * 70)
    print(
        f"\U0001F389 Done. {n_written} written, {n_cached} cached, "
        f"{n_skipped} skipped (of {len(sessions)} sessions)."
    )
    print(f"   Load results back with: PrivateLatentAnalyzer(trial_type={trial_type!r}).load_all()")
    print("   Part 1a/1b : az.get_region_pca(session, region, out_behaviour=True/False)")
    print("   Part 2a/2b : az.get_hub_pca_draws(session, hub, partner)  (list of 10)")
    print("   Part 2c    : az.get_pair_draws(session, region_i, region_j)  (list of 10)")
    print("   top-weight : az.get_selected_neurons(session, region_i, region_j, side='i'/'j')")
    print("=" * 70)


def main() -> None:
    run_all_sessions()


# =============================================================================
# 9.  Analyzer -- the Python-native, .pkl-loading counterpart of v1's
#     `PrivateLatentAnalyzer`, adapted for Part 2's now-10-draws-per-pair
#     results.
# =============================================================================

class PrivateLatentAnalyzer:
    """
    Indexes the `*_analysis_results.pkl` files written by `run_all_sessions`.

    Typical use
    -----------
        az = PrivateLatentAnalyzer(trial_type="cued_hit_long")
        az.load_all()
        az.summary()

        draws = az.get_pair_draws("yp021_220407", "MOp", "VPMPO")           # Part 2c, 10 draws
        draws[0].z_i_lat[:, :, 0]      # draw 0's component-0 private latent
        draws[0].neuron_idx_i          # draw 0's sampled region_i neuron indices

        r1a = az.get_region_pca("yp021_220407", "MOp")                      # Part 1a
        r1b = az.get_region_pca("yp021_220407", "MOp", out_behaviour=True)  # Part 1b

        hub_draws = az.get_hub_pca_draws("yp021_220407", "MOp", "VPMPO")    # Parts 2a+2b, 10 draws
        hub_draws[0].latent_network[:, :, 0]

        sel = az.get_selected_neurons("yp021_220407", "MOp", "VPMPO", side="i")  # Part 2c
        sel.weight_threshold          # pooled-|weight| cutoff used for this pair/side
        sel.neurons[0].neuron_idx     # a top-20%-weight MOp neuron's FULL-population index
        sel.neurons[0].residual       # (n_trials, T) its residualized activity, averaged
                                       # across every draw in which it qualified
    """

    def __init__(
            self,
            base_dir: Path = BASE_DIR,
            trial_type: str = TRIAL_TYPE,
            align_mode: str = ALIGN,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.trial_type = trial_type
        self.align_mode = align_mode
        self.results_dir = self.base_dir / out_subdir_name(trial_type, align_mode)
        self.sessions: Dict[str, PrivateLatentSessionResult] = {}

    def available_sessions(self) -> List[str]:
        """Session names with a cached .pkl on disk, without loading them."""
        if not self.results_dir.exists():
            return []
        return sorted(
            p.stem.replace("_analysis_results", "")
            for p in self.results_dir.glob("*_analysis_results.pkl")
        )

    def load_session(self, session_name: str) -> PrivateLatentSessionResult:
        path = self.results_dir / f"{session_name}_analysis_results.pkl"
        with open(path, "rb") as fh:
            result = pickle.load(fh)
        self.sessions[session_name] = result
        return result

    def load_all(self) -> Dict[str, PrivateLatentSessionResult]:
        for session_name in self.available_sessions():
            try:
                self.load_session(session_name)
            except Exception as exc:
                print(f"    [{session_name}] load error: {exc}")
        print(
            f"[PrivateLatentAnalyzer v2] loaded {len(self.sessions)} session(s) "
            f"from {self.results_dir}"
        )
        return self.sessions

    def get_pair_draws(
            self, session_name: str, region_i: str, region_j: str,
            regions_only: bool = False,
    ) -> List[PrivateLatentPairDrawResult]:
        """All `N_SAMPLE_DRAWS` Part 2c (`regions_only=False`, default) /
        2c' (`regions_only=True`) draws for one pair, one loaded session.
        Empty list if the pair/session isn't present."""
        session = self.sessions.get(session_name)
        if session is None:
            return []
        table = session.pairs_regions_only if regions_only else session.pairs
        pair_result = table.get(sort_pair_by_anatomy(region_i, region_j))
        return pair_result.draws if pair_result is not None else []

    def get_pair_draw(
            self, session_name: str, region_i: str, region_j: str,
            draw_idx: int, regions_only: bool = False,
    ) -> Optional[PrivateLatentPairDrawResult]:
        """One specific draw (0-indexed) of one pair, one loaded session."""
        draws = self.get_pair_draws(session_name, region_i, region_j, regions_only)
        for d in draws:
            if d.draw_idx == draw_idx:
                return d
        return None

    def get_selected_neurons(
            self, session_name: str, region_i: str, region_j: str, side: str,
            regions_only: bool = False,
    ) -> Optional[SelectedNeuronSet]:
        """Top-pCCA-weight neuron selection (+ saved residualized activity)
        for one region side of one pair, one loaded session: `side="i"`
        for region_i (the anatomically earlier of the pair, NOT
        necessarily whichever region name you passed as `region_i`) or
        `side="j"` for region_j; `regions_only=False`/`True` picks 2c/2c'
        as usual. `None` if the pair/session isn't present."""
        if side not in ("i", "j"):
            raise ValueError(f"side must be 'i' or 'j', got {side!r}")
        session = self.sessions.get(session_name)
        if session is None:
            return None
        table = session.pairs_regions_only if regions_only else session.pairs
        pair_result = table.get(sort_pair_by_anatomy(region_i, region_j))
        if pair_result is None:
            return None
        return pair_result.selected_neurons_i if side == "i" else pair_result.selected_neurons_j

    def get_region_pca(
            self, session_name: str, region: str, out_behaviour: bool = False,
    ) -> Optional[RegionPCAResult]:
        """Part 1a (`out_behaviour=False`, default) / Part 1b
        (`out_behaviour=True`) region-level PCA result -- unaffected by
        Part 2's sampling, unchanged from v1."""
        session = self.sessions.get(session_name)
        if session is None:
            return None
        table = session.region_pca_out_behaviour if out_behaviour else session.region_pca_raw
        return table.get(region)

    def get_hub_pca_draws(
            self, session_name: str, hub: str, partner: str,
    ) -> List[HubOrientationPCADrawResult]:
        """All `N_SAMPLE_DRAWS` Parts 2a+2b draws for one (hub, partner)
        orientation, one loaded session. Canonicalizes internally, so the
        caller never has to reason about pair ordering."""
        session = self.sessions.get(session_name)
        if session is None:
            return []
        region_i, region_j = sort_pair_by_anatomy(hub, partner)
        pair_result = session.hub_pca_pairs.get((region_i, region_j))
        if pair_result is None:
            return []
        return pair_result.region_i_as_hub_draws if hub == region_i else pair_result.region_j_as_hub_draws

    def get_hub_pca_draw(
            self, session_name: str, hub: str, partner: str, draw_idx: int,
    ) -> Optional[HubOrientationPCADrawResult]:
        """One specific draw (0-indexed) of one hub orientation."""
        draws = self.get_hub_pca_draws(session_name, hub, partner)
        for d in draws:
            if d.draw_idx == draw_idx:
                return d
        return None

    def iter_pair_across_sessions(self, region_i: str, region_j: str):
        """Yield (session_name, PrivateLatentPairResult) for every loaded
        session that has this pair -- `.draws` holds all N_SAMPLE_DRAWS
        draws for that session."""
        pair_key = sort_pair_by_anatomy(region_i, region_j)
        for session_name, session in self.sessions.items():
            if pair_key in session.pairs:
                yield session_name, session.pairs[pair_key]

    def summary(self) -> None:
        """Per-pair (Part 2c) and per-region (Part 1a/1b) session-coverage
        tables, in the spirit of v1's own `summary`."""
        if not self.sessions:
            print("[PrivateLatentAnalyzer v2] no sessions loaded -- call load_all() first.")
            return
        counts: Dict[Tuple[str, str], int] = {}
        draw_counts: Dict[Tuple[str, str], int] = {}
        for session in self.sessions.values():
            for pair_key, pair_result in session.pairs.items():
                counts[pair_key] = counts.get(pair_key, 0) + 1
                draw_counts[pair_key] = draw_counts.get(pair_key, 0) + len(pair_result.draws)
        print(f"[PrivateLatentAnalyzer v2]  trial_type={self.trial_type}  sessions={len(self.sessions)}")
        print("  Part 2c -- private pCCA, by pair (sessions covered / total draws):")
        for category, pairs in PAIR_CATEGORIES:
            for pair in pairs:
                n = counts.get(pair, 0)
                nd = draw_counts.get(pair, 0)
                print(
                    f"    [{category:<26s}] {pair[0]:>7s} <-> {pair[1]:<7s} : "
                    f"{n:3d}/{len(self.sessions)} sessions, {nd:4d} draws"
                )
        region_counts: Dict[str, int] = {}
        for session in self.sessions.values():
            for region in session.region_pca_raw:
                region_counts[region] = region_counts.get(region, 0) + 1
        print("  Part 1a/1b -- region-level PCA, by region:")
        for region in HUB_REGIONS:
            n = region_counts.get(region, 0)
            print(f"    {region:>7s} : {n:3d}/{len(self.sessions)} sessions")


if __name__ == "__main__":
    main()

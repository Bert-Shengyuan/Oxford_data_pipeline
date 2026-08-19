#!/usr/bin/env python3
r"""
pcca_all_regions_out_behaviour.py
================================================================================

Per-session, per-pair PRIVATE pCCA LATENT under the "AllRegions + Behaviour"
nuisance condition, computed natively in Python and cached to disk as one
pickle file per session.

--------------------------------------------------------------------------------
Relationship to the rest of the pipeline
--------------------------------------------------------------------------------
This script generalises a computation that already exists, for one hardcoded
target pair, inside `pCCA_sensitive_realsingle_Session_8panel.py`: the
`abl-AllRegions+Behaviour` step of `run_single_ablation`. There, for a fixed
pair (TARGET_I, TARGET_J), the nuisance design

    Z = [ X_r  for every anatomical region r not in {TARGET_I, TARGET_J} ]
        concatenated with [ position (x, y, z), speed ]

is regressed out of both target regions' activity via a joint ridge hat
matrix (`residualize`), and ridge CCA (`ridge_cca`) is then fit on the two
residuals (`pcca`). This script performs *exactly* that computation -- same
Z construction, same residualize/ridge_cca/pcca primitives, same crop to the
behavioural-tracking window `BEHAVIOR_TIME_RANGE_S = (-1.0, 2.0)` s -- but
loops it over every pair in `REGION_PAIRS` (the same 21-pair / 7-category
set used throughout this project's cross-session figures, e.g.
`pcca_cross_session_mi_bar.py` and `pCCA_latent_extrenal_variable_bar.py`,
copied verbatim from the latter) and over every session auto-discovered
under the relevant `pcca_sessions_{trial_type}_results` folder.

The result is called "private" because, once every other recorded region
AND the animal's kinematics have been partialled out, what remains in the
region_i <-> region_j canonical correlation is the coupling specific to that
one dyad, not a hub-wide signal shared with the rest of the network -- the
pCCA counterpart of a partner-specific ("private") communication channel.

--------------------------------------------------------------------------------
Mathematical framework
--------------------------------------------------------------------------------
For canonicalized pair (region_i, region_j) with flattened, z-scored
activity X_i, X_j in R^{(T n_trials) x n}, and nuisance design Z (every
other available region's activity plus position and speed, all flattened
and z-scored the same way):

  Residualization (one joint ridge hat matrix over the whole of Z -- not one
  fit per nuisance region; see "Nuisance regression must be joint" below):

      X_hat = X - Z (Z^T Z + lam_hat * n * I)^(-1) Z^T X

  Ridge CCA on the residuals:

      Cxx = X_hat_i^T X_hat_i / (n-1),  Cyy, Cxy analogous
      A = Cxx^(-1/2)_{lam_cca},  B = Cyy^(-1/2)_{lam_cca}
      U, S, V^T = SVD(A Cxy B)
      Wx = A U[:, :K],  Wy = B V[:, :K]^T,  rho = clip(S[:K], 0, 1)

  Private latent (per trial, per time bin, per canonical dimension k):

      z_i^(k)(trial, t) = [X_hat_i Wx]_{(trial, t), k}

--------------------------------------------------------------------------------
Storage
--------------------------------------------------------------------------------
Following this project's "primitives copied, not imported" convention --
every core numerical routine below is copied verbatim from
pCCA_sensitive_realsingle_Session_8panel.py / cross_trial_type_cca_analysis.py
so this script stays independently auditable and runnable on its own --
results are not kept only in memory. `run_all_sessions()` writes one
`{session}_analysis_results.pkl` file per session into

    {BASE_DIR}/pcca_all_regions_out_behaviour_sessions_{trial_type}_results/

directly mirroring the naming convention of `pcca_sessions_{trial_type}_
results` (the folder this script's own neural input already lives in); the
only difference is the file format -- .pkl, loaded with `pickle`, in place
of .mat, loaded with `mat73`. `PrivateLatentAnalyzer`, defined at the bottom
of this file, is the Python-native counterpart of `OxfordAdvancedAnalyzer`
(Useful_definition.py) / `CrossTrialTypeCCAAnalyzer`
(cross_trial_type_cca_analysis.py): it indexes these .pkl files the same
way those two classes index the .mat files, minus the MATLAB round-trip,
since the pCCA fit was already performed in Python.

Switching `TRIAL_TYPE` below from "cued_hit_long" to "spont_hit_long" (or
"spont_miss_long") re-targets the whole pipeline -- input folder, output
folder, and the *_task_label.npy filter string -- at that condition;
nothing else has to change.

Note on scope: the plotting/diagnostic machinery of the source ablation
script (StepResult, SupplementaryMetrics, Rastermap neuron ordering, the
8-panel figures, per-step sign correction) is deliberately NOT carried over
here. This script's only job is to compute and persist the private latent
itself, so only the primitives that computation actually needs are copied
in. Canonical-vector sign is left as computed (arbitrary up to a per-
component flip, a standard CCA/pCCA property) rather than heuristically
corrected: the one place downstream that is already known to consume this
kind of latent, `variance_explained` in pCCA_latent_extenal_variable_bar.py,
is explicitly documented as sign-invariant. A cross-session sign-alignment
step (analogous to `CrossSessionCCAAnalyzer._align_signs_spectral`) is a
cross-session operation and belongs in a future aggregation step built on
top of `PrivateLatentAnalyzer`, not in this per-session computation.

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import pickle
import warnings
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
# 1.  USER-CONFIGURABLE PARAMETERS
# =============================================================================

# ---- The switch requested in items (1)/(5): flip this to "spont_hit_long"
#      (or "spont_miss_long") to rerun the identical pipeline on a different
#      condition. Everything else below (input folder, output folder,
#      behavioural-label filter) is DERIVED from this one string, so there
#      is no second dict to keep in sync by hand. ---------------------------
TRIAL_TYPE: str = "cued_hit_long"          # "cued_hit_long" | "spont_hit_long" | "spont_miss_long"

BASE_DIR: Path = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
BEHAVIOR_DIR: Path = BASE_DIR / "Paper_output" / "tapproach_sessions"

# ---- CCA / pCCA dimensionality & regularisation (matches the 8-panel script)
N_COMPONENTS: int = 1
LAMBDA_CCA: float = 1e-4
LAMBDA_HAT: float = 1e-4

# ---- Time windows -----------------------------------------------------------
TIME_RANGE_S: Tuple[float, float] = (-1.5, 3.0)          # raw neural acquisition window
BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = (-1.0, 2.0)  # "for example just keep -1 to 2"
BEHAVIOR_FS: float = 50.0
BEHAVIOR_T_OFFSET: float = -1.0

# ---- Regime toggles (raw regime by default, matching this request; flip
#      SUBTRACT_PSTH if this pipeline is ever pointed at the residual regime)
SUBTRACT_PSTH: bool = False
SHUFFLE_TRIALS: bool = False

# ---- If a session has no usable position/speed tracking, REQUIRE_BEHAVIOR
#      = True (default) skips it outright, since "AllRegions+Behaviour"
#      without the behaviour term is a different quantity (AllRegions only).
#      Set False to fall back to AllRegions-only nuisance for such sessions
#      instead of dropping them -- every such session is printed clearly
#      either way so this never happens silently.
REQUIRE_BEHAVIOR: bool = True

# ---- Fixed nuisance blacklist, applied consistently across this project's
#      pCCA scripts (thin/noisy regions excluded from the nuisance design;
#      none of REGION_PAIRS' targets fall in this set). Extend if needed.

EXCLUDED_REGIONS: List[str] = []
#EXCLUDED_REGIONS: List[str] = None
# ---- Optional explicit session restriction. None => auto-discover every
#      session present under the source .mat folder for TRIAL_TYPE (glob,
#      matching OxfordAdvancedAnalyzer.load_all's own discovery pattern) --
#      the appropriate default here, since the request is "for each session".
SESSIONS: Optional[List[str]] = None


def mat_subdir_name(trial_type: str) -> str:
    """Existing MATLAB-pipeline pCCA source folder for `trial_type`."""
    return f"pcca_sessions_{trial_type}_results"


def out_subdir_name(trial_type: str) -> str:
    """This script's own output folder for `trial_type` -- item (4)'s
    naming pattern: same 'pcca_sessions_{trial_type}_results' stem, with
    the 'pcca_all_regions_out_behaviour' prefix identifying the condition."""
    return f"pcca_all_regions_out_behaviour_sessions_{trial_type}_results"


def behavior_label_for(trial_type: str) -> str:
    """'cued_hit_long' -> 'cued hit long', matching the *_task_label.npy
    string convention (BEHAVIOR_TRIAL_LABEL in the 8-panel script /
    _trial_type_to_behavior_label in pCCA_latent_extenal_variable_bar.py)."""
    return trial_type.replace("_", " ")


# =============================================================================
# 2.  Anatomical canonicalisation -- copied verbatim from
#     cross_trial_type_cca_analysis.py (get_anatomical_index /
#     sort_pair_by_anatomy), per this project's "primitive copying over
#     importing" convention. This is the mechanism behind item (3): every
#     pair is stored with region_i = anatomically earlier, region_j = later.
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
# 3.  Region-pair categories -- copied verbatim from
#     pCCA_latent_extenal_variable_bar.py (same 21 pairs / 7 categories used
#     throughout this project's cross-session figures), so this script's
#     output covers exactly the pairs the intended downstream consumer
#     already iterates over.
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


# =============================================================================
# 4.  Core pCCA/CCA primitives -- copied verbatim from
#     pCCA_sensitive_realsingle_Session_8panel.py.
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

    # ── 1. PSTH subtraction ───────────────────────────────────────────────
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

    # ── 3. Flatten → (n, T * n_trials), transpose (no re-zscore: already
    #      z-scored in step 1; re-zscoring after PSTH subtraction would be
    #      applied to a near-zero-mean residual) ────────────────────────────
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


def latent_projections(X_flat: np.ndarray, W: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """
    Project flattened, residualized activity onto every column of a
    canonical weight matrix at once.

    X_flat : (T * n_trials, n_neurons)   residualized, flattened activity
             (the X_res / Y_res returned by `pcca`).
    W      : (n_neurons, K)              canonical weight matrix (Wx or Wy).

    Returns (n_trials, T, K) -- one trace per trial, per canonical
    dimension. Generalises the single-column convention used throughout
    pCCA_sensitive_realsingle_Session_8panel.py (`latent_projections(...,
    Wx_p[:, 0], ...)` -> (n_trials, T)) to the full weight matrix in one
    matrix multiply, matching the (n_trials, T, K) convention already used
    by `_project_trials_single_region` in pCCA_latent_extenal_variable_bar.py
    -- so a future `latent[:, :, comp_idx]` slice works exactly the same way
    on either source.
    """
    K = W.shape[1]
    proj = X_flat @ W                                        # (T * n_trials, K)
    return proj.reshape(T, n_trials, K).transpose(1, 0, 2)    # (n_trials, T, K)


# =============================================================================
# 5.  Data loading -- copied verbatim from
#     pCCA_sensitive_realsingle_Session_8panel.py.
# =============================================================================

def load_region_spikes(session_path: str) -> Tuple[Dict[str, np.ndarray], int, int]:
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
        f"    [load_region_spikes]  {len(region_spikes)} regions loaded  "
        f"| n_trials={n_trials_out}  T={T_out}"
    )
    return region_spikes, int(n_trials_out), int(T_out)


def crop_time_window(
        region_spikes: Dict[str, np.ndarray],
        time_vec_full: np.ndarray,
        window: Tuple[float, float],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Crop the trailing time axis of every region's (n_trials, n, T) tensor
    to the closed interval `window`, using `time_vec_full` to build the mask."""
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
        fs: float = BEHAVIOR_FS,
        t_offset: float = BEHAVIOR_T_OFFSET,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-trial position (x, y, z) and speed traces for one session,
    filtered to trials matching `trial_label`.

    Returns
    -------
    pos_sel   : (n_trials_behav, 3, T_behav)  float32,  channels = [x, y, z]
    speed_sel : (n_trials_behav, 1, T_behav)  float32
    t_behav   : (T_behav,)  time vector in seconds
    """
    pos_path   = behavior_dir / f"{session_name}_pos.npy"
    speed_path = behavior_dir / f"{session_name}_speed.npy"
    label_path = behavior_dir / f"{session_name}_task_label.npy"
    for p in (pos_path, speed_path, label_path):
        if not p.exists():
            raise FileNotFoundError(f"Behaviour file not found: {p}")

    pos    = np.load(pos_path)                       # (N, 3, T_behav)
    speed  = np.load(speed_path)                      # (N, T_behav)
    labels = np.load(label_path, allow_pickle=True)   # (N,) object array

    if speed.ndim == 2:
        speed = speed[:, None, :]                      # → (N, 1, T_behav)

    sel = (labels == trial_label)
    if not np.any(sel):
        available = sorted(set(labels.tolist()))
        raise ValueError(
            f"No behaviour trials with label '{trial_label}' found in "
            f"{label_path.name} (available labels: {available})."
        )

    pos_sel   = pos[sel].astype(np.float32)
    speed_sel = speed[sel].astype(np.float32)
    T_behav   = pos_sel.shape[-1]
    t_behav   = np.arange(T_behav, dtype=np.float64) / fs + t_offset
    return pos_sel, speed_sel, t_behav


# =============================================================================
# 6.  Result containers -- plain dataclasses, matching the style of
#     PairWeights / SingleTrialPair in Useful_definition.py (the closest
#     existing analogue: a per-pair, per-session result feeding an
#     analyzer), rather than the __slots__ classes used for the ablation
#     script's ephemeral per-figure StepResult objects. Python 3.8 target,
#     so no `slots=True` (that dataclass kwarg is 3.10+).
# =============================================================================

@dataclass
class PrivateLatentPairResult:
    """Private pCCA latent for one canonicalized (region_i, region_j) pair,
    one session, under Z = AllRegions + Behaviour."""
    region_i: str
    region_j: str
    rho: np.ndarray                      # (K,)               canonical correlations
    Wx: np.ndarray                       # (n_neurons_i, K)   region_i canonical weights
    Wy: np.ndarray                       # (n_neurons_j, K)   region_j canonical weights
    z_i_lat: np.ndarray                  # (n_trials, T, K)   region_i private latent
    z_j_lat: np.ndarray                  # (n_trials, T, K)   region_j private latent
    n_neurons_i: int
    n_neurons_j: int
    nuisance_regions: List[str] = field(default_factory=list)
    z_dim_total: int = 0


@dataclass
class PrivateLatentSessionResult:
    """One session's worth of PrivateLatentPairResult objects, keyed by
    canonicalized (region_i, region_j) tuples -- the unit pickled to
    {session}_analysis_results.pkl, mirroring the one-.mat-file-per-session
    granularity of pcca_sessions_{trial_type}_results (each of whose files
    holds a `cca_results['pair_results']` list spanning every pair)."""
    session: str
    trial_type: str
    time_vec: np.ndarray                 # (T,) seconds, BEHAVIOR_TIME_RANGE_S-cropped
    n_trials: int
    T: int
    behavior_available: bool
    behavior_channel_labels: List[str] = field(default_factory=list)
    excluded_regions: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    pairs: Dict[Tuple[str, str], PrivateLatentPairResult] = field(default_factory=dict)

# Force the pickled module name to the real, importable one, independent of
# how this script happens to be invoked. Running `python
# pcca_all_regions_out_behaviour.py` directly executes it as module
# '__main__', and by default pickle bakes THAT name into every saved
# instance -- which then fails to unpickle from any other script's own
# '__main__' (the "Can't get attribute ... on <module '__main__' ...>"
# error). Overriding __module__ here makes every pickle reference
# 'pcca_all_regions_out_behaviour' instead, resolvable from anywhere this
# file is importable.




def config_fingerprint() -> Dict[str, Any]:
    """Snapshot of every parameter that changes the numerical result, stored
    inside each session's pickle so a stale cache is *detected*, not just
    hashed and ignored -- see 'Cache invalidation' (config hash must be
    wired into an actual comparison, not merely computed) in this project's
    working notes."""
    return dict(
        n_components=N_COMPONENTS,
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
    )


# =============================================================================
# 7.  Per-pair / per-session computation
# =============================================================================

def _compute_pair_result(
        region_i: str,
        region_j: str,
        region_flat: Dict[str, np.ndarray],
        behavior_Z_flat: Optional[np.ndarray],
        n_trials: int,
        T: int,
) -> Optional[PrivateLatentPairResult]:
    """Fit the AllRegions+Behaviour pCCA condition for one canonicalized
    region pair -- the per-pair generalisation of
    `_run_one_ablation_step(..., Z_name='AllRegions+Behaviour', ...)` in
    pCCA_sensitive_realsingle_Session_8panel.py."""
    if region_i not in region_flat or region_j not in region_flat:
        return None

    nuisance_all = [
        r for r in ANATOMICAL_ORDER
        if r in region_flat
        and r not in (region_i, region_j)
        and r not in EXCLUDED_REGIONS
    ]

    z_parts = [region_flat[r] for r in nuisance_all]
    if behavior_Z_flat is not None:
        z_parts.append(behavior_Z_flat)
    if not z_parts:
        # Neither other regions nor behaviour available for this session:
        # there is nothing to partial out, so AllRegions+Behaviour is not
        # defined for this pair here.
        return None
    Z_full = np.concatenate(z_parts, axis=1)

    Wx, Wy, rho, X_i_res, X_j_res = pcca(
        region_flat[region_i], region_flat[region_j], Z_full,
        lam_cca=LAMBDA_CCA, lam_hat=LAMBDA_HAT, n_components=N_COMPONENTS,
    )

    z_i_lat = latent_projections(X_i_res, Wx, n_trials, T)   # (n_trials, T, K)
    z_j_lat = latent_projections(X_j_res, Wy, n_trials, T)

    return PrivateLatentPairResult(
        region_i=region_i,
        region_j=region_j,
        rho=np.asarray(rho, dtype=np.float64),
        Wx=Wx.astype(np.float32),
        Wy=Wy.astype(np.float32),
        z_i_lat=z_i_lat.astype(np.float32),
        z_j_lat=z_j_lat.astype(np.float32),
        n_neurons_i=int(region_flat[region_i].shape[1]),
        n_neurons_j=int(region_flat[region_j].shape[1]),
        nuisance_regions=nuisance_all,
        z_dim_total=int(Z_full.shape[1]),
    )


def compute_private_latents_for_session(
        session_name: str,
        trial_type: str = TRIAL_TYPE,
        mat_dir: Optional[Path] = None,
) -> Optional[PrivateLatentSessionResult]:
    """Compute the AllRegions+Behaviour private pCCA latent for every pair
    in REGION_PAIRS, for one session of one trial type.

    Returns None if the session cannot be processed at all: missing source
    .mat file, a crop window with < 2 overlapping samples, or (when
    REQUIRE_BEHAVIOR=True, the default) missing/unmatched behavioural
    tracking.
    """
    mat_dir = mat_dir if mat_dir is not None else (BASE_DIR / mat_subdir_name(trial_type))
    session_file = mat_dir / f"{session_name}_analysis_results.mat"
    if not session_file.exists():
        print(f"  [skip] {session_name}: source file not found -> {session_file}")
        return None

    region_spikes, n_trials, T = load_region_spikes(str(session_file))
    if not region_spikes:
        print(f"  [skip] {session_name}: no regions loaded")
        return None

    # ---- Crop to the behavioural-tracking window, "-1 to 2" -------------
    time_vec_full = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)
    try:
        region_spikes, time_vec_full = crop_time_window(
            region_spikes, time_vec_full, BEHAVIOR_TIME_RANGE_S)
    except ValueError as exc:
        print(f"  [skip] {session_name}: {exc}")
        return None
    T = time_vec_full.shape[0]

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
            f"REQUIRE_BEHAVIOR=False -- this session's pairs are therefore "
            f"NOT the AllRegions+Behaviour condition."
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
        region_spikes = {r: X[:n_common, :, :T_common] for r, X in region_spikes.items()}
        time_vec_full = time_vec_full[:T_common]
        n_trials, T = n_common, T_common

        pos_raw   = pos_sel[:n_common, :, :T_common].astype(np.float32)
        speed_raw = speed_sel[:n_common, :, :T_common].astype(np.float32)
        behav_combined_raw = np.concatenate([pos_raw, speed_raw], axis=1)
        behavior_channel_labels = ["x", "y", "z", "speed"]

    # ---- Precompute once per session, reused across every pair ----------
    region_flat: Dict[str, np.ndarray] = {
        r: _zscore_flat(X, subtract_psth=SUBTRACT_PSTH, shuffle_trials=SHUFFLE_TRIALS)
        for r, X in region_spikes.items()
    }
    behavior_Z_flat: Optional[np.ndarray] = None
    if behav_combined_raw is not None:
        behavior_Z_flat = _zscore_flat(behav_combined_raw, subtract_psth=SUBTRACT_PSTH)

    # ---- Every pair in REGION_PAIRS --------------------------------------
    pairs: Dict[Tuple[str, str], PrivateLatentPairResult] = {}
    for ri_raw, rj_raw in REGION_PAIRS:
        # (3) storage-convention optimisation, applied defensively here (not
        # just relied upon from REGION_PAIRS already being pre-sorted above):
        # region_i is always the anatomically earlier region.
        region_i, region_j = sort_pair_by_anatomy(ri_raw, rj_raw)
        result = _compute_pair_result(
            region_i, region_j, region_flat, behavior_Z_flat, n_trials, T)
        if result is not None:
            pairs[(region_i, region_j)] = result

    print(
        f"  [{session_name}] {len(pairs)}/{len(REGION_PAIRS)} pairs computed  "
        f"(n_trials={n_trials}, T={T}, behaviour="
        f"{'yes' if behav_combined_raw is not None else 'no'})"
    )

    return PrivateLatentSessionResult(
        session=session_name,
        trial_type=trial_type,
        time_vec=time_vec_full.astype(np.float64),
        n_trials=n_trials,
        T=T,
        behavior_available=behav_combined_raw is not None,
        behavior_channel_labels=behavior_channel_labels,
        excluded_regions=list(EXCLUDED_REGIONS),
        config=config_fingerprint(),
        pairs=pairs,
    )


# =============================================================================
# 8.  Orchestration
# =============================================================================

def run_all_sessions(
        trial_type: str = TRIAL_TYPE,
        sessions: Optional[List[str]] = None,
        overwrite: bool = False,
) -> None:
    """Compute and pickle every session's PrivateLatentSessionResult for
    `trial_type`. `sessions=None` (default) auto-discovers every session
    under the source .mat folder; pass an explicit list to restrict to a
    subset (e.g. while testing).

    Caching: an existing {session}_analysis_results.pkl is reused as-is
    only if its stored `config` matches `config_fingerprint()` for the
    CURRENT settings above -- otherwise it is recomputed and overwritten,
    with a clear printed message either way (never a silent stale reload).
    """
    sessions = sessions if sessions is not None else SESSIONS
    mat_dir = BASE_DIR / mat_subdir_name(trial_type)
    out_dir = BASE_DIR / out_subdir_name(trial_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sessions is None:
        session_files = sorted(mat_dir.glob("*_analysis_results.mat"))
        sessions = [f.stem.replace("_analysis_results", "") for f in session_files]

    print("=" * 70)
    print("pCCA private latent  |  AllRegions + Behaviour ablation")
    print(f"  trial_type : {trial_type}   (behaviour label = '{behavior_label_for(trial_type)}')")
    print(f"  source dir : {mat_dir}")
    print(f"  output dir : {out_dir}")
    print(f"  sessions   : {len(sessions)}")
    print(f"  pairs      : {len(REGION_PAIRS)}  (across {len(PAIR_CATEGORIES)} categories)")
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
            result = compute_private_latents_for_session(session_name, trial_type, mat_dir)
        except Exception as exc:
            print(f"  \U0001F4A5 [ERROR] {session_name}: {exc}")
            n_skipped += 1
            continue

        if result is None or not result.pairs:
            n_skipped += 1
            continue

        with open(out_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  \u2728 saved -> {out_path}")
        n_written += 1

    print("\n" + "=" * 70)
    print(
        f"\U0001F389 Done. {n_written} written, {n_cached} cached, "
        f"{n_skipped} skipped (of {len(sessions)} sessions)."
    )
    print(f"   Load results back with: PrivateLatentAnalyzer(trial_type={trial_type!r}).load_all()")
    print("=" * 70)


def main() -> None:
    run_all_sessions()


# =============================================================================
# 9.  Analyzer -- the Python-native, .pkl-loading counterpart of
#     OxfordAdvancedAnalyzer / CrossTrialTypeCCAAnalyzer.
# =============================================================================

class PrivateLatentAnalyzer:
    """
    Indexes the `*_analysis_results.pkl` files written by `run_all_sessions`
    directly with `pickle`, the same way `OxfordAdvancedAnalyzer`
    (Useful_definition.py) and `CrossTrialTypeCCAAnalyzer`
    (cross_trial_type_cca_analysis.py) index `*_analysis_results.mat` files
    with `mat73` -- minus the MATLAB round-trip, since these results were
    fitted in Python to begin with.

    Typical use
    -----------
        az = PrivateLatentAnalyzer(trial_type="cued_hit_long")
        az.load_all()
        az.summary()
        pair = az.get_pair("yp021_220407", "MOp", "VPMPO")
        pair.z_i_lat[:, :, 0]     # (n_trials, T) component-0 private latent
        pair.rho[0]               # component-0 canonical correlation
    """

    def __init__(self, base_dir: Path = BASE_DIR, trial_type: str = TRIAL_TYPE) -> None:
        self.base_dir = Path(base_dir)
        self.trial_type = trial_type
        self.results_dir = self.base_dir / out_subdir_name(trial_type)
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
            #result: PrivateLatentSessionResult = pickle.load(fh)
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
            f"[PrivateLatentAnalyzer] loaded {len(self.sessions)} session(s) "
            f"from {self.results_dir}"
        )
        return self.sessions

    def get_pair(
            self, session_name: str, region_i: str, region_j: str,
    ) -> Optional[PrivateLatentPairResult]:
        """Look up one pair's result for one loaded session. Region order
        does not matter -- looked up via the same canonicalisation used at
        write time."""
        session = self.sessions.get(session_name)
        if session is None:
            return None
        return session.pairs.get(sort_pair_by_anatomy(region_i, region_j))

    def iter_pair_across_sessions(self, region_i: str, region_j: str):
        """Yield (session_name, PrivateLatentPairResult) for every loaded
        session that has this pair -- the natural starting point for a
        future cross-session aggregation step analogous to
        CrossSessionCCAAnalyzer.add_session_result / aggregate_projections."""
        pair_key = sort_pair_by_anatomy(region_i, region_j)
        for session_name, session in self.sessions.items():
            if pair_key in session.pairs:
                yield session_name, session.pairs[pair_key]

    def summary(self) -> None:
        """Per-pair session-coverage table, in the spirit of
        OxfordAdvancedAnalyzer._loading_summary."""
        if not self.sessions:
            print("[PrivateLatentAnalyzer] no sessions loaded -- call load_all() first.")
            return
        counts: Dict[Tuple[str, str], int] = {}
        for session in self.sessions.values():
            for pair_key in session.pairs:
                counts[pair_key] = counts.get(pair_key, 0) + 1
        print(f"[PrivateLatentAnalyzer]  trial_type={self.trial_type}  sessions={len(self.sessions)}")
        for category, pairs in PAIR_CATEGORIES:
            for pair in pairs:
                n = counts.get(pair, 0)
                print(
                    f"    [{category:<26s}] {pair[0]:>7s} <-> {pair[1]:<7s} : "
                    f"{n:3d}/{len(self.sessions)} sessions"
                )


if __name__ == "__main__":
    main()
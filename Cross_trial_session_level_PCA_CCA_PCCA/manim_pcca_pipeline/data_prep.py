"""
Real-data loader for the residualize()/pcca() Manim walkthrough (steps 1-7).

Source of example dynamics: pCCA_all_regions_out_behaviour.py in the parent
directory -- same session .mat layout, same region-selection convention
(spike_data indexed by selected_neurons), same movement-onset display window
(-1, 2) s, same residualize()/ridge_cca()/pcca() primitives (copied here
verbatim rather than imported, matching this project's own "primitives
copied, not imported" convention stated in that file's module docstring).

Hyperparameters (session / regions / neuron counts / trial count / seed) are
the block directly below -- edit these and every step script picks up the
change automatically. Neuron counts requested by the animation spec (10, 200
for the other region; 3, 20 for Region A) are clamped at load time to however
many curated (selected_neurons) units the chosen session actually has for
that region -- printed clearly when clamping happens, never padded with fake
neurons.

Architecture: steps 2-6 illustrate residualize() with Region A as the target
and OTHER_REGION (any region besides REGION_A/REGION_B) as the growing
predictor -- REGION_B is held back and only appears in step 7, where
ridge_cca() is computed between Region A and Region B after each is
independently residualized against OTHER_REGION (not against each other).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import zscore

try:
    import mat73
except ImportError as exc:
    raise SystemExit("mat73 is required: pip install mat73") from exc


# =============================================================================
# Hyperparameters -- adjust freely; every step script reads only this block.
# =============================================================================

BASE_DIR: Path = Path("/Users/shengyuancai/Downloads/Oxford_dataset")

TRIAL_TYPE: str = "cued_hit_long"
ALIGN_MODE: str = "default_move_onset"
SESSION_NAME: str = "yp020_220401"          # any session with both regions below p020_220401 p021_220404

REGION_A: str = "VPMPO"                     # Region A -- residualized against OTHER_REGION in steps 2-6
REGION_B: str = "MOp"                     # held back for step 7 only
OTHER_REGION: str = "MOs"                 # any region besides REGION_A/REGION_B: the
                                             # residualize() demo's single-neuron/10-neuron
                                             # predictor in steps 1-3, and the shared nuisance
                                             # both A and B are residualized against in step 7
                                             # (rather than A and B against each other)

# Steps 4-6's "other region" panel, once it grows past the single OTHER_REGION
# example: every region in this list besides REGION_A/REGION_B that the
# session actually has, concatenated along the neuron axis in this list's own
# order (cortical -> subcortical) -- see `load_other_regions_multi`. Region
# names not present in a given session are simply skipped, so this list can
# stay a superset across sessions.
ANATOMICAL_ORDER: List[str] = [
    "mPFC", "ORB", "MOp", "MOs", "OLF",
    "STR", "STRv",
    "MD", "LP", "VALVM", "VPMPO", "ILM",
    "HY",
]

# Friendlier labels for the region name shown below a region's colour block
# (e.g. in a colour-bar legend) -- anatomical code everywhere else (loading,
# ANATOMICAL_ORDER, region_qualitative_colors), display name only at the
# point of on-screen text. Regions not listed here fall back to their code.
DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
    "ORB": "OFC",
    "MOp": "M1 Ctx",
    "MOs": "preM Ctx",
}


def display_name(region: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(region, region)


TIME_RANGE_S: Tuple[float, float] = (-1.5, 3.0)     # raw acquisition window
DISPLAY_WINDOW_S: Tuple[float, float] = (-1.0, 2.0)  # movement-onset window to show

N_TRIALS_DEMO: int = 10                     # trials concatenated from step 2 onward
TRIAL_START_IDX: int = 0                    # first trial index used everywhere

TRAIN_FRACTION: float = 0.8                 # fraction of the session's FULL trial set (n_trials_full)
                                             # used to fit every beta/Beta shown in steps 2-6 -- the
                                             # remaining trials are held out. Steps 2-6 never fit beta
                                             # on just the N_TRIALS_DEMO trials being scrolled on screen
                                             # (see `fit_beta_train_apply_demo`); that would let the
                                             # displayed residual look artificially clean, tuned to
                                             # exactly the handful of trials the viewer can see.

# Neuron counts requested at each stage of the walkthrough (clamped to what's
# actually available for the given region in SESSION_NAME -- see `_clamp`).
N_REGION_A_STEPS: Tuple[int, int, int] = (1, 3, 20)      # steps 2 / 5 / 6
N_OTHER_REGION_STEPS: Tuple[int, int, int] = (1, 10, 200)  # steps 2 / 3 / 4
N_LATENT_NEURONS: int = 3                   # step 7's Region A / Region B neuron count

RANDOM_SEED: int = 0                        # neuron sampling order (every region)
ACTIVITY_TOP_FRACTION: float = 0.5          # neuron order favours each region's more active
                                             # half first (by std over the displayed trials),
                                             # random within that pool -- so the single "example
                                             # neuron" steps 2+ lead with isn't a flat one by luck

LAMBDA_HAT: float = 1e-4                    # ridge lambda for residualize()
LAMBDA_CCA: float = 1e-4                    # ridge lambda for ridge_cca()
N_COMPONENTS_CCA: int = 1                   # matches the source pipeline's N_COMPONENTS

CACHE_DIR: Path = Path(__file__).resolve().parent / ".cache"

# ---- Step 1 only: simulated (non-real) single-trial data -- five separated
# Gaussian peaks for the Region A neuron; the other-region neuron repeats
# exactly SIM_SHARED_PEAK_IDX of them, so residualizing removes precisely
# those two and leaves the other three (see `simulate_step1_traces`).
SIM_PEAK_CENTERS_S: Tuple[float, ...] = (-0.7, -0.2, 0.4, 1.0, 1.6)
SIM_PEAK_AMPS: Tuple[float, ...] = (1.0, 1.8, 0.7, 2.2, 1.3)
SIM_PEAK_SIGMA_S: float = 0.06
SIM_SHARED_PEAK_IDX: Tuple[int, int] = (1, 3)
SIM_FS: float = 50.0                        # matches the real pipeline's sampling rate


def mat_subdir_name(trial_type: str = TRIAL_TYPE, align_mode: str = ALIGN_MODE) -> str:
    return f"{trial_type}_{align_mode}_results"


# =============================================================================
# Numerical primitives -- copied verbatim (residualize/ridge_cca) or a direct
# extension (residualize_with_beta) of pCCA_all_regions_out_behaviour.py.
# =============================================================================

def _ridge_inv_sqrt(C: np.ndarray, lam: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(C + lam * np.eye(C.shape[0]))
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def ridge_cca(
        X: np.ndarray,
        Y: np.ndarray,
        lam: float = LAMBDA_CCA,
        n_components: int = N_COMPONENTS_CCA,
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


def residualize(X_flat: np.ndarray, Z_flat: np.ndarray, lam_hat: float = LAMBDA_HAT) -> np.ndarray:
    if Z_flat is None or Z_flat.ndim < 2 or Z_flat.shape[1] == 0:
        return X_flat.copy()
    n, m = Z_flat.shape
    ZtZ = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    Beta = np.linalg.solve(ZtZ, Z_flat.T)
    return X_flat - Z_flat @ (Beta @ X_flat)


def residualize_with_beta(
        X_flat: np.ndarray,
        Z_flat: np.ndarray,
        lam_hat: float = LAMBDA_HAT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same ridge hat-matrix solve as `residualize`, additionally returning
    the (n_Z_neurons, n_X_neurons) coefficient matrix Beta @ X_flat -- the
    beta_j values steps 1/3/4/5/6 draw on screen. `residual` here is
    numerically identical to `residualize(X_flat, Z_flat, lam_hat)`."""
    if Z_flat is None or Z_flat.ndim < 2 or Z_flat.shape[1] == 0:
        return X_flat.copy(), np.zeros((0, X_flat.shape[1]))
    n, m = Z_flat.shape
    ZtZ = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    Beta = np.linalg.solve(ZtZ, Z_flat.T)          # (m, n)
    coeff = Beta @ X_flat                          # (m, p_X)
    residual = X_flat - Z_flat @ coeff
    return residual, coeff


def train_test_trial_split(
        n_trials_full: int,
        train_fraction: float = TRAIN_FRACTION,
        seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """A fixed (seeded) train/test split of trial indices [0, n_trials_full),
    used to fit every beta/Beta shown in steps 2-6 on a held-in sample drawn
    from the FULL session rather than on just the N_TRIALS_DEMO trials being
    scrolled on screen."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials_full)
    n_train = int(round(n_trials_full * train_fraction))
    n_train = max(1, min(n_trials_full - 1, n_train))
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])
    return train_idx, test_idx


def fit_beta_train_apply_demo(
        bundle,
        region_x,
        neuron_x_idx: np.ndarray,
        region_z,
        neuron_z_idx: np.ndarray,
        lam_hat: float = LAMBDA_HAT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Beta on the TRAIN split of every trial in the session
    (`train_test_trial_split(bundle.n_trials_full)`) -- always via
    `residualize_with_beta`'s ridge hat-matrix solve (`np.linalg.solve(ZtZ,
    Z_flat.T)`), the same matrix-form regression regardless of how many
    columns `neuron_z_idx` has -- then apply that fixed Beta to
    `bundle.trials_demo` to get the residual actually scrolled on screen.
    Mirrors the source pipeline's train/held-out convention: beta is never
    fit on exactly the same handful of trials being visualized. Returns
    (resid_demo, coeff), coeff shaped (n_Z_neurons, n_X_neurons) as in
    `residualize_with_beta`."""
    train_idx, _test_idx = train_test_trial_split(bundle.n_trials_full)
    X_train = select_flat(region_x.zscored, train_idx, neuron_x_idx)
    Z_train = select_flat(region_z.zscored, train_idx, neuron_z_idx)
    _, coeff = residualize_with_beta(X_train, Z_train, lam_hat)

    X_demo = select_flat(region_x.zscored, bundle.trials_demo, neuron_x_idx)
    if coeff.shape[0] == 0:
        return X_demo, coeff
    Z_demo = select_flat(region_z.zscored, bundle.trials_demo, neuron_z_idx)
    resid_demo = X_demo - Z_demo @ coeff
    return resid_demo, coeff


def _gaussian_bumps(t: np.ndarray, centers, amps, sigma: float) -> np.ndarray:
    y = np.zeros_like(t)
    for c, a in zip(centers, amps):
        y = y + a * np.exp(-0.5 * ((t - c) / sigma) ** 2)
    return y


def simulate_step1_traces() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean, noiseless single-trial toy data for step 1: (t, x, z) where `x`
    (Region A's example neuron) is five separated Gaussian peaks and `z` (the
    other region's example neuron) repeats exactly two of them -- so
    `residualize_with_beta` recovers beta_hat ~ 1 and a 3-peak residual."""
    T = int(round((DISPLAY_WINDOW_S[1] - DISPLAY_WINDOW_S[0]) * SIM_FS)) + 1
    t = np.linspace(DISPLAY_WINDOW_S[0], DISPLAY_WINDOW_S[1], T)
    x = _gaussian_bumps(t, SIM_PEAK_CENTERS_S, SIM_PEAK_AMPS, SIM_PEAK_SIGMA_S)
    shared_centers = [SIM_PEAK_CENTERS_S[i] for i in SIM_SHARED_PEAK_IDX]
    shared_amps = [SIM_PEAK_AMPS[i] for i in SIM_SHARED_PEAK_IDX]
    z = _gaussian_bumps(t, shared_centers, shared_amps, SIM_PEAK_SIGMA_S)
    return t, x, z


# =============================================================================
# Data loading
# =============================================================================

def _raw_from_region_info(info: dict) -> np.ndarray:
    """(n_trials, n_neurons, T) raw spike tensor from one region's own entry
    in region_data.regions, with the session's curated `selected_neurons`
    index already applied -- same two steps as `load_region_spikes` in
    pCCA_all_regions_out_behaviour.py, minus the subregion-label bookkeeping
    this animation never uses."""
    spikes = np.asarray(info["spike_data"], dtype=np.float32)
    sel = info.get("selected_neurons")
    if sel is not None and np.asarray(sel).size > 0:
        idx0 = np.asarray(sel).ravel().astype(int) - 1
        spikes = spikes[:, idx0, :]
    return spikes


def _load_region_raw(session_path: Path, region: str) -> np.ndarray:
    """`_raw_from_region_info`, looking `region` up in the session's .mat
    file itself (single-region callers that don't already have `regs`)."""
    data = mat73.loadmat(str(session_path))
    regs = data.get("region_data", {}).get("regions", {})
    info = regs.get(region)
    if info is None:
        raise KeyError(f"Region {region!r} not found in {session_path.name}; available: {sorted(regs)}")
    return _raw_from_region_info(info)


def _crop_time(spikes: np.ndarray, time_vec_full: np.ndarray, window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = window
    mask = (time_vec_full >= lo - 1e-6) & (time_vec_full <= hi + 1e-6)
    if mask.sum() < 2:
        raise ValueError(f"window {window} has < 2 overlapping samples")
    return spikes[:, :, mask], time_vec_full[mask]


def _zscore_full(X: np.ndarray) -> np.ndarray:
    """Per-neuron z-score over every (trial, time) sample -- the raw-regime
    (SUBTRACT_PSTH=SHUFFLE_TRIALS=False) branch of `_zscore_flat` in
    pCCA_all_regions_out_behaviour.py, returning the (n_trials, n, T) tensor
    instead of that function's own flattened form: every step below slices
    its own trial/neuron subset out of this on demand (see `select_flat`)."""
    n_trials, n, T = X.shape
    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)   # (n, T*n_trials), time-major/trial-minor
    flat = zscore(flat, axis=1, nan_policy="omit")
    np.nan_to_num(flat, nan=0.0, copy=False)
    return flat.reshape(n, T, n_trials).transpose(2, 0, 1)  # back to (n_trials, n, T)


def select_flat(zscored: np.ndarray, trial_idx: np.ndarray, neuron_idx: np.ndarray) -> np.ndarray:
    """(n_trials, n_neurons, T) -> flat (T*len(trial_idx), len(neuron_idx)),
    in the SAME time-major/trial-minor row order `_zscore_flat` /
    `residualize` / `ridge_cca` use elsewhere in this project (row order
    never changes a ridge fit -- every sample is an exchangeable row -- but
    keeping it identical means these coefficients are the same numbers the
    full pipeline would compute for the same X/Z/samples)."""
    X = zscored[np.ix_(np.asarray(trial_idx), np.asarray(neuron_idx), np.arange(zscored.shape[2]))]
    n_sel, n_neu, T = X.shape
    return X.transpose(1, 2, 0).reshape(n_neu, T * n_sel).T   # (T*n_sel, n_neu)


def flat_to_trial_major(flat: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """Inverse of `select_flat`'s row order -> (n_trials, T, n_neurons), for
    display (a natural "trial 0 then trial 1 then ..." scrolling timeline).
    Matches `latent_projections`'s own unflatten in the source pipeline."""
    n_neu = flat.shape[1]
    return flat.reshape(T, n_trials, n_neu).transpose(1, 0, 2)


# =============================================================================
# Bundle assembly
# =============================================================================

@dataclass
class RegionData:
    name: str
    zscored: np.ndarray            # (n_trials_full, n_neurons_available, T) z-scored
    neuron_order: np.ndarray       # (n_neurons_available,) random sampling order (RANDOM_SEED)
    step_counts: Tuple[int, int, int]   # requested counts, clamped to n_neurons_available
    n_neurons_available: int

    def neurons(self, k: int) -> np.ndarray:
        """First `k` neuron indices (into this region's own column axis) in
        the fixed random sampling order -- steps only ever grow this prefix,
        never replace it, so a neuron already on screen stays put."""
        return self.neuron_order[:k]


@dataclass
class MultiRegionData:
    """Steps 4-6's "other region" panel once it grows past the single
    OTHER_REGION example: every ANATOMICAL_ORDER region besides Region
    A/Region B that this session has, concatenated along the neuron axis in
    that list's own cortical-to-subcortical order -- fixed, not grown via a
    `.neurons(k)` prefix the way `RegionData` is, since it's shown as a whole
    rather than added to one neuron at a time."""
    names: List[str]                       # one entry per contiguous region block, in concatenation order
    boundaries: List[Tuple[int, int]]      # (start, end) neuron-column index per block, parallel to `names`
    zscored: np.ndarray                    # (n_trials_full, n_neurons, T) z-scored, blocks concatenated

    @property
    def n_neurons(self) -> int:
        return self.zscored.shape[1]

    def region_at(self, neuron_idx: int) -> str:
        for name, (lo, hi) in zip(self.names, self.boundaries):
            if lo <= neuron_idx < hi:
                return name
        raise IndexError(neuron_idx)


def load_other_regions_multi(session_path: Path, exclude: Tuple[str, str],
                              use_cache: bool = True) -> MultiRegionData:
    """`ANATOMICAL_ORDER` regions besides `exclude` (Region A, Region B),
    concatenated in that list's order -- regions this session doesn't have
    are skipped rather than erroring, so `ANATOMICAL_ORDER` can stay a
    superset across sessions. Reuses the same per-region `.npz` cache
    `_load_region_zscored` writes, so a region already loaded for
    REGION_A/REGION_B/OTHER_REGION isn't re-read from the .mat file, and the
    .mat file itself is only opened once (lazily, on the first cache miss)
    rather than once per region."""
    candidates = [r for r in ANATOMICAL_ORDER if r not in exclude]
    mat_data: Optional[dict] = None

    def get_mat_data() -> dict:
        nonlocal mat_data
        if mat_data is None:
            mat_data = mat73.loadmat(str(session_path))
        return mat_data

    names: List[str] = []
    arrays: List[np.ndarray] = []
    for region in candidates:
        cache_path = _region_cache_path(region)
        if use_cache and cache_path.exists():
            arrays.append(np.load(cache_path)["zscored"])
            names.append(region)
            continue
        regs = get_mat_data().get("region_data", {}).get("regions", {})
        info = regs.get(region)
        if info is None:
            continue
        raw = _raw_from_region_info(info)
        T_raw = raw.shape[2]
        time_vec_full = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T_raw)
        cropped, _ = _crop_time(raw, time_vec_full, DISPLAY_WINDOW_S)
        zscored = _zscore_full(cropped).astype(np.float32)
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, zscored=zscored)
        arrays.append(zscored)
        names.append(region)

    if not arrays:
        raise ValueError(f"none of {candidates} available in {session_path.name} besides {exclude}")

    n_trials_full = min(a.shape[0] for a in arrays)
    arrays = [a[:n_trials_full] for a in arrays]
    boundaries: List[Tuple[int, int]] = []
    start = 0
    for a in arrays:
        end = start + a.shape[1]
        boundaries.append((start, end))
        start = end
    return MultiRegionData(names=names, boundaries=boundaries, zscored=np.concatenate(arrays, axis=1))


@dataclass
class PipelineBundle:
    session: str
    trial_type: str
    align_mode: str
    region_a: RegionData
    region_b: RegionData
    region_other: RegionData
    region_other_multi: MultiRegionData
    time_vec: np.ndarray            # (T,) seconds, DISPLAY_WINDOW_S-cropped
    fs: float
    n_trials_full: int
    trial_single: np.ndarray = field(default_factory=lambda: np.array([TRIAL_START_IDX]))
    trials_demo: np.ndarray = field(
        default_factory=lambda: np.arange(TRIAL_START_IDX, TRIAL_START_IDX + N_TRIALS_DEMO)
    )

    @property
    def T(self) -> int:
        return self.time_vec.shape[0]


def _clamp(requested: Tuple[int, ...], available: int, label: str) -> Tuple[int, ...]:
    clamped = tuple(min(k, available) for k in requested)
    if clamped != requested:
        warnings.warn(
            f"[data_prep] {label}: requested neuron counts {requested} clamped to "
            f"{clamped} -- only {available} curated (selected_neurons) units "
            f"available for this region/session; not padded with synthetic neurons."
        )
    return clamped


def _order_by_activity(zscored: np.ndarray, trial_idx: np.ndarray, rng: np.random.Generator,
                        top_fraction: float = ACTIVITY_TOP_FRACTION) -> np.ndarray:
    """Neuron order for `.neurons(k)`: the more active half of this region's
    units, themselves in random order, followed by the rest, also in random
    order -- so a low-firing neuron never becomes the single "example
    neuron" steps lead with just because it happened to draw position 0 in a
    flat random permutation, while every later "grow to N neurons" beat
    still draws from a random -- not hand-picked -- order.

    Activity is each neuron's 20th-percentile per-trial std (not the mean,
    and not the std pooled across all trials at once): a neuron can pool a
    high overall std from two or three big trials while sitting flat in the
    rest, which looks fine as a static summary but goes dead for multi-second
    stretches once that neuron scrolls across all of them -- the percentile
    instead asks for a trial that's still clearly active even on a below-
    average trial, which is what a *scrolling* multi-trial display needs."""
    sub = zscored[np.ix_(np.asarray(trial_idx), np.arange(zscored.shape[1]), np.arange(zscored.shape[2]))]
    per_trial_std = sub.std(axis=2)                          # (n_trials, n_neurons)
    activity = np.percentile(per_trial_std, 20, axis=0)      # (n_neurons,)
    n = len(activity)
    n_keep = max(1, int(np.ceil(n * top_fraction)))
    active_idx = np.argsort(-activity)[:n_keep]
    rest_idx = np.setdiff1d(np.arange(n), active_idx, assume_unique=True)
    return np.concatenate([rng.permutation(active_idx), rng.permutation(rest_idx)])


def _region_cache_path(region: str) -> Path:
    return CACHE_DIR / f"{SESSION_NAME}_{TRIAL_TYPE}_{ALIGN_MODE}_{region}.npz"


def _load_region_zscored(session_path: Path, region: str, use_cache: bool = True) -> np.ndarray:
    cache_path = _region_cache_path(region)
    if use_cache and cache_path.exists():
        return np.load(cache_path)["zscored"]

    raw = _load_region_raw(session_path, region)
    T_raw = raw.shape[2]
    time_vec_full = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T_raw)
    cropped, _ = _crop_time(raw, time_vec_full, DISPLAY_WINDOW_S)
    zscored = _zscore_full(cropped).astype(np.float32)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, zscored=zscored)
    return zscored


def load_pipeline_data(use_cache: bool = True) -> PipelineBundle:
    session_path = BASE_DIR / mat_subdir_name(TRIAL_TYPE, ALIGN_MODE) / f"{SESSION_NAME}_analysis_results.mat"
    if not session_path.exists():
        raise FileNotFoundError(session_path)

    zscored_a = _load_region_zscored(session_path, REGION_A, use_cache)
    zscored_b = _load_region_zscored(session_path, REGION_B, use_cache)
    zscored_o = _load_region_zscored(session_path, OTHER_REGION, use_cache)
    region_other_multi = load_other_regions_multi(session_path, exclude=(REGION_A, REGION_B), use_cache=use_cache)

    n_trials_full = min(zscored_a.shape[0], zscored_b.shape[0], zscored_o.shape[0],
                         region_other_multi.zscored.shape[0])
    zscored_a = zscored_a[:n_trials_full]
    zscored_b = zscored_b[:n_trials_full]
    zscored_o = zscored_o[:n_trials_full]
    region_other_multi.zscored = region_other_multi.zscored[:n_trials_full]
    if n_trials_full <= TRIAL_START_IDX + N_TRIALS_DEMO:
        raise ValueError(
            f"session {SESSION_NAME!r} has only {n_trials_full} trials; need at least "
            f"{TRIAL_START_IDX + N_TRIALS_DEMO} for TRIAL_START_IDX={TRIAL_START_IDX}, "
            f"N_TRIALS_DEMO={N_TRIALS_DEMO}."
        )
    trials_demo = np.arange(TRIAL_START_IDX, TRIAL_START_IDX + N_TRIALS_DEMO)

    n_avail_a = zscored_a.shape[1]
    n_avail_b = zscored_b.shape[1]
    n_avail_o = zscored_o.shape[1]
    steps_a = _clamp(N_REGION_A_STEPS, n_avail_a, REGION_A)
    steps_o = _clamp(N_OTHER_REGION_STEPS, n_avail_o, OTHER_REGION)
    steps_b = _clamp((N_LATENT_NEURONS,) * 3, n_avail_b, REGION_B)
    if N_LATENT_NEURONS > min(steps_a[1], steps_o[1], n_avail_b):
        raise ValueError("N_LATENT_NEURONS must be <= Region A's / the other region's step-2 "
                          "neuron count and Region B's available neuron count")

    rng = np.random.default_rng(RANDOM_SEED)
    order_a = _order_by_activity(zscored_a, trials_demo, rng)
    # Swap in the neuron that otherwise sorts to position 1: regressing out
    # every other region removes ~36% of its variance with the residual
    # still clearly fluctuating (std 0.90 vs 1.12), vs. ~13% for the neuron
    # that would otherwise lead -- a much clearer residualize() example.
    order_a[[0, 1]] = order_a[[1, 0]]
    order_b = _order_by_activity(zscored_b, trials_demo, rng)
    order_o = _order_by_activity(zscored_o, trials_demo, rng)

    T = zscored_a.shape[2]
    time_vec = np.linspace(DISPLAY_WINDOW_S[0], DISPLAY_WINDOW_S[1], T)

    region_a = RegionData(REGION_A, zscored_a, order_a, steps_a, n_avail_a)
    region_b = RegionData(REGION_B, zscored_b, order_b, steps_b, n_avail_b)
    region_other = RegionData(OTHER_REGION, zscored_o, order_o, steps_o, n_avail_o)

    fs = (T - 1) / (DISPLAY_WINDOW_S[1] - DISPLAY_WINDOW_S[0])
    return PipelineBundle(
        session=SESSION_NAME,
        trial_type=TRIAL_TYPE,
        align_mode=ALIGN_MODE,
        region_a=region_a,
        region_b=region_b,
        region_other=region_other,
        region_other_multi=region_other_multi,
        time_vec=time_vec,
        fs=fs,
        n_trials_full=n_trials_full,
        trial_single=np.array([TRIAL_START_IDX]),
        trials_demo=trials_demo,
    )


if __name__ == "__main__":
    bundle = load_pipeline_data()
    print(f"session={bundle.session}  trials_full={bundle.n_trials_full}  T={bundle.T}  fs={bundle.fs:.1f} Hz")
    print(f"Region A ({bundle.region_a.name}): {bundle.region_a.n_neurons_available} neurons, "
          f"step counts {bundle.region_a.step_counts}")
    print(f"Region B ({bundle.region_b.name}, step 7 only): {bundle.region_b.n_neurons_available} neurons")
    print(f"Other region ({bundle.region_other.name}): {bundle.region_other.n_neurons_available} neurons, "
          f"step counts {bundle.region_other.step_counts}")

    X = select_flat(bundle.region_a.zscored, bundle.trials_demo, bundle.region_a.neurons(1))
    Z = select_flat(bundle.region_other.zscored, bundle.trials_demo, bundle.region_other.neurons(1))
    resid, coeff = residualize_with_beta(X, Z)
    print(f"step2 sanity: X{X.shape} Z{Z.shape} -> resid{resid.shape} beta{coeff.shape} beta_1={coeff[0, 0]:.4f}")

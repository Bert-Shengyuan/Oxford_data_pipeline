#!/usr/bin/env python3
r"""
pCCA_all_regions_hubmode_explain_variable_v2.py
================================================================================

v2 counterpart of ``pCCA_all_regions_hubmode_explain_variable.py`` ("v1"):
same Tasks 3/4/5/6 (behavioural-variance bars, cross-session latent traces,
subregion/laminar enrichment boxplots), reading from
``pCCA_all_regions_out_behaviour_v2.py``'s pickles instead of v1's --
PLUS two NEW tasks, 7 and 8, that visualise the top-pCCA-weight neurons v2
already identifies and saves.

--------------------------------------------------------------------------------
Why a separate script, and what actually changes
--------------------------------------------------------------------------------
``pCCA_all_regions_out_behaviour_v2.py`` fits Part 2c/2c' (and 2a/2b) on
``N_SAMPLE_DRAWS`` (10) independently RESAMPLED neuron draws per pair, per
session, instead of v1's single fixed-neuron-set fit -- so every quantity
this script's v1 counterpart reads as ONE value per session is now 10
values per session (``PrivateLatentPairResult.draws``, one
``PrivateLatentPairDrawResult`` per draw). Three different things happen
to that "x10" depending on the task (per this revision's request):

    Tasks 3/4  (R^2 bars)              -- compute R^2 PER DRAW (10 ridge
                                           fits per session, same as
                                           before, just repeated), then
                                           AVERAGE the 10 R^2 SCALARS into
                                           ONE value per session -- so
                                           `aggregate_behavior_variance` /
                                           `hubmode_plot_task3_bars` /
                                           `hubmode_plot_task4_bars` /
                                           `hubmode_plot_multipanel_bars`
                                           are copied UNCHANGED from v1;
                                           only the per-session INPUT is
                                           now a 10-draw average instead
                                           of a single fit.
    Task 5     (latent traces)         -- the light per-session lines now
                                           show EVERY trial of EVERY draw
                                           (10x more lines than v1's
                                           one-line-per-session), while the
                                           dark cross-session mean line is
                                           unchanged in FORMULA -- it is
                                           still `CrossSessionCCAAnalyzer`'s
                                           own mean-of-session-means -- only
                                           each session's own mean is now
                                           computed over 10x more
                                           (draw, trial) samples than
                                           before. See "Task 5" below for
                                           how the per-session sign flip
                                           `CrossSessionCCAAnalyzer`
                                           computes (but does not expose)
                                           is recovered for the light lines.
    Task 6     (enrichment boxplots)   -- same average-the-10-draws-into-
                                           one-session-value pattern as
                                           Tasks 3/4, applied to each
                                           group's `enrichment_ratio`
                                           instead of an R^2.

Tasks 7 & 8 are NEW: v2 also identifies, per pair, per session, per
region side, the neurons whose pCCA weight fell in the top
`TOP_WEIGHT_FRACTION` (20%) pooled across the 10 draws, deduplicated, with
their residualized activity already saved
(``PrivateLatentPairResult.selected_neurons_i`` / ``_j``, a
``SelectedNeuronSet``). Task 7 pools these neurons' ORIGINAL (pre-
residualization, z-scored) firing-rate PSTH across every session that
contributed any, one heatmap per (hub, partner) pairing; Task 8 is the
same layout with RESIDUALIZED activity instead (already saved, no reload
needed). Like Tasks 3-6, Tasks 7/8 sweep every hub in
``HUB_MODE_HUB_REGIONS`` (one figure per hub), not a single hard-coded
hub. See Section 10 below for the full design, including why Rastermap is
now fit ONCE on the fully pooled (all sessions, all selected neurons)
trial-averaged PSTH matrix rather than per session block (sessions
generally have different trial counts, so their raw continuous-cross-trial
traces are not directly comparable in one joint Rastermap fit -- only the
trial-AVERAGED PSTH, sharing one common T, can be pooled across sessions
BEFORE the single Rastermap fit).

Everything else -- data source (``PrivateLatentAnalyzer`` reading v2's own
``pcca_all_regions_out_behaviour_v2_sampled_sessions_{trial_type}_
{align_mode}_results`` pickles), the two pCCA regress-out variants
(``PCCA_VARIANTS`` = 'regions_only' / 'regions_behavior', i.e. ``.pairs``
vs ``.pairs_regions_only``), the reward-kernel/B-spline machinery, the
hub-mode row/panel layout, and every plot's visual styling -- is copied
verbatim from v1's own file, per this revision's "keep the bar plot
style/display style unchanged" instructions for Tasks 3/4/6.

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import csv
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import BSpline
from scipy.stats import zscore

warnings.filterwarnings('ignore')

try:
    from rastermap import Rastermap
    _RASTERMAP_OK = True
except ImportError:
    _RASTERMAP_OK = False
    warnings.warn(
        "rastermap not found; Tasks 7/8 fall back to peak-time neuron ordering."
    )

# =============================================================================
# 0.  Imports. `cross_trial_type_cca_analysis` is the same stable library
#     module v1 uses for Task 5's sign-aligned cross-session aggregator.
#     `pCCA_all_regions_out_behaviour_v2.py` is this script's ONLY source
#     of neural data (Tasks 3-6/pCCA weights) and ALSO the source of the
#     raw-reload primitives Task 7 needs (`load_region_spikes_full` /
#     `crop_time_window` / `_zscore_flat` / `load_behavior_regressors`) --
#     the SAME functions v2 itself used to build the region_flat_full pool
#     `SelectedNeuronSet.neurons[].neuron_idx` indexes into, so reloading
#     with them (rather than reimplementing the load/crop/truncate
#     sequence a second, possibly-diverging way) guarantees the neuron
#     axis lines up. Task 8 needs no such reload -- its data
#     (`SelectedNeuronResidual.residual`) is already saved.
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_trial_type_cca_analysis import (   # noqa: E402
    CrossSessionCCAAnalyzer,
    TRIAL_TYPE_COLORS,
    MIN_SESSIONS_THRESHOLD,
    align_signs_spectral,
)
from pCCA_all_regions_out_behaviour_v2 import (  # noqa: E402
    PrivateLatentAnalyzer,
    PrivateLatentSessionResult,
    PrivateLatentPairResult,
    PrivateLatentPairDrawResult,
    RegionPCAResult,
    HubOrientationPCADrawResult,
    HubPairPCAResult,
    SubregionWeightMetrics,
    SelectedNeuronSet,
    SelectedNeuronResidual,
    REGION_PAIRS,
    N_COMPONENTS,
    N_SAMPLE_DRAWS,
    TOP_WEIGHT_FRACTION,
    CORTICAL_REGIONS,
    sort_pair_by_anatomy,
    get_anatomical_index,
    out_subdir_name,
    mat_subdir_name as v2_mat_subdir_name,
    load_region_spikes_full as v2_load_region_spikes_full,
    crop_time_window as v2_crop_time_window,
    _zscore_flat as v2_zscore_flat,
    load_behavior_regressors as v2_load_behavior_regressors,
    BASE_DIR as V2_BASE_DIR,
    SUBTRACT_PSTH as V2_SUBTRACT_PSTH,
    SHUFFLE_TRIALS as V2_SHUFFLE_TRIALS,
)

# `pCCA_all_regions_out_behaviour_v2.py` bakes '__main__' into every pickled
# dataclass instance's module reference (it is normally *run* directly);
# unpickling those files from THIS script's own '__main__' therefore needs
# the same classes reachable under `__main__` here too -- every dataclass
# that can appear nested inside a pickled `PrivateLatentSessionResult`
# needs registering, not just the top-level one.
for _cls in (
        PrivateLatentSessionResult, PrivateLatentPairResult,
        PrivateLatentPairDrawResult, RegionPCAResult,
        HubOrientationPCADrawResult, HubPairPCAResult,
        SubregionWeightMetrics, SelectedNeuronSet, SelectedNeuronResidual,
):
    setattr(sys.modules['__main__'], _cls.__name__, _cls)

try:
    import mat73  # noqa: F401  (transitively required by cross_trial_type_cca_analysis's own imports)
except Exception:
    warnings.warn("mat73 not importable -- cross_trial_type_cca_analysis.py may fail to "
                  "import; this script's own Task 7 reload also needs it.")


# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS -- copied from v1 unless noted.
# =============================================================================

REFERENCE_TYPE: str = 'cued_hit_long'
ACTIVE_TRIAL_TYPES: List[str] = [
    'cued_hit_long',
    # 'spont_hit_long',
    # 'spont_miss_long',
]
HUB_MODE_ENRICHMENT_YLIM_C: Optional[Tuple[float, float]] = [-0.05, 1.8]
HUB_MODE_ENRICHMENT_YLIM_SC: Optional[Tuple[float, float]] = [0, 2.1]

ALIGN_MODE: str = 'default_move_onset'  # default_move_onset | cue_onset | ...
Align_type_value = ALIGN_MODE.replace("_", " ")
if ALIGN_MODE == 'default_move_onset':
    Align_type_value = 'Move onset'

ALIGNMENT_WINDOWS_S: Dict[str, Tuple[float, float]] = {
    "default_move_onset": (-1.0, 2.0),
    "cue_onset":           (-0.8, 2.2),
    "bar_off_onset":       (-2.0, 1.0),
    "reward_onset":        (-1.2, 1.8),
}

TASK3_ALIGN_MODES: Tuple[str, ...] = ("default_move_onset",)

# ---- Paths ------------------------------------------------------------
BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
BEHAVIOR_DIR = BASE_DIR / "Paper_output" / f"tapproach_sessions_{ALIGN_MODE}"
OUTPUT_DIR = (BASE_DIR / "Paper_output"
              / f"pcca_all_regions_hubmode_v2_{REFERENCE_TYPE}_{ALIGN_MODE}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Sessions -----------------------------------------------------------
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

COMPONENT_INDICES: List[int] = [0]
MIN_SESSIONS: int = MIN_SESSIONS_THRESHOLD

PCCA_VARIANTS: Tuple[str, ...] = ('regions_only', 'regions_behavior')
VARIANT_DISPLAY: Dict[str, str] = {
    'regions_only':     'AllRegions only',
    'regions_behavior': 'AllRegions + Behaviour',
}
EXTERNAL_VARIABLES_BY_VARIANT: Dict[str, List[str]] = {
    'regions_behavior': ['reward_presence', 'reward_consumption'],
    'regions_only':      ['position', 'speed', 'reward_presence', 'reward_consumption'],
}
EXTERNAL_VARIABLES: List[str] = EXTERNAL_VARIABLES_BY_VARIANT['regions_behavior']

BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = ALIGNMENT_WINDOWS_S[ALIGN_MODE]
LAMBDA_R2: float = 1e-4
VARIANCE_METHOD: str = 'marginal'  # 'marginal' | 'leave_one_out'

REWARD_PRESENCE_WINDOW_S: Tuple[float, float] = (0.0, 0.5)
REWARD_CONSUMPTION_WINDOW_S: Tuple[float, float] = (0.5, 1.5)
REWARD_CONSUMPTION_DURATION_S: float = 1.0
N_REWARD_CONSUMPTION_BASIS: int = 7
REWARD_SPLINE_DEGREE: int = 2

HUB_MODE_HUB_REGIONS: List[str] = ['MOs', 'MOp', 'VALVM', 'VPMPO']
HUB_MODE_ROI_REGIONS: List[str] = sorted(
    {r for pair in REGION_PAIRS for r in pair}, key=get_anatomical_index)

HUB_MODE_BAND_COLORS: Dict[str, str] = {
    'MOs':   '#DD8452',
    'MOp':   '#55A868',
    'VALVM': '#4C72B0',
    'VPMPO': '#C44E52',
}

EXTERNAL_VAR_COLORS: Dict[str, str] = {
    'position':           "#DE6E4B",
    'speed':              "#4B7DDE",
    'reward_presence':    "#55A868",
    'reward_consumption': "#B07AA1",
}
BAR_LEN: float = 0.2
HUB_MODE_BAR_XLIM_SINGLE_TRIAL: Dict[str, Tuple[float, float]] = {
    'position':           (0.0, BAR_LEN),
    'speed':              (0.0, BAR_LEN),
    'reward_presence':    (0.0, BAR_LEN),
    'reward_consumption': (0.0, BAR_LEN),
}
HUB_MODE_BAR_XLIM_TRIAL_AVG: Dict[str, Tuple[float, float]] = {
    'position':           (0.0, 0.75),
    'speed':              (0.0, 0.75),
    'reward_presence':    (0.0, 0.75),
    'reward_consumption': (0.0, 0.75),
}

ENRICHMENT_GROUP_PALETTE: List[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#64B5CD", "#CCB974",
]
ENRICHMENT_BOX_WIDTH: float = 0.6
ENRICHMENT_DOT_JITTER: float = 0.16

LAMINAR_GROUP_ORDER: List[str] = ['layer-shallow', 'layer-deep']
LAMINAR_GROUP_DISPLAY_NAMES: Dict[str, str] = {
    'layer-shallow': 'Superficial',
    'layer-deep':    'Deep',
}

DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
}


def _display_name(region: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(region, region)


def _lighten(hex_color: str, amount: float = 0.45) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


DOT_COLOR = "#262626"
BAR_HEIGHT = 0.62
CLUSTER_GAP = 0.55
ROW_GAP = 0.15
DOT_JITTER_FRAC = 0.35
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 15
CLUSTER_HATCH_CYCLE = [None, "///", "xxx"]

SAVE_DPI: int = 400

# ---- Tasks 7/8 (NEW) -- top-pCCA-weight neuron PSTH heatmaps ------------
# Sweeps every hub in HUB_MODE_HUB_REGIONS (same hub-mode band layout
# Tasks 3-6 use, via `hubmode_band_pairs()`), one figure per hub -- NOT
# limited to a single hard-coded hub.

TASK78_TRIAL_TYPE: str = REFERENCE_TYPE
TASK78_PANEL_WIDTH: float = 3.4
TASK78_PANEL_HEIGHT: float = 5.2
# Rastermap fit knobs -- same defaults this project already uses
# (pCCA_sensitive_realsingle_Session_11panel.py's own get_neuron_order).
TASK78_RASTERMAP_KW: Dict = dict(locality=0.0, time_lag_window=10, grid_upsample=10)


# =============================================================================
# 2.  Anatomical / hub-mode helpers -- copied verbatim from v1.
# =============================================================================

def hubmode_band_pairs(
        hub_regions: List[str] = HUB_MODE_HUB_REGIONS,
        roi_regions: List[str] = HUB_MODE_ROI_REGIONS,
) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """One band per hub region (anatomically ordered), each listing every
    (hub, partner) row for every OTHER region in `roi_regions`."""
    ordered_hubs = sorted(dict.fromkeys(hub_regions), key=get_anatomical_index)
    ordered_rois = sorted(dict.fromkeys(roi_regions), key=get_anatomical_index)
    return [
        (hub, [(hub, partner) for partner in ordered_rois if partner != hub])
        for hub in ordered_hubs
    ]


def _hub_region_role(hub: str, pair_key: Tuple[str, str]) -> str:
    """Which of a canonical pair's two slots `hub` occupies."""
    return 'region_i' if pair_key[0] == hub else 'region_j'


# =============================================================================
# 3.  Behavioural regressors + variance-explained primitives -- copied
#     verbatim from v1 (Tasks 3/4's own machinery; nothing about the
#     10-draws change touches this section -- it operates on whichever
#     single (n_trials, T) latent it is handed, one draw at a time).
# =============================================================================

def load_behavior_regressors(
        session_name: str,
        behavior_dir: Path = BEHAVIOR_DIR,
        trial_label: str = "cued hit long",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load per-trial position (x, y, z), speed, and (align_mode ==
    'default_move_onset' only) each trial's own reward-onset time for one
    session -- copied verbatim from v1's own function of the same name."""
    pkl_path = Path(behavior_dir) / f"{session_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Behaviour file not found: {pkl_path}")
    with open(pkl_path, "rb") as fh:
        session_data = pickle.load(fh)

    pos     = np.asarray(session_data["pos"])
    speed   = np.asarray(session_data["speed"])
    labels  = np.asarray(session_data["task_label"], dtype=object)
    t_behav = np.asarray(session_data["time"], dtype=np.float64)
    trigger_times = session_data.get("trigger_times")

    if speed.ndim == 2:
        speed = speed[:, None, :]

    sel = (labels == trial_label)
    if not np.any(sel):
        available = sorted(set(labels.tolist()))
        raise ValueError(f"No behaviour trials labelled '{trial_label}' "
                         f"(available: {available}).")

    pos_sel   = pos[sel].astype(np.float32)
    speed_sel = speed[sel].astype(np.float32)
    reward_onset_sel = None
    if trigger_times is not None and "reward_onset" in trigger_times:
        reward_onset_sel = np.asarray(trigger_times["reward_onset"], dtype=np.float64)[sel]
    return pos_sel, speed_sel, t_behav, reward_onset_sel


def _trial_type_to_behavior_label(trial_type: str) -> str:
    return trial_type.replace('_', ' ')


def _load_behavior_safe(
        session_name: str, trial_type: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    label = _trial_type_to_behavior_label(trial_type)
    try:
        return load_behavior_regressors(session_name, trial_label=label)
    except (FileNotFoundError, ValueError) as exc:
        warnings.warn(f"[{session_name}/{trial_type}] behaviour unavailable ({exc}).")
        return None


def variance_explained(latent_2d: np.ndarray, design_3d: np.ndarray,
                       lam: float = LAMBDA_R2) -> float:
    """Ridge R^2 for a latent (n_trials, T) explained by a behavioural
    design (n_trials, C, T) -- copied verbatim from v1."""
    n_tr, T = latent_2d.shape
    ell = latent_2d.reshape(-1).astype(np.float64)
    Z = np.transpose(design_3d, (0, 2, 1)).reshape(n_tr * T, -1).astype(np.float64)
    finite = np.isfinite(ell) & np.all(np.isfinite(Z), axis=1)
    if finite.sum() < (Z.shape[1] + 2):
        return 0.0
    ell = ell[finite]
    Z = Z[finite]
    ell_c = ell
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
    """Unique (leave-one-out, 'no-refit') R^2 per predictor block -- copied
    verbatim from v1."""
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


def _bspline_basis_matrix(t: np.ndarray, n_basis: int, degree: int = REWARD_SPLINE_DEGREE) -> np.ndarray:
    """Cubic B-spline basis (matching R's bs()) evaluated at t -- copied
    verbatim from v1."""
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
        c[i] = 0.5
        spline = BSpline(knots, c, degree, extrapolate=False)
        basis[i] = np.nan_to_num(spline(t), nan=0.0)
    return basis[1:-1, :]


def build_reward_presence_design(
        t_behav: np.ndarray, n_trials: int,
        presence_window: Tuple[float, float] = REWARD_PRESENCE_WINDOW_S,
        reward_onset_per_trial: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(n_trials, 1, T) step-function 'reward presence' regressor -- copied
    verbatim from v1."""
    if reward_onset_per_trial is not None:
        out = np.zeros((n_trials, 1, t_behav.size), dtype=float)
        for i in range(n_trials):
            r = reward_onset_per_trial[i]
            if np.isfinite(r):
                out[i, 0] = (t_behav >= 0.0) & (t_behav <= r)
        return out
    presence = ((t_behav >= presence_window[0]) & (t_behav <= presence_window[1])).astype(float)
    return np.tile(presence[None, None, :], (n_trials, 1, 1))


def build_reward_consumption_design(
        t_behav: np.ndarray, n_trials: int,
        consumption_window: Tuple[float, float] = REWARD_CONSUMPTION_WINDOW_S,
        n_basis: int = N_REWARD_CONSUMPTION_BASIS,
        reward_onset_per_trial: Optional[np.ndarray] = None,
        consumption_duration: float = REWARD_CONSUMPTION_DURATION_S,
) -> np.ndarray:
    """(n_trials, n_basis, T) cubic B-spline 'reward consumption' kernel --
    copied verbatim from v1."""
    if reward_onset_per_trial is not None:
        out = np.zeros((n_trials, n_basis, t_behav.size), dtype=float)
        for i in range(n_trials):
            r = reward_onset_per_trial[i]
            if not np.isfinite(r):
                continue
            lo, hi = r, r + consumption_duration
            mask = (t_behav >= lo) & (t_behav <= hi)
            if mask.sum() >= (n_basis + REWARD_SPLINE_DEGREE + 1):
                out[i][:, mask] = _bspline_basis_matrix(t_behav[mask], n_basis)
        return out
    lo, hi = consumption_window
    mask = (t_behav >= lo) & (t_behav <= hi)
    consumption = np.zeros((n_basis, t_behav.size))
    if mask.sum() >= (n_basis + REWARD_SPLINE_DEGREE + 1):
        consumption[:, mask] = _bspline_basis_matrix(t_behav[mask], n_basis)
    return np.tile(consumption[None, :, :], (n_trials, 1, 1))


def _build_predictor_designs(
        pos: np.ndarray, speed: np.ndarray, t_behav: np.ndarray, n_trials: int,
        reward_onset_per_trial: Optional[np.ndarray] = None,
        external_variables: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    variables = external_variables if external_variables is not None else EXTERNAL_VARIABLES
    out: Dict[str, np.ndarray] = {}
    if 'position' in variables:
        out['position'] = pos
    if 'speed' in variables:
        out['speed'] = speed
    if 'reward_presence' in variables:
        out['reward_presence'] = build_reward_presence_design(
            t_behav, n_trials, reward_onset_per_trial=reward_onset_per_trial)
    if 'reward_consumption' in variables:
        out['reward_consumption'] = build_reward_consumption_design(
            t_behav, n_trials, reward_onset_per_trial=reward_onset_per_trial)
    return out


def _sem(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _prepare_regression_inputs(
        latent: np.ndarray, time_bins: np.ndarray,
        pos: np.ndarray, speed: np.ndarray, t_behav: np.ndarray,
        reward_onset_per_trial: Optional[np.ndarray] = None,
        external_variables: Optional[List[str]] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
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

    reward_onset_c = reward_onset_per_trial[:n] if reward_onset_per_trial is not None else None
    if reward_onset_c is not None:
        valid = np.isfinite(reward_onset_c)
        if valid.sum() < 3:
            return None
        L, P, S, reward_onset_c = L[valid], P[valid], S[valid], reward_onset_c[valid]

    design_dict = _build_predictor_designs(
        P, S, t_win, n_trials=L.shape[0],
        reward_onset_per_trial=reward_onset_c, external_variables=external_variables)
    return L, design_dict, t_win


def _prepare_averaged_regression_inputs(
        latent: np.ndarray, time_bins: np.ndarray,
        pos_full: np.ndarray, speed_full: np.ndarray, t_behav_full: np.ndarray,
        reward_onset_per_trial: Optional[np.ndarray] = None,
        external_variables: Optional[List[str]] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
    lo, hi = BEHAVIOR_TIME_RANGE_S
    neural_mask = (time_bins >= lo - 1e-6) & (time_bins <= hi + 1e-6)
    behav_mask = (t_behav_full >= lo - 1e-6) & (t_behav_full <= hi + 1e-6)
    if neural_mask.sum() < 2 or behav_mask.sum() < 2:
        return None

    L = latent[:, neural_mask]
    P = pos_full[:, :, behav_mask]
    S = speed_full[:, :, behav_mask]
    t_win = t_behav_full[behav_mask]

    n = min(L.shape[0], P.shape[0])
    T = min(L.shape[1], P.shape[2], t_win.size)
    if n < 1 or T < 2:
        return None
    L = L[:n, :T]
    P = P[:n, :, :T]
    S = S[:n, :, :T]
    t_win = t_win[:T]

    reward_onset_c = reward_onset_per_trial[:n] if reward_onset_per_trial is not None else None
    if reward_onset_c is not None:
        valid = np.isfinite(reward_onset_c)
        if not valid.any():
            return None
        L, P, S, reward_onset_c = L[valid], P[valid], S[valid], reward_onset_c[valid]

    L_avg = L.mean(axis=0, keepdims=True)
    P_avg = P.mean(axis=0, keepdims=True)
    S_avg = S.mean(axis=0, keepdims=True)
    reward_onset_avg = (np.array([float(np.median(reward_onset_c))])
                         if reward_onset_c is not None else None)

    design_dict = _build_predictor_designs(
        P_avg, S_avg, t_win, n_trials=1,
        reward_onset_per_trial=reward_onset_avg, external_variables=external_variables)
    return L_avg, design_dict, t_win


def _r2_by_var_for_latent(
        latent: np.ndarray, time_bins: np.ndarray,
        pos_full: np.ndarray, speed_full: np.ndarray, t_behav_full: np.ndarray,
        reward_onset_full: Optional[np.ndarray] = None,
        external_variables: Optional[List[str]] = None,
) -> Optional[Dict[str, float]]:
    prep = _prepare_regression_inputs(
        latent, time_bins, pos_full, speed_full, t_behav_full,
        reward_onset_per_trial=reward_onset_full, external_variables=external_variables)
    if prep is None:
        return None
    latent_c, design_dict, t_win = prep
    if VARIANCE_METHOD == 'marginal':
        return {name: variance_explained(latent_c, d) for name, d in design_dict.items()}
    elif VARIANCE_METHOD == 'leave_one_out':
        return variance_explained_unique_loo(latent_c, design_dict)
    raise ValueError(f"Unknown VARIANCE_METHOD: {VARIANCE_METHOD!r}")


def _r2_by_var_for_averaged_latent(
        trials: np.ndarray, comp_idx: int, time_bins: np.ndarray,
        pos_full: np.ndarray, speed_full: np.ndarray, t_behav_full: np.ndarray,
        reward_onset_full: Optional[np.ndarray] = None,
        external_variables: Optional[List[str]] = None,
) -> Optional[Dict[str, float]]:
    if comp_idx >= trials.shape[2]:
        return None
    latent = trials[:, :, comp_idx]
    prep = _prepare_averaged_regression_inputs(
        latent, time_bins, pos_full, speed_full, t_behav_full,
        reward_onset_per_trial=reward_onset_full, external_variables=external_variables)
    if prep is None:
        return None
    latent_c, design_dict, t_win = prep
    if VARIANCE_METHOD == 'marginal':
        return {name: variance_explained(latent_c, d) for name, d in design_dict.items()}
    elif VARIANCE_METHOD == 'leave_one_out':
        return variance_explained_unique_loo(latent_c, design_dict)
    raise ValueError(f"Unknown VARIANCE_METHOD: {VARIANCE_METHOD!r}")


class _PrivateLatentSessionAdapter:
    """Minimal duck-typed stand-in for CrossTrialTypeCCAAnalyzer -- copied
    verbatim from v1."""
    def __init__(self, projections: Dict[str, Dict[str, np.ndarray]], time_bins: np.ndarray):
        self.projections = projections
        self.statistical_results: Dict = {}
        self.time_bins = time_bins


# =============================================================================
# 4.  Data gathering -- ONE pass over (session, hub-mode pair, trial type,
#     PCCA_VARIANT), now reading `PrivateLatentPairResult.draws` (10
#     items) instead of a single fixed-neuron-set fit. Per this revision's
#     request:
#       - Task 5's `per_trial_type[trial_type]['u_mean'/'v_mean']` is the
#         mean over the CONCATENATED (N_SAMPLE_DRAWS * n_trials) block
#         (item 2's "only the underlying per-session data is now 10x
#         larger" -- the dark line's own FORMULA, mean-of-session-means,
#         does not change); `['u_trials'/'v_trials']` is that same
#         concatenated block, now the light-line source Task 5 itself
#         reads (see Section 7).
#       - Tasks 3/4's R^2 is computed ONCE PER DRAW (matching v1's own
#         per-session fit, just repeated on each draw's OWN (n_trials, T)
#         latent -- NOT on the concatenated block, since a ridge fit is
#         not linear in the sample set the way a plain mean is) and then
#         averaged into ONE value per session (item 1).
#       - Task 6's per-group `enrichment_ratio` is likewise averaged
#         across the 10 draws into one value per session (item 3).
#
#     Sign alignment across draws: each of the N_SAMPLE_DRAWS=10 draws is
#     an INDEPENDENT pCCA fit (`pcca()` in pCCA_all_regions_out_behaviour_
#     v2.py only sign-aligns its OWN internal CV folds via a `Wx_ref`
#     dot-product check -- that alignment is local to one draw and does
#     NOT extend across draws), so two draws can land on opposite signs
#     for the same component. Concatenating/averaging `z_i_lat`/`z_j_lat`
#     across draws without correcting for that would let those draws
#     partially cancel instead of reinforcing each other. `_sign_align_
#     and_pool_draws` below fixes this by reusing the SAME Z2 spectral-
#     sync `CrossSessionCCAAnalyzer` already applies across SESSIONS
#     (`align_signs_spectral`, imported from cross_trial_type_cca_
#     analysis.py), applied here across the 10 DRAWS instead -- BEFORE
#     the per-session concatenation/averaging this section performs.
# =============================================================================


def _sign_align_and_pool_draws(
        draws: List[PrivateLatentPairDrawResult],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool every draw's (n_trials, T, K) `z_i_lat`/`z_j_lat` into one
    (N_SAMPLE_DRAWS * n_trials, T, K) array each -- first sign-aligning
    the N_SAMPLE_DRAWS draws PER COMPONENT via `align_signs_spectral`
    (see the section docstring above for why this is needed: each draw is
    an independent pCCA fit with its own unrelated sign ambiguity that
    `pcca()`'s own fold-alignment does not resolve across draws).

    `align_signs_spectral` aligns u (`z_i_lat`) and v (`z_j_lat`)
    independently -- same convention `CrossSessionCCAAnalyzer` itself
    uses when aligning across sessions (separate eigendecompositions of
    the u/v correlation matrices) -- so a draw can be flipped on its u
    side, its v side, both, or neither.
    """
    u_draw_means = np.stack([d.z_i_lat.mean(axis=0) for d in draws], axis=0)  # (n_draws, T, K)
    v_draw_means = np.stack([d.z_j_lat.mean(axis=0) for d in draws], axis=0)
    T = u_draw_means.shape[1]
    _, _, flip_decisions = align_signs_spectral(u_draw_means, v_draw_means, epoch=(0, T))

    u_signed: List[np.ndarray] = []
    v_signed: List[np.ndarray] = []
    for i, d in enumerate(draws):
        u_arr = d.z_i_lat.copy()
        v_arr = d.z_j_lat.copy()
        for comp_idx, decision in flip_decisions[i].items():
            if decision['u_flip']:
                u_arr[:, :, comp_idx] *= -1.0
            if decision['v_flip']:
                v_arr[:, :, comp_idx] *= -1.0
        u_signed.append(u_arr)
        v_signed.append(v_arr)

    return np.concatenate(u_signed, axis=0), np.concatenate(v_signed, axis=0)


def run_hubmode_analysis(
        sessions: List[str] = SESSIONS,
        base_dir: Path = BASE_DIR,
        reference_type: str = REFERENCE_TYPE,
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        align_mode: str = ALIGN_MODE,
        component_indices: List[int] = COMPONENT_INDICES,
        min_sessions: int = MIN_SESSIONS,
) -> Tuple[
    Dict[str, Dict[Tuple[str, str], CrossSessionCCAAnalyzer]],
    List[dict], List[dict],
    Dict[str, PrivateLatentAnalyzer],
]:
    hub_bands = hubmode_band_pairs()
    pair_list = sorted({sort_pair_by_anatomy(hub, partner)
                        for _, rows in hub_bands for hub, partner in rows})

    analyzers_by_trial_type: Dict[str, PrivateLatentAnalyzer] = {
        t: PrivateLatentAnalyzer(base_dir=base_dir, trial_type=t, align_mode=align_mode)
        for t in active_trial_types
    }
    for t, az in analyzers_by_trial_type.items():
        print(f"[load] trial_type={t!r} align_mode={align_mode!r} <- {az.results_dir}")
        az.load_all()

    cross_session_analyzers: Dict[str, Dict[Tuple[str, str], CrossSessionCCAAnalyzer]] = {
        v: {} for v in PCCA_VARIANTS
    }
    behavior_records: List[dict] = []
    subregion_records: List[dict] = []

    all_session_names = sorted(set().union(
        *(az.sessions.keys() for az in analyzers_by_trial_type.values())
    )) if analyzers_by_trial_type else []
    if sessions:
        all_session_names = [s for s in all_session_names if s in sessions]

    for s_idx, session_name in enumerate(all_session_names, 1):
        print("\n" + "=" * 70)
        print(f"[hub-mode v2] SESSION {s_idx}/{len(all_session_names)}: {session_name}")
        print("=" * 70)

        behavior_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]] = {}

        for region_i, region_j in pair_list:
            pair_key = (region_i, region_j)

            for variant in PCCA_VARIANTS:
                regions_only = (variant == 'regions_only')
                external_vars = EXTERNAL_VARIABLES_BY_VARIANT[variant]

                per_trial_type: Dict[str, Dict[str, np.ndarray]] = {}
                time_vec_for_session: Optional[np.ndarray] = None
                pr_by_trial_type: Dict[str, PrivateLatentPairResult] = {}

                for trial_type, az in analyzers_by_trial_type.items():
                    session_result = az.sessions.get(session_name)
                    if session_result is None:
                        continue
                    table = session_result.pairs_regions_only if regions_only else session_result.pairs
                    pr = table.get(pair_key)
                    if pr is None or not pr.draws:
                        continue
                    pr_by_trial_type[trial_type] = pr

                    # ---- Sign-align the N_SAMPLE_DRAWS draws (see section
                    #      docstring), THEN concatenate every draw's per-trial
                    #      latent along the trial axis:
                    #      (N_SAMPLE_DRAWS * n_trials, T, K). Task 5's
                    #      dark-line mean/SEM and light-line pool both come
                    #      from this one array -- see Section 7. -----------
                    u_all, v_all = _sign_align_and_pool_draws(pr.draws)
                    n_tr_total = u_all.shape[0]
                    per_trial_type[trial_type] = dict(
                        u_mean=u_all.mean(axis=0), v_mean=v_all.mean(axis=0),
                        u_trials=u_all, v_trials=v_all,
                        u_std=u_all.std(axis=0), v_std=v_all.std(axis=0),
                        u_sem=u_all.std(axis=0) / np.sqrt(max(n_tr_total, 1)),
                        v_sem=v_all.std(axis=0) / np.sqrt(max(n_tr_total, 1)),
                        n_trials=n_tr_total,
                    )
                    if time_vec_for_session is None or trial_type == reference_type:
                        time_vec_for_session = session_result.time_vec

                if not per_trial_type:
                    continue

                # ---- Task 5 data path ---------------------------------------
                adapter = _PrivateLatentSessionAdapter(
                    projections=per_trial_type, time_bins=time_vec_for_session,
                )
                variant_analyzers = cross_session_analyzers[variant]
                if pair_key not in variant_analyzers:
                    variant_analyzers[pair_key] = CrossSessionCCAAnalyzer(
                        base_dir=str(base_dir), region_pair=pair_key,
                        reference_type=reference_type, n_components=N_COMPONENTS,
                        min_sessions=min_sessions,
                    )
                variant_analyzers[pair_key].add_session_result(
                    session_name, adapter, swap_uv=False,
                )

                # ---- Tasks 3/4 data path: R^2 per DRAW, averaged into ONE
                #      value per (session, pair, role, trial_type,
                #      component, predictor, metric) -- item 1. ------------
                for trial_type, pr in pr_by_trial_type.items():
                    if trial_type not in behavior_cache:
                        behavior_cache[trial_type] = _load_behavior_safe(session_name, trial_type)
                    behav = behavior_cache[trial_type]
                    if behav is None:
                        continue
                    pos_full, speed_full, t_behav_full, reward_onset_full = behav

                    for comp_idx in component_indices:
                        for region_role, region_name, draw_attr in (
                                ('region_i', pair_key[0], 'z_i_lat'),
                                ('region_j', pair_key[1], 'z_j_lat')):

                            single_trial_r2_per_draw: List[Dict[str, float]] = []
                            trial_avg_r2_per_draw: List[Dict[str, float]] = []
                            for d in pr.draws:
                                trials = getattr(d, draw_attr)   # (n_trials, T, K)
                                if comp_idx >= trials.shape[2]:
                                    continue
                                latent = trials[:, :, comp_idx]
                                r2 = _r2_by_var_for_latent(
                                    latent, time_vec_for_session, pos_full, speed_full, t_behav_full,
                                    reward_onset_full=reward_onset_full, external_variables=external_vars)
                                if r2 is not None:
                                    single_trial_r2_per_draw.append(r2)

                                r2_avg = _r2_by_var_for_averaged_latent(
                                    trials, comp_idx, time_vec_for_session, pos_full, speed_full, t_behav_full,
                                    reward_onset_full=reward_onset_full, external_variables=external_vars)
                                if r2_avg is not None:
                                    trial_avg_r2_per_draw.append(r2_avg)

                            if single_trial_r2_per_draw:
                                for var_name in single_trial_r2_per_draw[0]:
                                    mean_r2 = float(np.mean(
                                        [d.get(var_name, 0.0) for d in single_trial_r2_per_draw]))
                                    behavior_records.append(dict(
                                        session=session_name, pair=pair_key, region_role=region_role,
                                        region=region_name, trial_type=trial_type, component=comp_idx,
                                        predictor=var_name, r2=mean_r2, metric='single_trial',
                                        variant=variant, n_draws=len(single_trial_r2_per_draw),
                                    ))
                            if trial_avg_r2_per_draw:
                                for var_name in trial_avg_r2_per_draw[0]:
                                    mean_r2 = float(np.mean(
                                        [d.get(var_name, 0.0) for d in trial_avg_r2_per_draw]))
                                    behavior_records.append(dict(
                                        session=session_name, pair=pair_key, region_role=region_role,
                                        region=region_name, trial_type=trial_type, component=comp_idx,
                                        predictor=var_name, r2=mean_r2, metric='trial_avg',
                                        variant=variant, n_draws=len(trial_avg_r2_per_draw),
                                    ))

                # ---- Task 6 data path: per-group enrichment_ratio,
                #      averaged across the 10 draws into ONE value per
                #      (session, group, ...) -- item 3. --------------------
                for trial_type, pr in pr_by_trial_type.items():
                    for region_role, region_name, metrics_attr in (
                            ('region_i', pair_key[0], 'subregion_weight_metrics_i'),
                            ('region_j', pair_key[1], 'subregion_weight_metrics_j')):
                        for comp_idx in component_indices:
                            per_group_ratios: Dict[str, List[float]] = {}
                            is_cortical_flag: Optional[bool] = None
                            n_resolved_vals: List[int] = []
                            n_total_vals: List[int] = []
                            for d in pr.draws:
                                metrics_list = getattr(d, metrics_attr)
                                if comp_idx >= len(metrics_list):
                                    continue
                                m = metrics_list[comp_idx]
                                is_cortical_flag = m.is_cortical
                                n_resolved_vals.append(m.n_neurons_resolved)
                                n_total_vals.append(m.n_neurons_total)
                                for group, ratio in m.enrichment_ratio.items():
                                    if ratio is None or not np.isfinite(ratio) or ratio < 0:
                                        continue
                                    per_group_ratios.setdefault(group, []).append(float(ratio))
                            for group, ratios in per_group_ratios.items():
                                subregion_records.append(dict(
                                    session=session_name, pair=pair_key, region_role=region_role,
                                    region=region_name, trial_type=trial_type, component=comp_idx,
                                    group=group, enrichment_ratio=float(np.mean(ratios)),
                                    is_cortical=is_cortical_flag,
                                    n_neurons_resolved=(int(round(np.mean(n_resolved_vals)))
                                                        if n_resolved_vals else 0),
                                    n_neurons_total=(int(round(np.mean(n_total_vals)))
                                                    if n_total_vals else 0),
                                    variant=variant, n_draws=len(ratios),
                                ))

    print("\n" + "=" * 70)
    print("[hub-mode v2] CROSS-SESSION AGGREGATION (sign alignment + mean/SEM across sessions)")
    print("=" * 70)
    for variant, variant_analyzers in cross_session_analyzers.items():
        for pair_key, cs in variant_analyzers.items():
            n_sess = len(cs.session_projections)
            if n_sess < min_sessions:
                print(f"  [{variant}] {pair_key[0]} vs {pair_key[1]}: {n_sess} sessions (skipping, < {min_sessions})")
                continue
            cs.aggregate_projections()

    print("\n" + "=" * 70)
    print("[hub-mode v2] DATA-GATHERING COMPLETE")
    for variant in PCCA_VARIANTS:
        print(f"  [{variant}] pairs with >=1 session : {len(cross_session_analyzers[variant])}")
    print(f"  behavioural-variance records   : {len(behavior_records)}")
    print(f"  subregion-ratio records        : {len(subregion_records)}")
    print("=" * 70)
    return cross_session_analyzers, behavior_records, subregion_records, analyzers_by_trial_type


# =============================================================================
# 5.  Aggregation -- copied verbatim from v1 (operates purely on the
#     records list, agnostic to how each record's `r2`/`enrichment_ratio`
#     scalar was derived upstream).
# =============================================================================

def aggregate_behavior_variance(
        behavior_records: List[dict],
) -> Dict[Tuple[Tuple[str, str], str, str], Dict[str, dict]]:
    grouped: Dict[Tuple[Tuple[str, str], str, str, str], List[float]] = {}
    for rec in behavior_records:
        key = (rec['pair'], rec['region_role'], rec['trial_type'], rec['predictor'])
        grouped.setdefault(key, []).append(rec['r2'])

    out: Dict[Tuple[Tuple[str, str], str, str], Dict[str, dict]] = {}
    for (pair_key, region_role, trial_type, predictor), vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size < MIN_SESSIONS:
            continue
        out.setdefault((pair_key, region_role, trial_type), {})[predictor] = dict(
            mean=float(arr.mean()), sem=_sem(arr), values=arr, n=arr.size,
        )
    return out


def aggregate_enrichment_ratio(
        subregion_records: List[dict],
) -> Dict[Tuple[Tuple[str, str], str, str, str], np.ndarray]:
    grouped: Dict[Tuple[Tuple[str, str], str, str, str], List[float]] = {}
    for rec in subregion_records:
        key = (rec['pair'], rec['region_role'], rec['trial_type'], rec['group'])
        grouped.setdefault(key, []).append(rec['enrichment_ratio'])

    out: Dict[Tuple[Tuple[str, str], str, str, str], np.ndarray] = {}
    for key, vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size < MIN_SESSIONS:
            continue
        out[key] = arr
    return out


# =============================================================================
# 6.  Shared hub-mode bar-plotting engine -- copied verbatim from v1.
# =============================================================================

def hubmode_plot_multipanel_bars(
        hub: str,
        partner_rows: List[str],
        clusters_by_panel: Dict[int, Dict[str, List[List[dict]]]],
        panel_titles: List[str],
        panel_xlims: List[Optional[Tuple[float, float]]],
        save_path: Path,
        legend_entries: Optional[List[Tuple[str, str, Optional[str]]]] = None,
        panel_vline_zero: Optional[List[bool]] = None,
        panel_width: float = 4.6,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    def _footprint(clusters: List[List[dict]]) -> float:
        if not clusters:
            return 1.0
        return sum(len(cl) for cl in clusters) + max(0, len(clusters) - 1) * CLUSTER_GAP

    present_rows = [r for r in partner_rows
                    if any(clusters_by_panel.get(p, {}).get(r) for p in clusters_by_panel)]
    if not present_rows:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    y = 0.0
    row_y0: Dict[str, float] = {}
    row_label_pos: Dict[str, float] = {}
    for r in present_rows:
        footprint = max(
            (_footprint(clusters_by_panel.get(p, {}).get(r, [])) for p in clusters_by_panel),
            default=1.0)
        row_y0[r] = y
        row_label_pos[r] = y + footprint / 2.0 - 0.5
        y += footprint + ROW_GAP

    n_panels = len(panel_titles)
    if panel_vline_zero is None:
        panel_vline_zero = [False] * n_panels
    fig_h = max(4.0, 0.42 * y + 2.2)
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_width * n_panels, fig_h), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)

    for panel_idx, (title, ax) in enumerate(zip(panel_titles, axes)):
        ax.axhspan(-BAR_HEIGHT / 2 - 0.25, y - ROW_GAP + BAR_HEIGHT / 2 + 0.25,
                   color=HUB_MODE_BAND_COLORS.get(hub, '#888888'), alpha=0.07, zorder=0)

        means, sems, ys, dot_x, dot_y = [], [], [], [], []
        for r in present_rows:
            yy = row_y0[r]
            clusters = clusters_by_panel.get(panel_idx, {}).get(r, [])
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
        ax.set_title(title, fontsize=TICK_FONTSIZE)
        xlim = panel_xlims[panel_idx] if panel_idx < len(panel_xlims) else None
        if xlim is not None:
            ax.set_xlim(*xlim)
        if panel_vline_zero[panel_idx]:
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
        ax.tick_params(axis='x', labelsize=TICK_FONTSIZE - 2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if panel_idx > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)
        if legend_entries and panel_idx == n_panels - 1:
            handles = [Patch(facecolor=c, label=l, alpha=0.9, hatch=h) for l, c, h in legend_entries]
            ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=False, loc='lower right')

    axes[0].set_yticks(list(row_label_pos.values()))
    axes[0].set_yticklabels([_display_name(r) for r in row_label_pos.keys()], fontsize=TICK_FONTSIZE)
    axes[0].margins(y=0.015)
    axes[0].invert_yaxis()
    fig.suptitle(f"Hub: {_display_name(hub)} \n {Align_type_value}", fontsize=TICK_FONTSIZE)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 7.  Tasks 3 & 4 -- behavioural variance bars, copied verbatim from v1.
#     Both consume `behavior_records`, which now already carries the
#     10-draws-averaged-to-one R^2 per session (Section 4) -- nothing
#     about the plotting code itself needed to change.
# =============================================================================

def hubmode_plot_task3_bars(
        summary: dict,
        summary_trial_avg: dict,
        hub_bands: List[Tuple[str, List[Tuple[str, str]]]],
        output_dir: Path,
        reference_type: str = REFERENCE_TYPE,
        external_variables: List[str] = EXTERNAL_VARIABLES,
        file_suffix: str = '',
) -> Dict[str, Dict[str, Optional[plt.Figure]]]:
    metric_specs = (
        ('single_trial', summary, HUB_MODE_BAR_XLIM_SINGLE_TRIAL, ''),
        ('trial_avg', summary_trial_avg, HUB_MODE_BAR_XLIM_TRIAL_AVG, 'Trialavg'),
    )
    figs: Dict[str, Dict[str, Optional[plt.Figure]]] = {}
    for hub, hub_partner_pairs in hub_bands:
        partner_rows = [p for _, p in hub_partner_pairs]
        figs[hub] = {}
        for metric, src, xlim_by_var, suffix in metric_specs:
            clusters_by_panel: Dict[int, Dict[str, List[list]]] = {
                i: {} for i in range(len(external_variables))
            }
            for partner in partner_rows:
                pair_key = sort_pair_by_anatomy(hub, partner)
                role = _hub_region_role(hub, pair_key)
                per_pred = src.get((pair_key, role, reference_type))
                if per_pred is None:
                    continue
                for panel_idx, v in enumerate(external_variables):
                    if v not in per_pred:
                        continue
                    stats = per_pred[v]
                    bar = dict(mean=stats['mean'], sem=stats['sem'], values=stats['values'],
                              color=EXTERNAL_VAR_COLORS.get(v, 'gray'), alpha=0.9, hatch=None)
                    clusters_by_panel[panel_idx][partner] = [[bar]]
            titles = [v.replace('_', ' ') for v in external_variables]
            xlims = [xlim_by_var[v] for v in external_variables]
            save_path = output_dir / f"{suffix}{file_suffix}_hubmode_task3_variance_{hub}.png"
            figs[hub][metric] = hubmode_plot_multipanel_bars(
                hub, partner_rows, clusters_by_panel, titles, xlims, save_path,
            )
    return figs


def hubmode_plot_task4_bars(
        summary: dict,
        summary_trial_avg: dict,
        hub_bands: List[Tuple[str, List[Tuple[str, str]]]],
        output_dir: Path,
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        reference_type: str = REFERENCE_TYPE,
        task_label: str = 'task4',
) -> Dict[str, Dict[str, Optional[plt.Figure]]]:
    non_ref = [t for t in active_trial_types if t != reference_type]
    metric_specs = (
        ('single_trial', summary, HUB_MODE_BAR_XLIM_SINGLE_TRIAL, ''),
        ('trial_avg', summary_trial_avg, HUB_MODE_BAR_XLIM_TRIAL_AVG, 'Trialavg'),
    )
    legend = [(t.replace('_', ' '), '#bbbbbb', CLUSTER_HATCH_CYCLE[ti % len(CLUSTER_HATCH_CYCLE)])
              for ti, t in enumerate(non_ref)]
    figs: Dict[str, Dict[str, Optional[plt.Figure]]] = {}
    for hub, hub_partner_pairs in hub_bands:
        partner_rows = [p for _, p in hub_partner_pairs]
        figs[hub] = {}
        for metric, src, xlim_by_var, suffix in metric_specs:
            clusters_by_panel: Dict[int, Dict[str, List[list]]] = {
                i: {} for i in range(len(EXTERNAL_VARIABLES))
            }
            for partner in partner_rows:
                pair_key = sort_pair_by_anatomy(hub, partner)
                role = _hub_region_role(hub, pair_key)
                for panel_idx, v in enumerate(EXTERNAL_VARIABLES):
                    clusters: List[list] = []
                    for ti, trial_type in enumerate(non_ref):
                        hatch = CLUSTER_HATCH_CYCLE[ti % len(CLUSTER_HATCH_CYCLE)]
                        per_pred = src.get((pair_key, role, trial_type))
                        if per_pred and v in per_pred:
                            stats = per_pred[v]
                            bar = dict(mean=stats['mean'], sem=stats['sem'],
                                      values=stats['values'],
                                      color=EXTERNAL_VAR_COLORS.get(v, 'gray'),
                                      alpha=0.9, hatch=hatch)
                            clusters.append([bar])
                    if clusters:
                        clusters_by_panel[panel_idx][partner] = clusters
            titles = [v.replace('_', ' ') for v in EXTERNAL_VARIABLES]
            xlims = [xlim_by_var[v] for v in EXTERNAL_VARIABLES]
            save_path = output_dir / f"{suffix}_hubmode_{task_label}_variance_{hub}.png"
            figs[hub][metric] = hubmode_plot_multipanel_bars(
                hub, partner_rows, clusters_by_panel, titles, xlims, save_path,
                legend_entries=legend,
            )
    return figs


# =============================================================================
# 8.  Task 5 -- latent traces across sessions. Layout/styling copied from
#     v1's own `_hubmode_plot_task5_one_hub`: ONE light line per session
#     (its own mean across ALL trials x N_SAMPLE_DRAWS draws combined --
#     e.g. 100 trials x 10 draws -> one line averaged over 1000 values),
#     not one line per individual (draw, trial) sample -- an earlier
#     revision of this function drew every individual sample (up to
#     several thousand per panel: N_SAMPLE_DRAWS x n_trials x n_sessions),
#     and even at a low per-line alpha that many overlapping same-colour
#     lines saturate to full opacity almost immediately (compositing N
#     layers at alpha a leaves only (1-a)^N of the background showing
#     through -- already under 3% at a=0.035, N=100), which is what
#     produced the solid-colour block hiding the dashed t=0 line and the
#     mean trace itself in that revision's plots. Reverting to one line
#     per session removes that failure mode by construction.
#
#     That per-session line is exactly `CrossSessionCCAAnalyzer.
#     aggregate_projections()`'s own `u_sessions`/`v_sessions` -- each
#     session's `u_mean`/`v_mean` (already the mean over the concatenated
#     N_SAMPLE_DRAWS x n_trials pool, built in Section 4 above), sign-
#     aligned by that method's own Z2 spectral sync -- so this needs no
#     separate sign-recovery step; it is simply what v1's own light lines
#     already plotted (`agg[sessions_key]`), now sourced from a 10x
#     larger per-session pool upstream.
# =============================================================================

def _hubmode_plot_task5_one_hub(
        hub: str,
        hub_partner_pairs: List[Tuple[str, str]],
        cross_session_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer],
        save_path: Path,
        component_idx: int,
        active_trial_types: List[str],
        row_height: float,
        fig_width: float,
        dpi: int,
) -> Optional[plt.Figure]:
    rows: List[Tuple[Tuple[str, str], str, str]] = []
    for hub_r, partner in hub_partner_pairs:
        pair_key = sort_pair_by_anatomy(hub_r, partner)
        cs = cross_session_analyzers.get(pair_key)
        if cs is None or not cs.aggregated_projections:
            continue
        role = _hub_region_role(hub_r, pair_key)
        rows.append((pair_key, partner, role))

    if not rows:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, row_height * n_rows), sharex=True)
    axes = np.atleast_1d(axes)

    for r, (pair_key, partner, role) in enumerate(rows):
        ax = axes[r]
        cs = cross_session_analyzers[pair_key]
        ax.set_facecolor(HUB_MODE_BAND_COLORS.get(hub, '#888888'))
        ax.patch.set_alpha(0.001)

        mean_key, sem_key, sessions_key = (
            ('u_mean', 'u_sem', 'u_sessions') if role == 'region_i'
            else ('v_mean', 'v_sem', 'v_sessions')
        )

        n_light_lines_total = 0
        for trial_type in active_trial_types:
            if trial_type not in cs.aggregated_projections:
                continue
            agg = cs.aggregated_projections[trial_type]
            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')

            # ---- Light lines: ONE per session -- that session's own mean
            #      across ALL trials x N_SAMPLE_DRAWS draws combined,
            #      already sign-aligned by aggregate_projections() (see
            #      section docstring). ------------------------------------
            session_traces = agg[sessions_key][:, :, component_idx]
            n_light_lines_total += session_traces.shape[0]
            for sess_trace in session_traces:
                ax.plot(cs.time_bins, sess_trace, color=color, linewidth=0.5,
                        alpha=0.2, zorder=1)

            mean_trace = agg[mean_key][:, component_idx]
            sem_trace = agg[sem_key][:, component_idx]
            is_ref = (trial_type == REFERENCE_TYPE)
            ax.plot(cs.time_bins, mean_trace, color=color, linewidth=2.0 if is_ref else 1.4,
                    alpha=0.9 if is_ref else 0.8,
                    label=f"{trial_type.replace('_', ' ')} (n={agg['n_sessions']} sess)", zorder=3)
            ax.fill_between(cs.time_bins, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=color, alpha=0.18, zorder=2)

        print(f"    [{save_path.stem}] {_display_name(hub)} <-> {_display_name(partner)}: "
              f"{n_light_lines_total} light lines (1/session)")

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
        ax.set_xlim(cs.time_bins[0], cs.time_bins[-1])
        ax.set_ylim(-0.5, 0.75)
        ax.text(0.01, 0.90, f"{_display_name(hub)} ↔ {_display_name(partner)}",
                transform=ax.transAxes, fontsize=TICK_FONTSIZE - 6, va='top', ha='left')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)
        if r == 0:
            ax.legend(fontsize=LEGEND_FONTSIZE - 3, loc='upper right', frameon=False)

    axes[-1].set_xlabel("Time from reach (s)", fontsize=TICK_FONTSIZE)
    axes[-1].tick_params(axis='x', labelsize=TICK_FONTSIZE - 2)
    fig.suptitle(f"Hub: {_display_name(hub)} \n {Align_type_value}", fontsize=TICK_FONTSIZE)

    fig.tight_layout(h_pad=0.15, rect=(0.0, 0.0, 1.0, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


def hubmode_plot_task5_latent_traces(
        hub_bands: List[Tuple[str, List[Tuple[str, str]]]],
        cross_session_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer],
        output_dir: Path,
        component_idx: int = COMPONENT_INDICES[0],
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        row_height: float = 1.5,
        fig_width: float = 5.0,
        dpi: int = SAVE_DPI,
        task_label: str = 'task5',
) -> Dict[str, Optional[plt.Figure]]:
    figs: Dict[str, Optional[plt.Figure]] = {}
    for hub, hub_partner_pairs in hub_bands:
        save_path = output_dir / f"{task_label}_hubmode_latent_traces_comp{component_idx}_{hub}.png"
        figs[hub] = _hubmode_plot_task5_one_hub(
            hub, hub_partner_pairs, cross_session_analyzers, save_path,
            component_idx, active_trial_types, row_height, fig_width, dpi,
        )
    return figs


# =============================================================================
# 9.  Task 6 -- subregion/laminar enrichment-ratio boxplots, copied
#     verbatim from v1. `subregion_records` now already carries the
#     10-draws-averaged-to-one ratio per session (Section 4).
# =============================================================================

def _boxplot_one_panel(
        ax: plt.Axes,
        categories: List[str],
        values_by_category: Dict[str, np.ndarray],
        colors_by_category: Dict[str, str],
        rng: np.random.Generator,
) -> None:
    for j, cat in enumerate(categories):
        vals = values_by_category.get(cat)
        color = colors_by_category.get(cat, '#888888')
        if vals is None or vals.size == 0:
            continue
        ax.boxplot(
            [vals], positions=[j], widths=ENRICHMENT_BOX_WIDTH, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=_lighten(color, 0.55), edgecolor='black', linewidth=1.3),
            medianprops=dict(color='black', linewidth=2.0),
            whiskerprops=dict(color='black', linewidth=1.3),
            capprops=dict(color='black', linewidth=1.3),
        )
        jitter = rng.uniform(-ENRICHMENT_DOT_JITTER, ENRICHMENT_DOT_JITTER, size=vals.size)
        ax.scatter(np.full(vals.size, j) + jitter, vals, s=42, color=color,
                  edgecolor='black', linewidth=0.4, alpha=0.9, zorder=5)


def hubmode_plot_task6_enrichment_boxplots(
        subregion_records: List[dict],
        hub_bands: List[Tuple[str, List[Tuple[str, str]]]],
        output_dir: Path,
        reference_type: str = REFERENCE_TYPE,
        ylim1: Optional[Tuple[float, float]] = HUB_MODE_ENRICHMENT_YLIM_C,
        ylim2: Optional[Tuple[float, float]] = HUB_MODE_ENRICHMENT_YLIM_SC,
        panel_width: float = 3,
        panel_height: float = 5.0,
        dpi: int = SAVE_DPI,
        file_suffix: str = '',
) -> Dict[str, Optional[plt.Figure]]:
    grouped = aggregate_enrichment_ratio(subregion_records)
    figs: Dict[str, Optional[plt.Figure]] = {}
    rng = np.random.default_rng(0)

    for hub, hub_partner_pairs in hub_bands:
        partner_rows = [p for _, p in hub_partner_pairs]
        cortical = hub in CORTICAL_REGIONS

        if cortical:
            categories = list(LAMINAR_GROUP_ORDER)
            cat_labels = [LAMINAR_GROUP_DISPLAY_NAMES[c] for c in categories]
        else:
            seen = set()
            for partner in partner_rows:
                pair_key = sort_pair_by_anatomy(hub, partner)
                role = _hub_region_role(hub, pair_key)
                for (p_key, r, t, group) in grouped:
                    if p_key == pair_key and r == role and t == reference_type:
                        seen.add(group)
            categories = sorted(seen)
            cat_labels = list(categories)

        if not categories:
            print(f"  [plot] nothing to plot for hub {hub} (task 6); skipping.")
            figs[hub] = None
            continue
        colors_by_category = {
            cat: ENRICHMENT_GROUP_PALETTE[i % len(ENRICHMENT_GROUP_PALETTE)]
            for i, cat in enumerate(categories)
        }

        present_partners: List[str] = []
        panel_values: List[Dict[str, np.ndarray]] = []
        for partner in partner_rows:
            pair_key = sort_pair_by_anatomy(hub, partner)
            role = _hub_region_role(hub, pair_key)
            values_by_category = {
                cat: grouped[(pair_key, role, reference_type, cat)]
                for cat in categories
                if (pair_key, role, reference_type, cat) in grouped
            }
            if values_by_category:
                present_partners.append(partner)
                panel_values.append(values_by_category)

        if not present_partners:
            print(f"  [plot] nothing to plot for hub {hub} (task 6); skipping.")
            figs[hub] = None
            continue

        n_panels = len(present_partners)
        fig, axes = plt.subplots(1, n_panels, figsize=(panel_width * n_panels, panel_height), sharey=True)
        axes = np.atleast_1d(axes)

        for ax, partner, values_by_category in zip(axes, present_partners, panel_values):
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1.3, alpha=0.6, zorder=1)
            _boxplot_one_panel(ax, categories, values_by_category, colors_by_category, rng)
            ax.set_title(_display_name(partner), fontsize=TICK_FONTSIZE - 2)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(cat_labels, fontsize=TICK_FONTSIZE - 4,
                               rotation=30 if cortical else 30,
                               ha='right' if cortical else 'right')
            ax.set_xlim(-0.7, len(categories) - 0.3)
            ylim = ylim1 if cortical else ylim2
            if ylim is not None:
                ax.set_ylim(*ylim)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 4)

        axes[0].set_ylabel("Enrichment ratio", fontsize=TICK_FONTSIZE - 2)
        fig.suptitle(
            f"Hub: {_display_name(hub)}  "
            f"({'laminar depth' if cortical else 'subregion'} enrichment)\n{Align_type_value}",
            fontsize=TICK_FONTSIZE)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        save_path = output_dir / f"{file_suffix}_hubmode_task6_enrichment_ratio_{hub}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  [plot] saved: {save_path}")
        plt.close(fig)
        figs[hub] = fig

    return figs


# =============================================================================
# 10. Tasks 7 & 8 (NEW) -- top-pCCA-weight-neuron PSTH heatmaps, pooled
#     across sessions, swept over every hub region in
#     `HUB_MODE_HUB_REGIONS` (same hub-mode band layout Tasks 3-6 use, via
#     `hubmode_band_pairs()`) -- one figure per hub, each against every
#     region it pairs with -- NOT limited to a single hard-coded hub.
#
#     Neuron identification is already done -- `PrivateLatentPairResult.
#     selected_neurons_i`/`_j` (a `SelectedNeuronSet`, item 4's "these
#     have already been saved"). Task 7 shows these neurons' ORIGINAL
#     (pre-residualization, z-scored) activity; Task 8 shows their already
#     -saved RESIDUALIZED activity (`SelectedNeuronResidual.residual`).
#
#     Rastermap sorting (item 7d) is now applied ONCE, to the FULLY POOLED
#     matrix -- every selected neuron from every session that contributed
#     any, stacked first, THEN sorted -- rather than sorting each
#     session's own block independently before stacking (the earlier
#     design). Sorting still runs on the trial-AVERAGED PSTH (one shared T
#     per align_mode), not each session's raw "continuous cross-trial"
#     trace (T*n_trials samples, this project's own established Rastermap
#     input convention -- see pCCA_sensitive_realsingle_Session_11panel.py's
#     own `get_neuron_order`): different sessions generally have different
#     trial counts, so those raw continuous traces are different lengths
#     and still cannot be pooled into one joint Rastermap fit -- only the
#     PSTH's shared T lets every session's selected neurons sit in one
#     (total_neurons, T) matrix, which is what is now pooled BEFORE the
#     single Rastermap fit runs. A consequence: since the sort is now
#     global, rows from a given session are no longer a contiguous block
#     in the final row order -- `session_labels`/`session_counts` (see
#     `_gather_task78_matrix`) describe each session's CONTRIBUTION
#     (provenance) only, not a slice of matrix rows.
# =============================================================================

def get_neuron_order_2d(mat: np.ndarray) -> np.ndarray:
    """Rastermap sort order for an (n_neurons, n_obs) matrix -- for Tasks
    7/8 this is the fully pooled, trial-averaged PSTH (n_obs = T), fit
    ONCE across every session's selected neurons together. Falls back to
    peak-time ordering if rastermap is unavailable or too few neurons are
    present -- same fallback convention as this project's own
    `get_neuron_order` (pCCA_sensitive_realsingle_Session_11panel.py)."""
    n = mat.shape[0]
    if n < 2:
        return np.arange(n)
    if _RASTERMAP_OK and n >= 5:
        try:
            z = zscore(mat, axis=1, nan_policy="omit")
            np.nan_to_num(z, nan=0.0, copy=False)
            mdl = Rastermap(n_PCs=min(50, n, mat.shape[1]), **TASK78_RASTERMAP_KW)
            mdl.fit(z)
            return np.asarray(mdl.isort)
        except Exception as exc:
            warnings.warn(f"Rastermap failed ({exc}); using peak-time ordering.")
    return np.argsort(np.argmax(mat, axis=1))


_raw_region_cache: Dict[Tuple[str, str, str, str], Optional[Tuple[np.ndarray, np.ndarray]]] = {}


def _load_raw_zscored_region(
        session_name: str, region: str,
        trial_type: str = TASK78_TRIAL_TYPE, align_mode: str = ALIGN_MODE,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Reload ONE region's RAW (pre-residualization), z-scored, cross-trial
    activity for one session -- Task 7's "original firing rate" source,
    which `pCCA_all_regions_out_behaviour_v2.py` never persists
    (`region_flat_full` is transient there). Reproduces that script's OWN
    load -> crop -> behaviour-truncate -> z-score sequence with its OWN
    functions, in the SAME order, so the returned matrix's neuron axis
    lines up EXACTLY with `SelectedNeuronResidual.neuron_idx`. Cached per
    (session, region, trial_type, align_mode) -- reused across every
    partner pairing this hub appears in, since the hub's own raw data
    does not depend on which partner a given panel is about.

    Returns (region_flat, time_vec) -- (T*n_trials, n_full_neurons) and
    (T,) -- or None if unavailable (mirrors v2's own skip conditions).
    """
    key = (session_name, region, trial_type, align_mode)
    if key in _raw_region_cache:
        return _raw_region_cache[key]

    mat_dir = V2_BASE_DIR / v2_mat_subdir_name(trial_type, align_mode)
    session_file = mat_dir / f"{session_name}_analysis_results.mat"
    if not session_file.exists():
        _raw_region_cache[key] = None
        return None

    region_spikes_full, _labels_full, n_trials, T = v2_load_region_spikes_full(str(session_file))
    if region not in region_spikes_full:
        _raw_region_cache[key] = None
        return None

    window = ALIGNMENT_WINDOWS_S[align_mode]
    time_vec_raw = np.linspace(window[0], window[1], T)
    try:
        region_spikes_full, time_vec = v2_crop_time_window(region_spikes_full, time_vec_raw, window)
    except ValueError:
        _raw_region_cache[key] = None
        return None
    T = time_vec.shape[0]

    try:
        pos_sel, speed_sel, _t_behav = v2_load_behavior_regressors(
            session_name, trial_label=_trial_type_to_behavior_label(trial_type))
    except (FileNotFoundError, ValueError) as exc:
        warnings.warn(f"[{session_name}] {region}: behaviour unavailable for Task 7/8 "
                      f"raw reload ({exc}); skipping.")
        _raw_region_cache[key] = None
        return None

    n_trials_behav, T_behav = pos_sel.shape[0], pos_sel.shape[-1]
    n_common = min(n_trials, n_trials_behav)
    T_common = min(T, T_behav)
    if n_common < 1 or T_common < 2:
        _raw_region_cache[key] = None
        return None
    X = region_spikes_full[region][:n_common, :, :T_common]
    time_vec = time_vec[:T_common]

    X_flat = v2_zscore_flat(X, subtract_psth=V2_SUBTRACT_PSTH, shuffle_trials=V2_SHUFFLE_TRIALS)
    result = (X_flat, time_vec)
    _raw_region_cache[key] = result
    return result


def _gather_task78_matrix(
        analyzer: PrivateLatentAnalyzer,
        hub: str,
        partner: str,
        sessions: List[str],
        data_source: str,             # 'raw' (Task 7) | 'residual' (Task 8)
        regions_only: bool,
        trial_type: str = TASK78_TRIAL_TYPE,
        align_mode: str = ALIGN_MODE,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], List[int]]]:
    """Pool one (hub, partner) pairing's already-selected top-pCCA-weight
    neurons' trial-averaged PSTH across EVERY contributing session first,
    then Rastermap-sort the fully pooled (total_neurons, T) matrix ONCE
    (see section docstring for why sorting runs on the PSTH rather than
    each session's own raw continuous trace).

    Returns (matrix, time_vec, session_labels, session_neuron_counts):
    matrix is (total_neurons, T), already sorted by the single pooled
    Rastermap fit; the last two describe how many of those neurons each
    session contributed (provenance only -- post-sort rows are no longer
    grouped into contiguous per-session blocks). None if no session
    contributed any selected neuron.
    """
    pair_key = sort_pair_by_anatomy(hub, partner)
    role = _hub_region_role(hub, pair_key)

    blocks: List[np.ndarray] = []
    session_labels: List[str] = []
    session_counts: List[int] = []
    time_vec_common: Optional[np.ndarray] = None

    for session_name in sessions:
        session_result = analyzer.sessions.get(session_name)
        if session_result is None:
            continue
        table = session_result.pairs_regions_only if regions_only else session_result.pairs
        pr = table.get(pair_key)
        if pr is None:
            continue
        selected = pr.selected_neurons_i if role == 'region_i' else pr.selected_neurons_j
        if selected is None or not selected.neurons:
            continue

        if data_source == 'residual':
            # `.residual` is (n_trials, T) per neuron -- trial-averaged
            # here for the displayed/pooled PSTH row.
            psth_rows = np.stack(
                [nr.residual.mean(axis=0) for nr in selected.neurons], axis=0)   # (n, T)
            time_vec = session_result.time_vec
        elif data_source == 'raw':
            loaded = _load_raw_zscored_region(session_name, hub, trial_type, align_mode)
            if loaded is None:
                continue
            X_flat, raw_time_vec = loaded
            idx = np.asarray([nr.neuron_idx for nr in selected.neurons], dtype=int)
            if idx.size == 0 or idx.max() >= X_flat.shape[1]:
                warnings.warn(f"[{session_name}] {hub}: selected neuron index out of range "
                              f"for reloaded raw data; skipping this session.")
                continue
            T_raw = raw_time_vec.shape[0]
            n_trials_raw = X_flat.shape[0] // T_raw
            cols = X_flat[:, idx]                        # (T_raw*n_trials_raw, n)
            psth_rows = np.stack(
                [cols[:, k].reshape(T_raw, n_trials_raw).T.mean(axis=0)
                 for k in range(cols.shape[1])], axis=0)  # (n, T_raw)
            time_vec = raw_time_vec
        else:
            raise ValueError(f"Unknown data_source: {data_source!r}")

        if time_vec_common is None:
            time_vec_common = time_vec
        elif time_vec.shape[0] != time_vec_common.shape[0]:
            T_min = min(time_vec.shape[0], time_vec_common.shape[0])
            psth_rows = psth_rows[:, :T_min]
            time_vec_common = time_vec_common[:T_min]

        blocks.append(psth_rows.astype(np.float64))
        session_labels.append(session_name)
        session_counts.append(psth_rows.shape[0])

    if not blocks:
        return None

    T_final = time_vec_common.shape[0]
    blocks = [b[:, :T_final] for b in blocks]
    matrix = np.concatenate(blocks, axis=0)

    # ---- Rastermap runs ONCE, on the FULLY POOLED matrix (every session's
    #      selected neurons stacked first) -- not per session block. -------
    order = get_neuron_order_2d(matrix)
    matrix = matrix[order]

    return matrix, time_vec_common, session_labels, session_counts


def hubmode_plot_task78_heatmaps(
        analyzer: PrivateLatentAnalyzer,
        hub: str,
        partners: List[str],
        sessions: List[str],
        output_dir: Path,
        data_source: str,             # 'raw' (Task 7) | 'residual' (Task 8)
        task_label: str,              # 'task7' | 'task8'
        variant: str,                 # 'regions_only' | 'regions_behavior'
        panel_width: float = TASK78_PANEL_WIDTH,
        panel_height: float = TASK78_PANEL_HEIGHT,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """ONE figure for `hub`: 1xn panels, one per partner region with any
    selected-neuron data (item 7a: 5 partners -> 1x5)."""
    regions_only = (variant == 'regions_only')
    gathered_by_partner = [
        (partner, _gather_task78_matrix(analyzer, hub, partner, sessions, data_source, regions_only))
        for partner in partners
    ]
    present = [(p, g) for p, g in gathered_by_partner if g is not None]
    if not present:
        print(f"  [plot] nothing to plot for hub={hub} task={task_label} variant={variant}; skipping.")
        return None

    n_panels = len(present)
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_width * n_panels, panel_height))
    axes = np.atleast_1d(axes)
    cbar_label = 'z-scored firing rate' if data_source == 'raw' else 'residualized activity'

    for panel_idx, (ax, (partner, (matrix, time_vec, sess_labels, sess_counts))) in enumerate(
            zip(axes, present)):
        vmax = float(np.nanpercentile(np.abs(matrix), 99)) if matrix.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(
            matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
            extent=[time_vec[0], time_vec[-1], matrix.shape[0], 0], origin='upper',
        )
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        # No per-session boundary lines: Rastermap now sorts the fully
        # pooled matrix ONCE, so rows from a given session are no longer a
        # contiguous block (see `_gather_task78_matrix`).
        ax.set_title(f"{_display_name(partner)}\n"
                      f"n={matrix.shape[0]} neurons",
                      fontsize=TICK_FONTSIZE - 5)
        ax.set_xlabel("Time (s)", fontsize=TICK_FONTSIZE - 4)
        ax.tick_params(labelsize=TICK_FONTSIZE - 6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        # One colorbar per panel (matches this project's own established
        # per-panel-colorbar PSTH convention) -- a single shared colorbar
        # added across multiple Axes fights with tight_layout/suptitle
        # spacing and tends to overlap the last panel's title.
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE - 8)
        if panel_idx == n_panels - 1:
            cbar.set_label(cbar_label, fontsize=TICK_FONTSIZE - 6)
        print(f"    [{task_label}/{variant}] {_display_name(hub)} <-> {_display_name(partner)}: "
              f"{matrix.shape[0]} neurons from {len(sess_labels)} sessions "
              f"({dict(zip(sess_labels, sess_counts))})")

    axes[0].set_ylabel("Neurons pooled across sessions", fontsize=TICK_FONTSIZE - 4)

    label = 'original firing rate' if data_source == 'raw' else 'residual activity'
    fig.suptitle(
        f"Hub: {_display_name(hub)} -- {label}, top-{int(round(TOP_WEIGHT_FRACTION * 100))}% "
        f"pCCA-weight neurons ({VARIANT_DISPLAY[variant]})\n{Align_type_value}",
        fontsize=TICK_FONTSIZE)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    suffix = VARIANT_FILE_SUFFIX[variant]
    save_path = output_dir / f"{suffix}_hubmode_{task_label}_psth_{hub}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 11. CSV I/O -- copied verbatim from v1.
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


def _write_variance_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region_i', 'region_j', 'region_role', 'trial_type',
                    'predictor', 'n_sessions', 'mean_r2', 'sem_r2'])
        for (pair, role, trial_type), per_pred in sorted(summary.items()):
            for predictor, stats in per_pred.items():
                w.writerow([pair[0], pair[1], role, trial_type, predictor,
                           stats['n'], f"{stats['mean']:.6f}", f"{stats['sem']:.6f}"])
    print(f"  [csv] -> {path}")


def _write_enrichment_summary_csv(
        grouped: Dict[Tuple[Tuple[str, str], str, str, str], np.ndarray], path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region_i', 'region_j', 'region_role', 'trial_type', 'group',
                    'n_sessions', 'median', 'q1', 'q3', 'mean', 'sem'])
        for (pair, role, trial_type, group), arr in sorted(grouped.items()):
            w.writerow([pair[0], pair[1], role, trial_type, group, arr.size,
                       f"{float(np.median(arr)):.6f}", f"{float(np.percentile(arr, 25)):.6f}",
                       f"{float(np.percentile(arr, 75)):.6f}", f"{arr.mean():.6f}", f"{_sem(arr):.6f}"])
    print(f"  [csv] -> {path}")


# =============================================================================
# 12. Driver
# =============================================================================

VARIANT_FILE_SUFFIX: Dict[str, str] = {
    'regions_only':     'regions_only',
    'regions_behavior': 'regions_behavior',
}
TASK4_LABEL_BY_VARIANT: Dict[str, str] = {'regions_only': 'regions_only', 'regions_behavior': 'regions_behavior'}
TASK5_LABEL_BY_VARIANT: Dict[str, str] = {'regions_only': 'regions_only', 'regions_behavior': 'regions_behavior'}


def main() -> None:
    print("=" * 70)
    print("HUB-MODE v2 TASKS 3-8 -- PART 2c/2c' PRIVATE pCCA (10-draw resampled)")
    print("(sourced exclusively from pCCA_all_regions_out_behaviour_v2.py's own")
    print(" pcca_all_regions_out_behaviour_v2_sampled_sessions_{trial_type}_{align_mode}_results pickles)")
    print("=" * 70)
    print(f"  reference type     : {REFERENCE_TYPE}")
    print(f"  active trial types : {ACTIVE_TRIAL_TYPES}")
    print(f"  align mode         : {ALIGN_MODE}")
    print(f"  pcca variants      : {[VARIANT_DISPLAY[v] for v in PCCA_VARIANTS]}")
    print(f"  component indices  : {COMPONENT_INDICES}  (of {N_COMPONENTS} fit)")
    print(f"  samples per pair   : {N_SAMPLE_DRAWS} draws/session (v2 resampling)")
    print(f"  behaviour window   : {BEHAVIOR_TIME_RANGE_S}")
    print(f"  variance method    : {VARIANCE_METHOD}")
    print(f"  hub-mode hubs      : {HUB_MODE_HUB_REGIONS}")
    print(f"  hub-mode ROIs      : {HUB_MODE_ROI_REGIONS}")
    print(f"  task 7/8 hubs      : {HUB_MODE_HUB_REGIONS}  (same sweep as tasks 3-6)")
    print(f"  output directory   : {OUTPUT_DIR}")
    print("=" * 70)

    cross_session_analyzers, behavior_records, subregion_records, analyzers_by_trial_type = run_hubmode_analysis()
    hub_bands = hubmode_band_pairs()

    # ---- Tasks 3 & 4 --------------------------------------------------------
    print("\n--- Tasks 3-4: behavioural variance explained (hub-mode, 10-draw averaged) ---")
    _write_records_csv(behavior_records, OUTPUT_DIR / "hubmode_behavior_variance_records.csv")
    single_trial_records = [r for r in behavior_records if r.get('metric', 'single_trial') == 'single_trial']
    trial_avg_records = [r for r in behavior_records if r.get('metric') == 'trial_avg']

    for variant in PCCA_VARIANTS:
        suffix = VARIANT_FILE_SUFFIX[variant]
        variant_single = [r for r in single_trial_records if r['variant'] == variant]
        variant_avg = [r for r in trial_avg_records if r['variant'] == variant]
        variance_summary = aggregate_behavior_variance(variant_single)
        variance_summary_trial_avg = aggregate_behavior_variance(variant_avg)
        _write_variance_summary_csv(
            variance_summary, OUTPUT_DIR / f"hubmode_task3_4_variance_summary{suffix}.csv")
        _write_variance_summary_csv(
            variance_summary_trial_avg, OUTPUT_DIR / f"hubmode_task3_4_variance_summary_trial_avg{suffix}.csv")

        if ALIGN_MODE in TASK3_ALIGN_MODES:
            hubmode_plot_task3_bars(
                variance_summary, variance_summary_trial_avg, hub_bands, OUTPUT_DIR,
                external_variables=EXTERNAL_VARIABLES_BY_VARIANT[variant], file_suffix=suffix,
            )
        else:
            print(f"  [task 3] skipped for align_mode={ALIGN_MODE!r} "
                  f"(only runs for {TASK3_ALIGN_MODES}) -- variant={variant}")

        hubmode_plot_task4_bars(
            variance_summary, variance_summary_trial_avg, hub_bands, OUTPUT_DIR,
            task_label=TASK4_LABEL_BY_VARIANT[variant],
        )

    # ---- Task 5 ---------------------------------------------------------------
    print("\n--- Task 5: latent traces across sessions (hub-mode, per hub, all draws x trials) ---")
    for variant in PCCA_VARIANTS:
        for comp_idx in COMPONENT_INDICES:
            hubmode_plot_task5_latent_traces(
                hub_bands, cross_session_analyzers[variant], OUTPUT_DIR, component_idx=comp_idx,
                task_label=TASK5_LABEL_BY_VARIANT[variant],
            )

    # ---- Task 6 -----------------------------------------------------------------
    print("\n--- Task 6: subregion/laminar ENRICHMENT-ratio boxplots (hub-mode, 10-draw averaged) ---")
    _write_records_csv(subregion_records, OUTPUT_DIR / "hubmode_enrichment_ratio_records.csv")
    for variant in PCCA_VARIANTS:
        suffix = VARIANT_FILE_SUFFIX[variant]
        variant_subregion = [r for r in subregion_records if r['variant'] == variant]
        enrichment_grouped = aggregate_enrichment_ratio(variant_subregion)
        _write_enrichment_summary_csv(
            enrichment_grouped, OUTPUT_DIR / f"hubmode_task6_enrichment_ratio_summary{suffix}.csv")
        hubmode_plot_task6_enrichment_boxplots(
            variant_subregion, hub_bands, OUTPUT_DIR, file_suffix=suffix)

    # ---- Tasks 7 & 8 (NEW) -------------------------------------------------------
    # Sweeps every hub in `hub_bands` (== HUB_MODE_HUB_REGIONS), same as
    # Tasks 3-6 -- NOT limited to a single hard-coded hub.
    print(f"\n--- Tasks 7-8: top-{int(round(TOP_WEIGHT_FRACTION*100))}%-pCCA-weight-neuron "
          f"PSTH heatmaps (hubs={HUB_MODE_HUB_REGIONS}) ---")
    az78 = analyzers_by_trial_type.get(TASK78_TRIAL_TYPE)
    if az78 is None:
        az78 = PrivateLatentAnalyzer(base_dir=BASE_DIR, trial_type=TASK78_TRIAL_TYPE, align_mode=ALIGN_MODE)
        az78.load_all()

    for hub, hub_partner_pairs in hub_bands:
        partners78 = [p for _, p in hub_partner_pairs]
        if not partners78:
            print(f"  [task 7/8] {hub!r} has no partners in HUB_MODE_ROI_REGIONS; skipping.")
            continue
        for variant in PCCA_VARIANTS:
            hubmode_plot_task78_heatmaps(
                az78, hub, partners78, SESSIONS, OUTPUT_DIR,
                data_source='raw', task_label='task7', variant=variant,
            )
            hubmode_plot_task78_heatmaps(
                az78, hub, partners78, SESSIONS, OUTPUT_DIR,
                data_source='residual', task_label='task8', variant=variant,
            )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Figures and CSVs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

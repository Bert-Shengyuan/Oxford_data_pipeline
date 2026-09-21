#!/usr/bin/env python3
r"""
pCCA_all_regions_hubmode_task345.py
================================================================================

Hub-mode Tasks 3, 4, 5 (behavioural-variance-explained bars and cross-session
latent traces) PLUS Task 6 (subregion/laminar weight-ratio boxplots) for
Part 2c/2c' (the private pCCA latent) of ``pCCA_all_regions_out_
behaviour.py``.

--------------------------------------------------------------------------------
Two pCCA regress-out variants, everywhere (latest revision)
--------------------------------------------------------------------------------
``pCCA_all_regions_out_behaviour.py`` now stores TWO independently-fit
versions of Part 2c's private latent per pair, per session: ``.pairs``
("AllRegions + Behaviour" -- the nuisance Z includes every other recorded
region AND behaviour, unchanged from before) and ``.pairs_regions_only``
("AllRegions only" -- same computation, behaviour never enters the nuisance
Z). ``PCCA_VARIANTS = ('regions_only', 'regions_behavior')`` drives BOTH
the data-gathering pass (``run_hubmode_analysis``, which now reads and
tags records from both dicts via ``PrivateLatentAnalyzer.get_pair(...,
regions_only=True/False)``) and every plotting task below, each producing
TWO output artifacts (filenames disambiguated by ``VARIANT_FILE_SUFFIX``
for Tasks 3/6, or by a renumbered task label -- ``task4_1``/``task4_2``,
``task5_1``/``task5_2`` -- for Tasks 4/5, per this revision's request):

    Task 3  (``hubmode_plot_task3_bars``)             -- default_move_onset ONLY
                                                          (TASK3_ALIGN_MODES; skipped
                                                          for every other align_mode)
              regions_only      -> 4 panels: position, speed, reward_presence,
                                   reward_consumption (position/speed are meaningful
                                   again here, since behaviour was NOT regressed out --
                                   EXTERNAL_VARIABLES_BY_VARIANT)
              regions_behavior  -> 2 panels: reward_presence, reward_consumption
                                   (unchanged from before this revision)
    Task 4.1 / 4.2  (``hubmode_plot_task4_bars``)      -- both variants, both align
                                                          modes; always the 2-panel
                                                          reward-only view
    Task 5.1 / 5.2  (``hubmode_plot_task5_latent_traces``) -- both variants' own
                                                          cross-session latent traces
    Task 6  (``hubmode_plot_task6_enrichment_boxplots``) -- both variants, same
                                                          enrichment_ratio boxplots
                                                          (no log2/dominant_ratio
                                                          reduction, unchanged)

Reward-kernel definition, ``align_mode == 'default_move_onset'`` only: the
fixed REWARD_PRESENCE_WINDOW_S/REWARD_CONSUMPTION_WINDOW_S this script
otherwise falls back to assume t=0 IS the reward (only true when align_mode
== 'reward_onset'). In default_move_onset, each trial's OWN reward-onset
time is available instead (t_approach_python.py's per-trial
``trigger_times['reward_onset']``, seconds relative to movement onset), so
``build_reward_presence_design``/``build_reward_consumption_design`` use a
PER-TRIAL kernel there: presence = (0, that trial's reward onset),
consumption = (that trial's reward onset, + REWARD_CONSUMPTION_DURATION_S).
A trial with no recorded reward (NaN) contributes an all-zero row rather
than being dropped.

This script is a PURE downstream consumer of ``pCCA_all_regions_out_
behaviour.py``: every quantity it plots is read straight out of that
script's pickled ``*_analysis_results.pkl`` files via ``PrivateLatentAnalyzer``
(``.pairs`` / ``.pairs_regions_only`` / ``PrivateLatentPairResult`` -- Part
2c / 2c'). Behavioural (position/speed/reward-timing) data is read from
t_approach_python.py's consolidated ``{session}.pkl`` per session (NOT the
older separate ``{session}_pos.npy`` etc. files) via this file's own
``load_behavior_regressors``. Unlike
``pCCA_latent_extrenal_variable_bar.py`` (which can also fit/reproject a
CCA/pCCA subspace straight from the raw ``.mat`` pipeline when
``KERNEL_MODE`` is ``'pcca'``/``'cca'``) and unlike ``PCA_latent_extrenal_
variable_part.py``'s own hub-based mode (whose Task-(c) reference-projection
path reloads raw region tensors from ``.mat`` whenever more than one trial
type is active -- see that file's Section 12), this script never imports
``mat73``, ``load_region_spikes``, ``residualize``, or any other raw-``.mat``
primitive: it only ever calls ``PrivateLatentAnalyzer.load_all()`` /
``.get_pair()``. If a session/pair/trial-type combination has no cached
pickle, it is skipped -- never recomputed.

--------------------------------------------------------------------------------
What this script reproduces, and what is new
--------------------------------------------------------------------------------
Tasks 3 & 4 (``hubmode_plot_task3_bars`` / ``hubmode_plot_task4_bars``) and
the underlying data-gathering pass are carried over from
``pCCA_latent_extrenal_variable_bar.py``'s own Section 9b (hub-mode display)
essentially unchanged -- same row layout (``hubmode_band_pairs``: one band
per ``HUB_MODE_HUB_REGIONS`` entry, one row per ``HUB_MODE_ROI_REGIONS``
partner), same two-panel (reward presence / reward consumption) bar-plot
engine, same single-trial/trial-averaged split, same file-naming convention
(``hubmode_task3_variance_{hub}{suffix}.png`` / ``hubmode_task4_variance_
{hub}{suffix}.png``).

Task 5 (``hubmode_plot_task5_latent_traces``) reproduces the SAME trace
styling (per-session thin traces, bold mean, SEM shading, dashed t=0 line)
but, per this version's request, is split into ONE FIGURE PER HUB REGION
instead of one figure pooling every hub's rows together -- saved as
``hubmode_task5_latent_traces_comp{component_idx}_{hub}.png`` -- mirroring
the per-hub split ``PCA_latent_extrenal_variable_part.py``'s own hub-mode
Task 5 already uses for Parts 2a/2b.

Task 6 (``hubmode_plot_task6_enrichment_boxplots``) is NEW: it visualises
Part 3's subregion/laminar-depth weight metric (``compute_subregion_
weight_metrics`` in ``pCCA_all_regions_out_behaviour.py``, stored per pair
as ``PrivateLatentPairResult.subregion_weight_metrics_i`` / ``_j``) for the
hub's own Wx/Wy canonical weight column -- a quantity neither ``pCCA_
latent_extrenal_variable_bar.py`` nor ``PCA_latent_extrenal_variable_
part.py`` ever plots (both only import ``SubregionWeightMetrics`` for
typing).

Task 6 plots the ``enrichment_ratio`` field -- NOT ``dominant_ratio`` (the
single-scalar "top group : rest" reduction an earlier revision of this
script used): ``enrichment_ratio[g] = weight_mass_fraction[g] /
neuron_count_fraction[g]`` is reported PER GROUP, so every group's own
cross-session distribution is shown side by side, not collapsed to
whichever group happened to dominate. Layout, per this version's request:
ONE FIGURE per hub region, laid out 1xn -- one PANEL per partner ROI
region (not one row per partner, the way Tasks 3/4/the earlier Task 6
revision lay out their bars). For a cortical hub (``CORTICAL_REGIONS``),
every panel is a 2-box plot ('Superficial' / 'Deep'); for a subcortical
hub, every panel has one box per subregion label actually observed for
that hub -- the SAME category list and category-to-colour assignment is
shared across every partner's panel within one hub's figure, so panels
stay directly comparable at a glance. Each box is a cross-session boxplot
(median / IQR / whiskers, matplotlib's default 1.5x-IQR rule) of that
group's per-session enrichment ratio, with individual sessions overlaid as
jittered dots -- the style requested, matching the attached reference
figure (pale, category-tinted box fill; black outline/median/whiskers;
solid, category-coloured dots; see ``_boxplot_one_panel``). A dashed
horizontal line at y=1.0 marks "no enrichment" (a group carrying exactly
its numerical fair share of |W|) -- the natural neutral point for a ratio
metric, analogous to Tasks 3/4's own y=0 baseline for R^2.

--------------------------------------------------------------------------------
Direct trial_type / align_mode configuration
--------------------------------------------------------------------------------
``pCCA_all_regions_out_behaviour.py`` ties its neural-data source folder to
``mat_subdir_name(trial_type, align_mode) == f"{trial_type}_{align_mode}_
results"`` and its own pickle output folder to ``out_subdir_name(trial_type,
align_mode)``. This script never touches the first (it has no raw-``.mat``
path at all), but it reads the SECOND for every trial type it loads, so it
exposes ``REFERENCE_TYPE`` / ``ACTIVE_TRIAL_TYPES`` (trial_type) and
``ALIGN_MODE`` (align_mode) as direct, top-level configuration constants --
not derived from each other or from any other hardcoded string -- feeding
``PrivateLatentAnalyzer(trial_type=t, align_mode=ALIGN_MODE)`` for every
entry of ``ACTIVE_TRIAL_TYPES`` (one independently-computed pickle folder
per trial type, all sharing the one ``ALIGN_MODE``, exactly mirroring how
``ALIGN`` is a single global in ``pCCA_all_regions_out_behaviour.py`` even
though ``TRIAL_TYPE`` there is swapped per run). ``BEHAVIOR_DIR`` is derived
from ``ALIGN_MODE`` the same align-aware way ``pCCA_all_regions_out_
behaviour.py``'s own ``BEHAVIOR_DIR`` is (``tapproach_sessions_{align_mode}``),
NOT the fixed, align-oblivious ``tapproach_sessions`` folder
``pCCA_latent_extrenal_variable_bar.py`` happens to hardcode.

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

warnings.filterwarnings('ignore')

# =============================================================================
# 0.  Imports. `cross_trial_type_cca_analysis.py` is treated as a stable
#     library module (per this project's convention) purely for its
#     cross-session, sign-aligned aggregator (`CrossSessionCCAAnalyzer`) --
#     Task 5's own trace-styling engine needs its spectral sign-alignment
#     step, exactly as every sibling script's own Task 5 does. Nothing in
#     THIS file ever touches that module's `.mat`-loading machinery.
#     `pCCA_all_regions_out_behaviour.py` is this script's ONLY source of
#     neural data -- see module docstring.
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_trial_type_cca_analysis import (   # noqa: E402
    CrossSessionCCAAnalyzer,
    TRIAL_TYPE_COLORS,
    MIN_SESSIONS_THRESHOLD,
)
from pCCA_all_regions_out_behaviour import (  # noqa: E402
    PrivateLatentAnalyzer,
    PrivateLatentSessionResult,
    PrivateLatentPairResult,
    RegionPCAResult,
    HubOrientationPCAResult,
    HubPairPCAResult,
    SubregionWeightMetrics,
    REGION_PAIRS,
    N_COMPONENTS,
    CORTICAL_REGIONS,
    sort_pair_by_anatomy,
    get_anatomical_index,
    out_subdir_name,
)

# `pcca_all_regions_out_behaviour.py` bakes '__main__' into every pickled
# dataclass instance's module reference (it is normally *run* directly);
# unpickling those files from THIS script's own '__main__' therefore needs
# the same classes reachable under `__main__` here too -- identical fix,
# identical reasoning, to every sibling script's own copy of this block.
sys.modules['__main__'].PrivateLatentSessionResult = PrivateLatentSessionResult
sys.modules['__main__'].PrivateLatentPairResult = PrivateLatentPairResult
sys.modules['__main__'].RegionPCAResult = RegionPCAResult
sys.modules['__main__'].HubOrientationPCAResult = HubOrientationPCAResult
sys.modules['__main__'].HubPairPCAResult = HubPairPCAResult

try:
    import mat73  # noqa: F401  (transitively required by cross_trial_type_cca_analysis's own imports)
except Exception:
    warnings.warn("mat73 not importable -- cross_trial_type_cca_analysis.py may fail to "
                  "import; this script itself never reads a .mat file directly.")


# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS
# =============================================================================

# ---- trial_type / align_mode -- specified DIRECTLY (item 4 of the request),
#      feeding PrivateLatentAnalyzer(trial_type=..., align_mode=ALIGN_MODE)
#      the same way pCCA_all_regions_out_behaviour.py's own
#      mat_subdir_name(trial_type, align_mode) / out_subdir_name(trial_type,
#      align_mode) take both as explicit arguments -- neither is derived
#      from the other, and ALIGN_MODE is a single global shared by every
#      entry of ACTIVE_TRIAL_TYPES (mirroring ALIGN's own single-global role
#      upstream). --------------------------------------------------------
REFERENCE_TYPE: str = 'cued_hit_long'
ACTIVE_TRIAL_TYPES: List[str] = [
    'cued_hit_long',
    # 'spont_hit_long',
    # 'spont_miss_long',
]
HUB_MODE_ENRICHMENT_YLIM_C: Optional[Tuple[float, float]] = [-0.05,1.8]
HUB_MODE_ENRICHMENT_YLIM_SC: Optional[Tuple[float, float]] = [0,2.1]

ALIGN_MODE: str = 'default_move_onset' #default_move_onset cue_onset
Align_type_value = ALIGN_MODE.replace("_", " ")
if ALIGN_MODE == 'default_move_onset':
    Align_type_value = 'Move onset'

# Per-alignment-mode trial window (seconds, relative to the aligned event) --
# must match segment_mdl_to_trials.m (neural), t_approach_python.py, and
# pCCA_all_regions_out_behaviour.py's own ALIGNMENT_WINDOWS_S EXACTLY, since
# BEHAVIOR_TIME_RANGE_S below crops both the neural latent and the
# behavioural tensors to this same window (`_prepare_regression_inputs`).
ALIGNMENT_WINDOWS_S: Dict[str, Tuple[float, float]] = {
    "default_move_onset": (-1.0, 2.0),
    "cue_onset":           (-0.8, 2.2),
    "bar_off_onset":       (-2.0, 1.0),
    "reward_onset":        (-1.2, 1.8),
}

# Task 3 (behavioural-variance bars against a fixed reward kernel) is only
# meaningful in a mode where "time since reward" is well defined for every
# trial without per-trial adjustment; in every OTHER alignment mode it is
# skipped entirely (see hubmode_plot_task3_bars's caller in main()).
TASK3_ALIGN_MODES: Tuple[str, ...] = ("default_move_onset",)

# ---- Paths ------------------------------------------------------------
BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
BEHAVIOR_DIR = BASE_DIR / "Paper_output" / f"tapproach_sessions_{ALIGN_MODE}"
OUTPUT_DIR = (BASE_DIR / "Paper_output"
              / f"pcca_all_regions_hubmode_{REFERENCE_TYPE}_{ALIGN_MODE}")
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

# ---- pCCA dimensionality (Part 2c) -- COMPONENT_INDICES bounded by
#      N_COMPONENTS, imported from pCCA_all_regions_out_behaviour.py itself
#      (the K every PrivateLatentPairResult was actually fit with), not
#      redeclared, so there is no second constant to drift out of sync. ---
COMPONENT_INDICES: List[int] = [0]

MIN_SESSIONS: int = MIN_SESSIONS_THRESHOLD

# ---- Tasks 3/4: behavioural variance. Two "regress-out" variants of Part
#      2c's private latent are now available upstream (`.pairs` =
#      AllRegions+Behaviour, `.pairs_regions_only` = AllRegions only -- see
#      pCCA_all_regions_out_behaviour.py's own `pairs_regions_only`), and
#      they warrant DIFFERENT predictor sets here: the AllRegions+Behaviour
#      latent has already had position/speed regressed out of it, so their
#      R^2 there is expected near zero and uninformative (the established
#      hub-mode rationale in pCCA_latent_extrenal_variable_bar.py, unchanged
#      for that variant); the AllRegions-only latent has NOT had behaviour
#      removed, so position/speed become meaningful predictors again for
#      IT specifically -- Task 3's own 4-column display (see
#      hubmode_plot_task3_bars) is the one place that difference is shown;
#      every other task (4/5/6) still only ever displays the 2-variable
#      reward-only view for both variants. ---------------------------------
PCCA_VARIANTS: Tuple[str, ...] = ('regions_only', 'regions_behavior')
VARIANT_DISPLAY: Dict[str, str] = {
    'regions_only':     'AllRegions only',
    'regions_behavior': 'AllRegions + Behaviour',
}
EXTERNAL_VARIABLES_BY_VARIANT: Dict[str, List[str]] = {
    'regions_behavior': ['reward_presence', 'reward_consumption'],
    'regions_only':      ['position', 'speed', 'reward_presence', 'reward_consumption'],
}
# Reward-only view shared by Tasks 4/5/6 regardless of variant (see above).
EXTERNAL_VARIABLES: List[str] = EXTERNAL_VARIABLES_BY_VARIANT['regions_behavior']

BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = ALIGNMENT_WINDOWS_S[ALIGN_MODE]
LAMBDA_R2: float = 1e-4
VARIANCE_METHOD: str = 'marginal'  # 'marginal' | 'leave_one_out'

# Reward-kernel construction -- identical constants to
# pCCA_latent_extrenal_variable_bar.py's own (see that file's "Reward
# kernel" docstring note for the fixed-window assumption this inherits),
# EXCEPT in align_mode == 'default_move_onset': there, per-trial reward
# timing is actually available (t_approach_python.py's own per-trial
# `trigger_times['reward_onset']`, seconds relative to movement onset), so
# `build_reward_presence_design` / `build_reward_consumption_design` use
# THAT instead of these fixed windows -- see their own docstrings. These
# two constants remain the fallback for every other alignment mode (where
# t=0 already IS the aligned event the window is defined relative to, so a
# fixed window is correct without any per-trial adjustment).
REWARD_PRESENCE_WINDOW_S: Tuple[float, float] = (0.0, 0.5)
REWARD_CONSUMPTION_WINDOW_S: Tuple[float, float] = (0.5, 1.5)
REWARD_CONSUMPTION_DURATION_S: float = 1.0  # default_move_onset only: window = (reward_onset, reward_onset + this)
N_REWARD_CONSUMPTION_BASIS: int = 7
REWARD_SPLINE_DEGREE: int = 2

# ---- Hub-mode row layout -- one band per HUB_MODE_HUB_REGIONS entry, each
#      paired against every OTHER region in HUB_MODE_ROI_REGIONS. Both
#      default to the same 4-hub / 7-ROI choice used throughout this
#      project's other hub-mode figures. ----------------------------------
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
    'position':           "#DE6E4B",  # matches pCCA_latent_extrenal_variable_bar.py's own EXTERNAL_VAR_COLORS
    'speed':              "#4B7DDE",  # (Task 3's regions_only variant only -- see EXTERNAL_VARIABLES_BY_VARIANT)
    'reward_presence':    "#55A868",
    'reward_consumption': "#B07AA1",
}
BAR_LEN: float = 0.2  # single-trial R^2 x-axis half-scale, matches pCCA_latent_extrenal_variable_bar.py
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

# ---- Task 6 (enrichment-ratio boxplots) styling -- a categorical palette
#      reused from this project's own PAIR_CATEGORY_COLORS/CATEGORY_COLORS
#      hue set (pcca_cross_session_mi_bar.py / pCCA_latent_extrenal_
#      variable_bar.py's own 7-colour category palette), so a group's
#      colour reads consistently with the rest of the pipeline's figures
#      even though it now encodes a laminar/subregion GROUP, not a pair
#      category. -----------------------------------------------------------
ENRICHMENT_GROUP_PALETTE: List[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#64B5CD", "#CCB974",
]
ENRICHMENT_BOX_WIDTH: float = 0.6
ENRICHMENT_DOT_JITTER: float = 0.16
# None = matplotlib autoscale; set e.g. (0.0, 4.0) to pin every Task-6 panel
# to the same y-range.

# Fixed category order/labels for a cortical hub -- 'Superficial' matches
# the request's own wording ("superficial and deep layers"); the underlying
# dict key is still 'layer-shallow', matching LAMINAR_DEPTH_MAP in
# pCCA_all_regions_out_behaviour.py.
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
    """Blend `hex_color` toward white by `amount` -- copied verbatim from
    pCCA_latent_extrenal_variable_bar.py's own helper. Used by Task 6's
    boxplot engine so a box's pale fill is a lightened tint of its dots'
    full-saturation colour, matching the attached reference figure's style."""
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

# ---- Caching / output -----------------------------------------------------
SAVE_DPI: int = 400


# =============================================================================
# 2.  Low-level primitives, copied verbatim from pCCA_latent_extrenal_
#     variable_bar.py (project convention: primitives copied, not imported,
#     so this script stays independently auditable and runnable).
# =============================================================================

def load_behavior_regressors(
        session_name: str,
        behavior_dir: Path = BEHAVIOR_DIR,
        trial_label: str = "cued hit long",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load per-trial position (x, y, z), speed, and (align_mode ==
    'default_move_onset' only) each trial's own reward-onset time for one
    session, filtered to trials matching `trial_label`.

    Reads t_approach_python.py's consolidated `{session}.pkl` (one dict per
    session: "pos", "task_label", "movement_label", "speed", "time",
    "trigger_times") -- the SAME format pCCA_all_regions_out_behaviour.py's
    own `load_behavior_regressors` now reads, replacing the four separate
    `{session}_*.npy` files this function used to read directly.

    Returns
    -------
    pos_sel          : (n, 3, T)  float32,  channels = [x, y, z]
    speed_sel        : (n, 1, T)  float32
    t_behav          : (T,)  seconds, this alignment mode's own window
    reward_onset_sel : (n,) float64 or None -- per-trial reward-onset time,
                       seconds relative to movement onset (t=0 in that
                       frame); None unless the pkl's "trigger_times" is
                       populated, which t_approach_python.py only does for
                       align_mode == 'default_move_onset'. NaN for a trial
                       with no recorded reward (e.g. a miss).
    """
    pkl_path = Path(behavior_dir) / f"{session_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Behaviour file not found: {pkl_path}")
    with open(pkl_path, "rb") as fh:
        session_data = pickle.load(fh)

    pos     = np.asarray(session_data["pos"])                      # (N, 3, T)
    speed   = np.asarray(session_data["speed"])                     # (N, T)
    labels  = np.asarray(session_data["task_label"], dtype=object)  # (N,)
    t_behav = np.asarray(session_data["time"], dtype=np.float64)    # (T,)
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
    r"""Ridge R^2 for a latent (n_trials, T) explained by a behavioural design
    (n_trials, C, T). Flattened over (trial, time), finite-masked, ridge-
    regularised. R^2 in [0, 1], clipped; invariant to a global sign flip of
    the latent (so no flip-alignment step is required upstream)."""
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


def _bspline_basis_matrix(t: np.ndarray, n_basis: int, degree: int = REWARD_SPLINE_DEGREE) -> np.ndarray:
    """Cubic B-spline basis (matching R's bs()) evaluated at t."""
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
    return basis[1:-1, :]  # (n_basis, T)


def build_reward_presence_design(
        t_behav: np.ndarray, n_trials: int,
        presence_window: Tuple[float, float] = REWARD_PRESENCE_WINDOW_S,
        reward_onset_per_trial: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(n_trials, 1, T) step-function 'reward presence' regressor.

    `reward_onset_per_trial` (align_mode == 'default_move_onset' only,
    seconds relative to movement onset): when given, trial i's own kernel
    is (0, reward_onset_per_trial[i]) instead of the fixed
    `presence_window` -- "kernel length (0, reward onset)", per-trial,
    rather than one window shared by every trial. Trials with no recorded
    reward (NaN) are expected to already have been dropped by
    `_prepare_regression_inputs`/`_prepare_averaged_regression_inputs`
    before this is called, so every row here gets a real kernel."""
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
    """(n_trials, n_basis, T) cubic B-spline 'reward consumption' kernel.

    `reward_onset_per_trial` (align_mode == 'default_move_onset' only):
    when given, trial i's own window is (reward_onset_per_trial[i],
    reward_onset_per_trial[i] + consumption_duration) instead of the fixed
    `consumption_window` -- "(reward onset, reward onset + 1s)", per trial.
    Trials with no recorded reward (NaN) are expected to already have
    been dropped by `_prepare_regression_inputs`/`_prepare_averaged_
    regression_inputs` before this is called, same as
    `build_reward_presence_design`."""
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
    """Map `external_variables` names (default: EXTERNAL_VARIABLES, the
    reward-only view) to (n_trials, C, T) design blocks. `position`/`speed`
    (Task 3's regions_only variant only -- EXTERNAL_VARIABLES_BY_VARIANT)
    are the raw traces themselves, matching pCCA_latent_extrenal_variable_
    bar.py's own `_build_predictor_designs`."""
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
    """Crop the per-trial neural latent (n_trials, T_neural) and the
    behavioural tensors to BEHAVIOR_TIME_RANGE_S, match trial/time counts
    (shorter-of-the-two truncation), drop any trial with a missing
    (NaN) `reward_onset_per_trial` -- from the latent AND position/speed
    alike, not just from the reward regressors -- and build the predictor
    design dict from what remains."""
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
    """Trial-averaged counterpart of `_prepare_regression_inputs`: regresses
    one session's TRIAL-AVERAGED latent trace against this session's own
    trial-averaged position/speed-derived design -- feeds the 'trial_avg'
    metric split, distinct from the 'single_trial' regression above (see
    pCCA_latent_extrenal_variable_bar.py's own identical helper for why
    these are two different statistics, not a duplicate computation).

    Takes the per-trial `latent` (not yet averaged) so that, exactly as in
    `_prepare_regression_inputs`, any trial with a missing (NaN)
    `reward_onset_per_trial` can be dropped -- from the latent AND
    position/speed alike -- *before* averaging, rather than the average
    silently baking in trials with no recorded reward.

    There is no single "trial-averaged" reward-onset time the way there is
    for position/speed (every trial's own reward lands at a different
    moment); when `reward_onset_per_trial` is given, this session's own
    MEDIAN reward-onset time (over the surviving trials) stands in for it
    here -- one scalar window, reused the same way REWARD_PRESENCE_
    WINDOW_S/REWARD_CONSUMPTION_WINDOW_S already are for align modes with
    no per-trial timing at all."""
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
    """Minimal duck-typed stand-in for CrossTrialTypeCCAAnalyzer, exposing
    only what CrossSessionCCAAnalyzer.add_session_result actually reads
    (.projections, .statistical_results, .time_bins), copied verbatim from
    pCCA_latent_extrenal_variable_bar.py's own identically-named class."""
    def __init__(self, projections: Dict[str, Dict[str, np.ndarray]], time_bins: np.ndarray):
        self.projections = projections
        self.statistical_results: Dict = {}
        self.time_bins = time_bins


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
# 3.  Data gathering -- ONE pass over (session, hub-mode pair, trial type,
#     PCCA_VARIANT) that reads BOTH Part 2c result sets straight out of
#     PrivateLatentAnalyzer.get_pair(..., regions_only=True/False) and feeds
#     THREE outputs, each now carrying a `variant` tag ('regions_only' /
#     'regions_behavior'): `cross_session_analyzers` (Task 5's sign-aligned
#     traces, one dict PER VARIANT since the two variants' Wx/Wy are
#     genuinely different fits, not just a relabelling), `behavior_records`
#     (Tasks 3/4's R^2), `subregion_records` (Task 6's enrichment ratios).
#     Only the pairs that can actually appear as a hub-mode row (some member
#     in HUB_MODE_HUB_REGIONS) are iterated, rather than the full
#     REGION_PAIRS set.
# =============================================================================

def run_hubmode_analysis(
        sessions: List[str] = SESSIONS,
        base_dir: Path = BASE_DIR,
        reference_type: str = REFERENCE_TYPE,
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        align_mode: str = ALIGN_MODE,
        component_indices: List[int] = COMPONENT_INDICES,
        min_sessions: int = MIN_SESSIONS,
) -> Tuple[Dict[str, Dict[Tuple[str, str], CrossSessionCCAAnalyzer]], List[dict], List[dict]]:
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
        print(f"[hub-mode] SESSION {s_idx}/{len(all_session_names)}: {session_name}")
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
                    if pr is None:
                        continue
                    pr_by_trial_type[trial_type] = pr
                    n_tr = pr.z_i_lat.shape[0]
                    per_trial_type[trial_type] = dict(
                        u_mean=pr.z_i_lat.mean(axis=0), v_mean=pr.z_j_lat.mean(axis=0),
                        u_trials=pr.z_i_lat, v_trials=pr.z_j_lat,
                        u_std=pr.z_i_lat.std(axis=0), v_std=pr.z_j_lat.std(axis=0),
                        u_sem=pr.z_i_lat.std(axis=0) / np.sqrt(max(n_tr, 1)),
                        v_sem=pr.z_j_lat.std(axis=0) / np.sqrt(max(n_tr, 1)),
                        n_trials=n_tr,
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

                # ---- Tasks 3/4 data path (R^2 against reward regressors,
                #      PLUS position/speed for the regions_only variant --
                #      EXTERNAL_VARIABLES_BY_VARIANT) -----------------------
                for trial_type, proj in per_trial_type.items():
                    if trial_type not in behavior_cache:
                        behavior_cache[trial_type] = _load_behavior_safe(session_name, trial_type)
                    behav = behavior_cache[trial_type]
                    if behav is None:
                        continue
                    pos_full, speed_full, t_behav_full, reward_onset_full = behav

                    for comp_idx in component_indices:
                        for region_role, region_name, trials in (
                                ('region_i', pair_key[0], proj['u_trials']),
                                ('region_j', pair_key[1], proj['v_trials'])):
                            if comp_idx >= trials.shape[2]:
                                continue
                            latent = trials[:, :, comp_idx]
                            r2_by_var = _r2_by_var_for_latent(
                                latent, time_vec_for_session, pos_full, speed_full, t_behav_full,
                                reward_onset_full=reward_onset_full, external_variables=external_vars)
                            if r2_by_var is not None:
                                for var_name, r2_val in r2_by_var.items():
                                    behavior_records.append(dict(
                                        session=session_name, pair=pair_key, region_role=region_role,
                                        region=region_name, trial_type=trial_type, component=comp_idx,
                                        predictor=var_name, r2=r2_val, metric='single_trial',
                                        variant=variant,
                                    ))

                            r2_by_var_avg = _r2_by_var_for_averaged_latent(
                                trials, comp_idx, time_vec_for_session, pos_full, speed_full, t_behav_full,
                                reward_onset_full=reward_onset_full, external_variables=external_vars)
                            if r2_by_var_avg is not None:
                                for var_name, r2_val in r2_by_var_avg.items():
                                    behavior_records.append(dict(
                                        session=session_name, pair=pair_key, region_role=region_role,
                                        region=region_name, trial_type=trial_type, component=comp_idx,
                                        predictor=var_name, r2=r2_val, metric='trial_avg',
                                        variant=variant,
                                    ))

                # ---- Task 6 data path (subregion/laminar ENRICHMENT ratio,
                #      one record per session x GROUP -- not collapsed to a
                #      single "dominant" scalar, since Task 6 shows every
                #      group's own cross-session distribution side by side). -
                for trial_type, pr in pr_by_trial_type.items():
                    for region_role, region_name, metrics_list in (
                            ('region_i', pair_key[0], pr.subregion_weight_metrics_i),
                            ('region_j', pair_key[1], pr.subregion_weight_metrics_j)):
                        for comp_idx in component_indices:
                            if comp_idx >= len(metrics_list):
                                continue
                            m = metrics_list[comp_idx]
                            for group, ratio in m.enrichment_ratio.items():
                                if ratio is None or not np.isfinite(ratio) or ratio < 0:
                                    continue
                                subregion_records.append(dict(
                                    session=session_name, pair=pair_key, region_role=region_role,
                                    region=region_name, trial_type=trial_type, component=comp_idx,
                                    group=group, enrichment_ratio=float(ratio),
                                    is_cortical=m.is_cortical,
                                    n_neurons_resolved=m.n_neurons_resolved,
                                    n_neurons_total=m.n_neurons_total,
                                    variant=variant,
                                ))

    print("\n" + "=" * 70)
    print("[hub-mode] CROSS-SESSION AGGREGATION (sign alignment + mean/SEM across sessions)")
    print("=" * 70)
    for variant, variant_analyzers in cross_session_analyzers.items():
        for pair_key, cs in variant_analyzers.items():
            n_sess = len(cs.session_projections)
            if n_sess < min_sessions:
                print(f"  [{variant}] {pair_key[0]} vs {pair_key[1]}: {n_sess} sessions (skipping, < {min_sessions})")
                continue
            cs.aggregate_projections()

    print("\n" + "=" * 70)
    print("[hub-mode] DATA-GATHERING COMPLETE")
    for variant in PCCA_VARIANTS:
        print(f"  [{variant}] pairs with >=1 session : {len(cross_session_analyzers[variant])}")
    print(f"  behavioural-variance records   : {len(behavior_records)}")
    print(f"  subregion-ratio records        : {len(subregion_records)}")
    print("=" * 70)
    return cross_session_analyzers, behavior_records, subregion_records


# =============================================================================
# 4.  Aggregation
# =============================================================================

def aggregate_behavior_variance(
        behavior_records: List[dict],
) -> Dict[Tuple[Tuple[str, str], str, str], Dict[str, dict]]:
    """Returns {(pair, region_role, trial_type): {predictor: {mean, sem,
    values, n}}}, pooled across sessions -- copied verbatim from
    pCCA_latent_extrenal_variable_bar.py's own aggregator."""
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
    """Returns {(pair, region_role, trial_type, group): values}, one array
    of per-session enrichment_ratio values per GROUP, pooled across
    sessions -- feeds Task 6's boxplots directly. A boxplot's own
    median/IQR/whiskers ARE the requested "cross-session statistical
    result", so no mean/SEM reduction happens here, unlike Tasks 3/4's
    `aggregate_behavior_variance` (which feeds a bar+errorbar, not a box)."""
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
# 5.  Shared hub-mode plotting engine -- copied-and-extended from
#     pCCA_latent_extrenal_variable_bar.py's own `hubmode_plot_multipanel_
#     bars`: the only addition is `panel_vline_zero`, an optional per-panel
#     dashed zero-reference line. Used by Tasks 3/4 (which simply leave it
#     off, so behaviour there is identical to the copied original); Task 6
#     has its OWN, box-plot-based engine (`_boxplot_one_panel` /
#     `hubmode_plot_task6_enrichment_boxplots`, Section 8) since a per-group
#     cross-session distribution needs a different visual grammar than a
#     mean+SEM bar.
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
    """ONE figure per hub region, ONE row per partner ROI, panels given by
    `panel_titles`/`panel_xlims`. Shared by Tasks 3, 4, and 6."""
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
# 6.  Tasks 3 & 4 -- behavioural variance bars, reproduced from pCCA_latent_
#     extrenal_variable_bar.py's own hub-mode section (same row layout, same
#     two-panel/reward-only display, same single-trial / trial-averaged
#     split, same file-naming convention) -- now each called ONCE PER PCCA
#     VARIANT (`PCCA_VARIANTS`). Task 3's own `external_variables` differs
#     by variant (EXTERNAL_VARIABLES_BY_VARIANT: 4 panels -- position,
#     speed, reward_presence, reward_consumption -- for regions_only, since
#     that latent has NOT had behaviour regressed out of it; unchanged
#     2-panel reward-only view for regions_behavior); Task 4 keeps the
#     2-panel reward-only view for BOTH variants (renamed 4.1/4.2 per this
#     version's request, not 4-column -- see module docstring).
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
    """TWO figures per hub region, reference-condition-only bars: single-
    trial R^2 (from `summary`) and trial-averaged R^2 (from
    `summary_trial_avg`), each with one panel per `external_variables`
    entry -- 2 panels (reward only) for the regions_behavior variant, 4
    panels (position, speed, reward_presence, reward_consumption) for
    regions_only (see EXTERNAL_VARIABLES_BY_VARIANT). `file_suffix`
    (e.g. '_regions_only' / '_regions_behavior') keeps the two variants'
    output files apart."""
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
    """Task-4 counterpart of `hubmode_plot_task3_bars`: every row holds one
    CLUSTER per non-reference trial type (hatch-coded). Always the 2-panel
    reward-only view (EXTERNAL_VARIABLES), regardless of variant -- only the
    caller-supplied `task_label` ('task4_1' for regions_only, 'task4_2' for
    regions_behavior, per this version's request) and `summary`/
    `summary_trial_avg` (already filtered to one variant by the caller)
    differ between the two calls."""
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
# 7.  Task 5 -- latent traces across sessions. Reproduces the SAME trace
#     styling as pCCA_latent_extrenal_variable_bar.py's own
#     `hubmode_plot_task5_latent_traces`, but split into ONE FIGURE PER HUB
#     REGION (per this version's request) instead of one figure pooling
#     every hub's rows together -- saved as
#     `hubmode_task5_latent_traces_comp{component_idx}_{hub}.png`.
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
    rows: List[Tuple[Tuple[str, str], str, str]] = []  # (pair_key, partner, role)
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
        ax.patch.set_alpha(0.07)

        mean_key, sem_key, sessions_key = (
            ('u_mean', 'u_sem', 'u_sessions') if role == 'region_i'
            else ('v_mean', 'v_sem', 'v_sessions')
        )
        for trial_type in active_trial_types:
            if trial_type not in cs.aggregated_projections:
                continue
            agg = cs.aggregated_projections[trial_type]
            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')

            session_traces = agg[sessions_key][:, :, component_idx]
            for sess_trace in session_traces:
                ax.plot(cs.time_bins, sess_trace, color=color, linewidth=0.5, alpha=0.2, zorder=1)

            mean_trace = agg[mean_key][:, component_idx]
            sem_trace = agg[sem_key][:, component_idx]
            is_ref = (trial_type == REFERENCE_TYPE)
            ax.plot(cs.time_bins, mean_trace, color=color, linewidth=2.0 if is_ref else 1.4,
                    alpha=0.85 if is_ref else 0.75,
                    label=f"{trial_type.replace('_', ' ')} (n={agg['n_sessions']})", zorder=3)
            ax.fill_between(cs.time_bins, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=color, alpha=0.15, zorder=2)

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
    """ONE figure PER HUB REGION: one row per (hub, partner) combination,
    showing the hub's own latent trace for that pairing. Saved as
    `hubmode_{task_label}_latent_traces_comp{component_idx}_{hub}.png`
    ('task5_1' for regions_only, 'task5_2' for regions_behavior, per this
    version's request -- `cross_session_analyzers` is already the ONE
    variant's own dict, picked by the caller)."""
    figs: Dict[str, Optional[plt.Figure]] = {}
    for hub, hub_partner_pairs in hub_bands:
        save_path = output_dir / f"{task_label}_hubmode_latent_traces_comp{component_idx}_{hub}.png"
        figs[hub] = _hubmode_plot_task5_one_hub(
            hub, hub_partner_pairs, cross_session_analyzers, save_path,
            component_idx, active_trial_types, row_height, fig_width, dpi,
        )
    return figs


# =============================================================================
# 8.  Task 6 (NEW) -- subregion/laminar-depth weight ENRICHMENT-ratio
#     boxplots. Its own, dedicated plotting engine (NOT
#     `hubmode_plot_multipanel_bars` -- a per-group cross-session
#     distribution needs boxes, not a mean+SEM bar): ONE figure per hub
#     region, laid out 1xn -- one PANEL per partner ROI region, each panel
#     one box per group ('Superficial'/'Deep' for a cortical hub, one box
#     per observed subregion label for a subcortical hub). See module
#     docstring for the full layout rationale.
# =============================================================================

def _boxplot_one_panel(
        ax: plt.Axes,
        categories: List[str],
        values_by_category: Dict[str, np.ndarray],
        colors_by_category: Dict[str, str],
        rng: np.random.Generator,
) -> None:
    """Draw one panel's worth of category boxplots + jittered dots, in the
    style of the attached reference figure: a pale, category-tinted box
    fill (`_lighten`) with a black outline/median/whiskers, and solid,
    category-coloured dots scattered within the box width -- one call per
    category present in `values_by_category` (a category missing from this
    particular panel, e.g. a subregion label never observed for THIS
    partner even though it was observed for a sibling partner in the same
    figure, is simply left blank, not zero-filled)."""
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
    """ONE figure per hub region, 1xn panels (n = number of partner ROI
    regions with any data for that hub), reference-condition-only. Category
    order/colours are fixed per hub (cortical: ['Superficial', 'Deep'];
    subcortical: every subregion label observed for that hub across ANY
    partner, alphabetically ordered) and shared across every panel in the
    figure, so a given colour/x-position always means the same group no
    matter which partner's panel it appears in. `subregion_records` is
    expected to already be filtered to ONE pCCA variant by the caller (see
    PCCA_VARIANTS / main()); `file_suffix` (e.g. '_regions_only' /
    '_regions_behavior') keeps the two variants' output files apart. No
    log2(dominant-group weight ratio) reduction here -- this plots
    `enrichment_ratio` directly, per group, as it already did before this
    variant split (see module docstring)."""
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
# 9.  CSV I/O
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
# 10.  Driver
# =============================================================================

# Filename suffix / display label for each pCCA variant, used throughout
# main() to keep the two variants' output files and console messages apart.
VARIANT_FILE_SUFFIX: Dict[str, str] = {
    'regions_only':     'regions_only',
    'regions_behavior': 'regions_behavior',
}
# Task 4/5's own renumbering, per this version's request ("Task 4 and 5
# become 4.1, 4.2, 5.1, 5.2"): variant order follows PCCA_VARIANTS
# ('regions_only' first, matching the request's own "one using the
# 'regress out only all other regions' result, one using the 'regress out
# all other regions and behavior' result" ordering).
TASK4_LABEL_BY_VARIANT: Dict[str, str] = {'regions_only': 'regions_only', 'regions_behavior': 'regions_behavior'}
TASK5_LABEL_BY_VARIANT: Dict[str, str] = {'regions_only': 'regions_only', 'regions_behavior': 'regions_behavior'}


def main() -> None:
    print("=" * 70)
    print("HUB-MODE TASKS 3-6 -- PART 2c / 2c' PRIVATE pCCA (two regress-out variants)")
    print("(sourced exclusively from pCCA_all_regions_out_behaviour.py's own")
    print(" pcca_all_regions_out_behaviour_sessions_{trial_type}_{align_mode}_results pickles)")
    print("=" * 70)
    print(f"  reference type     : {REFERENCE_TYPE}")
    print(f"  active trial types : {ACTIVE_TRIAL_TYPES}")
    print(f"  align mode         : {ALIGN_MODE}")
    print(f"  pcca variants      : {[VARIANT_DISPLAY[v] for v in PCCA_VARIANTS]}")
    print(f"  component indices  : {COMPONENT_INDICES}  (of {N_COMPONENTS} fit)")
    print(f"  behaviour window   : {BEHAVIOR_TIME_RANGE_S}")
    print(f"  variance method    : {VARIANCE_METHOD}")
    print(f"  external variables : {EXTERNAL_VARIABLES_BY_VARIANT}")
    print(f"  hub-mode hubs      : {HUB_MODE_HUB_REGIONS}")
    print(f"  hub-mode ROIs      : {HUB_MODE_ROI_REGIONS}")
    print(f"  output directory   : {OUTPUT_DIR}")
    print("=" * 70)

    cross_session_analyzers, behavior_records, subregion_records = run_hubmode_analysis()
    hub_bands = hubmode_band_pairs()

    # ---- Tasks 3 & 4 --------------------------------------------------------
    print("\n--- Tasks 3-4: behavioural variance explained (hub-mode) ---")
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

        # Task 3: default_move_onset only (TASK3_ALIGN_MODES) -- a fixed
        # reward kernel is only well defined there without per-trial
        # adjustment for every OTHER alignment mode's own t=0 (see
        # TASK3_ALIGN_MODES's own docstring comment). regions_only gets the
        # 4-column view (position, speed, reward_presence,
        # reward_consumption); regions_behavior keeps the original 2-column
        # reward-only view.
        if ALIGN_MODE in TASK3_ALIGN_MODES:
            hubmode_plot_task3_bars(
                variance_summary, variance_summary_trial_avg, hub_bands, OUTPUT_DIR,
                external_variables=EXTERNAL_VARIABLES_BY_VARIANT[variant], file_suffix=suffix,
            )
        else:
            print(f"  [task 3] skipped for align_mode={ALIGN_MODE!r} "
                  f"(only runs for {TASK3_ALIGN_MODES}) -- variant={variant}")

        # Task 4 -- renamed 4.1 (regions_only) / 4.2 (regions_behavior).
        hubmode_plot_task4_bars(
            variance_summary, variance_summary_trial_avg, hub_bands, OUTPUT_DIR,
            task_label=TASK4_LABEL_BY_VARIANT[variant],
        )

    # ---- Task 5 -- renamed 5.1 (regions_only) / 5.2 (regions_behavior) ------
    print("\n--- Task 5: latent traces across sessions (hub-mode, per hub) ---")
    for variant in PCCA_VARIANTS:
        for comp_idx in COMPONENT_INDICES:
            hubmode_plot_task5_latent_traces(
                hub_bands, cross_session_analyzers[variant], OUTPUT_DIR, component_idx=comp_idx,
                task_label=TASK5_LABEL_BY_VARIANT[variant],
            )

    # ---- Task 6 -------------------------------------------------------------
    print("\n--- Task 6: subregion/laminar ENRICHMENT-ratio boxplots (hub-mode) ---")
    _write_records_csv(subregion_records, OUTPUT_DIR / "hubmode_enrichment_ratio_records.csv")
    for variant in PCCA_VARIANTS:
        suffix = VARIANT_FILE_SUFFIX[variant]
        variant_subregion = [r for r in subregion_records if r['variant'] == variant]
        enrichment_grouped = aggregate_enrichment_ratio(variant_subregion)
        _write_enrichment_summary_csv(
            enrichment_grouped, OUTPUT_DIR / f"hubmode_task6_enrichment_ratio_summary{suffix}.csv")
        hubmode_plot_task6_enrichment_boxplots(
            variant_subregion, hub_bands, OUTPUT_DIR, file_suffix=suffix)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Figures and CSVs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

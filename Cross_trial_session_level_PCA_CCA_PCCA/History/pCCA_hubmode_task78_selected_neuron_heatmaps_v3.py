#!/usr/bin/env python3
r"""
pCCA_hubmode_task78_selected_neuron_heatmaps_v3.py
================================================================================

Standalone script that extracts Tasks 7 & 8 out of
``pCCA_all_regions_hubmode_explain_variable_v3.py`` (top-pCCA-weight-neuron
PSTH heatmaps, pooled across sessions, one figure per hub) into their own
file, sourced from ``pCCA_all_regions_out_behaviour_v3.py``'s pickles
instead of v2's -- so the neurons shown here are the ones v3's
CUMULATIVE-weight selection rule picked (`_select_cumulative_weight_neurons`,
"smallest set of neurons whose pooled weight contribution reaches
`CUMULATIVE_WEIGHT_FRACTION`"), not v2's fixed-percentile top-`
TOP_WEIGHT_FRACTION` cut. Nothing about Task 7/8's own logic changes
because of that swap -- both tasks only ever read whichever neurons
`PrivateLatentPairResult.selected_neurons_i`/`_j` (a `SelectedNeuronSet`)
already names; which selection RULE produced that set is entirely v3's
concern (`pCCA_all_regions_out_behaviour_v3.py`), not this script's.

Task 7 pools these neurons' ORIGINAL (pre-residualization, z-scored)
firing-rate PSTH across every session that contributed any, one heatmap
per (hub, partner) pairing; Task 8 is the same layout with RESIDUALIZED
activity instead (already saved on `SelectedNeuronResidual.residual`, no
reload needed). Both sweep every hub in `HUB_MODE_HUB_REGIONS` (one figure
per hub), not a single hard-coded hub -- same as the original hubmode
script.

--------------------------------------------------------------------------------
Sub-task 2a: shared colour bar per figure
--------------------------------------------------------------------------------
The original hubmode script gave every (hub, partner) panel its OWN
colour bar, each independently scaled to that panel's own 99th-percentile
|activity| -- deliberately, per its own comment, to avoid a shared
colorbar fighting `tight_layout`/`suptitle` spacing. This script instead
gives every panel in ONE figure (one hub, one data source, one pCCA
variant -- i.e. the hub region's heatmap against every paired region it
is shown with) a single COMMON colour scale (`vmin`/`vmax` = the max
99th-percentile |activity| across every panel in that figure) and ONE
shared colour bar for the whole figure, so heatmaps are now visually
comparable panel-to-panel. Everything else about the plot (layout, panel
sizes, titles, axis styling, Rastermap sorting, file naming) is
unchanged from the original.

Data source, dataclasses, and the raw-reload primitives Task 7 needs are
all imported directly from ``pCCA_all_regions_out_behaviour_v3.py`` (the
SAME functions that script itself used to build the `region_flat_full`
pool `SelectedNeuronSet.neurons[].neuron_idx` indexes into, so reloading
with them guarantees the neuron axis lines up) -- nothing about the data
layer is reimplemented differently here.

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
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
# 0.  Imports. `pCCA_all_regions_out_behaviour_v3.py` is this script's ONLY
#     source of neural data: the cumulative-weight-selected neurons
#     (`selected_neurons_i`/`_j`) AND the raw-reload primitives Task 7
#     needs (`load_region_spikes_full` / `crop_time_window` /
#     `_zscore_flat` / `load_behavior_regressors`) -- the SAME functions
#     v3 itself used to build the `region_flat_full` pool
#     `SelectedNeuronSet.neurons[].neuron_idx` indexes into, so reloading
#     with them (rather than reimplementing the load/crop/truncate
#     sequence a second, possibly-diverging way) guarantees the neuron
#     axis lines up. Task 8 needs no such reload -- its data
#     (`SelectedNeuronResidual.residual`) is already saved.
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_trial_type_cca_analysis import (  # noqa: E402
    CrossSessionCCAAnalyzer,
    align_signs_spectral,
    MIN_SESSIONS_THRESHOLD,
)
from pCCA_all_regions_out_behaviour_v3 import (  # noqa: E402
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
    N_SAMPLE_DRAWS,
    N_COMPONENTS,
    CUMULATIVE_WEIGHT_FRACTION,
    sort_pair_by_anatomy,
    get_anatomical_index,
    out_subdir_name,
    mat_subdir_name as v3_mat_subdir_name,
    load_region_spikes_full as v3_load_region_spikes_full,
    crop_time_window as v3_crop_time_window,
    _zscore_flat as v3_zscore_flat,
    load_behavior_regressors as v3_load_behavior_regressors,
    BASE_DIR as V3_BASE_DIR,
    SUBTRACT_PSTH as V3_SUBTRACT_PSTH,
    SHUFFLE_TRIALS as V3_SHUFFLE_TRIALS,
)

# `pCCA_all_regions_out_behaviour_v3.py` bakes '__main__' into every pickled
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
    import mat73  # noqa: F401  (transitively required by v3's own load_region_spikes_full)
except Exception:
    warnings.warn("mat73 not importable -- this script's own Task 7 raw reload may fail.")


# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS -- the subset of
#     `pCCA_all_regions_hubmode_explain_variable_v3.py`'s Section 1 that
#     Tasks 7/8 actually depend on, copied verbatim (values unchanged).
# =============================================================================

REFERENCE_TYPE: str = 'cued_hit_long'

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

# ---- Paths ------------------------------------------------------------
BASE_DIR = V3_BASE_DIR
OUTPUT_DIR = (BASE_DIR / "Paper_output"
              / f"pcca_all_regions_task78_v3_{REFERENCE_TYPE}_{ALIGN_MODE}")
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

PCCA_VARIANTS: Tuple[str, ...] = ('regions_only', 'regions_behavior')
VARIANT_DISPLAY: Dict[str, str] = {
    'regions_only':     'AllRegions only',
    'regions_behavior': 'AllRegions + Behaviour',
}
VARIANT_FILE_SUFFIX: Dict[str, str] = {
    'regions_only':     'regions_only',
    'regions_behavior': 'regions_behavior',
}

HUB_MODE_HUB_REGIONS: List[str] = ['MOs', 'MOp', 'VALVM', 'VPMPO']
HUB_MODE_ROI_REGIONS: List[str] = sorted(
    {r for pair in REGION_PAIRS for r in pair}, key=get_anatomical_index)

DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
}


def _display_name(region: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(region, region)


TICK_FONTSIZE = 18
SAVE_DPI: int = 400

# ---- Tasks 7/8 -- top-pCCA-weight neuron PSTH heatmaps ------------------
# Sweeps every hub in HUB_MODE_HUB_REGIONS (same hub-mode band layout the
# original hubmode script uses, via `hubmode_band_pairs()`), one figure
# per hub -- NOT limited to a single hard-coded hub.
TASK78_TRIAL_TYPE: str = REFERENCE_TYPE
TASK78_PANEL_WIDTH: float = 3.4
TASK78_PANEL_HEIGHT: float = 5.2
# Rastermap fit knobs -- same defaults this project already uses
# (pCCA_sensitive_realsingle_Session_11panel.py's own get_neuron_order).
TASK78_RASTERMAP_KW: Dict = dict(locality=0.0, time_lag_window=10, grid_upsample=10)

# ---- New row: per-panel cross-session private-latent trace, added below
#     each partner's PSTH heatmap (same column, `TASK78_PANEL_WIDTH` wide).
#     Plotting approach/style copied from `pCCA_all_regions_hubmode_
#     explain_variable_v3.py`'s `hubmode_plot_task5_latent_traces` (light
#     per-session lines + bold cross-session mean +/- SEM band, same y-axis
#     limits) -- the only differences are this row's width (matches this
#     figure's own panel width, not Task 5's own `fig_width`) and its line
#     colour, which is keyed by PAIRED REGION here instead of by trial type
#     (Task 7/8 only ever plot ONE trial type, `TASK78_TRIAL_TYPE`).
TASK78_LATENT_ROW_HEIGHT: float = 1.6
TASK78_LATENT_COMPONENT_IDX: int = 0
TASK78_LATENT_YLIM: Tuple[float, float] = (-0.5, 0.75)
_PARTNER_COLOR_CYCLE = plt.cm.tab20.colors
PARTNER_COLORS: Dict[str, Tuple[float, float, float]] = {
    region: _PARTNER_COLOR_CYCLE[i % len(_PARTNER_COLOR_CYCLE)]
    for i, region in enumerate(HUB_MODE_ROI_REGIONS)
}


# =============================================================================
# 2.  Anatomical / hub-mode helpers -- copied verbatim from the original
#     hubmode script.
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


def _trial_type_to_behavior_label(trial_type: str) -> str:
    return trial_type.replace('_', ' ')


# =============================================================================
# 2b.  New row's data path -- cross-session-aggregated private-latent trace
#      for the hub side of one (hub, partner) pCCA pair, restricted to this
#      script's single `TASK78_TRIAL_TYPE` (Tasks 7/8 never sweep multiple
#      trial types the way `pCCA_all_regions_hubmode_explain_variable_v3.
#      py`'s Task 5 does). Copied/adapted from that script's own Task 5
#      data path (`_PrivateLatentSessionAdapter`, `_sign_align_and_pool_
#      draws`, `CrossSessionCCAAnalyzer`) -- same sign-alignment-across-
#      draws-then-across-sessions machinery, just fed one trial type
#      instead of `ACTIVE_TRIAL_TYPES`.
# =============================================================================

class _PrivateLatentSessionAdapter:
    """Minimal duck-typed stand-in for CrossTrialTypeCCAAnalyzer -- copied
    verbatim from `pCCA_all_regions_hubmode_explain_variable_v3.py`."""
    def __init__(self, projections: Dict[str, Dict[str, np.ndarray]], time_bins: np.ndarray):
        self.projections = projections
        self.statistical_results: Dict = {}
        self.time_bins = time_bins


def _sign_align_and_pool_draws(
        draws: List[PrivateLatentPairDrawResult],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool every draw's (n_trials, T, K) `z_i_lat`/`z_j_lat` into one
    (N_SAMPLE_DRAWS * n_trials, T, K) array each, first sign-aligning the
    draws PER COMPONENT via `align_signs_spectral` -- copied verbatim from
    `pCCA_all_regions_hubmode_explain_variable_v3.py` (see that script's
    own docstring for why each independent-fit draw needs this before
    pooling)."""
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


def _gather_task78_latent_trace(
        analyzer: PrivateLatentAnalyzer,
        hub: str,
        partner: str,
        sessions: List[str],
        regions_only: bool,
        trial_type: str = TASK78_TRIAL_TYPE,
        component_idx: int = TASK78_LATENT_COMPONENT_IDX,
        min_sessions: int = MIN_SESSIONS_THRESHOLD,
) -> Optional[Dict[str, object]]:
    """Cross-session-aggregated private-latent trace (component
    `component_idx`) for `hub`'s own side of the (hub, partner) pCCA pair --
    same data path as Task 5's own (per-session sign-aligned-pooled-draws
    mean, then cross-session sign alignment + mean/SEM via
    `CrossSessionCCAAnalyzer.aggregate_projections`), just scoped to this
    script's single `trial_type` instead of every `ACTIVE_TRIAL_TYPES`.

    Returns a dict with 'time_bins' (T,), 'mean' (T,), 'sem' (T,), and
    'sessions' (n_sessions, T) -- the same fields `_hubmode_plot_task5_
    one_hub` reads off `cs.aggregated_projections[trial_type]` -- or None
    if fewer than `min_sessions` sessions contributed.
    """
    pair_key = sort_pair_by_anatomy(hub, partner)
    role = _hub_region_role(hub, pair_key)

    cs = CrossSessionCCAAnalyzer(
        base_dir=str(BASE_DIR), region_pair=pair_key,
        reference_type=trial_type, n_components=N_COMPONENTS,
        min_sessions=min_sessions,
    )
    for session_name in sessions:
        session_result = analyzer.sessions.get(session_name)
        if session_result is None:
            continue
        table = session_result.pairs_regions_only if regions_only else session_result.pairs
        pr = table.get(pair_key)
        if pr is None or not pr.draws:
            continue

        u_all, v_all = _sign_align_and_pool_draws(pr.draws)
        n_tr_total = u_all.shape[0]
        per_trial_type = {
            trial_type: dict(
                u_mean=u_all.mean(axis=0), v_mean=v_all.mean(axis=0),
                u_trials=u_all, v_trials=v_all,
                u_std=u_all.std(axis=0), v_std=v_all.std(axis=0),
                u_sem=u_all.std(axis=0) / np.sqrt(max(n_tr_total, 1)),
                v_sem=v_all.std(axis=0) / np.sqrt(max(n_tr_total, 1)),
                n_trials=n_tr_total,
            )
        }
        adapter = _PrivateLatentSessionAdapter(
            projections=per_trial_type, time_bins=session_result.time_vec,
        )
        cs.add_session_result(session_name, adapter, swap_uv=False)

    if len(cs.session_projections) < min_sessions:
        return None
    cs.aggregate_projections()
    agg = cs.aggregated_projections.get(trial_type)
    if agg is None:
        return None

    mean_key, sem_key, sessions_key = (
        ('u_mean', 'u_sem', 'u_sessions') if role == 'region_i'
        else ('v_mean', 'v_sem', 'v_sessions')
    )
    return dict(
        time_bins=cs.time_bins,
        mean=agg[mean_key][:, component_idx],
        sem=agg[sem_key][:, component_idx],
        sessions=agg[sessions_key][:, :, component_idx],
        n_sessions=agg['n_sessions'],
    )


# =============================================================================
# 3.  Tasks 7 & 8 -- top-pCCA-weight-neuron PSTH heatmaps, pooled across
#     sessions, swept over every hub region in `HUB_MODE_HUB_REGIONS` (same
#     hub-mode band layout via `hubmode_band_pairs()`) -- one figure per
#     hub, each against every region it pairs with -- NOT limited to a
#     single hard-coded hub. Extracted verbatim from
#     `pCCA_all_regions_hubmode_explain_variable_v3.py`'s own Section 10,
#     except `hubmode_plot_task78_heatmaps` now gives every panel in a
#     figure one common colour scale + one shared colour bar (sub-task 2a)
#     instead of each panel its own independently-scaled colour bar.
#
#     Neuron identification is already done -- `PrivateLatentPairResult.
#     selected_neurons_i`/`_j` (a `SelectedNeuronSet`, v3's cumulative-
#     weight selection). Task 7 shows these neurons' ORIGINAL (pre-
#     residualization, z-scored) activity; Task 8 shows their already
#     -saved RESIDUALIZED activity (`SelectedNeuronResidual.residual`).
#
#     Rastermap sorting is applied ONCE, to the FULLY POOLED matrix --
#     every selected neuron from every session that contributed any,
#     stacked first, THEN sorted -- rather than sorting each session's own
#     block independently before stacking. Sorting still runs on the
#     trial-AVERAGED PSTH (one shared T per align_mode), not each
#     session's raw "continuous cross-trial" trace (T*n_trials samples):
#     different sessions generally have different trial counts, so those
#     raw continuous traces are different lengths and still cannot be
#     pooled into one joint Rastermap fit -- only the PSTH's shared T lets
#     every session's selected neurons sit in one (total_neurons, T)
#     matrix, which is what is pooled BEFORE the single Rastermap fit
#     runs. A consequence: since the sort is global, rows from a given
#     session are no longer a contiguous block in the final row order --
#     `session_labels`/`session_counts` (see `_gather_task78_matrix`)
#     describe each session's CONTRIBUTION (provenance) only, not a slice
#     of matrix rows.
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
    which `pCCA_all_regions_out_behaviour_v3.py` never persists
    (`region_flat_full` is transient there). Reproduces that script's OWN
    load -> crop -> behaviour-truncate -> z-score sequence with its OWN
    functions, in the SAME order, so the returned matrix's neuron axis
    lines up EXACTLY with `SelectedNeuronResidual.neuron_idx`. Cached per
    (session, region, trial_type, align_mode) -- reused across every
    partner pairing this hub appears in, since the hub's own raw data
    does not depend on which partner a given panel is about.

    Returns (region_flat, time_vec) -- (T*n_trials, n_full_neurons) and
    (T,) -- or None if unavailable (mirrors v3's own skip conditions).
    """
    key = (session_name, region, trial_type, align_mode)
    if key in _raw_region_cache:
        return _raw_region_cache[key]

    mat_dir = V3_BASE_DIR / v3_mat_subdir_name(trial_type, align_mode)
    session_file = mat_dir / f"{session_name}_analysis_results.mat"
    if not session_file.exists():
        _raw_region_cache[key] = None
        return None

    region_spikes_full, _labels_full, n_trials, T = v3_load_region_spikes_full(str(session_file))
    if region not in region_spikes_full:
        _raw_region_cache[key] = None
        return None

    window = ALIGNMENT_WINDOWS_S[align_mode]
    time_vec_raw = np.linspace(window[0], window[1], T)
    try:
        region_spikes_full, time_vec = v3_crop_time_window(region_spikes_full, time_vec_raw, window)
    except ValueError:
        _raw_region_cache[key] = None
        return None
    T = time_vec.shape[0]

    try:
        pos_sel, speed_sel, _t_behav = v3_load_behavior_regressors(
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

    X_flat = v3_zscore_flat(X, subtract_psth=V3_SUBTRACT_PSTH, shuffle_trials=V3_SHUFFLE_TRIALS)
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
    """Pool one (hub, partner) pairing's already-selected cumulative-weight
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
        latent_row_height: float = TASK78_LATENT_ROW_HEIGHT,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """ONE figure for `hub`: 2xn panels, one COLUMN per partner region with
    any selected-neuron data -- the top row is that partner's PSTH heatmap,
    the new bottom row (same column width) is the cross-session private-
    latent trace for the (hub, partner) pCCA pair, plotted the same way as
    `pCCA_all_regions_hubmode_explain_variable_v3.py`'s `hubmode_plot_
    task5_latent_traces` (light per-session lines + bold mean +/- SEM band,
    same y-axis limits) -- only the panel width (this figure's own column
    width) and the line colour (keyed by paired region here, not trial
    type) differ. Sub-task 2a: every heatmap panel shares ONE common
    colour scale (`vmin`/`vmax` = the max per-panel 99th-percentile
    |activity| across the whole figure) and ONE shared colour bar, instead
    of each panel its own independently-scaled colour bar."""
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
    fig, axes = plt.subplots(
        2, n_panels, figsize=(panel_width * n_panels, panel_height + latent_row_height),
        gridspec_kw=dict(height_ratios=[panel_height, latent_row_height]),
        squeeze=False,
    )
    heatmap_axes = axes[0]
    latent_axes = axes[1]
    cbar_label = 'z-scored firing rate' if data_source == 'raw' else 'residualized activity'

    # ---- Sub-task 2a: ONE common colour scale for every panel in this
    #      figure (hub + every paired region shown alongside it), computed
    #      the same way each panel's own scale used to be (99th percentile
    #      of |activity|), just maxed across panels instead of per-panel. --
    per_panel_vmax = [
        float(np.nanpercentile(np.abs(matrix), 99)) if matrix.size else 0.0
        for _, (matrix, _tv, _sl, _sc) in present
    ]
    vmax_shared = max(per_panel_vmax) if per_panel_vmax else 1.0
    vmax_shared = vmax_shared if vmax_shared > 0 else 1.0

    im = None
    for panel_idx, (ax, (partner, (matrix, time_vec, sess_labels, sess_counts))) in enumerate(
            zip(heatmap_axes, present)):
        im = ax.imshow(
            matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax_shared, vmax=vmax_shared,
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
        print(f"    [{task_label}/{variant}] {_display_name(hub)} <-> {_display_name(partner)}: "
              f"{matrix.shape[0]} neurons from {len(sess_labels)} sessions "
              f"({dict(zip(sess_labels, sess_counts))})")

        # ---- New row: cross-session private-latent trace for this
        #      (hub, partner) pair, same plotting style as Task 5's own
        #      `_hubmode_plot_task5_one_hub` (light per-session lines, bold
        #      mean +/- SEM, same y-axis limits) -- colour keyed by
        #      `partner` (this figure's column) instead of by trial type. --
        lat_ax = latent_axes[panel_idx]
        color = PARTNER_COLORS.get(partner, '#888888')
        trace = _gather_task78_latent_trace(analyzer, hub, partner, sessions, regions_only)
        if trace is not None:
            for sess_trace in trace['sessions']:
                lat_ax.plot(trace['time_bins'], sess_trace, color=color, linewidth=0.5,
                            alpha=0.2, zorder=1)
            lat_ax.plot(trace['time_bins'], trace['mean'], color=color, linewidth=2.0,
                        alpha=0.9, zorder=3)
            lat_ax.fill_between(trace['time_bins'], trace['mean'] - trace['sem'],
                                trace['mean'] + trace['sem'], color=color, alpha=0.18, zorder=2)
            lat_ax.set_xlim(trace['time_bins'][0], trace['time_bins'][-1])
            print(f"    [{task_label}/{variant}] {_display_name(hub)} <-> {_display_name(partner)}: "
                  f"latent trace from {trace['n_sessions']} sessions")
        else:
            print(f"    [{task_label}/{variant}] {_display_name(hub)} <-> {_display_name(partner)}: "
                  f"latent trace unavailable (< {MIN_SESSIONS_THRESHOLD} sessions); row left blank.")
        lat_ax.axvline(x=0, color='black', linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
        lat_ax.set_ylim(*TASK78_LATENT_YLIM)
        lat_ax.text(0.01, 0.90, f"{_display_name(hub)} ↔ {_display_name(partner)}",
                    transform=lat_ax.transAxes, fontsize=TICK_FONTSIZE - 6, va='top', ha='left')
        for sp in ('top', 'right'):
            lat_ax.spines[sp].set_visible(False)
        lat_ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)
        lat_ax.set_xlabel("Time (s)", fontsize=TICK_FONTSIZE - 4)
        lat_ax.tick_params(axis='x', labelsize=TICK_FONTSIZE - 6)

    heatmap_axes[0].set_ylabel("Neurons pooled across sessions", fontsize=TICK_FONTSIZE - 4)

    label = 'original firing rate' if data_source == 'raw' else 'residual activity'
    fig.suptitle(
        f"Hub: {_display_name(hub)} - {label}, >= {int(round(CUMULATIVE_WEIGHT_FRACTION * 100))}% cumulative-weight-selected ",
        fontsize=TICK_FONTSIZE)
    # Reserve a slim strip on the right for the colour bar BEFORE
    # `tight_layout` packs the panels, so the panels never claim that space
    # and the bar always lands flush against the true right edge of the
    # figure (rather than being squeezed in between panels afterwards).
    fig.tight_layout(rect=(0.0, 0.0, 0.90, 0.95))

    # ---- ONE shared colour bar for the whole figure (sub-task 2a),
    #      spanning only the heatmap row's vertical extent. -----------------
    heatmap_top = heatmap_axes[0].get_position().y1
    heatmap_bottom = heatmap_axes[0].get_position().y0
    cbar_ax = fig.add_axes((0.90, heatmap_bottom, 0.012, heatmap_top - heatmap_bottom))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE - 6)
    cbar.set_label(cbar_label, fontsize=TICK_FONTSIZE - 6)

    suffix = VARIANT_FILE_SUFFIX[variant]
    save_path = output_dir / f"{suffix}_hubmode_{task_label}_psth_{hub}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 3b.  Task 9 -- ONE figure per hub combining Task 7's original PSTH and
#      Task 8's residual PSTH (same pooled neurons/rows) with the Task
#      7/8 latent-trace row below them: row 0 = raw, row 1 = residual,
#      row 2 = cross-session private-latent trace. As in Tasks 7/8,
#      `variant` ('regions_only' vs 'regions_behavior') only changes the
#      saved figure's path, nothing about the plot itself.
# =============================================================================

def _gather_task78_matrix_pair(
        analyzer: PrivateLatentAnalyzer,
        hub: str,
        partner: str,
        sessions: List[str],
        regions_only: bool,
        trial_type: str = TASK78_TRIAL_TYPE,
        align_mode: str = ALIGN_MODE,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int]]]:
    """Task 9's data path: gather the RAW (Task 7) and RESIDUAL (Task 8)
    pooled matrices for one (hub, partner) pairing in a SINGLE pass over
    sessions, so both matrices are built from the IDENTICAL per-session
    neuron blocks (`selected.neurons`, in the same order) -- letting the
    residual panel reuse the Rastermap order fit on the raw panel (this
    task's own request) instead of an independent Rastermap fit. The raw
    reload is the gating condition (same skip conditions as Task 7's own
    `_gather_task78_matrix`): a session only contributes if its raw
    z-scored reload succeeds -- the residual side (already saved on
    `SelectedNeuronResidual.residual`) never itself fails that way, so
    restricting to raw-successful sessions is what keeps both matrices'
    rows in lockstep.

    Returns (matrix_raw, matrix_residual, time_vec, session_labels,
    session_neuron_counts) -- both matrices (total_neurons, T), already
    sorted by the SAME single pooled Rastermap fit (computed on
    `matrix_raw`). None if no session contributed any selected neuron.
    """
    pair_key = sort_pair_by_anatomy(hub, partner)
    role = _hub_region_role(hub, pair_key)

    raw_blocks: List[np.ndarray] = []
    residual_blocks: List[np.ndarray] = []
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
        raw_rows = np.stack(
            [cols[:, k].reshape(T_raw, n_trials_raw).T.mean(axis=0)
             for k in range(cols.shape[1])], axis=0)  # (n, T_raw)

        # Residual side reuses the SAME `selected.neurons` list -> the SAME
        # neurons, in the SAME order, as `raw_rows` above -- only their
        # OWN time axis length can still differ (residual's T comes from
        # v3's own private-latent fit, raw's T from this script's
        # reload), so truncate both to whichever is shorter before either
        # joins the cross-session pool below.
        residual_rows_full = np.stack(
            [nr.residual.mean(axis=0) for nr in selected.neurons], axis=0)   # (n, T_res)
        T_session = min(T_raw, residual_rows_full.shape[1])
        raw_rows = raw_rows[:, :T_session]
        residual_rows = residual_rows_full[:, :T_session]
        time_vec = raw_time_vec[:T_session]

        if time_vec_common is None:
            time_vec_common = time_vec
        elif time_vec.shape[0] != time_vec_common.shape[0]:
            T_min = min(time_vec.shape[0], time_vec_common.shape[0])
            raw_rows = raw_rows[:, :T_min]
            residual_rows = residual_rows[:, :T_min]
            time_vec_common = time_vec_common[:T_min]

        raw_blocks.append(raw_rows.astype(np.float64))
        residual_blocks.append(residual_rows.astype(np.float64))
        session_labels.append(session_name)
        session_counts.append(raw_rows.shape[0])

    if not raw_blocks:
        return None

    T_final = time_vec_common.shape[0]
    raw_blocks = [b[:, :T_final] for b in raw_blocks]
    residual_blocks = [b[:, :T_final] for b in residual_blocks]
    matrix_raw = np.concatenate(raw_blocks, axis=0)
    matrix_residual = np.concatenate(residual_blocks, axis=0)

    # ---- ONE Rastermap fit, on the pooled RAW matrix -- the residual
    #      matrix reuses this SAME order (this task's own request) instead
    #      of an independent Rastermap fit of its own. ---------------------
    order = get_neuron_order_2d(matrix_raw)
    matrix_raw = matrix_raw[order]
    matrix_residual = matrix_residual[order]

    return matrix_raw, matrix_residual, time_vec_common, session_labels, session_counts


def hubmode_plot_task9_combined_psth(
        analyzer: PrivateLatentAnalyzer,
        hub: str,
        partners: List[str],
        sessions: List[str],
        output_dir: Path,
        variant: str,                 # 'regions_only' | 'regions_behavior'
        task_label: str = 'task9',
        panel_width: float = TASK78_PANEL_WIDTH,
        panel_height: float = TASK78_PANEL_HEIGHT,
        latent_row_height: float = TASK78_LATENT_ROW_HEIGHT,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """ONE figure for `hub`: 3xn panels, one COLUMN per partner region with
    any selected-neuron data -- row 0 is Task 7's original (pre-
    residualization) PSTH heatmap, row 1 is Task 8's residualized PSTH
    heatmap for the SAME pooled neurons/rows (reusing row 0's own
    Rastermap order -- see `_gather_task78_matrix_pair`), row 2 is the
    same cross-session private-latent trace `hubmode_plot_task78_heatmaps`
    already adds below its own single heatmap. Each heatmap row gets its
    OWN shared colour scale/colour bar across this figure's columns (sub-
    task 2a, applied per row since raw and residual activity live on very
    different scales)."""
    regions_only = (variant == 'regions_only')
    gathered_by_partner = [
        (partner, _gather_task78_matrix_pair(analyzer, hub, partner, sessions, regions_only))
        for partner in partners
    ]
    present = [(p, g) for p, g in gathered_by_partner if g is not None]
    if not present:
        print(f"  [plot] nothing to plot for hub={hub} task={task_label} variant={variant}; skipping.")
        return None

    n_panels = len(present)
    fig, axes = plt.subplots(
        3, n_panels,
        figsize=(panel_width * n_panels, 2 * panel_height + latent_row_height),
        gridspec_kw=dict(height_ratios=[panel_height, panel_height, latent_row_height]),
        squeeze=False,
    )
    raw_axes = axes[0]
    residual_axes = axes[1]
    latent_axes = axes[2]

    # ---- Sub-task 2a, applied PER ROW: one shared colour scale across
    #      every column for the raw row, another (independent) one for the
    #      residual row -- raw and residual activity live on very
    #      different magnitude scales, so one scale shared across BOTH
    #      rows would wash one of them out. -------------------------------
    raw_vmax = max(
        (float(np.nanpercentile(np.abs(m_raw), 99)) if m_raw.size else 0.0
         for _, (m_raw, _m_res, _tv, _sl, _sc) in present), default=1.0)
    raw_vmax = raw_vmax if raw_vmax > 0 else 1.0
    residual_vmax = max(
        (float(np.nanpercentile(np.abs(m_res), 99)) if m_res.size else 0.0
         for _, (_m_raw, m_res, _tv, _sl, _sc) in present), default=1.0)
    residual_vmax = residual_vmax if residual_vmax > 0 else 1.0

    im_raw = None
    im_residual = None
    for panel_idx, (partner, (matrix_raw, matrix_residual, time_vec, sess_labels, sess_counts)) in enumerate(present):
        extent = [time_vec[0], time_vec[-1], matrix_raw.shape[0], 0]

        ax = raw_axes[panel_idx]
        im_raw = ax.imshow(
            matrix_raw, aspect='auto', cmap='RdBu_r', vmin=-raw_vmax, vmax=raw_vmax,
            extent=extent, origin='upper',
        )
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.set_title(f"{_display_name(partner)}\n"
                      f"n={matrix_raw.shape[0]} neurons",
                      fontsize=TICK_FONTSIZE - 5)
        ax.set_xlabel("Time (s)", fontsize=TICK_FONTSIZE - 4)
        ax.tick_params(labelsize=TICK_FONTSIZE - 6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

        # ---- Residual row: SAME rows/order as the raw row above (both
        #      matrices came out of `_gather_task78_matrix_pair` already
        #      sorted by the raw panel's own Rastermap fit). ---------------
        ax2 = residual_axes[panel_idx]
        im_residual = ax2.imshow(
            matrix_residual, aspect='auto', cmap='RdBu_r', vmin=-residual_vmax, vmax=residual_vmax,
            extent=extent, origin='upper',
        )
        ax2.axvline(0.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        ax2.set_xlabel("Time (s)", fontsize=TICK_FONTSIZE - 4)
        ax2.tick_params(labelsize=TICK_FONTSIZE - 6)
        for sp in ('top', 'right'):
            ax2.spines[sp].set_visible(False)

        print(f"    [{task_label}/{variant}] {_display_name(hub)} <-> {_display_name(partner)}: "
              f"{matrix_raw.shape[0]} neurons from {len(sess_labels)} sessions "
              f"({dict(zip(sess_labels, sess_counts))})")

        # ---- Row 2: cross-session private-latent trace, identical data
        #      path/style to `hubmode_plot_task78_heatmaps`'s own new row. --
        lat_ax = latent_axes[panel_idx]
        color = PARTNER_COLORS.get(partner, '#888888')
        trace = _gather_task78_latent_trace(analyzer, hub, partner, sessions, regions_only)
        if trace is not None:
            for sess_trace in trace['sessions']:
                lat_ax.plot(trace['time_bins'], sess_trace, color=color, linewidth=0.5,
                            alpha=0.2, zorder=1)
            lat_ax.plot(trace['time_bins'], trace['mean'], color=color, linewidth=2.0,
                        alpha=0.9, zorder=3)
            lat_ax.fill_between(trace['time_bins'], trace['mean'] - trace['sem'],
                                trace['mean'] + trace['sem'], color=color, alpha=0.18, zorder=2)
            lat_ax.set_xlim(trace['time_bins'][0], trace['time_bins'][-1])
            print(f"    [{task_label}/{variant}] {_display_name(hub)} <-> {_display_name(partner)}: "
                  f"latent trace from {trace['n_sessions']} sessions")
        else:
            print(f"    [{task_label}/{variant}] {_display_name(hub)} <-> {_display_name(partner)}: "
                  f"latent trace unavailable (< {MIN_SESSIONS_THRESHOLD} sessions); row left blank.")
        lat_ax.axvline(x=0, color='black', linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
        lat_ax.set_ylim(*TASK78_LATENT_YLIM)
        lat_ax.text(0.01, 0.90, f"{_display_name(hub)} ↔ {_display_name(partner)}",
                    transform=lat_ax.transAxes, fontsize=TICK_FONTSIZE - 6, va='top', ha='left')
        for sp in ('top', 'right'):
            lat_ax.spines[sp].set_visible(False)
        lat_ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)
        lat_ax.set_xlabel("Time (s)", fontsize=TICK_FONTSIZE - 4)
        lat_ax.tick_params(axis='x', labelsize=TICK_FONTSIZE - 6)

    raw_axes[0].set_ylabel("Neurons pooled across sessions", fontsize=TICK_FONTSIZE - 4)
    residual_axes[0].set_ylabel("Neurons pooled across sessions", fontsize=TICK_FONTSIZE - 4)

    # ---- Title kept in the same style as Tasks 7/8's own suptitle, just
    #      naming BOTH activity types shown together (raw and residual)
    #      instead of picking whichever single `data_source` that figure
    #      was for. -----------------------------------------------------
    fig.suptitle(
        f"Hub: {_display_name(hub)} - original firing rate & residual activity, "
        f">= {int(round(CUMULATIVE_WEIGHT_FRACTION * 100))}% cumulative-weight-selected ",
        fontsize=TICK_FONTSIZE)
    # Reserve a slim strip on the right for the two colour bars BEFORE
    # `tight_layout` packs the panels (same rationale as Tasks 7/8's own).
    fig.tight_layout(rect=(0.0, 0.0, 0.90, 0.95))

    # ---- TWO shared colour bars (sub-task 2a, one per heatmap row), each
    #      spanning only that row's own vertical extent. --------------------
    raw_top = raw_axes[0].get_position().y1
    raw_bottom = raw_axes[0].get_position().y0
    cbar_raw_ax = fig.add_axes((0.90, raw_bottom, 0.012, raw_top - raw_bottom))
    cbar_raw = fig.colorbar(im_raw, cax=cbar_raw_ax)
    cbar_raw.ax.tick_params(labelsize=TICK_FONTSIZE - 6)
    cbar_raw.set_label('z-scored firing rate', fontsize=TICK_FONTSIZE - 6)

    residual_top = residual_axes[0].get_position().y1
    residual_bottom = residual_axes[0].get_position().y0
    cbar_res_ax = fig.add_axes((0.90, residual_bottom, 0.012, residual_top - residual_bottom))
    cbar_res = fig.colorbar(im_residual, cax=cbar_res_ax)
    cbar_res.ax.tick_params(labelsize=TICK_FONTSIZE - 6)
    cbar_res.set_label('residualized activity', fontsize=TICK_FONTSIZE - 6)

    # ---- As in Tasks 7/8, `variant` only changes the saved path. ---------
    suffix = VARIANT_FILE_SUFFIX[variant]
    save_path = output_dir / f"{suffix}_hubmode_{task_label}_psth_{hub}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 4.  Driver
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("HUB-MODE v3 TASKS 7-9 -- cumulative-weight-selected pCCA neurons")
    print("(sourced exclusively from pCCA_all_regions_out_behaviour_v3.py's own")
    print(f" {out_subdir_name(TASK78_TRIAL_TYPE, ALIGN_MODE)} pickles)")
    print("=" * 70)
    print(f"  reference type          : {REFERENCE_TYPE}")
    print(f"  align mode              : {ALIGN_MODE}")
    print(f"  pcca variants           : {[VARIANT_DISPLAY[v] for v in PCCA_VARIANTS]}")
    print(f"  samples per pair        : {N_SAMPLE_DRAWS} draws/session (v3 resampling)")
    print(f"  cumulative weight frac. : {CUMULATIVE_WEIGHT_FRACTION}")
    print(f"  hub-mode hubs           : {HUB_MODE_HUB_REGIONS}")
    print(f"  hub-mode ROIs           : {HUB_MODE_ROI_REGIONS}")
    print(f"  output directory        : {OUTPUT_DIR}")
    print("=" * 70)

    analyzer = PrivateLatentAnalyzer(base_dir=BASE_DIR, trial_type=TASK78_TRIAL_TYPE, align_mode=ALIGN_MODE)
    print(f"[load] trial_type={TASK78_TRIAL_TYPE!r} align_mode={ALIGN_MODE!r} <- {analyzer.results_dir}")
    analyzer.load_all()

    hub_bands = hubmode_band_pairs()

    print("\n--- Tasks 7-9: cumulative-weight-selected pCCA-neuron PSTH heatmaps "
          f"(hubs={HUB_MODE_HUB_REGIONS}) ---")
    for hub, hub_partner_pairs in hub_bands:
        partners78 = [p for _, p in hub_partner_pairs]
        if not partners78:
            print(f"  [task 7/8/9] {hub!r} has no partners in HUB_MODE_ROI_REGIONS; skipping.")
            continue
        for variant in PCCA_VARIANTS:
            hubmode_plot_task78_heatmaps(
                analyzer, hub, partners78, SESSIONS, OUTPUT_DIR,
                data_source='raw', task_label='task7', variant=variant,
            )
            hubmode_plot_task78_heatmaps(
                analyzer, hub, partners78, SESSIONS, OUTPUT_DIR,
                data_source='residual', task_label='task8', variant=variant,
            )
            hubmode_plot_task9_combined_psth(
                analyzer, hub, partners78, SESSIONS, OUTPUT_DIR, variant=variant,
            )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

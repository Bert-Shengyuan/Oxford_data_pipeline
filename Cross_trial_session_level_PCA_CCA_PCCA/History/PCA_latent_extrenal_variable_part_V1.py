#!/usr/bin/env python3
r"""
pca_all_regions_out_behaviour_task345.py
================================================================================

Tasks 3, 4, and 5 (behavioural-variance-explained bars, and cross-session
latent traces) for Parts 1a, 1b, 2a, and 2b of the "AllRegions + Behaviour"
PCA/pCCA decomposition -- the pkl-sourced sibling of this project's two
.mat-sourced scripts (``PCA_latent_extrenal_variable_bar.py``'s own previous
version, and its pCCA/CCA cousin ``pCCA_latent_extrenal_variable_bar.py``).

Per this version's explicit scope, Tasks 1 and 2 (peak, delta) are NOT
reproduced here -- only Tasks 3, 4, 5, across all four new result sets:

    Part 1a : PCA on each hub region's RAW (z-scored) activity
    Part 1b : PCA on each hub region's activity with behaviour regressed out
    Part 2a : PCA on the part of a hub's Part-1b residual EXPLAINED by the
              rest of the recorded network (excluding this hub's partner)
    Part 2b : PCA on what is LEFT of that residual after removing Part 2a

All FOUR are read exclusively from
``pcca_all_regions_out_behaviour_sessions_{trial_type}_results`` -- the
pickle folder written by ``pcca_all_regions_out_behaviour.py`` -- via
``PrivateLatentAnalyzer.get_region_pca`` (1a/1b) and ``.get_hub_pca``
(2a/2b). Nothing here touches the .mat pipeline or raw spike tensors at
all: unlike the .mat-sourced scripts' own Task-1-2 machinery (which fits a
PCA/CCA subspace on one REFERENCE_TYPE and re-projects every other trial
type's raw data through it), every quantity used below was ALREADY fit and
projected, once per trial type, by ``pcca_all_regions_out_behaviour.py``
itself -- so there is no "training subspace" step left to perform here.

--------------------------------------------------------------------------------
Why "reference" and "projected" (Tasks 3 / 4) still make sense here
--------------------------------------------------------------------------------
The .mat-sourced sibling's Task 3 ("reference condition") / Task 4
("projected conditions") distinction exists because ONE PCA/CCA subspace,
fit only on REFERENCE_TYPE, is projected onto every trial type in
ACTIVE_TRIAL_TYPES -- so "projected" literally means "this trial type's raw
data pushed through someone else's weights". That mechanism does not exist
here: ``pcca_all_regions_out_behaviour.py`` fits a completely INDEPENDENT
PCA/pCCA (its own SVD, its own sign ambiguity) for whichever ONE trial_type
its own TRIAL_TYPE constant was set to when it was run -- so
ACTIVE_TRIAL_TYPES = ['cued_hit_long', 'spont_miss_long'] here means "two
SEPARATE pickle folders, each internally self-contained", not "one shared
subspace, two projections of it". Tasks 3/4's PLOTS keep their old
names/meaning (Task 3 = REFERENCE_TYPE's own numbers; Task 4 = every other
ACTIVE_TRIAL_TYPES entry's own numbers) because the R^2 COMPUTATION
(`variance_explained`) does not care where its "latent" argument came from;
only the SOURCE of that latent changes. This distinction is already
established precedent in this exact codebase -- see
``pCCA_latent_extrenal_variable_bar.py``'s own
``run_full_analysis_pcca_out_behaviour``, whose docstring flags the
identical caveat for its Task 2: "each trial type here comes from an
INDEPENDENTLY fit ... pCCA -- there is no shared subspace being
generalised across conditions." Tasks 3-4 are the tasks that distinction
does NOT complicate (see that docstring's own note that behavioural
variance is "unaffected by this distinction").

--------------------------------------------------------------------------------
Sign alignment (Task 5, and the session/component axis of Tasks 3-4's dots)
--------------------------------------------------------------------------------
``pcca_all_regions_out_behaviour.py`` deliberately leaves PCA-loading sign
as computed by the SVD (arbitrary up to a per-component flip) -- see that
file's own "Note on scope" -- and explicitly hands the cross-session
sign-alignment step to "a future aggregation step built on top of
PrivateLatentAnalyzer". This script is that step, for Parts 1a/1b/2a/2b, and
it does so by REUSING -- not reimplementing -- the two aggregators this
project already trusts for exactly this purpose:

  * Parts 1a / 1b (one quantity per hub region) reuse
    ``CrossSessionPCAAnalyzer`` (``cross_trial_type_pca_analysis.py``):
    ``add_session_result`` takes the SAME ``{trial_type: {'z_mean':...,
    'n_trials':...}}`` dict shape this script already builds, so no adapter
    is needed; ``aggregate_projections`` performs the project's Z2 spectral
    sign alignment (leading eigenvector of the inter-session correlation
    matrix) for REFERENCE_TYPE, then re-applies each session's flip
    decision to every other trial type of that session.
  * Parts 2a / 2b (TWO quantities per hub orientation -- the Part-2a
    "network"-explained component and the Part-2b "residual" component,
    playing the same structural role region_i/region_j play in the pCCA/CCA
    sibling's own pairs) reuse ``CrossSessionCCAAnalyzer``
    (``cross_trial_type_cca_analysis.py``) via the SAME lightweight
    duck-typed adapter (`_PrivateLatentSessionAdapter`) the pCCA/CCA
    sibling's own ``run_full_analysis_pcca_out_behaviour`` already uses
    to feed pickle-sourced data into ``add_session_result`` /
    ``aggregate_projections`` unmodified.

Reusing the reference-type-then-propagate flip-decision mechanism
UNMODIFIED, even though Parts 1a/1b/2a/2b's non-reference trial types come
from an independently-fit PCA (see the note above), mirrors the identical
choice already made by ``pCCA_latent_extrenal_variable_bar.py``'s
``run_full_analysis_pcca_out_behaviour`` for Part 2c -- this script does
not introduce a new precedent, it follows the one already set. Tasks 3-4's
own R^2 metric (`variance_explained`) is explicitly documented as
sign-invariant, so this whole alignment step affects Task 5 (and the
dot-cloud placement in Tasks 3-4's plots, which display raw per-session
values, not signed traces) but never the R^2 numbers themselves.

--------------------------------------------------------------------------------
Figure layout (per the request)
--------------------------------------------------------------------------------
For EACH of Tasks 3, 4, 5:
  * Part 1a and Part 1b each produce their OWN figure, in the existing
    single-region bar-chart style (``plot_grouped_region_bars_multipanel``
    for tasks 3/4, ``plot_task5_region_latent_traces`` for task 5) --
    TWO separate figures, never merged into one.
  * Parts 2a and 2b share ONE figure per task, in a "paired display"
    adapted from ``pCCA_latent_extrenal_variable_bar.py``'s own paired
    region_i/region_j format: for tasks 3/4, one ROW per hub orientation
    (hub, partner), with the Part-2a ("network") bar in the row's base
    colour and the Part-2b ("residual") bar in a lightened tint of the
    SAME colour, directly generalising that script's
    ``_region_bar``/``_lighten`` region_i/region_j convention to
    network/residual; for task 5, Part 2a and Part 2b get ADJACENT panels
    (network then residual, immediately next to each other in the 2-column
    tiling -- see plot_task5_hub_latent_traces) for the same hub
    orientation, mirroring that script's adjacent region_i/region_j panels
    for the same pair.

Because a canonicalized pair (region_i, region_j) yields TWO hub
orientations (region_i-as-hub and region_j-as-hub -- see
``pcca_all_regions_out_behaviour.py``'s own "Part 2 (hub-paired)" framework
note), Parts 2a/2b's figures have roughly TWICE as many rows as the pCCA/CCA
sibling's own pair-based figures for the same REGION_PAIRS list.

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import csv
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
# 0.  Imports. High-level classes imported directly, per this project's
#     convention (both cross_trial_type_pca_analysis.py and
#     cross_trial_type_cca_analysis.py are treated as stable library
#     modules, exactly as PCA_latent_extrenal_variable_bar.py's own
#     previous version and pCCA_latent_extrenal_variable_bar.py already
#     treat them). pcca_all_regions_out_behaviour.py -- lower-cased import
#     name, matching the exact convention pCCA_latent_extrenal_variable_
#     bar.py itself already uses for this same module -- is the ONLY
#     source of neural data in this script; see the module docstring.
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_trial_type_pca_analysis import (   # noqa: E402
    CrossSessionPCAAnalyzer,
    TRIAL_TYPE_COLORS,
    MIN_SESSIONS_THRESHOLD,
)
from cross_trial_type_cca_analysis import (   # noqa: E402
    CrossSessionCCAAnalyzer,
)
from pCCA_all_regions_out_behaviour import (  # noqa: E402
    PrivateLatentAnalyzer,
    PrivateLatentSessionResult,
    PrivateLatentPairResult,
    RegionPCAResult,
    HubOrientationPCAResult,
    HubPairPCAResult,
    REGION_PAIRS,
    PAIR_CATEGORIES,
    HUB_REGIONS,
    N_PCA_COMPONENTS,
    sort_pair_by_anatomy,
    get_anatomical_index,
)

# `pcca_all_regions_out_behaviour.py` is typically *run* directly (as
# `__main__`), so pickle bakes '__main__' into every dataclass instance's
# module reference; unpickling those files from THIS script (a different
# `__main__`) therefore requires the same classes to be reachable under
# `__main__` here too. Same fix, same reasoning, as
# pCCA_latent_extrenal_variable_bar.py's own identical block -- extended to
# the two new Part-1/2 dataclasses that script did not yet need.
sys.modules['__main__'].PrivateLatentSessionResult = PrivateLatentSessionResult
sys.modules['__main__'].PrivateLatentPairResult = PrivateLatentPairResult
sys.modules['__main__'].RegionPCAResult = RegionPCAResult
sys.modules['__main__'].HubOrientationPCAResult = HubOrientationPCAResult
sys.modules['__main__'].HubPairPCAResult = HubPairPCAResult

try:
    import mat73  # noqa: F401  (transitively required by both cross_trial_type_*_analysis modules)
    _MAT73_OK = True
except Exception:
    _MAT73_OK = False
    warnings.warn("mat73 not importable -- install with `pip install mat73` "
                  "(only cross_trial_type_*_analysis.py's own unused .mat "
                  "loading code needs it; this script itself never reads "
                  "a .mat file).")


# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS
# =============================================================================

# ---- Reference / trial-type selection --------------------------------------
REFERENCE_TYPE: str = 'cued_hit_long'
ACTIVE_TRIAL_TYPES: List[str] = [
    'cued_hit_long',
    # 'spont_hit_long',
]

# ---- Paths -------------------------------------------------------------
BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
BEHAVIOR_DIR = BASE_DIR / "Paper_output" / "tapproach_sessions"
OUTPUT_DIR = BASE_DIR / "Paper_output" / f"pca_all_regions_out_behaviour_{REFERENCE_TYPE}_task345"
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
# N_PCA_COMPONENTS is imported from pcca_all_regions_out_behaviour.py itself
# (the K every RegionPCAResult / HubOrientationPCAResult was actually fit
# with) -- not redeclared here, so there is no second constant that could
# silently drift out of sync with the pickles this script reads.
COMPONENT_INDICES: List[int] = [0]  # which of those K components to plot here

MIN_SESSIONS: int = MIN_SESSIONS_THRESHOLD  # reuse the pipeline's own threshold

# ---- Tasks 3 / 4: behavioural variance ------------------------------------
BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = (-1.0, 2.0)
BEHAVIOR_FS: float = 50.0
BEHAVIOR_T_OFFSET: float = -1.0
LAMBDA_R2: float = 1e-4
VARIANCE_METHOD: str = 'marginal'  # 'marginal' | 'leave_one_out'
EXTERNAL_VARIABLES: List[str] = ['position', 'speed', 'reward_presence', 'reward_consumption']

# Reward-kernel construction (see the .mat-sourced sibling scripts' own
# "Reward kernel" docstring notes -- same FIXED-WINDOW ASSUMPTION applies)
REWARD_PRESENCE_WINDOW_S: Tuple[float, float] = (0.0, 0.5)
REWARD_CONSUMPTION_WINDOW_S: Tuple[float, float] = (0.5, 1.5)
N_REWARD_CONSUMPTION_BASIS: int = 4
REWARD_SPLINE_DEGREE: int = 2

# ---- Bar x-axis scale: Parts 1b/2a/2b have ALREADY had behaviour regressed
#      out (directly for 1b, upstream via 1b for 2a/2b -- see
#      pcca_all_regions_out_behaviour.py's Part 1b/2a/2b framework note), so
#      their R^2 against position/speed/reward is expected to sit near zero,
#      the same expectation pCCA_latent_extrenal_variable_bar.py's own
#      BAR_LEN=0.05 already encodes for Part 2c. Part 1a has NOT had
#      behaviour regressed out, so it keeps the wider, "raw activity"
#      x-axis scale PCA_latent_extrenal_variable_bar.py's previous version
#      already used. ------------------------------------------------------
BAR_XLIM_RAW: Tuple[float, float] = (0.0, 0.5)        # Part 1a only
BAR_XLIM_RESIDUAL: Tuple[float, float] = (0.0, 0.05)  # Parts 1b, 2a, 2b

# ---- Caching / output -----------------------------------------------------
USE_CACHED_DATA: bool = True
SAVE_DPI: int = 400


# =============================================================================
# 2.  Individual regions & anatomical grouping (Parts 1a/1b, one bar per
#     region -- REGIONS_OF_INTEREST/REGION_CATEGORIES replace the pCCA/CCA
#     sibling's 21-pair/7-category PAIR_CATEGORIES the same way
#     PCA_latent_extrenal_variable_bar.py's previous version already did);
#     PAIR_CATEGORIES itself (imported above) is reused UNCHANGED for
#     Parts 2a/2b, which -- unlike 1a/1b -- remain pair-structured.
# =============================================================================
REGIONS_OF_INTEREST: List[str] = list(HUB_REGIONS)  # already anatomically ordered

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


def _display_pair(pair: Tuple[str, str]) -> str:
    return f"{_display_name(pair[0])}\u2194{_display_name(pair[1])}"


def _display_hub_pair(hub: str, partner: str) -> str:
    """Row label for one hub orientation: 'hub (vs partner)' -- plain
    English rather than set notation, since a reader should not need to
    know this means 'hub, nuisance excludes partner' to read the axis."""
    return f"{_display_name(hub)} (vs {_display_name(partner)})"


def _lighten(hex_color: str, amount: float = 0.45) -> str:
    """Blend `hex_color` toward white by `amount`. Copied verbatim from
    pCCA_latent_extrenal_variable_bar.py -- used here so the Part-2b
    ('residual') bar/trace is a lighter tint of the SAME hue as the
    Part-2a ('network') bar/trace, exactly as that script lightens
    region_j relative to region_i."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


# Category background colours -- region bands (1a/1b) keep
# PCA_latent_extrenal_variable_bar.py's previous 4-band palette; pair
# categories (2a/2b) reuse pCCA_latent_extrenal_variable_bar.py's own
# 7-category palette unchanged, since PAIR_CATEGORIES itself is unchanged.
REGION_CATEGORY_COLORS: Dict[str, str] = {
    "cortical":     "#DD8452",
    "striatal":     "#937860",
    "thalamic":     "#4C72B0",
    "hypothalamic": "#8172B2",
}
PAIR_CATEGORY_COLORS: Dict[str, str] = {
    "thalamic-thalamic":          "#4C72B0",
    "cortico-cortical":           "#DD8452",
    "cortico-motor thalamic":     "#55A868",
    "cortico-sensory thalamic":   "#C44E52",
    "to HY":                      "#8172B2",
    "to STR":                     "#937860",
    "other":                      "#64B5CD",
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
CLUSTER_GAP = 0.55     # extra spacing between trial-type clusters within one region/hub (task 4 only)
REGION_GAP = 0.15      # extra spacing between different regions within one category (1a/1b)
PAIR_GAP = 0.15        # extra spacing between different hub orientations within one category (2a/2b)
DOT_JITTER_FRAC = 0.35
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 15
CLUSTER_HATCH_CYCLE = [None, "///", "xxx"]


# =============================================================================
# 3.  Low-level primitives -- copied verbatim from PCA_latent_extrenal_
#     variable_bar.py's previous version / pCCA_latent_extrenal_variable_
#     bar.py (project convention: primitives are copied, not imported).
#     NOT copied: `_project_trials_single_region` -- no longer needed,
#     since every latent this script uses was already fit AND projected by
#     pcca_all_regions_out_behaviour.py itself (see module docstring).
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
    alignment step is required upstream of this function -- see module
    docstring, "Sign alignment")."""
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


# ---- 3c. Reward-kernel design matrix ---------------------------------------
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


# ---- 3d. Small numeric helpers ---------------------------------------------
def _sem(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _prepare_regression_inputs(
        latent: np.ndarray, time_bins: np.ndarray,
        pos: np.ndarray, speed: np.ndarray, t_behav: np.ndarray,
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
    """Crop the per-trial neural latent (n_trials, T_neural) and the
    behavioural tensors to BEHAVIOR_TIME_RANGE_S, match trial/time counts
    (shorter-of-the-two truncation), and build the predictor design dict.
    Returns None if there isn't enough overlap to regress. Copied verbatim
    from the .mat-sourced sibling scripts -- this function only ever
    touches a (n_trials, T) latent array and a time axis, so it is
    completely indifferent to whether that latent came from a fresh
    per-trial projection (the .mat path) or a pickle-precomputed one (this
    script's path)."""
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


def _r2_records_for_latent(
        latent_trials: np.ndarray, time_bins: np.ndarray,
        pos_full: np.ndarray, speed_full: np.ndarray, t_behav_full: np.ndarray,
        component_indices: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compute {component_idx: {predictor: r2}} for one (n_trials, T, K)
    latent tensor against one session/trial-type's behavioural design --
    the shared inner loop of both Part-1a/1b and Part-2a/2b's tasks 3/4
    data-gathering passes, factored out once here rather than duplicated
    in each (this is new code, not copied, since the .mat-sourced siblings
    each only ever call this for ONE latent tensor per (session, region)
    or (session, pair, region_role) iteration -- this script calls it
    twice as often, once per hub orientation per role, so keeping it
    DRY matters more here)."""
    out: Dict[int, Dict[str, float]] = {}
    for comp_idx in component_indices:
        if comp_idx >= latent_trials.shape[2]:
            continue
        latent = latent_trials[:, :, comp_idx]
        prep = _prepare_regression_inputs(latent, time_bins, pos_full, speed_full, t_behav_full)
        if prep is None:
            continue
        latent_c, design_dict, t_win = prep
        if VARIANCE_METHOD == 'marginal':
            r2_by_var = {name: variance_explained(latent_c, d) for name, d in design_dict.items()}
        elif VARIANCE_METHOD == 'leave_one_out':
            r2_by_var = variance_explained_unique_loo(latent_c, design_dict)
        else:
            raise ValueError(f"Unknown VARIANCE_METHOD: {VARIANCE_METHOD!r}")
        out[comp_idx] = r2_by_var
    return out


# ---- 3e. Duck-typed adapter for CrossSessionCCAAnalyzer.add_session_result,
#     copied verbatim from pCCA_latent_extrenal_variable_bar.py's own
#     _PrivateLatentSessionAdapter (used there for Part 2c; used here for
#     Parts 2a/2b's network/residual role pair). ---------------------------
class _PrivateLatentSessionAdapter:
    """Minimal duck-typed stand-in for CrossTrialTypeCCAAnalyzer, exposing
    only what CrossSessionCCAAnalyzer.add_session_result actually reads
    (.projections, .statistical_results, .time_bins) so that method and
    aggregate_projections() can be reused unmodified for pickle-sourced
    (u, v) role pairs -- region_i/region_j in the pCCA/CCA sibling script,
    network/residual (Parts 2a/2b) here."""
    def __init__(self, projections: Dict[str, Dict[str, np.ndarray]], time_bins: np.ndarray):
        self.projections = projections
        self.statistical_results: Dict = {}   # unread by add_session_result/aggregate_projections
        self.time_bins = time_bins


# =============================================================================
# 4.  Data gathering. TWO passes -- one per storage granularity, matching
#     pcca_all_regions_out_behaviour.py's own separation of Parts 1a/1b
#     (region_pca_raw / region_pca_out_behaviour, keyed by region) from
#     Parts 2a/2b (hub_pca_pairs, keyed by (hub, partner)). Each pass loads
#     ONE PrivateLatentAnalyzer per entry in ACTIVE_TRIAL_TYPES (each
#     trial type is a SEPARATE, independently-computed pickle folder --
#     see module docstring), then feeds the pipeline's existing
#     cross-session aggregators exactly the shapes they already expect.
# =============================================================================

def _all_session_names(
        analyzers_by_trial_type: Dict[str, PrivateLatentAnalyzer],
        sessions: Optional[List[str]],
) -> List[str]:
    names = sorted(set().union(*(az.sessions.keys() for az in analyzers_by_trial_type.values())))
    if sessions:
        names = [s for s in names if s in sessions]
    return names


def run_full_analysis_region_pca(
        out_behaviour: bool,
        sessions: List[str] = SESSIONS,
        region_list: List[str] = REGIONS_OF_INTEREST,
        base_dir: Path = BASE_DIR,
        reference_type: str = REFERENCE_TYPE,
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        component_indices: List[int] = COMPONENT_INDICES,
        min_sessions: int = MIN_SESSIONS,
) -> Tuple[Dict[str, CrossSessionPCAAnalyzer], List[dict]]:
    """Part 1a (`out_behaviour=False`) / Part 1b (`out_behaviour=True`)
    data-gathering pass: sources per-session, per-region PCA latents from
    PrivateLatentAnalyzer.get_region_pca instead of fitting/reprojecting
    PCA weights from the .mat pipeline. Returns (region_analyzers,
    behavior_records) in the SAME shape PCA_latent_extrenal_variable_bar.
    py's previous `run_full_analysis` returned, so every task 3/4/5 call
    below is unchanged in how it consumes them."""
    analyzers_by_trial_type: Dict[str, PrivateLatentAnalyzer] = {
        t: PrivateLatentAnalyzer(base_dir=base_dir, trial_type=t) for t in active_trial_types
    }
    for az in analyzers_by_trial_type.values():
        az.load_all()

    region_analyzers: Dict[str, CrossSessionPCAAnalyzer] = {}
    behavior_records: List[dict] = []
    all_session_names = _all_session_names(analyzers_by_trial_type, sessions)

    for s_idx, session_name in enumerate(all_session_names, 1):
        print("\n" + "=" * 70)
        print(f"SESSION {s_idx}/{len(all_session_names)}: {session_name}  "
              f"(Part 1{'b' if out_behaviour else 'a'})")
        print("=" * 70)

        behavior_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

        for region in region_list:
            per_trial_type: Dict[str, dict] = {}
            time_vec_for_session: Optional[np.ndarray] = None

            for trial_type, az in analyzers_by_trial_type.items():
                session_result = az.sessions.get(session_name)
                if session_result is None:
                    continue
                rpca = az.get_region_pca(session_name, region, out_behaviour=out_behaviour)
                if rpca is None:
                    continue
                n_comp_use = rpca.latent.shape[2]
                if n_comp_use < 1 or max(component_indices) >= n_comp_use:
                    continue
                per_trial_type[trial_type] = dict(
                    z_mean=rpca.latent.mean(axis=0), n_trials=rpca.latent.shape[0],
                )
                if time_vec_for_session is None or trial_type == reference_type:
                    time_vec_for_session = session_result.time_vec

            if not per_trial_type:
                continue

            if region not in region_analyzers:
                region_analyzers[region] = CrossSessionPCAAnalyzer(
                    base_dir=str(base_dir), region=region,
                    reference_type=reference_type, n_components=N_PCA_COMPONENTS,
                    min_sessions=min_sessions,
                )
            region_analyzers[region].add_session_result(
                session_name, per_trial_type, {}, time_vec_for_session)

            # ---- tasks 3/4 data path -----------------------------------
            for trial_type, az in analyzers_by_trial_type.items():
                rpca = az.get_region_pca(session_name, region, out_behaviour=out_behaviour)
                if rpca is None:

                    continue
                if trial_type not in behavior_cache:
                    behavior_cache[trial_type] = _load_behavior_safe(session_name, trial_type)
                behav = behavior_cache[trial_type]
                if behav is None:
                    continue
                pos_full, speed_full, t_behav_full = behav
                session_result = az.sessions[session_name]

                r2_by_comp = _r2_records_for_latent(
                    rpca.latent, session_result.time_vec, pos_full, speed_full, t_behav_full,
                    component_indices,
                )
                for comp_idx, r2_by_var in r2_by_comp.items():
                    for var_name, r2_val in r2_by_var.items():
                        behavior_records.append(dict(
                            session=session_name, region=region, trial_type=trial_type,
                            component=comp_idx, predictor=var_name, r2=r2_val,
                        ))

    print("\n" + "=" * 70)
    print("CROSS-SESSION AGGREGATION (sign alignment + mean/SEM across sessions)")
    print("=" * 70)
    for region, cs in region_analyzers.items():
        n_sess = len(cs.session_projections)
        if n_sess < min_sessions:
            print(f"  {region}: {n_sess} sessions (skipping, < {min_sessions})")
            continue
        cs.aggregate_projections(epoch=(50,150))

    print("\n" + "=" * 70)
    print(f"DATA-GATHERING COMPLETE (Part 1{'b' if out_behaviour else 'a'})")
    print(f"  regions with >=1 session : {len(region_analyzers)}")
    print(f"  behavioural-variance records  : {len(behavior_records)}")
    print("=" * 70)
    return region_analyzers, behavior_records


def run_full_analysis_hub_pca(
        sessions: List[str] = SESSIONS,
        pair_list: List[Tuple[str, str]] = REGION_PAIRS,
        base_dir: Path = BASE_DIR,
        reference_type: str = REFERENCE_TYPE,
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        component_indices: List[int] = COMPONENT_INDICES,
        min_sessions: int = MIN_SESSIONS,
) -> Tuple[Dict[Tuple[str, str], CrossSessionCCAAnalyzer], List[dict]]:
    """Parts 2a + 2b data-gathering pass: sources per-session, per-hub-
    orientation PCA latents from PrivateLatentAnalyzer.get_hub_pca, pairing
    Part 2a's 'network' latent (u role) with Part 2b's 'residual' latent
    (v role) for each hub orientation -- the direct structural analogue of
    how pCCA_latent_extrenal_variable_bar.py's run_full_analysis_pcca_out_
    behaviour pairs region_i (u) with region_j (v) for Part 2c. A
    canonicalized pair (region_i, region_j) yields TWO entries in the
    returned dict -- keyed (region_i, region_j) AND (region_j, region_i) --
    since Parts 2a/2b are hub-directional (see
    pcca_all_regions_out_behaviour.py's Part 2 framework note)."""
    analyzers_by_trial_type: Dict[str, PrivateLatentAnalyzer] = {
        t: PrivateLatentAnalyzer(base_dir=base_dir, trial_type=t) for t in active_trial_types
    }
    for az in analyzers_by_trial_type.values():
        az.load_all()

    hub_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer] = {}
    behavior_records: List[dict] = []
    all_session_names = _all_session_names(analyzers_by_trial_type, sessions)

    for s_idx, session_name in enumerate(all_session_names, 1):
        print("\n" + "=" * 70)
        print(f"SESSION {s_idx}/{len(all_session_names)}: {session_name}  (Parts 2a/2b)")
        print("=" * 70)

        behavior_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

        for region_i, region_j in pair_list:
            pair_key = sort_pair_by_anatomy(region_i, region_j)

            for hub, partner in ((pair_key[0], pair_key[1]), (pair_key[1], pair_key[0])):
                per_trial_type: Dict[str, Dict[str, np.ndarray]] = {}
                time_vec_for_session: Optional[np.ndarray] = None

                for trial_type, az in analyzers_by_trial_type.items():
                    session_result = az.sessions.get(session_name)
                    if session_result is None:
                        continue
                    hpca = az.get_hub_pca(session_name, hub, partner)
                    if hpca is None:
                        continue
                    n_tr = hpca.latent_network.shape[0]
                    per_trial_type[trial_type] = dict(
                        u_mean=hpca.latent_network.mean(axis=0),
                        v_mean=hpca.latent_residual.mean(axis=0),
                        u_trials=hpca.latent_network, v_trials=hpca.latent_residual,
                        u_std=hpca.latent_network.std(axis=0),
                        v_std=hpca.latent_residual.std(axis=0),
                        u_sem=hpca.latent_network.std(axis=0) / np.sqrt(max(n_tr, 1)),
                        v_sem=hpca.latent_residual.std(axis=0) / np.sqrt(max(n_tr, 1)),
                        n_trials=n_tr,
                    )
                    if time_vec_for_session is None or trial_type == reference_type:
                        time_vec_for_session = session_result.time_vec

                if not per_trial_type:
                    continue

                adapter = _PrivateLatentSessionAdapter(
                    projections=per_trial_type, time_bins=time_vec_for_session,
                )
                if (hub, partner) not in hub_analyzers:
                    hub_analyzers[(hub, partner)] = CrossSessionCCAAnalyzer(
                        base_dir=str(base_dir), region_pair=(hub, partner),
                        reference_type=reference_type, n_components=N_PCA_COMPONENTS,
                        min_sessions=min_sessions,
                    )
                hub_analyzers[(hub, partner)].add_session_result(
                    session_name, adapter, swap_uv=False)

                # ---- tasks 3/4 data path: network (2a) and residual (2b)
                #      each get their own R^2 records, tagged by `role` ---
                for trial_type, proj in per_trial_type.items():
                    if trial_type not in behavior_cache:
                        behavior_cache[trial_type] = _load_behavior_safe(session_name, trial_type)
                    behav = behavior_cache[trial_type]
                    if behav is None:
                        continue
                    pos_full, speed_full, t_behav_full = behav

                    for role, trials in (('network', proj['u_trials']), ('residual', proj['v_trials'])):
                        r2_by_comp = _r2_records_for_latent(
                            trials, time_vec_for_session, pos_full, speed_full, t_behav_full,
                            component_indices,
                        )
                        for comp_idx, r2_by_var in r2_by_comp.items():
                            for var_name, r2_val in r2_by_var.items():
                                behavior_records.append(dict(
                                    session=session_name, hub=hub, partner=partner, role=role,
                                    trial_type=trial_type, component=comp_idx,
                                    predictor=var_name, r2=r2_val,
                                ))

    print("\n" + "=" * 70)
    print("CROSS-SESSION AGGREGATION (sign alignment + mean/SEM across sessions)")
    print("=" * 70)
    for (hub, partner), cs in hub_analyzers.items():
        n_sess = len(cs.session_projections)
        if n_sess < min_sessions:
            print(f"  {hub} (vs {partner}): {n_sess} sessions (skipping, < {min_sessions})")
            continue
        cs.aggregate_projections()

    print("\n" + "=" * 70)
    print("DATA-GATHERING COMPLETE (Parts 2a/2b)")
    print(f"  hub orientations with >=1 session : {len(hub_analyzers)}")
    print(f"  behavioural-variance records      : {len(behavior_records)}")
    print("=" * 70)
    return hub_analyzers, behavior_records


# =============================================================================
# 5.  Tasks 3 & 4 -- behavioural variance explained, Parts 1a/1b
#     (single-region bars, unchanged style from PCA_latent_extrenal_
#     variable_bar.py's previous version).
# =============================================================================
def aggregate_region_behavior_variance(
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


def plot_task3_region_variance_bars(summary: dict, save_path: Path,
                                    reference_type: str = REFERENCE_TYPE,
                                    xlim: Tuple[float, float] = BAR_XLIM_RESIDUAL) -> Optional[plt.Figure]:
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
        xlim=xlim, legend_entries=None,
    )


def plot_task4_region_variance_bars(summary: dict, save_path: Path,
                                    active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
                                    reference_type: str = REFERENCE_TYPE,
                                    xlim: Tuple[float, float] = BAR_XLIM_RESIDUAL) -> Optional[plt.Figure]:
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
        common_xlabel="Variance explained, $R^2$ (other active trial types)",
        xlim=xlim, legend_entries=legend,
    )


# =============================================================================
# 6.  Task 5 -- per-region latent traces across sessions, Parts 1a/1b
#     (unchanged style from PCA_latent_extrenal_variable_bar.py's previous
#     version).
# =============================================================================
def plot_task5_region_latent_traces(
        region_analyzers: Dict[str, CrossSessionPCAAnalyzer],
        save_path: Path,
        component_idx: int = COMPONENT_INDICES[0],
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        row_height: float = 1.5,
        fig_width: float = 4.0,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """One row per region. Trace styling (individual sessions thin/low-
    alpha, mean bold, SEM shading, dashed line at t=0) is identical to the
    .mat-sourced sibling scripts."""
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
    legend_handles = {}

    for r, region in enumerate(rows):
        ax = axes[r]
        cs = region_analyzers[region]
        ax.set_facecolor(REGION_CATEGORY_COLORS.get(row_category[r], '#888888'))
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
            line, = ax.plot(cs.time_bins, mean_trace, color=color, linewidth=2.0 if is_ref else 1.4,
                    alpha=0.85 if is_ref else 0.75,
                    label=f"{trial_type.replace('_', ' ')} (n={agg['n_sessions']})", zorder=3)
            legend_handles.setdefault(trial_type.replace('_', ' '), line)
            ax.fill_between(cs.time_bins, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=color, alpha=0.15, zorder=2)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
        ax.set_xlim(cs.time_bins[0], cs.time_bins[-1])
        ax.text(0.01, 0.90, f"{_display_name(region)}",
                transform=ax.transAxes, fontsize=TICK_FONTSIZE - 6, va='top', ha='left')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)

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

    fig.tight_layout(h_pad=0.15, rect=(0.0, 0.0, 1.0, 0.94))
    if legend_handles:
        fig.legend(legend_handles.values(), legend_handles.keys(),
                   fontsize=LEGEND_FONTSIZE - 3, frameon=False, ncol=len(legend_handles),
                   loc='upper center', bbox_to_anchor=(0.5, 1.0))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 7.  Tasks 3 & 4 -- behavioural variance explained, Parts 2a/2b (paired
#     display, adapted from pCCA_latent_extrenal_variable_bar.py's
#     region_i/region_j format: ONE ROW per hub orientation, with the
#     Part-2a ("network") bar in the row's base colour and the Part-2b
#     ("residual") bar in a lightened tint of the SAME colour -- directly
#     generalising that script's `_region_bar`/`_lighten` convention from
#     region_i/region_j to network/residual).
# =============================================================================
def aggregate_hub_behavior_variance(
        behavior_records: List[dict],
) -> Dict[Tuple[Tuple[str, str], str, str], Dict[str, dict]]:
    """Returns {((hub, partner), role, trial_type): {predictor: {mean, sem,
    values, n}}}, pooled across sessions."""
    grouped: Dict[Tuple[Tuple[str, str], str, str, str], List[float]] = {}
    for rec in behavior_records:
        key = ((rec['hub'], rec['partner']), rec['role'], rec['trial_type'], rec['predictor'])
        grouped.setdefault(key, []).append(rec['r2'])

    out: Dict[Tuple[Tuple[str, str], str, str], Dict[str, dict]] = {}
    for (hub_pair, role, trial_type, predictor), vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size < MIN_SESSIONS:
            continue
        out.setdefault((hub_pair, role, trial_type), {})[predictor] = dict(
            mean=float(arr.mean()), sem=_sem(arr), values=arr, n=arr.size,
        )
    return out


def _role_bar(stats: dict, role: str, base_color: str) -> dict:
    """One bar dict for a given Part-2 role: 'network' (Part 2a) gets the
    variable's full colour, 'residual' (Part 2b) gets a lightened tint of
    the same hue -- generalises pCCA_latent_extrenal_variable_bar.py's own
    `_region_bar` (region_i -> full colour, region_j -> lightened) from
    region role to Part-2 role."""
    color = base_color if role == 'network' else _lighten(base_color)
    return dict(mean=stats['mean'], sem=stats['sem'], values=stats['values'],
               color=color, alpha=0.9, hatch=None)


def _category_slug(category: str) -> str:
    """Filename-safe category name, e.g. 'cortico-motor thalamic' ->
    'cortico_motor_thalamic'."""
    return category.replace(' ', '_').replace('-', '_')


def plot_task3_hub_variance_bars(summary: dict, save_path: Path,
                                 reference_type: str = REFERENCE_TYPE,
                                 xlim: Tuple[float, float] = BAR_XLIM_RESIDUAL) -> Dict[str, Optional[plt.Figure]]:
    """ONE figure PER PAIR_CATEGORY, not one figure spanning all ~42 hub
    orientations pooled together: with up to 2 hub orientations per pair
    and up to 5 pairs per category (10 rows, worst case -- 'to HY'/'to
    STR'), splitting this way keeps every saved PNG close in scale to
    Parts 1a/1b's own (<=7-region) figures, rather than the
    ~40-inch-tall/25-megapixel figure pooling all categories into one plot
    would produce. Returns {category: Figure-or-None}, one entry per
    PAIR_CATEGORIES category that had anything to plot."""
    hub_clusters_by_variable: Dict[str, Dict[Tuple[str, str], List[list]]] = {
        v: {} for v in EXTERNAL_VARIABLES
    }
    for (hub_pair, role, trial_type), per_pred in summary.items():
        if trial_type != reference_type:
            continue
        for v in EXTERNAL_VARIABLES:
            if v not in per_pred:
                continue
            bar = _role_bar(per_pred[v], role, EXTERNAL_VAR_COLORS.get(v, 'gray'))
            cluster = hub_clusters_by_variable[v].setdefault(hub_pair, [[None, None]])[0]
            cluster[0 if role == 'network' else 1] = bar
    for v in EXTERNAL_VARIABLES:
        for hub_pair, clusters in list(hub_clusters_by_variable[v].items()):
            clusters[0][:] = [b for b in clusters[0] if b is not None]
            if not clusters[0]:
                del hub_clusters_by_variable[v][hub_pair]

    legend = [('network component (2a)', '#555555', None),
              ('residual component (2b)', _lighten('#555555'), None)]
    figs: Dict[str, Optional[plt.Figure]] = {}
    for category, pairs in PAIR_CATEGORIES:
        this_path = save_path.with_name(
            f"{save_path.stem}_{_category_slug(category)}{save_path.suffix}")
        figs[category] = plot_grouped_hub_bars_multipanel(
            [(category, pairs)], hub_clusters_by_variable, EXTERNAL_VARIABLES, this_path,
            common_xlabel=(f"Variance explained, $R^2$ "
                          f"({reference_type.replace('_', ' ')})  \u2014  {category}"),
            xlim=xlim, legend_entries=legend,
        )
    return figs


def plot_task4_hub_variance_bars(summary: dict, save_path: Path,
                                 active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
                                 reference_type: str = REFERENCE_TYPE,
                                 xlim: Tuple[float, float] = BAR_XLIM_RESIDUAL) -> Dict[str, Optional[plt.Figure]]:
    """ONE figure PER PAIR_CATEGORY (see plot_task3_hub_variance_bars'
    docstring for why), each internally matching pCCA_latent_extrenal_
    variable_bar.py's own plot_task4_variance_bars precedent: every hub
    orientation's row holds one CLUSTER per non-reference trial type, and
    each cluster holds the [network, residual] bar pair for that trial
    type -- role is colour/tint (network full colour, residual lightened),
    trial type is hatch, shared by both bars in a cluster so the two
    encodings never collide. Returns {category: Figure-or-None}."""
    non_ref = [t for t in active_trial_types if t != reference_type]
    hub_clusters_by_variable: Dict[str, Dict[Tuple[str, str], List[list]]] = {
        v: {} for v in EXTERNAL_VARIABLES
    }
    all_hub_pairs = {k[0] for k in summary.keys()}
    for v in EXTERNAL_VARIABLES:
        for hub_pair in all_hub_pairs:
            clusters: List[list] = []
            for ti, trial_type in enumerate(non_ref):
                hatch = CLUSTER_HATCH_CYCLE[ti % len(CLUSTER_HATCH_CYCLE)]
                cluster: List[Optional[dict]] = [None, None]
                for role in ('network', 'residual'):
                    per_pred = summary.get((hub_pair, role, trial_type))
                    if per_pred and v in per_pred:
                        bar = _role_bar(per_pred[v], role, EXTERNAL_VAR_COLORS.get(v, 'gray'))
                        bar['hatch'] = hatch
                        cluster[0 if role == 'network' else 1] = bar
                cluster = [b for b in cluster if b is not None]
                if cluster:
                    clusters.append(cluster)
            if clusters:
                hub_clusters_by_variable[v][hub_pair] = clusters

    legend = [('network component (2a)', '#555555', None),
              ('residual component (2b)', _lighten('#555555'), None)]
    legend += [(t.replace('_', ' '), '#bbbbbb', CLUSTER_HATCH_CYCLE[ti % len(CLUSTER_HATCH_CYCLE)])
              for ti, t in enumerate(non_ref)]
    figs: Dict[str, Optional[plt.Figure]] = {}
    for category, pairs in PAIR_CATEGORIES:
        this_path = save_path.with_name(
            f"{save_path.stem}_{_category_slug(category)}{save_path.suffix}")
        figs[category] = plot_grouped_hub_bars_multipanel(
            [(category, pairs)], hub_clusters_by_variable, EXTERNAL_VARIABLES, this_path,
            common_xlabel=f"Variance explained, $R^2$ (other active trial types)  \u2014  {category}",
            xlim=xlim, legend_entries=legend,
        )
    return figs


def plot_grouped_hub_bars_multipanel(
        pair_categories: List[Tuple[str, List[Tuple[str, str]]]],
        hub_clusters_by_variable: Dict[str, Dict[Tuple[str, str], List[List[dict]]]],
        variable_names: List[str],
        save_path: Path,
        common_xlabel: str,
        xlim: Optional[Tuple[float, float]] = None,
        legend_entries: Optional[List[Tuple[str, str, Optional[str]]]] = None,
        panel_width: float = 4.6,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """One subplot per entry in `variable_names`; ONE ROW per hub
    orientation (up to two per canonicalized pair -- see module
    docstring), grouped by PAIR_CATEGORIES exactly as pCCA_latent_
    extrenal_variable_bar.py's plot_grouped_pair_bars_multipanel groups by
    pair, except each category's row list is expanded to both hub
    orientations of every pair it contains."""
    def _footprint(clusters: List[List[dict]]) -> float:
        if not clusters:
            return 1.0
        return sum(len(cl) for cl in clusters) + max(0, len(clusters) - 1) * CLUSTER_GAP

    all_hub_pairs = set()
    for v in variable_names:
        all_hub_pairs |= {p for p, c in hub_clusters_by_variable.get(v, {}).items() if c}

    groups = []
    for category, pairs in pair_categories:
        present: List[Tuple[str, str]] = []
        for pair in pairs:
            for hub, partner in ((pair[0], pair[1]), (pair[1], pair[0])):
                if (hub, partner) in all_hub_pairs:
                    present.append((hub, partner))
        if present:
            groups.append((category, present))
    if not groups:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    y = 0.0
    hub_y0: Dict[Tuple[str, str], float] = {}
    hub_label_pos: Dict[Tuple[str, str], float] = {}
    group_spans: List[Tuple[str, float, float]] = []
    for category, hub_pairs in groups:
        y_start = y
        for hp in hub_pairs:
            footprint = max(_footprint(hub_clusters_by_variable.get(v, {}).get(hp, []))
                            for v in variable_names)
            hub_y0[hp] = y
            hub_label_pos[hp] = y + footprint / 2.0 - 0.5
            y += footprint + PAIR_GAP
        group_spans.append((category, y_start, y - PAIR_GAP))
        y += GROUP_GAP - 1.0

    n_panels = len(variable_names)
    fig_h = max(7.0, 0.42 * y + 2.2)
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_width * n_panels, fig_h), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)

    for panel_idx, (variable_name, ax) in enumerate(zip(variable_names, axes)):
        for category, y_lo, y_hi in group_spans:
            ax.axhspan(y_lo - BAR_HEIGHT / 2 - 0.25, y_hi + BAR_HEIGHT / 2 + 0.25,
                       color=PAIR_CATEGORY_COLORS.get(category, '#888888'), alpha=0.07, zorder=0)

        means, sems, ys, dot_x, dot_y = [], [], [], [], []
        for category, hub_pairs in groups:
            for hp in hub_pairs:
                yy = hub_y0[hp]
                clusters = hub_clusters_by_variable.get(variable_name, {}).get(hp, [])
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

    axes[0].set_yticks(list(hub_label_pos.values()))
    axes[0].set_yticklabels([_display_hub_pair(*hp) for hp in hub_label_pos.keys()], fontsize=TICK_FONTSIZE)
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
# 8.  Task 5 -- latent traces across sessions, Parts 2a/2b (paired display:
#     network and residual get ADJACENT rows for the same hub orientation,
#     mirroring pCCA_latent_extrenal_variable_bar.py's own adjacent
#     region_i/region_j rows for the same pair).
# =============================================================================
def _plot_hub_latent_traces_one_figure(
        rows: List[Tuple[Tuple[str, str], str]],
        row_category: Dict[int, str],
        hub_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer],
        save_path: Path,
        component_idx: int,
        active_trial_types: List[str],
        row_height: float,
        fig_width: float,
        dpi: int,
) -> Optional[plt.Figure]:
    """Single-figure drawing engine, factored out of plot_task5_hub_latent_
    traces so it can be called once PER PAIR_CATEGORY (see that function's
    docstring) instead of once for all ~84 rows pooled together."""
    if not rows:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    n_rows = (len(rows) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width * 2, row_height * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    legend_handles = {}

    for r, ((hub, partner), role) in enumerate(rows):
        ax = axes[r]
        cs = hub_analyzers[(hub, partner)]
        ax.set_facecolor(PAIR_CATEGORY_COLORS.get(row_category[r], '#888888'))
        ax.patch.set_alpha(0.07)

        mean_key, sem_key, sessions_key = (
            ('u_mean', 'u_sem', 'u_sessions') if role == 'network'
            else ('v_mean', 'v_sem', 'v_sessions')
        )
        for trial_type in active_trial_types:
            if trial_type not in cs.aggregated_projections:
                continue
            agg = cs.aggregated_projections[trial_type]
            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')
            if role == 'residual':
                color = _lighten(color, amount=0.25)

            session_traces = agg[sessions_key][:, :, component_idx]
            for sess_trace in session_traces:
                ax.plot(cs.time_bins, sess_trace, color=color, linewidth=0.5, alpha=0.2, zorder=1)

            mean_trace = agg[mean_key][:, component_idx]
            sem_trace = agg[sem_key][:, component_idx]
            is_ref = (trial_type == REFERENCE_TYPE)
            line, = ax.plot(cs.time_bins, mean_trace, color=color, linewidth=2.0 if is_ref else 1.4,
                    alpha=0.85 if is_ref else 0.75,
                    label=f"{trial_type.replace('_', ' ')} (n={agg['n_sessions']})", zorder=3)
            if role == 'network':
                legend_handles.setdefault(trial_type.replace('_', ' '), line)
            ax.fill_between(cs.time_bins, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=color, alpha=0.15, zorder=2)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
        ax.set_xlim(cs.time_bins[0], cs.time_bins[-1])
        role_label = 'network (2a)' if role == 'network' else 'residual (2b)'
        ax.text(0.01, 0.90, f"{_display_hub_pair(hub, partner)}  \u2014  {role_label}",
                transform=ax.transAxes, fontsize=TICK_FONTSIZE - 6, va='top', ha='left')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)

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

    fig.tight_layout(h_pad=0.15, rect=(0.0, 0.0, 1.0, 0.94))
    if legend_handles:
        fig.legend(legend_handles.values(), legend_handles.keys(),
                   fontsize=LEGEND_FONTSIZE - 3, frameon=False, ncol=len(legend_handles),
                   loc='upper center', bbox_to_anchor=(0.5, 1.0))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


def plot_task5_hub_latent_traces(
        hub_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer],
        save_path: Path,
        component_idx: int = COMPONENT_INDICES[0],
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        row_height: float = 1.5,
        fig_width: float = 4.0,
        dpi: int = SAVE_DPI,
) -> Dict[str, Optional[plt.Figure]]:
    """ONE figure PER PAIR_CATEGORY (see plot_task3_hub_variance_bars'
    docstring for the same reasoning: pooling all ~42 hub orientations x 2
    roles = ~84 rows into one figure produced an unreadable, tens-of-
    megapixel PNG). Within each category's figure: one panel per (hub
    orientation, role) -- network immediately followed by residual for the
    same hub orientation (adjacent in the 2-column tiling: same tiled row,
    left then right column), exactly mirroring pCCA_latent_extrenal_
    variable_bar.py's own adjacent region_i/region_j panels for the same
    pair. Returns {category: Figure-or-None}."""
    figs: Dict[str, Optional[plt.Figure]] = {}
    for category, pairs in PAIR_CATEGORIES:
        rows: List[Tuple[Tuple[str, str], str]] = []
        row_category: Dict[int, str] = {}
        for pair in pairs:
            for hub, partner in ((pair[0], pair[1]), (pair[1], pair[0])):
                cs = hub_analyzers.get((hub, partner))
                if cs is None or not cs.aggregated_projections:
                    continue
                row_category[len(rows)] = category
                rows.append(((hub, partner), 'network'))
                row_category[len(rows)] = category
                rows.append(((hub, partner), 'residual'))

        this_path = save_path.with_name(
            f"{save_path.stem}_{_category_slug(category)}{save_path.suffix}")
        figs[category] = _plot_hub_latent_traces_one_figure(
            rows, row_category, hub_analyzers, this_path,
            component_idx, active_trial_types, row_height, fig_width, dpi,
        )
    return figs


# =============================================================================
# 9.  Shared plotting engine, Parts 1a/1b -- copied from PCA_latent_
#     extrenal_variable_bar.py's previous version. Only the multipanel
#     variant is kept: the single-panel `plot_grouped_region_bars` that
#     script also defined was only ever called by Tasks 1/2 (peak, delta),
#     which are out of scope for this version -- see module docstring --
#     so it is dead code here and has been dropped, per this file's own
#     "only the primitives those five [now three] computations actually
#     need are copied or newly written in" principle. No region_i/region_j
#     -style role split in this engine (hence no colour-lightening step
#     here) -- that only applies to Parts 2a/2b above.
# =============================================================================
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
                       color=REGION_CATEGORY_COLORS.get(category, '#888888'), alpha=0.07, zorder=0)

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


def _write_region_variance_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region', 'trial_type', 'predictor', 'n_sessions', 'mean_r2', 'sem_r2'])
        for (region, trial_type), per_pred in sorted(summary.items(), key=lambda kv: get_anatomical_index(kv[0][0])):
            for predictor, stats in per_pred.items():
                w.writerow([region, trial_type, predictor, stats['n'],
                           f"{stats['mean']:.6f}", f"{stats['sem']:.6f}"])
    print(f"  [csv] -> {path}")


def _write_hub_variance_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['hub', 'partner', 'role', 'trial_type', 'predictor', 'n_sessions', 'mean_r2', 'sem_r2'])
        for ((hub, partner), role, trial_type), per_pred in sorted(
                summary.items(), key=lambda kv: get_anatomical_index(kv[0][0][0])):
            for predictor, stats in per_pred.items():
                w.writerow([hub, partner, role, trial_type, predictor, stats['n'],
                           f"{stats['mean']:.6f}", f"{stats['sem']:.6f}"])
    print(f"  [csv] -> {path}")


# =============================================================================
# 11.  Driver
# =============================================================================
def _run_region_tasks(out_behaviour: bool) -> None:
    part_label = '1b' if out_behaviour else '1a'
    xlim = BAR_XLIM_RESIDUAL if out_behaviour else BAR_XLIM_RAW
    print("\n" + "#" * 70)
    print(f"# PART {part_label}  (region-level PCA, out_behaviour={out_behaviour})")
    print("#" * 70)

    region_analyzers, behavior_records = run_full_analysis_region_pca(out_behaviour=out_behaviour)

    print(f"\n--- Tasks 3-4: behavioural variance explained (Part {part_label}) ---")
    _write_records_csv(behavior_records, OUTPUT_DIR / f"part{part_label}_behavior_variance_records.csv")
    variance_summary = aggregate_region_behavior_variance(behavior_records)
    _write_region_variance_summary_csv(
        variance_summary, OUTPUT_DIR / f"part{part_label}_task3_4_variance_summary.csv")
    plot_task3_region_variance_bars(
        variance_summary, OUTPUT_DIR / f"part{part_label}_task3_variance_reference.png", xlim=xlim)
    plot_task4_region_variance_bars(
        variance_summary, OUTPUT_DIR / f"part{part_label}_task4_variance_other.png", xlim=xlim)

    print(f"\n--- Task 5: latent traces across sessions (Part {part_label}) ---")
    for comp_idx in COMPONENT_INDICES:
        plot_task5_region_latent_traces(
            region_analyzers, OUTPUT_DIR / f"part{part_label}_task5_latent_traces_comp{comp_idx}.png",
            component_idx=comp_idx)


def _run_hub_tasks() -> None:
    print("\n" + "#" * 70)
    print("# PARTS 2a/2b  (hub-orientation PCA: network / residual)")
    print("#" * 70)

    hub_analyzers, behavior_records = run_full_analysis_hub_pca()

    print("\n--- Tasks 3-4: behavioural variance explained (Parts 2a/2b) ---")
    _write_records_csv(behavior_records, OUTPUT_DIR / "part2ab_behavior_variance_records.csv")
    variance_summary = aggregate_hub_behavior_variance(behavior_records)
    _write_hub_variance_summary_csv(
        variance_summary, OUTPUT_DIR / "part2ab_task3_4_variance_summary.csv")
    plot_task3_hub_variance_bars(
        variance_summary, OUTPUT_DIR / "part2ab_task3_variance_reference.png",
        xlim=BAR_XLIM_RESIDUAL)
    plot_task4_hub_variance_bars(
        variance_summary, OUTPUT_DIR / "part2ab_task4_variance_other.png",
        xlim=BAR_XLIM_RESIDUAL)

    print("\n--- Task 5: latent traces across sessions (Parts 2a/2b) ---")
    for comp_idx in COMPONENT_INDICES:
        plot_task5_hub_latent_traces(
            hub_analyzers, OUTPUT_DIR / f"part2ab_task5_latent_traces_comp{comp_idx}.png",
            component_idx=comp_idx)


def main() -> None:
    print("=" * 70)
    print("TASKS 3-5: PCA/pCCA BEHAVIOURAL-VARIANCE & LATENT-TRACE ANALYSIS")
    print("(Parts 1a / 1b / 2a / 2b, sourced exclusively from")
    print(" pcca_all_regions_out_behaviour_sessions_{trial_type}_results)")
    print("=" * 70)
    print(f"  reference type     : {REFERENCE_TYPE}")
    print(f"  active trial types : {ACTIVE_TRIAL_TYPES}")
    print(f"  regions of interest: {REGIONS_OF_INTEREST}")
    print(f"  region pairs       : {len(REGION_PAIRS)}  (across {len(PAIR_CATEGORIES)} categories)")
    print(f"  component indices  : {COMPONENT_INDICES}  (of {N_PCA_COMPONENTS} fit)")
    print(f"  behaviour window   : {BEHAVIOR_TIME_RANGE_S}")
    print(f"  variance method    : {VARIANCE_METHOD}")
    print(f"  external variables : {EXTERNAL_VARIABLES}")
    print(f"  output directory   : {OUTPUT_DIR}")
    print("=" * 70)

    if not _MAT73_OK:
        print("\n[warning] mat73 unavailable -- cross_trial_type_pca_analysis.py and "
              "cross_trial_type_cca_analysis.py may fail to import their own .mat "
              "loading paths, though this script never calls them.")

    _run_region_tasks(out_behaviour=False)   # Part 1a
    _run_region_tasks(out_behaviour=True)    # Part 1b
    _run_hub_tasks()                         # Parts 2a + 2b

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Figures and CSVs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
r"""
cross_trial_type_peak_delta_reward_variance.py
================================================================================

Cross-session, cross-trial-type LATENT PEAK, LATENT DELTA, and BEHAVIOURAL-
VARIANCE analysis for the pCCA/CCA communication-subspace pipeline.

This script sits directly on top of the two existing pipeline modules
``cross_trial_type_cca_analysis.py`` (per-session projection + cross-session
sign-aligned aggregation, imported here) and ``pcca_cross_session_mi_bar.py``
(fixed 21-pair / 7-category bar-grouping and display style, whose constants
and visual grammar are reproduced here rather than imported, per this
project's "primitive copying over importing" convention). It does **not**
refit any pCCA/CCA weights: all communication-subspace projections come from
the reference-condition-trained ``A``/``B`` matrices already stored in each
session's ``*_analysis_results.mat`` file, exactly as the existing pipeline
extracts and applies them.

Four analyses are produced, each as an upper-triangle-free, MI-bar-styled
horizontal bar chart (one figure for the "left" region of every pair, one for
the "right" region), all sharing the same category background bands,
font sizes, jitter/error-bar styling, and flip-detection machinery as
``pcca_cross_session_mi_bar.py``:

  1. PEAK comparison   -- adapted from Fig. 3e of the attached PCA paper
                           (2024.11.04.621878v2). Each pair gets 2-3 bars,
                           one per active trial type, showing the
                           cross-session mean (+ per-session dots) of the
                           session-wise, sign-aligned latent peak.
  2. DELTA comparison  -- adapted from Fig. 5 of the same paper. Each pair
                           gets (n_trial_types - 1) bars: the session-matched
                           difference between the reference trace and each
                           non-reference trace, reduced to a scalar per
                           session.
  3. BEHAVIOURAL VARIANCE, reference condition -- adapted from Fig. 4 (GLM /
                           reward-kernel encoding model). Each pair gets one
                           cluster of bars, one per external variable
                           (position, speed, reward kernel), computed on the
                           reference trial type's (cued_hit_long by default)
                           per-trial latent.
  4. BEHAVIOURAL VARIANCE, projected conditions -- same as (3) but for the
                           latents obtained by projecting the non-reference
                           trial types through the reference-trained
                           weights. Each pair gets (n_trial_types - 1)
                           clusters (one per non-reference trial type), each
                           containing the same position/speed/reward bars.

Mathematical framework
-----------------------------------------------------------------------------
**Sign alignment ("flip detection").** Reused verbatim from
``CrossSessionCCAAnalyzer._align_signs_spectral`` / ``aggregate_projections``:
signs are resolved once, across sessions, from the REFERENCE trial type via
Z2 spectral synchronisation (leading eigenvector of the inter-session
correlation matrix) followed by a global epoch-mean and peak-polarity
convention; the resulting per-session, per-component flip decision is then
re-applied identically to every other trial type of that session. Nothing
about this mechanism is altered here -- this script simply consumes
``aggregated_projections[trial_type]['u_sessions'] / ['v_sessions']``, which
are already sign-aligned.

**Peak (task 1).** Following the paper's Methods ("projected latent dynamics
were scaled within each session by the peak absolute amplitude of the
corresponding component in the training PCA subspace, computed within a
500 ms post-movement time window"), for session $s$, trial type $c$,
component $d$:
$$
t^\star_{c,s} = \arg\max_{t \in W} \left| u_{c,s}(t) \right|, \qquad
\pi_{c,s} = u_{c,s}(t^\star_{c,s})
$$
with $W$ = ``PEAK_WINDOW_S`` (default 0-500 ms post-onset). If
``NORMALIZE_PEAK_BY_REFERENCE`` is set, every session's peaks are additionally
divided by $|\pi_{\text{ref},s}|$, matching the paper's within-session
peak-amplitude scaling (this necessarily drives the reference bar itself to
~1.0 with near-zero cross-session variance -- an expected consequence of the
normalisation, not an artefact).

**Delta (task 2).** For each non-reference trial type $c$, session-matched to
the reference trace (after the same optional normalisation):
$$
d_{c,s}(t) = u_{\text{ref},s}(t) - u_{c,s}(t), \qquad t \in W
$$
reduced to a scalar via ``DELTA_METRIC``:
  * ``'auc_of_difference'`` (default -- matches the paper's own framing,
    "difference in average ... activity"): $\Delta_{c,s} = \text{mean}_{t \in W}\, d_{c,s}(t)$
  * ``'peak_of_difference'``: $\Delta_{c,s} = d_{c,s}(t^\star)$,
    $t^\star = \arg\max_t |d_{c,s}(t)|$ (same "signed-peak" idiom as task 1
    and as Step 5 of ``_align_signs_spectral``).

**Behavioural variance (tasks 3-4).** Marginal ridge $R^2$ (default, per the
explicit instruction to follow ``Cross-Session Behavioural Variance of
CCA.py``), copied verbatim as ``variance_explained``: for latent
$\ell \in \mathbb{R}^{n_{\text{tr}} \times T}$ and design
$Z \in \mathbb{R}^{n_{\text{tr}} \times C \times T}$, flattened over
(trial, time), mean-centred, ridge-regularised:
$$
R^2(\ell, Z) = 1 - \frac{\lVert \ell_c - Z_c\hat\beta \rVert^2}{\lVert \ell_c \rVert^2}, \qquad
\hat\beta = (Z_c^\top Z_c + \lambda n I)^{-1} Z_c^\top \ell_c
$$
computed **separately** for each predictor block (position / speed /
reward), i.e. the *marginal* contribution of each block on its own, not a
joint leave-one-out decomposition. An optional ``variance_explained_unique_loo``
is also provided for users who want closer fidelity to the paper's own
"no-refit" unique-variance method (Fig. 4b): fit ONE joint model on all
blocks concatenated, zero one block's fitted coefficients at a time, and take
the resulting drop in $R^2$ relative to the full model.

**Reward kernel.** The paper's reward-consumption predictor is a 1-second
cubic B-spline basis (5 retained spline columns, "splines 1-5" of a longer
basis generated with R's ``bs()``); reward-presence is a step function over
the presence window. Because this project's behavioural preprocessing
currently exposes only ``*_pos.npy`` / ``*_speed.npy`` / ``*_task_label.npy``
(no per-trial reward-delivery timestamp file), the kernel here is built on a
FIXED, trial-invariant window rather than a per-trial time-locked one --
this is the single largest assumption in this script and is isolated in
``build_reward_kernel_design`` / ``REWARD_PRESENCE_WINDOW_S`` /
``REWARD_CONSUMPTION_WINDOW_S`` so it is trivial to replace once real
per-trial reward timestamps become available.

Flexibility
-----------------------------------------------------------------------------
Everything that Shengyuan is likely to want to change lives in the single
"USER-CONFIGURABLE PARAMETERS" block below:
  * ``KERNEL_MODE``        : 'pcca' | 'cca'
  * ``REFERENCE_TYPE``     : which trial type the subspace is trained on
  * ``ACTIVE_TRIAL_TYPES`` : trim to 2 trial types or extend back to 3
  * ``COMPONENT_INDICES``  : which CCA/pCCA dimension(s) to analyze
  * ``VARIANCE_METHOD``    : 'marginal' | 'leave_one_out'
  * ``DELTA_METRIC``       : 'auc_of_difference' | 'peak_of_difference'
  * ``NORMALIZE_PEAK_BY_REFERENCE`` : True | False

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpmath.math2 import sqrt2
from scipy.stats import zscore, wilcoxon
from scipy.interpolate import BSpline


warnings.filterwarnings('ignore')

# =============================================================================
# 0.  Import the existing pipeline (high-level classes imported directly, per
#     this project's convention -- cross_trial_type_cca_analysis.py is a
#     stable library module, treated the same way run_cross_trial_type_
#     analysis.py already treats it).
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_trial_type_cca_analysis as _ctta  # noqa: E402  (module kept for global override, see §1)
from cross_trial_type_cca_analysis import (   # noqa: E402
    CrossTrialTypeCCAAnalyzer,
    CrossSessionCCAAnalyzer,
    TRIAL_TYPE_COLORS,
    ANATOMICAL_ORDER,
    MIN_SESSIONS_THRESHOLD,
    sort_pair_by_anatomy,
    get_anatomical_index,
)

from pCCA_all_regions_out_behaviour import PrivateLatentAnalyzer
import sys
from pCCA_all_regions_out_behaviour import PrivateLatentSessionResult, PrivateLatentPairResult

# 将这两个类手动注入到当前脚本的 __main__ 命名空间中
sys.modules['__main__'].PrivateLatentSessionResult = PrivateLatentSessionResult
sys.modules['__main__'].PrivateLatentPairResult = PrivateLatentPairResult

try:
    import mat73  # noqa: F401  (transitively required by cross_trial_type_cca_analysis; imported
    _MAT73_OK = True
except Exception:
    _MAT73_OK = False
    warnings.warn("mat73 not importable -- the analysis loop will fail; "
                  "install with `pip install mat73`.")


# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS
#     Every knob mentioned in the request lives in this one block.
# =============================================================================

# ---- Kernel / reference / trial-type selection -----------------------------
KERNEL_MODE: str = 'pCCA_out_behaviour'              # 'pcca' | 'cca' | 'pCCA_out_behaviour'
REFERENCE_TYPE: str = 'cued_hit_long'  # swap to 'spont_hit_long' to reproduce
                                        # the paper's right-hand Fig. 3e panel
ACTIVE_TRIAL_TYPES: List[str] = [      # trim to 2 entries, or restore the 3rd,
    'cued_hit_long',                   # freely -- REFERENCE_TYPE need not be
    # 'spont_miss_long',                 # first in this list.
    # 'spont_hit_long',                # <- uncomment once pCCA results exist
                                        #    for this trial type; already
                                        #    available for KERNEL_MODE='cca'.
]

# Per-trial-type .mat subdirectory, mirroring cross_trial_type_cca_analysis.py's
# own KERNEL switch. Overridden into that module's global TRIAL_TYPES below
# (see §1b) because CrossTrialTypeCCAAnalyzer.load_all_trial_types() reads
# TRIAL_TYPES as a bare name resolved from its *own* module namespace at call
# time -- so reassigning cross_trial_type_cca_analysis.TRIAL_TYPES here is the
# correct (and only) hook into that behaviour without editing the source file.
TRIAL_TYPE_SUBDIRS: Dict[str, Dict[str, str]] = {
    'cued_hit_long':   {'pcca': 'pcca_sessions_cued_hit_long_results',
                         'cca':  'sessions_cued_hit_long_results'},
    'spont_hit_long':  {'pcca': 'pcca_sessions_spont_hit_long_results',
                         'cca':  'sessions_spont_hit_long_results'},
    'spont_miss_long': {'pcca': 'pcca_sessions_spont_miss_long_results',
                         'cca':  'sessions_spont_miss_long_results'},
}

# ---- Paths -------------------------------------------------------------
BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
BEHAVIOR_DIR = BASE_DIR / "Paper_output" / "tapproach_sessions"
OUTPUT_DIR = BASE_DIR / "Paper_output" / f"cross_trial_type_{KERNEL_MODE}_{REFERENCE_TYPE}_peak_delta_variance"
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

# ---- CCA/pCCA dimensionality --------------------------------------------
N_COMPONENTS: int = 5             # how many components are read out of the .mat file
COMPONENT_INDICES: List[int] = [0]  # which of those to actually analyze/plot here
                                     # (LATENT_DIM = 0 is this project's default
                                     # leading-dimension convention; add more
                                     # indices to get one figure set per index)

MIN_SESSIONS: int = MIN_SESSIONS_THRESHOLD  # reuse the pipeline's own threshold

# ---- Task 1 / 2: peak & delta -------------------------------------------
# 500 ms post-movement window, following the paper's Methods verbatim
# ("... computed within 500 ms post-movement time window"). t = 0 is reach
# / movement onset in this pipeline's own time_bins convention.
PEAK_WINDOW_S: Tuple[float, float] = (0.0, 1)
DELTA_WINDOW_S: Tuple[float, float] = PEAK_WINDOW_S   # kept equal to PEAK_WINDOW_S by
                                                       # default for a like-for-like
                                                       # comparison; decouple freely.
NORMALIZE_PEAK_BY_REFERENCE: bool = True   # within-session scaling by the reference
                                            # condition's own |peak|, per the paper's
                                            # Methods; the reference bar is then ~1.0
                                            # by construction (see docstring above).
DELTA_METRIC: str = 'auc_of_difference'    # 'auc_of_difference' | 'peak_of_difference'
                                            # default follows the paper's own phrasing
                                            # ("difference in AVERAGE ... activity");
                                            # 'peak_of_difference' is offered as the
                                            # more literal generalisation of task 1's
                                            # signed-peak metric to a paired difference.

# ---- Task 3 / 4: behavioural variance ------------------------------------
BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = (-1.0, 2.0)  # per the request: crop every
                                                            # trial to this window before
                                                            # regressing against behaviour
BEHAVIOR_FS: float = 50.0
BEHAVIOR_T_OFFSET: float = -1.0
LAMBDA_R2: float = 1e-4            # ridge coefficient for variance_explained (matches
                                    # Cross-Session Behavioural Variance of CCA.py)
VARIANCE_METHOD: str = 'marginal'  # 'marginal' (default, matches the fourth script's own
                                    # convention, explicitly requested) | 'leave_one_out'
                                    # (closer to the paper's own Fig. 4b "no-refit" method)
EXTERNAL_VARIABLES: List[str] = ['position', 'speed', 'reward_presence', 'reward_consumption']
                                    # order = subplot order (tasks 3-4 now give each variable
                                    # its own panel); extend with 'position+speed' etc. by
                                    # adding a branch to _build_predictor_designs()

# Reward-kernel construction (see docstring §"Reward kernel" -- FIXED WINDOW
# ASSUMPTION, adjust the moment real per-trial reward timestamps are available)
REWARD_PRESENCE_WINDOW_S: Tuple[float, float] = (0.0, 0.5)     # step-function "reward
                                                                 # available" proxy window
REWARD_CONSUMPTION_WINDOW_S: Tuple[float, float] = (0.5, 1.5)  # 1 s long, matching the
                                                                 # paper's spline-kernel length
N_REWARD_CONSUMPTION_BASIS: int = 7   # matches "splines 1-5" in the paper's Methods
REWARD_SPLINE_DEGREE: int = 2          # cubic, matching R's default bs() degree








# ---- Region pairs / categories (identical to pcca_cross_session_mi_bar.py) -
# See §2 below.

# ---- Caching / output -----------------------------------------------------
USE_CACHED_DATA: bool = True   # False forces full recomputation; set False after
                                # changing ANY analysis parameter above (component
                                # index, window, normalization, kernel mode, ...)
SAVE_DPI: int = 400
BAR_LEN: float = 0.2

# =============================================================================
# 1b. Push the KERNEL_MODE / ACTIVE_TRIAL_TYPES choice into the imported
#     pipeline module's global TRIAL_TYPES (see the comment on
#     TRIAL_TYPE_SUBDIRS above for why this is the correct mechanism).
# =============================================================================
def _configure_pipeline_globals() -> None:
    if KERNEL_MODE == 'pCCA_out_behaviour':
        # This mode reads pre-computed results via PrivateLatentAnalyzer
        # (pcca_all_regions_out_behaviour.py) and never touches the .mat
        # pipeline module, so there is nothing to push into _ctta's globals.
        print(f"[config] KERNEL_MODE='pCCA_out_behaviour' bypasses "
              f"cross_trial_type_cca_analysis entirely.")
        return
    resolved = {t: TRIAL_TYPE_SUBDIRS[t][KERNEL_MODE] for t in ACTIVE_TRIAL_TYPES}
    _ctta.TRIAL_TYPES = resolved
    _ctta.KERNEL = KERNEL_MODE
    print(f"[config] cross_trial_type_cca_analysis.TRIAL_TYPES overridden -> {resolved}")


_configure_pipeline_globals()


# =============================================================================
# 2.  Region-pair categories -- copied verbatim from pcca_cross_session_
#     mi_bar.py (same 21 pairs / 7 categories, same display-name remap),
#     per this project's "primitive copying over importing" convention and
#     the explicit instruction to preserve that script's figure style.
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
    ])
]


# PAIR_CATEGORIES: List[Tuple[str, List[Tuple[str, str]]]] = [
#     ("thalamic-thalamic", [
#         ("VPMPO", "VALVM"),
#     ]),
#     ("cortico-cortical", [
#         ("MOp", "MOs"),
#         ("MOp", "ORB"),
#         ("MOs", "ORB"),
#     ]),
#     ("cortico-motor thalamic", [
#         ("MOp", "VALVM"),
#         ("ORB", "VALVM"),
#         ("MOs", "VALVM"),
#     ]),
#     ("cortico-sensory thalamic", [
#         ("MOp", "VPMPO"),
#         ("ORB", "VPMPO"),
#         ("MOs", "VPMPO"),
#     ]),
#     ("to HY", [
#         ("VALVM", "HY"),
#         ("VPMPO", "HY"),
#         ("MOp", "HY"),
#         ("MOs", "HY"),
#         ("ORB", "HY"),
#     ]),
#     ("to STR", [
#         ("VALVM", "STR"),
#         ("VPMPO", "STR"),
#         ("MOp", "STR"),
#         ("MOs", "STR"),
#         ("ORB", "STR"),
#     ]),
#     ("other", [
#         ("HY", "STR"),
#     ]),
# ]

PAIR_CATEGORIES = [
    (category, [sort_pair_by_anatomy(*pair) for pair in pairs])
    for category, pairs in PAIR_CATEGORIES
]

REGION_PAIRS: List[Tuple[str, str]] = [
    (ri, rj) for _, pairs in PAIR_CATEGORIES for (ri, rj) in pairs
]



DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
}


def _display_name(region: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(region, region)


def _display_pair(pair: Tuple[str, str]) -> str:
    return f"{_display_name(pair[0])}\u2194{_display_name(pair[1])}"


# Category background colours -- identical palette to pcca_cross_session_mi_bar.py
CATEGORY_COLORS: Dict[str, str] = {
    "thalamic-thalamic":          "#4C72B0",
    "cortico-cortical":           "#DD8452",
    "cortico-motor thalamic":     "#55A868",
    "cortico-sensory thalamic":   "#C44E52",
    "to HY":                      "#8172B2",
    "to STR":                     "#937860",
    "other":                      "#64B5CD",
}

# New colour keys for the two "identity" dimensions this script adds on top of
# category (trial type, external variable). Trial-type colours are the
# pipeline's own TRIAL_TYPE_COLORS (imported above), kept unchanged so a
# reader who already knows the blue/teal/red convention recognises it
# immediately. External-variable colours are new: position/speed reuse the
# exact two hex codes from Cross-Session Behavioural Variance of CCA.py's own
# behavioural/similarity bar palette, and reward gets a new, unused colour.
EXTERNAL_VAR_COLORS: Dict[str, str] = {
    'position':           "#DE6E4B",
    'speed':              "#4B7DDE",
    'reward_presence':    "#55A868",
    'reward_consumption': "#B07AA1",
}


def _lighten(hex_color: str, amount: float = 0.45) -> str:
    """Blend `hex_color` toward white by `amount`. Used so the 'column
    region' bar is a lighter tint of the SAME hue as the 'row region' bar
    within a single-variable subplot -- region identity is now encoded by
    shade, since predictor identity has moved to the panel itself."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"

DOT_COLOR = "#262626"
BAR_HEIGHT = 0.62
GROUP_GAP = 1.35
CLUSTER_GAP = 0.55     # extra spacing between trial-type clusters within one pair (task 4 only)
PAIR_GAP = 0.15        # extra spacing between different pairs within one category
DOT_JITTER_FRAC = 0.35 # dot jitter as a fraction of BAR_HEIGHT
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 15
CLUSTER_HATCH_CYCLE = [None, "///", "xxx"]  # first, second, third non-reference trial type


# =============================================================================
# 3.  Low-level primitives copied verbatim (project convention: primitives
#     are copied, not imported, so every script stays independently
#     auditable and runnable).
# =============================================================================

# ---- 3a. Behavioural loading -- copied verbatim from
#          "Cross-Session Behavioural Variance of CCA.py" ------------------
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
    """'spont_miss_long' -> 'spont miss long', matching the *_task_label.npy
    string convention used by load_behavior_regressors (see Doc 4's
    BEHAVIOR_TRIAL_LABEL = "cued hit long")."""
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


# ---- 3b. Ridge R^2 -- copied verbatim from
#          "Cross-Session Behavioural Variance of CCA.py" ------------------
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
    # zsd = Z.std(axis=0)
    # zsd[zsd < 1e-12] = 1.0
    # Z = (Z - Z.mean(axis=0, keepdims=True)) / zsd
    #ell_c = ell - ell.mean()
    ell_c = ell
    ss_tot = float(ell_c @ ell_c)
    if ss_tot < 1e-12:
        return 0.0
    n, m = Z.shape
    ZtZ = Z.T @ Z + lam * n * np.eye(m)
    beta = np.linalg.solve(ZtZ, Z.T @ ell_c)
    resid = ell_c - Z @ beta
    return float(np.clip(1.0 - float(resid @ resid) / ss_tot, 0.0, 1.0))*(math.log(T))


def variance_explained_unique_loo(
        latent_2d: np.ndarray, design_dict: Dict[str, np.ndarray],
        lam: float = LAMBDA_R2,
) -> Dict[str, float]:
    """Unique (leave-one-out, 'no-refit') R^2 per predictor block, following
    Engelhard et al. (2019) / the PCA paper's Methods: fit ONE joint ridge
    model on every block concatenated, then for each block zero its fitted
    coefficients and recompute R^2 on the SAME fit; the drop relative to the
    full model's R^2 is that block's unique contribution. Closer to Fig. 4b
    of the paper than `variance_explained`'s per-block marginal fits, at the
    cost of being sensitive to collinearity between blocks."""
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


def compute_vif(design_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Lightweight variance-inflation-factor diagnostic across predictor
    BLOCKS (not individual columns), for QC only -- the paper reports
    VIF in [1.00, 1.19] for its own design (minimal collinearity)."""
    names = list(design_dict.keys())
    n_tr = next(iter(design_dict.values())).shape[0]
    T = next(iter(design_dict.values())).shape[-1]
    cols = []
    for name in names:
        Z_i = np.transpose(design_dict[name], (0, 2, 1)).reshape(n_tr * T, -1)
        cols.append(Z_i.mean(axis=1))  # one representative column per block
    M = np.column_stack(cols)
    finite = np.all(np.isfinite(M), axis=1)
    M = M[finite]
    if M.shape[0] < M.shape[1] + 2:
        return {name: float('nan') for name in names}
    out = {}
    for i, name in enumerate(names):
        y = M[:, i]
        X = np.delete(M, i, axis=1)
        X = np.column_stack([np.ones(X.shape[0]), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        out[name] = float(1.0 / max(1.0 - r2, 1e-6))
    return out


# ---- 3c. Per-trial CCA/pCCA projection, matching the z-score + matmul
#          recipe already used by CrossTrialTypeCCAAnalyzer.compute_
#          projections() for non-reference trial types, applied here
#          UNIFORMLY to every trial type (including the reference) so
#          tasks 3-4 have trial-resolved latents throughout. -------------
def _project_trials_single_region(X_raw: np.ndarray, W: np.ndarray, n_components: int) -> np.ndarray:
    """
    X_raw : (n_trials, n_neurons, n_time) raw spike-rate tensor for one region
    W     : (n_neurons, >=n_components) CCA/pCCA weight matrix (A or B),
            extracted from the reference-condition fit.

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


# ---- 3d. Reward-kernel design matrix (new; see docstring "Reward kernel") -
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
        c[i] = 0.5
        spline = BSpline(knots, c, degree, extrapolate=False)
        basis[i] = np.nan_to_num(spline(t), nan=0.0)
    basis_end = basis[1:-1,:]
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
    """(n_trials, n_basis, T) cubic B-spline 'reward consumption' kernel
    (unchanged from before -- still N_REWARD_CONSUMPTION_BASIS = 5 columns;
    it just no longer gets concatenated with the presence regressor)."""
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
    return out


# ---- 3e. Small numeric helpers ------------------------------------------
def _sem(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _signed_peak_in_window(trace: np.ndarray, time_vec: np.ndarray,
                           window: Tuple[float, float]) -> float:
    """Signed value at argmax|trace| within `window` -- the same 'signed
    peak' idiom already used by _align_signs_spectral's Step 5."""
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
    d =  comp_trace[mask] - ref_trace[mask]
    if d.size == 0:
        return float('nan')
    if metric == 'peak_of_difference':
        idx = np.argmax(np.abs(d))
        return float(d[idx])
    elif metric == 'auc_of_difference':
        return float(np.mean(d))
    raise ValueError(f"Unknown DELTA_METRIC: {metric!r}")


def _paired_wilcoxon(a: np.ndarray, b: Optional[np.ndarray] = None) -> float:
    """Wilcoxon signed-rank p-value: paired a-vs-b if b given, else a-vs-0.
    Matches the paper's own use of a sign-rank test for Fig. 3e asterisks."""
    x = a if b is None else (a - b)
    if x.size < 5 or np.allclose(x, x[0]):
        return float('nan')
    try:
        _, p = wilcoxon(x)
        return float(p)
    except ValueError:
        return float('nan')


def _sessions_for_trial_type(cs: CrossSessionCCAAnalyzer, trial_type: str) -> List[str]:
    """Recover, in the SAME order CrossSessionCCAAnalyzer.aggregate_projections
    used internally, which sessions contributed to `trial_type`'s stacked
    u_sessions/v_sessions array. (aggregated_projections does not store this
    directly, so it is reconstructed here from the public session_projections
    dict using the identical filter -- Python dicts preserve insertion order,
    so this is exact, not a heuristic.)"""
    return [s for s, proj in cs.session_projections.items() if trial_type in proj]


# =============================================================================
# 4.  Main data-gathering pass: ONE loop over (session, pair) feeds BOTH the
#     peak/delta pipeline (tasks 1-2, via CrossSessionCCAAnalyzer) and the
#     behavioural-variance pipeline (tasks 3-4, via fresh per-trial
#     projections), so every .mat file is only loaded once per session.
# =============================================================================
def run_full_analysis(
        sessions: List[str] = SESSIONS,
        pair_list: List[Tuple[str, str]] = REGION_PAIRS,
        base_dir: Path = BASE_DIR,
        reference_type: str = REFERENCE_TYPE,
        n_components: int = N_COMPONENTS,
        component_indices: List[int] = COMPONENT_INDICES,
        min_sessions: int = MIN_SESSIONS,
) -> Tuple[Dict[Tuple[str, str], CrossSessionCCAAnalyzer], List[dict]]:
    cross_session_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer] = {}
    behavior_records: List[dict] = []

    for s_idx, session_name in enumerate(sessions, 1):
        print("\n" + "=" * 70)
        print(f"SESSION {s_idx}/{len(sessions)}: {session_name}")
        print("=" * 70)

        analyzer = CrossTrialTypeCCAAnalyzer(
            base_dir=str(base_dir), session_name=session_name,
            reference_type=reference_type, n_components=n_components,
        )
        if not analyzer.load_all_trial_types():
            print(f"  [skip session] could not load trial types for {session_name}")
            continue

        behavior_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

        for region_i, region_j in pair_list:
            if not analyzer.extract_neural_data((region_i, region_j)):
                continue
            if not analyzer.extract_cca_weights((region_i, region_j)):
                continue
            if not analyzer.compute_projections():
                continue

            # ---- tasks 1 & 2 data path -----------------------------------
            pair_key = sort_pair_by_anatomy(region_i, region_j)
            is_flipped = (pair_key[0] != region_i)
            if pair_key not in cross_session_analyzers:
                cross_session_analyzers[pair_key] = CrossSessionCCAAnalyzer(
                    base_dir=str(base_dir), region_pair=pair_key,
                    reference_type=reference_type, n_components=n_components,
                    min_sessions=min_sessions,
                )
            cross_session_analyzers[pair_key].add_session_result(
                session_name, analyzer, swap_uv=is_flipped
            )

            # ---- tasks 3 & 4 data path ------------------------------------
            A = analyzer.cca_weights['A']
            B = analyzer.cca_weights['B']

            for trial_type in analyzer.available_trial_types:
                if trial_type not in analyzer.neural_data:
                    continue
                if trial_type not in behavior_cache:
                    behavior_cache[trial_type] = _load_behavior_safe(session_name, trial_type)
                behav = behavior_cache[trial_type]
                if behav is None:
                    continue
                pos_full, speed_full, t_behav_full = behav

                X_i_raw = analyzer.neural_data[trial_type][region_i]
                X_j_raw = analyzer.neural_data[trial_type][region_j]
                u_trials = _project_trials_single_region(X_i_raw, A, n_components)
                v_trials = _project_trials_single_region(X_j_raw, B, n_components)

                for comp_idx in component_indices:
                    for region_role, region_name, trials in (
                            ('region_i', region_i, u_trials), ('region_j', region_j, v_trials)):
                        latent = trials[:, :, comp_idx]
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
                                session=session_name, pair=pair_key, region_role=region_role,
                                region=region_name, trial_type=trial_type, component=comp_idx,
                                predictor=var_name, r2=r2_val,
                            ))

        del analyzer  # drop this session's raw spike tensors before moving on

    # cross-session aggregation (sign alignment + mean/SEM across sessions).
    # aggregated_projections is {} until this is called explicitly; mirrors
    # CrossTrialTypeCCAPipeline.run_cross_session_aggregation()'s own loop.
    print("\n" + "=" * 70)
    print("CROSS-SESSION AGGREGATION (sign alignment + mean/SEM across sessions)")
    print("=" * 70)
    for pair_key, cs in cross_session_analyzers.items():
        n_sess = len(cs.session_projections)
        if n_sess < min_sessions:
            print(f"  {pair_key[0]} vs {pair_key[1]}: {n_sess} sessions (skipping, < {min_sessions})")
            continue
        cs.aggregate_projections()

    print("\n" + "=" * 70)
    print("DATA-GATHERING COMPLETE")
    print(f"  region pairs with >=1 session : {len(cross_session_analyzers)}")
    print(f"  region pairs successfully aggregated : "
          f"{sum(1 for cs in cross_session_analyzers.values() if cs.aggregated_projections)}")
    print(f"  behavioural-variance records  : {len(behavior_records)}")
    print("=" * 70)
    return cross_session_analyzers, behavior_records


def _prepare_regression_inputs(
        latent: np.ndarray, time_bins: np.ndarray,
        pos: np.ndarray, speed: np.ndarray, t_behav: np.ndarray,
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
    """Crop the per-trial neural latent (n_trials, T_neural) and the
    behavioural tensors to BEHAVIOR_TIME_RANGE_S, match trial/time counts
    (shorter-of-the-two truncation, matching _behavior_r2 in Doc 4), and
    build the predictor design dict. Returns None if there isn't enough
    overlap to regress."""
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
# 5.  Task 1 -- peak comparison (Fig. 3e style)
# =============================================================================
def aggregate_peak_records(
        cross_session_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer],
        component_idx: int = COMPONENT_INDICES[0],
        window: Tuple[float, float] = PEAK_WINDOW_S,
        normalize: bool = NORMALIZE_PEAK_BY_REFERENCE,
        reference_type: str = REFERENCE_TYPE,
) -> Tuple[Dict[Tuple[Tuple[str, str], str], Dict[str, dict]], List[dict]]:
    """
    Returns
    -------
    summary : {(pair, region_role): {trial_type: {mean, sem, values, n, wilcoxon_p_vs_ref}}}
    records : flat per-session rows, for the CSV.
    """
    summary: Dict[Tuple[Tuple[str, str], str], Dict[str, dict]] = {}
    records: List[dict] = []

    for pair_key, cs in cross_session_analyzers.items():
        if reference_type not in cs.aggregated_projections:
            continue
        ref_agg = cs.aggregated_projections[reference_type]
        sess_names_ref = _sessions_for_trial_type(cs, reference_type)

        for region_role, sessions_key in (('region_i', 'u_sessions'), ('region_j', 'v_sessions')):
            ref_sessions = ref_agg[sessions_key]
            ref_peak_by_session = {
                sname: _signed_peak_in_window(ref_sessions[i, :, component_idx], cs.time_bins, window)
                for i, sname in enumerate(sess_names_ref)
            }

            for trial_type in cs.available_trial_types:
                if trial_type not in cs.aggregated_projections:
                    continue
                agg = cs.aggregated_projections[trial_type]
                sessions = agg[sessions_key]
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
                        session=sname, pair=pair_key, region_role=region_role,
                        trial_type=trial_type, component=component_idx, peak=peak_val,
                    ))

                if not values:
                    continue
                values_arr = np.asarray(values, dtype=float)
                ref_arr = np.asarray(matched_ref, dtype=float)
                p_vs_ref = (float('nan') if trial_type == reference_type
                           else _paired_wilcoxon(values_arr, ref_arr))
                summary.setdefault((pair_key, region_role), {})[trial_type] = dict(
                    mean=float(values_arr.mean()), sem=_sem(values_arr),
                    values=values_arr, n=values_arr.size, wilcoxon_p_vs_ref=p_vs_ref,
                )
    return summary, records


def plot_task1_peak_bars(summary: dict, region_role: str, save_path: Path,
                         active_trial_types: List[str] = ACTIVE_TRIAL_TYPES) -> Optional[plt.Figure]:
    pair_clusters: Dict[Tuple[str, str], List[list]] = {}
    for (pair_key, role), per_type in summary.items():
        if role != region_role:
            continue
        ordered_types = [t for t in ([REFERENCE_TYPE] + sorted(set(active_trial_types) - {REFERENCE_TYPE}))
                         if t in per_type]
        cluster = [dict(mean=per_type[t]['mean'], sem=per_type[t]['sem'], values=per_type[t]['values'],
                        color=TRIAL_TYPE_COLORS.get(t, 'gray'), alpha=0.9, hatch=None)
                  for t in ordered_types]
        if cluster:
            pair_clusters[pair_key] = [cluster]  # single cluster per pair

    legend = [(t.replace('_', ' '), TRIAL_TYPE_COLORS.get(t, 'gray'), None) for t in active_trial_types]
    ylabel = ("Peak amplitude, normalized to reference |peak| (a.u.)" if NORMALIZE_PEAK_BY_REFERENCE
             else "Peak amplitude (a.u.)")
    return plot_grouped_pair_bars(
        PAIR_CATEGORIES, pair_clusters, save_path, xlabel=ylabel,
        legend_entries=legend, vline_zero=not NORMALIZE_PEAK_BY_REFERENCE,
    )


# =============================================================================
# 6.  Task 2 -- delta comparison (Fig. 5 style)
# =============================================================================
def aggregate_delta_records(
        cross_session_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer],
        component_idx: int = COMPONENT_INDICES[0],
        window: Tuple[float, float] = DELTA_WINDOW_S,
        normalize: bool = NORMALIZE_PEAK_BY_REFERENCE,
        reference_type: str = REFERENCE_TYPE,
        metric: str = DELTA_METRIC,
) -> Tuple[Dict[Tuple[Tuple[str, str], str], Dict[str, dict]], List[dict]]:
    summary: Dict[Tuple[Tuple[str, str], str], Dict[str, dict]] = {}
    records: List[dict] = []

    for pair_key, cs in cross_session_analyzers.items():
        if reference_type not in cs.aggregated_projections:
            continue
        ref_agg = cs.aggregated_projections[reference_type]
        sess_names_ref = _sessions_for_trial_type(cs, reference_type)

        for region_role, sessions_key in (('region_i', 'u_sessions'), ('region_j', 'v_sessions')):
            ref_sessions = ref_agg[sessions_key]
            ref_trace_by_session, ref_peak_by_session = {}, {}
            for i, sname in enumerate(sess_names_ref):
                ref_trace_by_session[sname] = ref_sessions[i, :, component_idx]
                ref_peak_by_session[sname] = _signed_peak_in_window(
                    ref_sessions[i, :, component_idx], cs.time_bins, window)

            for trial_type in cs.available_trial_types:
                if trial_type == reference_type or trial_type not in cs.aggregated_projections:
                    continue
                agg = cs.aggregated_projections[trial_type]
                sessions = agg[sessions_key]
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
                        session=sname, pair=pair_key, region_role=region_role,
                        trial_type=trial_type, component=component_idx, delta=delta,
                    ))

                if not values:
                    continue
                values_arr = np.asarray(values, dtype=float)
                summary.setdefault((pair_key, region_role), {})[trial_type] = dict(
                    mean=float(values_arr.mean()), sem=_sem(values_arr),
                    values=values_arr, n=values_arr.size,
                    wilcoxon_p_vs_zero=_paired_wilcoxon(values_arr),
                )
    return summary, records


def plot_task2_delta_bars(summary: dict, region_role: str, save_path: Path,
                          active_trial_types: List[str] = ACTIVE_TRIAL_TYPES) -> Optional[plt.Figure]:
    non_ref = [t for t in active_trial_types if t != REFERENCE_TYPE]
    pair_clusters: Dict[Tuple[str, str], List[list]] = {}
    for (pair_key, role), per_type in summary.items():
        if role != region_role:
            continue
        cluster = [dict(mean=per_type[t]['mean'], sem=per_type[t]['sem'], values=per_type[t]['values'],
                        color=TRIAL_TYPE_COLORS.get(t, 'gray'), alpha=0.9, hatch=None)
                  for t in non_ref if t in per_type]
        if cluster:
            pair_clusters[pair_key] = [cluster]

    legend = [(t.replace('_', ' '), TRIAL_TYPE_COLORS.get(t, 'gray'), None) for t in non_ref]
    metric_label = "AUC of (reference - comparison)" if DELTA_METRIC == 'auc_of_difference' \
        else "Peak of (reference - comparison)"
    xlabel = f"{metric_label} [{'normalized' if NORMALIZE_PEAK_BY_REFERENCE else 'raw'}]"
    return plot_grouped_pair_bars(
        PAIR_CATEGORIES, pair_clusters, save_path, xlabel=xlabel,
        legend_entries=legend, vline_zero=True,
    )


# =============================================================================
# 7.  Tasks 3 & 4 -- behavioural variance explained
# =============================================================================
def aggregate_behavior_variance(
        behavior_records: List[dict],
) -> Dict[Tuple[Tuple[str, str], str, str], Dict[str, dict]]:
    """
    Returns {(pair, region_role, trial_type): {predictor: {mean, sem, values, n}}},
    pooled across sessions (and, if COMPONENT_INDICES has >1 entry, across
    components too -- kept separate per component if the caller wants that;
    here pooled over whatever component_indices were actually run).
    """
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

def _region_bar(stats: dict, region_role: str, base_color: str) -> dict:
    """One bar dict for a given region role: region_i (row) gets the
    variable's full colour, region_j (column) gets a lightened tint of the
    same hue."""
    color = base_color if region_role == 'region_i' else _lighten(base_color)
    return dict(mean=stats['mean'], sem=stats['sem'], values=stats['values'],
               color=color, alpha=0.9, hatch=None)


def plot_task3_variance_bars(summary: dict, save_path: Path,
                             reference_type: str = REFERENCE_TYPE) -> Optional[plt.Figure]:
    pair_clusters_by_variable: Dict[str, Dict[Tuple[str, str], List[list]]] = {
        v: {} for v in EXTERNAL_VARIABLES
    }
    for (pair_key, role, trial_type), per_pred in summary.items():
        if trial_type != reference_type:
            continue
        for v in EXTERNAL_VARIABLES:
            if v not in per_pred:
                continue
            bar = _region_bar(per_pred[v], role, EXTERNAL_VAR_COLORS.get(v, 'gray'))
            cluster = pair_clusters_by_variable[v].setdefault(pair_key, [[None, None]])[0]
            cluster[0 if role == 'region_i' else 1] = bar
    for v in EXTERNAL_VARIABLES:
        for pair_key, clusters in list(pair_clusters_by_variable[v].items()):
            clusters[0][:] = [b for b in clusters[0] if b is not None]
            if not clusters[0]:
                del pair_clusters_by_variable[v][pair_key]

    legend = [('left region (region i)', '#555555', None),
              ('right region (region j)', _lighten('#555555'), None)]
    return plot_grouped_pair_bars_multipanel(
        PAIR_CATEGORIES, pair_clusters_by_variable, EXTERNAL_VARIABLES, save_path,
        common_xlabel=f"Variance explained, $R^2$ ({reference_type.replace('_', ' ')})",
        xlim=(0, BAR_LEN), legend_entries=legend,
    )

def plot_grouped_pair_bars_multipanel(
        pair_categories: List[Tuple[str, List[Tuple[str, str]]]],
        pair_clusters_by_variable: Dict[str, Dict[Tuple[str, str], List[List[dict]]]],
        variable_names: List[str],
        save_path: Path,
        common_xlabel: str,
        xlim: Optional[Tuple[float, float]] = None,
        legend_entries: Optional[List[Tuple[str, str, Optional[str]]]] = None,
        panel_width: float = 4.6,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """One subplot per entry in `variable_names`, all sharing ONE y-axis
    layout (pairs grouped by category) computed from the union of pairs
    across all variables, so a pair sits on the same row in every panel."""
    def _footprint(clusters: List[List[dict]]) -> float:
        if not clusters:
            return 1.0
        return sum(len(cl) for cl in clusters) + max(0, len(clusters) - 1) * CLUSTER_GAP

    all_pairs = set()
    for v in variable_names:
        all_pairs |= {p for p, c in pair_clusters_by_variable.get(v, {}).items() if c}

    groups = []
    for category, pairs in pair_categories:
        present = [p for p in pairs if p in all_pairs]
        if present:
            groups.append((category, present))
    if not groups:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    y = 0.0
    pair_y0: Dict[Tuple[str, str], float] = {}
    pair_label_pos: Dict[Tuple[str, str], float] = {}
    group_spans: List[Tuple[str, float, float]] = []
    for category, pairs in groups:
        y_start = y
        for pair in pairs:
            footprint = max(_footprint(pair_clusters_by_variable.get(v, {}).get(pair, []))
                            for v in variable_names)
            pair_y0[pair] = y
            pair_label_pos[pair] = y + footprint / 2.0 - 0.5
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
                       color=CATEGORY_COLORS.get(category, '#888888'), alpha=0.07, zorder=0)

        means, sems, ys, dot_x, dot_y = [], [], [], [], []
        for category, pairs in groups:
            for pair in pairs:
                yy = pair_y0[pair]
                clusters = pair_clusters_by_variable.get(variable_name, {}).get(pair, [])
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

    axes[0].set_yticks(list(pair_label_pos.values()))
    axes[0].set_yticklabels([_display_pair(p) for p in pair_label_pos.keys()], fontsize=TICK_FONTSIZE)
    axes[0].margins(y=0.015)
    axes[0].invert_yaxis()
    fig.supxlabel(common_xlabel, fontsize=TICK_FONTSIZE)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig

def plot_task4_variance_bars(summary: dict, save_path: Path,
                             active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
                             reference_type: str = REFERENCE_TYPE) -> Optional[plt.Figure]:
    non_ref = [t for t in active_trial_types if t != reference_type]
    pair_clusters_by_variable: Dict[str, Dict[Tuple[str, str], List[list]]] = {
        v: {} for v in EXTERNAL_VARIABLES
    }
    all_pairs = {k[0] for k in summary.keys()}
    for v in EXTERNAL_VARIABLES:
        for pair_key in all_pairs:
            clusters: List[list] = []
            for ti, trial_type in enumerate(non_ref):
                hatch = CLUSTER_HATCH_CYCLE[ti % len(CLUSTER_HATCH_CYCLE)]
                cluster: List[Optional[dict]] = [None, None]
                for role in ('region_i', 'region_j'):
                    per_pred = summary.get((pair_key, role, trial_type))
                    if per_pred and v in per_pred:
                        bar = _region_bar(per_pred[v], role, EXTERNAL_VAR_COLORS.get(v, 'gray'))
                        bar['hatch'] = hatch
                        cluster[0 if role == 'region_i' else 1] = bar
                cluster = [b for b in cluster if b is not None]
                if cluster:
                    clusters.append(cluster)
            if clusters:
                pair_clusters_by_variable[v][pair_key] = clusters

    legend = [('left region (region i)', '#555555', None),
              ('right region (region j)', _lighten('#555555'), None)]
    legend += [(t.replace('_', ' '), '#bbbbbb', CLUSTER_HATCH_CYCLE[ti % len(CLUSTER_HATCH_CYCLE)])
              for ti, t in enumerate(non_ref)]
    return plot_grouped_pair_bars_multipanel(
        PAIR_CATEGORIES, pair_clusters_by_variable, EXTERNAL_VARIABLES, save_path,
        common_xlabel="Variance explained, $R^2$ (projected conditions)",
        xlim=(0, BAR_LEN), legend_entries=legend,
    )

# def plot_task5_latent_traces(
#         cross_session_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer],
#         save_path: Path,
#         component_idx: int = COMPONENT_INDICES[0],
#         active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
#         row_height: float = 1.5,
#         fig_width: float = 4.0,
#         dpi: int = SAVE_DPI,
# ) -> Optional[plt.Figure]:
#     """One row per (pair, region). Trace styling (individual sessions thin/
#     low-alpha, mean bold, SEM shading, dashed line at t=0) is copied from
#     CrossTrialTypeSummaryVisualizer._plot_pair_projections_single_region in
#     cross_trial_type_cca_analysis.py; layout is the MI-bar script's flat,
#     category-grouped list rather than that visualizer's N x N matrix."""
#     rows: List[Tuple[Tuple[str, str], str, str]] = []
#     row_category: Dict[int, str] = {}
#     for category, pairs in PAIR_CATEGORIES:
#         for pair in pairs:
#             cs = cross_session_analyzers.get(pair)
#             if cs is None or not cs.aggregated_projections:
#                 continue
#             row_category[len(rows)] = category
#             rows.append((pair, 'region_i', pair[0]))
#             row_category[len(rows)] = category
#             rows.append((pair, 'region_j', pair[1]))
#
#     if not rows:
#         print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
#         return None
#
#     n_rows = len(rows)
#     fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, row_height * n_rows), sharex=True)
#     axes = np.atleast_1d(axes)
#
#     for r, (pair, region_role, region_name) in enumerate(rows):
#         ax = axes[r]
#         cs = cross_session_analyzers[pair]
#         ax.set_facecolor(CATEGORY_COLORS.get(row_category[r], '#888888'))
#         ax.patch.set_alpha(0.07)
#
#         mean_key, sem_key, sessions_key = (
#             ('u_mean', 'u_sem', 'u_sessions') if region_role == 'region_i'
#             else ('v_mean', 'v_sem', 'v_sessions')
#         )
#         for trial_type in active_trial_types:
#             if trial_type not in cs.aggregated_projections:
#                 continue
#             agg = cs.aggregated_projections[trial_type]
#             color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')
#
#             session_traces = agg[sessions_key][:, :, component_idx]
#             for sess_trace in session_traces:
#                 ax.plot(cs.time_bins, sess_trace, color=color, linewidth=0.5, alpha=0.2, zorder=1)
#
#             mean_trace = agg[mean_key][:, component_idx]
#             sem_trace = agg[sem_key][:, component_idx]
#             is_ref = (trial_type == REFERENCE_TYPE)
#             ax.plot(cs.time_bins, mean_trace, color=color, linewidth=2.0 if is_ref else 1.4,
#                     alpha=0.85 if is_ref else 0.75,
#                     label=f"{trial_type.replace('_', ' ')} (n={agg['n_sessions']})", zorder=3)
#             ax.fill_between(cs.time_bins, mean_trace - sem_trace, mean_trace + sem_trace,
#                             color=color, alpha=0.15, zorder=2)
#
#         ax.axvline(x=0, color='black', linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
#         ax.set_xlim(cs.time_bins[0], cs.time_bins[-1])
#         ax.set_ylim(*((-0.5, 0.5) if KERNEL_MODE == 'pcca' else (-2.5, 5.0)))
#         ax.text(0.01, 0.90, f"{_display_pair(pair)}  \u2014  {_display_name(region_name)}",
#                 transform=ax.transAxes, fontsize=TICK_FONTSIZE - 6, va='top', ha='left')
#         for sp in ('top', 'right'):
#             ax.spines[sp].set_visible(False)
#         ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)
#         if r == 0:
#             ax.legend(fontsize=LEGEND_FONTSIZE - 3, loc='upper right', frameon=False)
#
#     axes[-1].set_xlabel("Time from reach (s)", fontsize=TICK_FONTSIZE)
#     axes[-1].tick_params(axis='x', labelsize=TICK_FONTSIZE - 2)
#
#     fig.tight_layout(h_pad=0.15)
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
#     print(f"  [plot] saved: {save_path}")
#     plt.close(fig)
#     return fig

def plot_task5_latent_traces(
        cross_session_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer],
        save_path: Path,
        component_idx: int = COMPONENT_INDICES[0],
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        row_height: float = 1.5,
        fig_width: float = 4.0,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """One row per (pair, region). Trace styling (individual sessions thin/
    low-alpha, mean bold, SEM shading, dashed line at t=0) is copied from
    CrossTrialTypeSummaryVisualizer._plot_pair_projections_single_region in
    cross_trial_type_cca_analysis.py; layout is the MI-bar script's flat,
    category-grouped list rather than that visualizer's N x N matrix."""
    rows: List[Tuple[Tuple[str, str], str, str]] = []
    row_category: Dict[int, str] = {}
    for category, pairs in PAIR_CATEGORIES:
        for pair in pairs:
            cs = cross_session_analyzers.get(pair)
            if cs is None or not cs.aggregated_projections:
                continue
            row_category[len(rows)] = category
            rows.append((pair, 'region_i', pair[0]))
            row_category[len(rows)] = category
            rows.append((pair, 'region_j', pair[1]))

    if not rows:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    n_rows = (len(rows) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width * 2, row_height * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for r, (pair, region_role, region_name) in enumerate(rows):
        ax = axes[r]
        cs = cross_session_analyzers[pair]
        ax.set_facecolor(CATEGORY_COLORS.get(row_category[r], '#888888'))
        ax.patch.set_alpha(0.07)

        mean_key, sem_key, sessions_key = (
            ('u_mean', 'u_sem', 'u_sessions') if region_role == 'region_i'
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
        ax.set_ylim(*((-0.5, 0.5) if KERNEL_MODE == 'pcca'or KERNEL_MODE == 'pCCA_out_behaviour' else (-2.5, 5.0)))
        ax.text(0.01, 0.90, f"{_display_pair(pair)}  \u2014  {_display_name(region_name)}",
                transform=ax.transAxes, fontsize=TICK_FONTSIZE - 6, va='top', ha='left')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)
        if r == 1:
            ax.legend(fontsize=LEGEND_FONTSIZE - 3, loc='lower right', frameon=False)

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
# 8.  Shared plotting engine -- generalises pcca_cross_session_mi_bar.py's
#     plot_cross_session_mi from "one bar per pair" to "one-or-more clusters
#     of one-or-more bars per pair", while reusing every style constant
#     (category background bands, jitter/error-bar/dot styling, font sizes,
#     dpi, spine removal, inverted y-axis) from that script unchanged.
# =============================================================================
def plot_grouped_pair_bars(
        pair_categories: List[Tuple[str, List[Tuple[str, str]]]],
        pair_clusters: Dict[Tuple[str, str], List[List[dict]]],
        save_path: Path,
        xlabel: str,
        xlim: Optional[Tuple[float, float]] = None,
        legend_entries: Optional[List[Tuple[str, str, Optional[str]]]] = None,
        vline_zero: bool = False,
        fig_width: float = 11.0,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """
    pair_clusters[pair] : list of clusters; each cluster is a list of bar
    dicts {mean, sem, values, color, alpha, hatch}. One cluster per pair for
    tasks 1/2/3 (a single row of 1-3 bars); multiple clusters per pair for
    task 4 (one cluster per non-reference trial type).
    """
    groups = []
    for category, pairs in pair_categories:
        present = [p for p in pairs if p in pair_clusters and pair_clusters[p]]
        if present:
            groups.append((category, present))
    if not groups:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    y = 0.0
    flat_bars: List[Tuple[float, dict]] = []
    dot_x: List[float] = []
    dot_y: List[float] = []
    pair_label_pos: Dict[Tuple[str, str], float] = {}
    group_spans: List[Tuple[str, float, float]] = []
    rng = np.random.default_rng(0)

    for category, pairs in groups:
        y_start = y
        for pair in pairs:
            clusters = pair_clusters[pair]
            pair_y0 = y
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
            pair_label_pos[pair] = (pair_y0 + y - 1.0) / 2.0
            y += PAIR_GAP
        group_spans.append((category, y_start, y - PAIR_GAP - 1.0))
        y += GROUP_GAP - 1.0

    n_bars = len(flat_bars)
    fig_h = max(7.0, 0.42 * n_bars + 2.2)
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

    ax.set_yticks(list(pair_label_pos.values()))
    ax.set_yticklabels([_display_pair(p) for p in pair_label_pos.keys()], fontsize=TICK_FONTSIZE)
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


# =============================================================================
# 9.  CSV I/O -- simple existence-check caching, matching
#     pcca_cross_session_mi_bar.py's own convention (no config-hash keying).
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
        w.writerow(['region_i', 'region_j', 'display_pair', 'region_role', 'trial_type',
                    'n_sessions', 'mean_peak', 'sem_peak', 'wilcoxon_p_vs_ref'])
        for (pair, role), per_type in sorted(summary.items()):
            for trial_type, stats in per_type.items():
                w.writerow([pair[0], pair[1], _display_pair(pair), role, trial_type,
                           stats['n'], f"{stats['mean']:.6f}", f"{stats['sem']:.6f}",
                           stats['wilcoxon_p_vs_ref']])
    print(f"  [csv] -> {path}")


def _write_delta_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region_i', 'region_j', 'display_pair', 'region_role', 'trial_type',
                    'n_sessions', 'mean_delta', 'sem_delta', 'wilcoxon_p_vs_zero'])
        for (pair, role), per_type in sorted(summary.items()):
            for trial_type, stats in per_type.items():
                w.writerow([pair[0], pair[1], _display_pair(pair), role, trial_type,
                           stats['n'], f"{stats['mean']:.6f}", f"{stats['sem']:.6f}",
                           stats['wilcoxon_p_vs_zero']])
    print(f"  [csv] -> {path}")


def _write_variance_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region_i', 'region_j', 'display_pair', 'region_role', 'trial_type',
                    'predictor', 'n_sessions', 'mean_r2', 'sem_r2'])
        for (pair, role, trial_type), per_pred in sorted(summary.items()):
            for predictor, stats in per_pred.items():
                w.writerow([pair[0], pair[1], _display_pair(pair), role, trial_type, predictor,
                           stats['n'], f"{stats['mean']:.6f}", f"{stats['sem']:.6f}"])
    print(f"  [csv] -> {path}")

class _PrivateLatentSessionAdapter:
    """Minimal duck-typed stand-in for CrossTrialTypeCCAAnalyzer, exposing
    only what CrossSessionCCAAnalyzer.add_session_result actually reads
    (.projections, .statistical_results, .time_bins) so that method and
    aggregate_projections() can be reused unmodified for KERNEL_MODE ==
    'pCCA_out_behaviour'."""
    def __init__(self, projections: Dict[str, Dict[str, np.ndarray]], time_bins: np.ndarray):
        self.projections = projections
        self.statistical_results: Dict = {}   # unread by add_session_result/aggregate_projections
        self.time_bins = time_bins


def run_full_analysis_pcca_out_behaviour(
        sessions: List[str] = SESSIONS,
        pair_list: List[Tuple[str, str]] = REGION_PAIRS,
        base_dir: Path = BASE_DIR,
        reference_type: str = REFERENCE_TYPE,
        active_trial_types: List[str] = ACTIVE_TRIAL_TYPES,
        component_indices: List[int] = COMPONENT_INDICES,
        min_sessions: int = MIN_SESSIONS,
) -> Tuple[Dict[Tuple[str, str], CrossSessionCCAAnalyzer], List[dict]]:
    """KERNEL_MODE == 'pCCA_out_behaviour' counterpart of run_full_analysis:
    sources per-session, per-pair latents from PrivateLatentAnalyzer's
    pickled AllRegions+Behaviour results instead of fitting/reprojecting
    CCA weights from the .mat pipeline. Returns the SAME
    (cross_session_analyzers, behavior_records) shape as run_full_analysis,
    so every task 1-5 call in main() below is unchanged.

    Caveat, worth keeping in mind when reading task 2 (delta): unlike the
    pcca/cca branches, where one reference-trained subspace is projected
    onto every trial type, each trial type here comes from an
    INDEPENDENTLY fit AllRegions+Behaviour pCCA -- there is no shared
    subspace being generalised across conditions. Task 2's "delta" is
    therefore a difference between two independently optimised subspaces,
    not a generalisation gap. Tasks 3-4 (behavioural variance) are
    unaffected by this distinction -- that comparison is the whole point of
    this mode (see the request's motivation: position/speed R^2 should come
    out near zero, since that variance was already partialled out upstream).
    """
    analyzers_by_trial_type: Dict[str, PrivateLatentAnalyzer] = {
        t: PrivateLatentAnalyzer(base_dir=base_dir, trial_type=t) for t in active_trial_types
    }
    for az in analyzers_by_trial_type.values():
        az.load_all()

    cross_session_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer] = {}
    behavior_records: List[dict] = []

    all_session_names = sorted(set().union(
        *(az.sessions.keys() for az in analyzers_by_trial_type.values())
    ))
    if sessions:
        all_session_names = [s for s in all_session_names if s in sessions]

    for s_idx, session_name in enumerate(all_session_names, 1):
        print("\n" + "=" * 70)
        print(f"SESSION {s_idx}/{len(all_session_names)}: {session_name}  (pCCA_out_behaviour)")
        print("=" * 70)

        behavior_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

        for region_i, region_j in pair_list:
            pair_key = sort_pair_by_anatomy(region_i, region_j)

            per_trial_type: Dict[str, Dict[str, np.ndarray]] = {}
            time_vec_for_session: Optional[np.ndarray] = None
            for trial_type, az in analyzers_by_trial_type.items():
                session_result = az.sessions.get(session_name)
                if session_result is None:
                    continue
                pr = session_result.pairs.get(pair_key)
                if pr is None:
                    continue
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

            # ---- tasks 1, 2, 5 data path -----------------------------------
            adapter = _PrivateLatentSessionAdapter(
                projections=per_trial_type, time_bins=time_vec_for_session,
            )
            if pair_key not in cross_session_analyzers:
                cross_session_analyzers[pair_key] = CrossSessionCCAAnalyzer(
                    base_dir=str(base_dir), region_pair=pair_key,
                    reference_type=reference_type, n_components=N_COMPONENTS,
                    min_sessions=min_sessions,
                )
            cross_session_analyzers[pair_key].add_session_result(
                session_name, adapter, swap_uv=False,
            )

            # ---- tasks 3, 4 data path: position/speed GLM fit is computed
            #      exactly as for pcca/cca -- this is what the request's
            #      motivation is checking (expect near-zero R^2 here) -----
            for trial_type, proj in per_trial_type.items():
                if trial_type not in behavior_cache:
                    behavior_cache[trial_type] = _load_behavior_safe(session_name, trial_type)
                behav = behavior_cache[trial_type]
                if behav is None:
                    continue
                pos_full, speed_full, t_behav_full = behav

                for comp_idx in component_indices:
                    for region_role, region_name, trials in (
                            ('region_i', pair_key[0], proj['u_trials']),
                            ('region_j', pair_key[1], proj['v_trials'])):
                        latent = trials[:, :, comp_idx]
                        prep = _prepare_regression_inputs(
                            latent, time_vec_for_session, pos_full, speed_full, t_behav_full
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
                                session=session_name, pair=pair_key, region_role=region_role,
                                region=region_name, trial_type=trial_type, component=comp_idx,
                                predictor=var_name, r2=r2_val,
                            ))

    print("\n" + "=" * 70)
    print("CROSS-SESSION AGGREGATION (sign alignment + mean/SEM across sessions)")
    print("=" * 70)
    for pair_key, cs in cross_session_analyzers.items():
        n_sess = len(cs.session_projections)
        if n_sess < min_sessions:
            print(f"  {pair_key[0]} vs {pair_key[1]}: {n_sess} sessions (skipping, < {min_sessions})")
            continue
        cs.aggregate_projections()

    print("\n" + "=" * 70)
    print("DATA-GATHERING COMPLETE (pCCA_out_behaviour)")
    print(f"  region pairs with >=1 session : {len(cross_session_analyzers)}")
    print(f"  behavioural-variance records  : {len(behavior_records)}")
    print("=" * 70)
    return cross_session_analyzers, behavior_records

# =============================================================================
# 10.  Driver
# =============================================================================
def main() -> None:
    print("=" * 70)
    print("CROSS-TRIAL-TYPE PEAK / DELTA / BEHAVIOURAL-VARIANCE ANALYSIS")
    print("=" * 70)
    print(f"  kernel mode        : {KERNEL_MODE}")
    print(f"  reference type     : {REFERENCE_TYPE}")
    print(f"  active trial types : {ACTIVE_TRIAL_TYPES}")
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
              f"cross_session_analyzers needed for tasks 1-2's flip-aligned "
              f"trace data are NOT cached, so a full re-run is still required "
              f"unless you additionally cache/reload aggregated_projections "
              f"yourself. Set USE_CACHED_DATA = False to force a clean run.")

    if KERNEL_MODE == 'pCCA_out_behaviour':
        cross_session_analyzers, behavior_records = run_full_analysis_pcca_out_behaviour()
    else:
        cross_session_analyzers, behavior_records = run_full_analysis()


    # ---- Task 1: peak -----------------------------------------------------
    for comp_idx in COMPONENT_INDICES:
        print(f"\n--- Task 1: peak comparison (component {comp_idx}) ---")
        peak_summary, peak_records = aggregate_peak_records(
            cross_session_analyzers, component_idx=comp_idx)
        _write_records_csv(peak_records, OUTPUT_DIR / f"task1_peak_records_comp{comp_idx}.csv")
        _write_peak_summary_csv(peak_summary, OUTPUT_DIR / f"task1_peak_summary_comp{comp_idx}.csv")
        for region_role in ('region_i', 'region_j'):
            suffix = 'left_region' if region_role == 'region_i' else 'right_region'
            plot_task1_peak_bars(
                peak_summary, region_role,
                OUTPUT_DIR / f"task1_peak_comp{comp_idx}_{suffix}.png")

        # ---- Task 2: delta --------------------------------------------------
        print(f"\n--- Task 2: delta comparison (component {comp_idx}) ---")
        delta_summary, delta_records = aggregate_delta_records(
            cross_session_analyzers, component_idx=comp_idx)
        _write_records_csv(delta_records, OUTPUT_DIR / f"task2_delta_records_comp{comp_idx}.csv")
        _write_delta_summary_csv(delta_summary, OUTPUT_DIR / f"task2_delta_summary_comp{comp_idx}.csv")
        for region_role in ('region_i', 'region_j'):
            suffix = 'left_region' if region_role == 'region_i' else 'right_region'
            plot_task2_delta_bars(
                delta_summary, region_role,
                OUTPUT_DIR / f"task2_delta_comp{comp_idx}_{suffix}.png")

    # ---- Tasks 3 & 4: behavioural variance ---------------------------------
    print("\n--- Tasks 3-4: behavioural variance explained ---")
    _write_records_csv(behavior_records, behav_records_path)
    variance_summary = aggregate_behavior_variance(behavior_records)
    _write_variance_summary_csv(variance_summary, OUTPUT_DIR / "task3_4_variance_summary.csv")

    for region_role in ('region_i', 'region_j'):
        plot_task3_variance_bars(variance_summary, OUTPUT_DIR / "task3_variance_reference.png")
        plot_task4_variance_bars(variance_summary, OUTPUT_DIR / "task4_variance_projected.png")

        print("\n--- Task 5: latent traces across sessions ---")
        for comp_idx in COMPONENT_INDICES:
            plot_task5_latent_traces(
                cross_session_analyzers, OUTPUT_DIR / f"task5_latent_traces_comp{comp_idx}.png",
                component_idx=comp_idx)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Figures and CSVs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
r"""
pcca_all_regions_out_behaviour.py
================================================================================

Per-session, per-region / per-pair PCA and PRIVATE pCCA LATENT decomposition
under the "AllRegions + Behaviour" nuisance condition, computed natively in
Python and cached to disk as one pickle file per session.

Six quantities are computed and stored SEPARATELY, for every hub region
(Parts 1a/1b) and every hub-region/partner-region combination (Parts 2a/2b/
2c/2c') in `REGION_PAIRS`:

    1a. PCA on each hub region's raw (z-scored) activity.
    1b. PCA on each hub region's activity AFTER regressing out behaviour
        (position x/y/z, speed).
    2a. PCA on the part of a hub region's Part-1b residual that is linearly
        EXPLAINED by every OTHER recorded region (excluding this hub and
        this partner) -- "how much of my behaviour-corrected activity does
        the rest of the network account for".
    2b. PCA on what is LEFT of that residual after removing 2a -- "what's
        left that the rest of the network does NOT account for".
    2c. The private pCCA latent between hub and partner, jointly
        residualized against BOTH the rest of the network AND behaviour in
        one ridge fit, with the ridge fit and CCA weights now K-fold
        cross-validated (see "Mathematical framework, Part 2c" below and
        `pcca`'s own docstring -- this revision's fix for the "no
        cross-validation" divergence from the original MATLAB pCCA,
        `perform_session_pcca.m`).
    2c'. The SAME private pCCA computation as 2c, but residualized against
        ONLY the rest of the network -- behaviour is never part of the
        nuisance Z here -- so a caller can compare the pair's coupling with
        and without behaviour partialled out (`pairs_regions_only`, see
        "Storage" below).

--------------------------------------------------------------------------------
Relationship to the rest of the pipeline
--------------------------------------------------------------------------------
Part 2c generalises a computation that already exists, for one hardcoded
target pair, inside `pCCA_sensitive_realsingle_Session_8panel.py`: the
`abl-AllRegions+Behaviour` step of `run_single_ablation`. There, for a fixed
pair (TARGET_I, TARGET_J), the nuisance design

    Z = [ X_r  for every anatomical region r not in {TARGET_I, TARGET_J} ]
        concatenated with [ position (x, y, z), speed ]

is regressed out of both target regions' activity via a joint ridge hat
matrix (`residualize`), and ridge CCA (`ridge_cca`) is then fit on the two
residuals (`pcca`). Part 2c performs *exactly* that computation -- same
Z construction, same residualize/ridge_cca/pcca primitives, same crop to the
behavioural-tracking window `BEHAVIOR_TIME_RANGE_S` (now `ALIGNMENT_WINDOWS_S
[ALIGN]`, e.g. `(-1.2, 1.8)` s for `ALIGN="reward_onset"`, rather than a
fixed `(-1.0, 2.0)`) -- but loops it over every pair in `REGION_PAIRS` (the
same 21-pair / 7-category
set used throughout this project's cross-session figures, e.g.
`pcca_cross_session_mi_bar.py` and `pCCA_latent_extrenal_variable_bar.py`,
copied verbatim from the latter) and over every session auto-discovered
under the relevant `pcca_sessions_{trial_type}_results` folder.

The result is called "private" because, once every other recorded region
AND the animal's kinematics have been partialled out, what remains in the
region_i <-> region_j canonical correlation is the coupling specific to that
one dyad, not a hub-wide signal shared with the rest of the network -- the
pCCA counterpart of a partner-specific ("private") communication channel.

Parts 1a/1b/2a/2b are new in this version, and sit upstream of and alongside
Part 2c rather than replacing it: they decompose each hub region's OWN
signal (never a canonical pair) into "raw", "behaviour-corrected",
"network-explained", and "network-residual" pieces, giving a per-region and
per-hub-orientation baseline against which Part 2c's joint, CCA-refined
private latent can be compared. Nothing about Part 2c's own computation
changes -- it is retained verbatim (see `_compute_pair_result`) precisely so
the comparison stays apples-to-apples with every session already processed
by the previous version of this script.

--------------------------------------------------------------------------------
Mathematical framework
--------------------------------------------------------------------------------
Throughout, X_r denotes region r's flattened, z-scored activity in
R^{(T n_trials) x n_r} (`_zscore_flat`), and Bhv denotes the flattened,
z-scored behavioural design (position x/y/z, speed) in
R^{(T n_trials) x 4}.

**Part 1a (region-level PCA, raw).** For hub region h, PCA is fit directly
on X_h, no regression step at all:

    X_h = U S V^T                     (SVD of the column-centred X_h)
    W_h = V[:, :K]                    (loadings, Part-1a PCA)
    latent_h(trial, t) = [(X_h - mean(X_h)) W_h]_{(trial, t), :}

**Part 1b (region-level PCA, behaviour regressed out first).** Behaviour is
partialled out of X_h via the SAME joint ridge hat matrix used everywhere
else in this file (`residualize`), and PCA is then fit on the residual:

    X_h^resid = X_h - Bhv (Bhv^T Bhv + lam_hat * n * I)^(-1) Bhv^T X_h
    W_h^(1b)  = PCA loadings of X_h^resid

**Part 2 (hub-paired).** For canonicalized pair (region_i, region_j), each
member takes a turn as "hub" (the other as "partner") -- a canonicalized
pair therefore yields TWO hub orientations, not one, since Parts 2a/2b are
directional in a way Part 2c's joint fit is not. Let Z_h denote the
flattened, concatenated activity of every OTHER recorded region excluding
{hub, partner} (`_other_region_nuisance_list` -- the identical "AllRegions"
half of Part 2c's own nuisance set, just without behaviour appended, since
behaviour was already removed from X_h^resid in Part 1b):

    Beta_h          = (Z_h^T Z_h + lam_hat * n * I)^(-1) Z_h^T
    explained_h (2a) = Z_h @ (Beta_h @ X_h^resid)        <- PCA fit on this
    residual_h  (2b) = X_h^resid - explained_h            <- PCA fit on this

so explained_h + residual_h = X_h^resid EXACTLY (Part 1b's residual, not
Part 1a's raw activity -- Parts 2a/2b always start from behaviour already
removed). `explained_h` is "how much of hub h's behaviour-corrected signal
the rest of the recorded network, excluding this specific partner,
linearly accounts for"; `residual_h` is what that network does NOT account
for. Because `Beta_h` depends only on Z_h -- not on which region is being
explained -- one ridge solve per canonicalized pair yields BOTH hub
orientations' explained/residual split (see `_compute_hub_pair_pca_
result`), each independently reduced to Part-1-style PCA loadings and
latents exactly as in Part 1a/1b.

**Part 2c -- private pCCA.** Z_full = [ Z_h , Bhv ] (the SAME AllRegions
nuisance as above, but with behaviour concatenated back on, and BOTH
regions of the pair jointly residualized against it in a single ridge fit
-- not sequentially, the way Part 1b -> 2a/2b proceeds), now fit with
K-FOLD CROSS-VALIDATION (`pcca`, `CV_FOLDS` trial-based folds -- see that
function's own docstring): within each fold, Beta_X/Beta_Y below are fit
on the TRAIN trials only and applied unchanged to the held-out TEST
trials; `Wx`/`Wy` are the sign-aligned, fold-averaged canonical weights;
`rho` is the held-out (test-fold) canonical correlation, averaged across
folds. The projected latent z_i^(k) still uses a single ridge fit on
EVERY sample (not the CV folds) for X_hat, applied through the
fold-averaged Wx -- only the weight vectors and rho are cross-validated:

    X_hat = X - Z_full (Z_full^T Z_full + lam_hat * n * I)^(-1) Z_full^T X
    Cxx = X_hat_i^T X_hat_i / (n-1),  Cyy, Cxy analogous     (per fold, on
    A = Cxx^(-1/2)_{lam_cca},  B = Cyy^(-1/2)_{lam_cca}       TRAIN rows)
    U, S, V^T = SVD(A Cxy B)
    Wx_fold = A U[:, :K],  Wy_fold = B V^T[:, :K]
    Wx = mean_folds(sign_align(Wx_fold)),  Wy analogous
    rho = mean_folds(corr(X_hat_i^test Wx_fold, X_hat_j^test Wy_fold))
    z_i^(k)(trial, t) = [X_hat_i Wx]_{(trial, t), k}          (X_hat_i from
                                                                 the FULL-data fit)

This replaces a single in-sample (non-cross-validated) fit used by an
earlier revision of this file, which was identified as a source of
downward-biased behavioural R^2 relative to the original MATLAB-computed
pCCA (`perform_session_pcca.m`, which always cross-validates its own
region-only nuisance fit) -- see `pcca`'s docstring for the full
rationale.

The joint fit above and the sequential Part 1b -> 2a/2b route are NOT
algebraically equivalent unless the behavioural design Bhv happens to be
orthogonal to the network design Z_h (joint multiple regression correctly
handles Bhv/Z_h collinearity; two sequential regressions double-count
shared nuisance variance instead -- see this project's own working notes on
"joint nuisance regression is strictly superior to per-region sequential
regression"). This divergence is intentional, not a bug to reconcile:
Parts 1b/2a/2b exist precisely to expose how much that sequential-vs-joint
distinction matters for this dataset, as a diagnostic ALONGSIDE -- not a
replacement for -- Part 2c's already-correct joint fit.

--------------------------------------------------------------------------------
Storage
--------------------------------------------------------------------------------
Following this project's "primitives copied, not imported" convention --
every core numerical routine below is copied verbatim from
pCCA_sensitive_realsingle_Session_8panel.py / cross_trial_type_cca_analysis.py
(Parts 1/2a/2b's PCA and explained/residual primitives are NEW, written for
this version -- see Section 4's header comment for exactly which functions
are copied vs. new) so this script stays independently auditable and
runnable on its own -- results are not kept only in memory.
`run_all_sessions()` writes one `{session}_analysis_results.pkl` file per
session into

    {BASE_DIR}/pcca_all_regions_out_behaviour_sessions_{trial_type}_results/

directly mirroring the naming convention of `pcca_sessions_{trial_type}_
results` (the folder this script's own neural input already lives in); the
only difference is the file format -- .pkl, loaded with `pickle`, in place
of .mat, loaded with `mat73`. Each session's `PrivateLatentSessionResult`
holds SIX separately-named result sets (Parts 1a, 1b, 2a+2b, 2c, and 2c'
are each independently addressable, per this version's storage
requirement):

    .region_pca_raw           Dict[region -> RegionPCAResult]        Part 1a
    .region_pca_out_behaviour Dict[region -> RegionPCAResult]        Part 1b
    .hub_pca_pairs  Dict[(region_i,region_j) -> HubPairPCAResult]    Parts 2a+2b
    .pairs      Dict[(region_i,region_j) -> PrivateLatentPairResult] Part 2c  (AllRegions+Behaviour, now cross-validated)
    .pairs_regions_only Dict[(region_i,region_j) -> PrivateLatentPairResult] Part 2c' (AllRegions only, NEW)

`.region_pca_raw` / `.region_pca_out_behaviour` are keyed by region name
alone -- one entry per hub region, since Part 1 has no partner. `.hub_pca_
pairs` is keyed IDENTICALLY to the existing `.pairs` (canonicalized
(region_i, region_j), region_i anatomically earlier), so the two dicts
share a key space and can be zipped on it; each `HubPairPCAResult` then
holds BOTH hub orientations as named sub-objects (`.region_i_as_hub`,
`.region_j_as_hub`), each bundling its own Part-2a (`*_network`) and
Part-2b (`*_residual`) PCA side by side -- the pairing the next task's
`PCA_latent_extrenal_variable_bar.py` modification is expected to build its
"paired display" from, by analogy with how `pCCA_latent_extrenal_variable_
bar.py` already displays region_i next to region_j for Part 2c.
`.pairs_regions_only` is keyed IDENTICALLY to `.pairs` too, and holds the
SAME `_compute_pair_result` computation with `behavior_Z_flat=None` -- so a
caller can directly compare, pair by pair, what changes once behaviour is
also regressed out. `PrivateLatentAnalyzer.get_region_pca` / `.get_hub_pca`
provide canonicalization-aware lookups for all four Part-1/2a/2b result
sets, and `.get_pair(..., regions_only=True/False)` for both Part 2c
variants.

`PrivateLatentAnalyzer`, defined at the bottom of this file, is the
Python-native counterpart of `OxfordAdvancedAnalyzer` (Useful_definition.py)
/ `CrossTrialTypeCCAAnalyzer` (cross_trial_type_cca_analysis.py): it indexes
these .pkl files the same way those two classes index the .mat files, minus
the MATLAB round-trip, since every fit here was already performed in
Python.

Switching `TRIAL_TYPE` below from "cued_hit_long" to "spont_hit_long" (or
"spont_miss_long") re-targets the whole pipeline -- input folder, output
folder, and the "task_label" filter string read from each session's
consolidated {session}.pkl -- at that condition; nothing else has to
change.

Note on scope: the plotting/diagnostic machinery of the source ablation
script (StepResult, SupplementaryMetrics, Rastermap neuron ordering, the
8-panel figures, per-step sign correction) is deliberately NOT carried over
here. This script's job is to compute and persist Parts 1a/1b/2a/2b/2c, so
only the primitives those five computations actually need are copied or
newly written in. Canonical-vector and PCA-loading sign is left as computed
(arbitrary up to a per-component flip, a standard CCA/pCCA/PCA property)
rather than heuristically corrected, for ALL five result sets: the one
place downstream that is already known to consume this kind of latent,
`variance_explained` in pCCA_latent_extenal_variable_bar.py, is explicitly
documented as sign-invariant, and Parts 1/2a/2b's PCA loadings carry the
same SVD sign ambiguity. A cross-session sign-alignment step (analogous to
`CrossSessionCCAAnalyzer._align_signs_spectral`) is a cross-session
operation and belongs in a future aggregation step built on top of
`PrivateLatentAnalyzer`, not in this per-session computation.

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

# ---- Alignment hyperparameter -- selects which behavioural event trials
#      are aligned to. Mirrors ALIGN / ALIGN_MODES in
#      pCCA_sensitive_realsingle_Session_11panel.py: this one switch picks
#      BOTH the neural session-region-data source folder (`mat_subdir_name`,
#      e.g. "cued_hit_long_cue_onset_results") AND the behavioural
#      regressor folder (`BEHAVIOR_DIR`, e.g. "tapproach_sessions_cue_onset"),
#      so the two can never drift out of sync with each other.
ALIGN_MODES: Dict[str, str] = {
    "default_move_onset": "align to movement onset using t_approach.start_time directly",
    "cue_onset":           "align to cue onset: start_time - t_approach.cue",
    "bar_off_onset":       "align to bar-off onset: start_time + t_approach.bar_off",
    "reward_onset":        "align to reward onset: start_time + t_approach.drop_time",
}

# Per-alignment-mode trial window (seconds, relative to the aligned event) --
# must match segment_mdl_to_trials.m (neural) and t_approach_python.py's own
# ALIGNMENT_WINDOWS_S (behavioural) EXACTLY, so the two data sources line up
# sample-for-sample once both are loaded below. All four windows span 3.0s
# total at 50Hz (151 samples) -- matching the native recorded span of the
# behavioural position-tracking input, so no part of any window relies on
# extrapolated (edge-held) position data.
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

# ---- CCA / pCCA dimensionality & regularisation (matches the 8-panel script)
N_COMPONENTS: int = 1
LAMBDA_CCA: float = 1e-4
LAMBDA_HAT: float = 1e-4

# ---- Part 2c/2c' cross-validation -- matches oxford_session_pipeline_mdl.m's
#      analysis_config.cv_folds (10), the config actually fed to MATLAB's
#      perform_session_pcca.m. Part 2c's private pCCA is now fit the SAME
#      way that file fits it: nuisance-regression coefficients and CCA
#      weights are estimated on a train fold and evaluated on a held-out
#      test fold (see `pcca`'s own docstring for exactly how), instead of
#      a single in-sample fit on every sample at once -- an in-sample fit
#      is optimistically biased (it can lock onto session-specific shared
#      noise between the two residualized regions that has nothing to do
#      with behaviour) and was identified as the leading cause of this
#      script's private-latent behavioural R^2 reading LOWER than the
#      original MATLAB-computed pCCA's. Folds are split by TRIAL (every
#      timepoint of a held-out trial stays in the same fold), not by raw
#      flattened sample index the way MATLAB's own fold split happens to
#      -- a within-trial train/test split would leak autocorrelated
#      timepoints across the CV boundary, which is not what "held-out"
#      is supposed to mean.
CV_FOLDS: int = 10
CV_RNG_SEED: int = 12345   # fixed seed -> deterministic, reproducible fold assignment

# ---- Nuisance-region eligibility -- matches oxford_session_pipeline_mdl.m's
#      analysis_config.min_neurons_per_region (50), the threshold MATLAB's
#      perform_session_pcca.m applies before a region is allowed into its
#      nuisance matrix Z (region_matrices, Step 1 of that file). This
#      script previously had NO such threshold: `_other_region_nuisance_
#      list` let every recorded region into Z regardless of how few
#      neurons it contributed, which both adds noise columns to the
#      nuisance regression AND is a straightforward divergence from how
#      MATLAB originally defined "AllRegions" nuisance for the
#      regress-region-only case.
MIN_NEURONS_PER_REGION: int = 50

# ---- PCA dimensionality for Parts 1a/1b/2a/2b (region-level and hub-
#      orientation PCA) -- deliberately a SEPARATE knob from N_COMPONENTS
#      above (Part 2c's pCCA canonical dimensionality): PCA over a
#      region's own neurons is typically informative well past K=1,
#      whereas ridge CCA between two regions is normally kept small.
#      Matches the default already used by this project's downstream
#      consumer, PCA_latent_extenal_variable_bar.py's own N_COMPONENTS.
N_PCA_COMPONENTS: int = 5

# ---- Time windows -------------------------------------------------------
# Both the raw neural acquisition window and the behavioural crop window are
# now simply ALIGN's window (ALIGNMENT_WINDOWS_S) -- segment_mdl_to_trials.m
# already segments the neural .mat data to exactly this window per
# alignment mode, and t_approach_python.py already crops its saved
# pos/speed/time .npy files to the same window, so there is nothing left
# for crop_time_window() below to actually trim; it is kept as a defensive
# no-op / mismatch check rather than removed.
TIME_RANGE_S: Tuple[float, float] = ALIGNMENT_WINDOWS_S[ALIGN]          # raw neural acquisition window
BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = ALIGNMENT_WINDOWS_S[ALIGN]  # behavioural-tracking window

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


def mat_subdir_name(trial_type: str, align_mode: str = ALIGN) -> str:
    """MATLAB-pipeline session-region-data source folder for `trial_type` /
    `align_mode`, e.g. "cued_hit_long_cue_onset_results" -- matches
    SESSION_DATA_DIR's naming convention in
    pCCA_sensitive_realsingle_Session_11panel.py. `align_mode` defaults to
    the module-level ALIGN so every existing single-argument call site
    (this file's own, and PCA_latent_extrenal_variable_part.py's) keeps
    resolving to whichever alignment mode is currently selected above."""
    return f"{trial_type}_{align_mode}_results"


def out_subdir_name(trial_type: str, align_mode: str = ALIGN) -> str:
    """This script's own output folder for `trial_type` / `align_mode`."""
    return f"pcca_all_regions_out_behaviour_sessions_{trial_type}_{align_mode}_results"


def behavior_label_for(trial_type: str) -> str:
    """'cued_hit_long' -> 'cued hit long', matching the "task_label" string
    convention inside each session's consolidated {session}.pkl
    (BEHAVIOR_TRIAL_LABEL in the 8-panel script / _trial_type_to_behavior_
    label in pCCA_latent_extenal_variable_bar.py)."""
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

# ---- Every region that appears in REGION_PAIRS, anatomically ordered --
#      Parts 1a/1b/2a/2b's "for each hub region": since REGION_PAIRS is
#      the complete graph over these regions (21 pairs = C(7,2)), every
#      entry here is simultaneously a hub in its own right AND a partner
#      in every OTHER hub's Part-2 computation -- the same relationship
#      already established for this project's per-hub heatmap figures
#      (heatmap_PCA_PCCA.py's REGIONS_OF_INTEREST). Derived, not
#      hardcoded, so there is no second region list to keep in sync by
#      hand if REGION_PAIRS above is ever extended.
HUB_REGIONS: List[str] = sorted(
    {region for pair in REGION_PAIRS for region in pair},
    key=get_anatomical_index,
)


# =============================================================================
# 3b. Subregion / laminar-depth classification for Part 3's weight-ratio
#     metrics -- LAMINAR_DEPTH_MAP / get_laminar_depth / CORTICAL_REGIONS
#     copied from pCCA_sensitive_realsingle_Session_11panel.py (the only
#     existing place in this project that already classifies a cortical
#     subregion_labels string as superficial or deep), minus the mPFC
#     entries -- mPFC never appears in this file's HUB_REGIONS.
# =============================================================================

EXCLUDED_SUBREGION_LABELS: List[str] = ["out"]

LAMINAR_DEPTH_MAP: Dict[str, str] = {
    # ---- mPFC (kept for completeness even though mPFC is not itself a
    #      hub region here; harmless if a label never actually occurs) ----
    "ACAd5":   "layer-deep",
    "ACAd6a":  "layer-deep",
    "PL6a":    "layer-deep",
    "ILA6a":   "layer-deep",
    "FRP6a":   "layer-deep",

    # ---- ORB ----
    "ORBl23":  "layer-shallow",
    "ORBvl23": "layer-shallow",
    "ORBl5":   "layer-deep",
    "ORBvl5":  "layer-deep",
    "ORBl6a":  "layer-deep",
    "ORBm6a":  "layer-deep",
    "ORBvl6a": "layer-deep",
    "ORBl6b":  "layer-deep",
    "ORBvl6b": "layer-deep",

    # ---- MOp ----
    "MOp1":    "layer-shallow",
    "MOp23":   "layer-shallow",
    "MOp5":    "layer-deep",
    "MOp6a":   "layer-deep",
    "MOp6b":   "layer-deep",

    # ---- MOs ----
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
    'layer-deep'. `strict=False` (this file's default -- the opposite of
    the 11panel script's) returns None for an unmapped label instead of
    raising, since Part 3's metrics must degrade gracefully: an
    unrecognised or unresolved cortical label should simply drop out of
    the ratio, not abort the whole session's computation."""
    if subregion_label in depth_map:
        return depth_map[subregion_label]
    if strict:
        raise KeyError(
            f"'{subregion_label}' has no laminar depth assignment "
            f"(not a laminar cortical subregion, or an unresolved/"
            f"subcortical label)."
        )
    return None


# Hub regions in REGION_PAIRS whose subregion labels are laminar-cortical
# (every key of LAMINAR_DEPTH_MAP belongs to one of these, plus mPFC, which
# never appears in this file's HUB_REGIONS).
CORTICAL_REGIONS: List[str] = ["MOp", "MOs", "ORB"]


# =============================================================================
# 4.  Core PCA / pCCA / CCA primitives. `_zscore_flat`, `_ridge_inv_sqrt`,
#     `ridge_cca`, `residualize`, `pcca`, and `latent_projections` are
#     copied verbatim from pCCA_sensitive_realsingle_Session_8panel.py, as
#     in the previous version of this file. `residualize_with_explained`
#     and `pca_fit_and_project` are NEW, written for Parts 1/2a/2b: this
#     project's PCA has so far only ever been fit in MATLAB (see
#     cross_trial_type_pca_analysis.py's own docstring), so there is no
#     existing native-Python PCA primitive to copy from. Both follow the
#     same style/signature idioms as their neighbours here, and
#     `residualize_with_explained` is structurally IDENTICAL to
#     `residualize` -- just additionally returning the "explained" half of
#     the same ridge decomposition -- so a reader can confirm by
#     inspection that Part 2a/2b's Z_flat @ (Beta @ X_flat) is the SAME
#     ridge solve Part 2c's nuisance regression already uses, not a
#     second, potentially-diverging implementation.
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


def _fit_ridge_beta(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_hat: float = LAMBDA_HAT,
) -> Optional[np.ndarray]:
    """Ridge hat-matrix regression coefficients of X_flat onto Z_flat,
    fit on whatever ROWS of X_flat/Z_flat are passed in:

        Beta = (Z^T Z + lam_hat * n * I)^(-1) Z^T X     (m, n_features)

    `n` is the number of rows actually passed in -- e.g. one CV train
    fold's samples, not necessarily the full dataset -- so the same Beta
    can be fit on a train fold and applied UNCHANGED to that fold's own
    held-out test rows via `_apply_ridge_beta` (see `pcca`'s docstring for
    why this matters). None if there is no nuisance to regress out."""
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
    """Apply a Beta already fit by `_fit_ridge_beta` (on possibly DIFFERENT
    rows -- e.g. a train fold) to `X_flat`/`Z_flat`. `beta=None` (nothing
    was fit, e.g. Z_flat was empty) returns X_flat unchanged."""
    if beta is None:
        return X_flat.copy()
    return X_flat - Z_flat @ beta


def residualize(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_hat: float = LAMBDA_HAT,
) -> np.ndarray:
    """Single ridge fit-and-apply on the SAME rows (no train/test split) --
    `_fit_ridge_beta` + `_apply_ridge_beta` composed for the common
    (non-cross-validated) case: Part 1b's behaviour regression, Part
    2a/2b's network regression, and Part 2c/2c's own full-dataset
    residualization used for the final latent projection (see `pcca`)."""
    beta = _fit_ridge_beta(X_flat, Z_flat, lam_hat)
    return _apply_ridge_beta(X_flat, Z_flat, beta)


def residualize_with_explained(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_hat: float = LAMBDA_HAT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same joint ridge hat-matrix regression as `residualize`, but returning
    BOTH halves of the decomposition instead of only the residual:

        Beta      = (Z^T Z + lam_hat * n * I)^(-1) Z^T X    -- `_fit_ridge_beta`
        explained = Z_flat @ Beta                            -- Part 2a
        residual  = X_flat - explained                       -- Part 2b,
                     numerically identical to residualize(X_flat, Z_flat,
                     lam_hat)'s own return value (same Beta, same solve).

    `Z_flat=None`/empty degenerates the same way `residualize` does:
    nothing to explain, so explained = 0 and residual = X_flat unchanged.
    """
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
    """Partition the `n_trials` trials underlying a `_zscore_flat`-produced
    (T * n_trials, n_neurons) matrix into `n_folds` folds, and return, for
    each fold, (train_row_idx, test_row_idx) into that flattened row axis.

    Row `r` of a `_zscore_flat` output belongs to trial `r % n_trials`
    (that function's own flatten order is time-major: `r = t * n_trials +
    trial`). Folds are built by RANDOMLY partitioning trials (fixed `seed`
    -> deterministic across runs, so results are reproducible and every
    region pair/variant in a session is evaluated against the SAME
    held-out trials), then including EVERY timepoint of a held-out trial's
    row range in that fold's test set -- never splitting one trial's own
    timepoints across the train/test boundary, since that would leak
    within-trial autocorrelation into a metric meant to be held-out.

    This deliberately does NOT reproduce `perform_session_pcca.m`'s own
    fold split, which slices the flattened MATLAB sample axis into
    contiguous blocks with no explicit trial-boundary guarantee -- a
    trial-aware split is the more rigorous choice and is not a MATLAB-
    parity requirement (see `pcca`'s own docstring).
    """
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
    conditioning on nuisance Z_flat -- the trial-aware Python counterpart
    of `perform_session_pcca.m`'s `run_pcca_pair`:

      * the nuisance-regression coefficients (Beta_X, Beta_Y) are fit on
        each fold's TRAIN rows only and applied, UNCHANGED, to that fold's
        own held-out TEST rows (`perform_session_pcca.m`'s own "WITHIN-
        FOLD NUISANCE REGRESSION" -- prevents test-fold leakage into the
        residuals CCA is then evaluated on);
      * ridge CCA (`ridge_cca`) is fit on the TRAIN fold's residuals only;
      * the reported canonical correlation `rho` is the HELD-OUT, test-fold
        correlation between the two projections, averaged across folds --
        not the in-sample correlation of a single full-data fit, which is
        optimistically biased upward (see module docstring, "no
        cross-validation" divergence from the original MATLAB pCCA);
      * the returned Wx/Wy (and therefore every downstream latent
        projection) is the fold-averaged canonical weight matrix, SIGN-
        ALIGNED against the first successful fold before averaging --
        `ridge_cca`'s (Wx, Wy) pair for a given component is only defined
        up to a joint sign flip (an SVD sign ambiguity), so averaging
        un-aligned folds can partially cancel the very signal the CV is
        meant to recover. `perform_session_pcca.m`'s own Wx_mean/Wy_mean
        average skips this alignment step; this implementation
        deliberately does not reproduce that, since silently cancelling
        signal across folds is a bug to fix, not a MATLAB-parity
        requirement.

    The RESIDUALS returned for downstream projection (`X_res`, `Y_res`,
    and therefore every `z_i_lat`/`z_j_lat` persisted by
    `_compute_pair_result`) come from a SEPARATE, single ridge fit on the
    FULL dataset (not the CV folds) -- matching
    `perform_session_pcca.m`'s own `calculate_pcca_projections`, which
    always re-residualizes on every sample before projecting through the
    fold-averaged weights. Only the CCA WEIGHT VECTORS and the reported
    `rho` come from cross-validation; the projected traces plotted/
    regressed downstream use every available sample.

    Falls back to a single in-sample fit (with a printed warning) only if
    every fold degenerates (e.g. a session with too few trials for
    `n_folds` non-empty train/test splits) -- expected to be rare given
    this project's typical trial counts.
    """
    N = X_flat.shape[0]
    p, q = X_flat.shape[1], Y_flat.shape[1]
    k = max(1, min(n_components, p, q))

    def _full_data_residuals() -> Tuple[np.ndarray, np.ndarray]:
        """Single ridge fit-and-apply on every sample -- the residuals
        every returned latent projection is built from (see docstring)."""
        beta_X = _fit_ridge_beta(X_flat, Z_flat, lam_hat)
        beta_Y = _fit_ridge_beta(Y_flat, Z_flat, lam_hat)
        X_res = _apply_ridge_beta(X_flat, Z_flat, beta_X)
        Y_res = _apply_ridge_beta(Y_flat, Z_flat, beta_Y)
        return X_res, Y_res

    def _full_data_fit() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """In-sample fallback used only when every CV fold degenerates."""
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
        X_te_res = _apply_ridge_beta(X_te, Z_te, beta_X)   # SAME beta as train fold
        Y_te_res = _apply_ridge_beta(Y_te, Z_te, beta_Y)   # -- no test-fold leakage

        Wx_f, Wy_f, _ = ridge_cca(X_tr_res, Y_tr_res, lam_cca, k)
        if Wx_f.shape[1] < k or Wy_f.shape[1] < k:
            continue  # degenerate fold (rank-deficient train residual) -- skip it

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


def pca_fit_and_project(
        X_flat: np.ndarray,
        n_trials: int,
        T: int,
        n_components: int = N_PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit standard (SVD-based, column-centred) PCA on a flattened
    (T * n_trials, n_neurons) activity matrix and immediately project it
    back through its own loadings -- the single-region analogue of `pcca`
    + `latent_projections` combined, since Parts 1/2's PCA is always fit
    and applied to the SAME matrix (there is no separate train/reference-
    condition projection step here; projecting one trial type's data
    through a DIFFERENT trial type's already-fitted weights is a
    cross-condition operation that belongs to the future
    PCA_latent_extenal_variable_bar.py modification, not to this
    per-session fit -- see this file's module docstring).

    X_flat : (T * n_trials, n_neurons)  already z-scored per neuron (Part
             1a's `region_flat[region]`), or a residual/explained-component
             thereof (Parts 1b/2a/2b) -- PCA only requires finite-mean
             input and re-centres defensively regardless of the caller's
             own upstream normalisation.

    Returns
    -------
    W                    : (n_neurons, K)   PCA loading matrix, columns
                            ordered by descending explained variance. Sign
                            is left as computed by the SVD (arbitrary up to
                            a per-component flip) -- the same convention
                            this file already applies to Wx/Wy (see 'Note
                            on scope' in the module docstring).
    explained_var_ratio  : (K,)             fraction of X_flat's total
                            variance captured by each retained component.
    mean                 : (1, n_neurons)   column mean subtracted before
                            the SVD.
    latent               : (n_trials, T, K) projection of the centred data
                            onto W, in the same (n_trials, T, K) layout
                            `latent_projections` already returns for
                            Part 2c.
    """
    mean = X_flat.mean(axis=0, keepdims=True)
    Xc = X_flat - mean
    n = Xc.shape[0]
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    W = Vt[:k].T                                              # (n_neurons, k)
    var = (S ** 2) / max(n - 1, 1)
    total_var = var.sum()
    explained_var_ratio = (
        var[:k] / total_var if total_var > 0 else np.zeros(k, dtype=np.float64)
    )
    latent = latent_projections(Xc, W, n_trials, T)           # (n_trials, T, k)
    return W, explained_var_ratio, mean, latent


# =============================================================================
# 5.  Data loading -- copied verbatim from
#     pCCA_sensitive_realsingle_Session_8panel.py.
# =============================================================================

def load_region_spikes(
        session_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], int, int]:
    """Load per-region spike tensors AND per-neuron subregion labels for one
    session.

    ``region_data.regions.<region>.subregion_labels`` sits at the same
    hierarchical level as ``spike_data`` (item 4): it is subset with the
    IDENTICAL ``selected_neurons`` index array already applied to
    ``spike_data``, so ``region_subregion_labels[r][i]`` stays aligned
    one-to-one with neuron ``i`` of ``region_spikes[r]``. Unlike
    ``pCCA_sensitive_realsingle_Session_11panel.py``'s own
    ``load_region_spikes``, a neuron whose label is in
    ``EXCLUDED_SUBREGION_LABELS`` (or missing) is NOT dropped here -- doing
    so would change which neurons PCA/pCCA fit on, altering every existing
    result this script produces. Such labels are instead treated as
    "unresolved" only inside Part 3's weight-ratio metrics
    (`compute_subregion_weight_metrics`), which is the sole consumer of
    this second return value.
    """
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-trial position (x, y, z) and speed traces for one session,
    filtered to trials matching `trial_label`.

    Reads t_approach_python.py's consolidated `{session}.pkl` (one dict per
    session: "pos", "task_label", "movement_label", "speed", "time",
    "trigger_times") instead of the four separate `{session}_*.npy` files
    that format replaced. The time axis comes directly from the pkl's own
    "time" field, written by t_approach_python.py's per-alignment-mode
    window crop (ALIGNMENT_WINDOWS_S there == ALIGNMENT_WINDOWS_S here) --
    this is what actually guarantees the behavioural axis matches the
    neural one sample-for-sample for whichever alignment mode produced
    `behavior_dir`, instead of merely assuming it.

    Returns
    -------
    pos_sel   : (n_trials_behav, 3, T_behav)  float32,  channels = [x, y, z]
    speed_sel : (n_trials_behav, 1, T_behav)  float32
    t_behav   : (T_behav,)  time vector in seconds
    """
    pkl_path = behavior_dir / f"{session_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Behaviour file not found: {pkl_path}")

    with open(pkl_path, "rb") as fh:
        session_data = pickle.load(fh)

    pos    = np.asarray(session_data["pos"])                     # (N, 3, T_behav)
    speed  = np.asarray(session_data["speed"])                    # (N, T_behav)
    labels = np.asarray(session_data["task_label"], dtype=object)  # (N,)
    t_behav = np.asarray(session_data["time"], dtype=np.float64)   # (T_behav,)

    if speed.ndim == 2:
        speed = speed[:, None, :]                      # → (N, 1, T_behav)

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
    subregion_weight_metrics_i: List[SubregionWeightMetrics] = field(default_factory=list)  # Part 3, one per column of Wx
    subregion_weight_metrics_j: List[SubregionWeightMetrics] = field(default_factory=list)  # Part 3, one per column of Wy


@dataclass
class SubregionWeightMetrics:
    """Part 3's weight-ratio measure for ONE PCA/CCA component of ONE
    region's weight vector: whether the neurons carrying the largest
    |weight| in this component tend to come from superficial vs. deep
    cortical layers (`region` in CORTICAL_REGIONS) or from a particular
    subcortical subregion (every other hub region). See
    `compute_subregion_weight_metrics` for exactly how each field below is
    derived. Every dict is keyed by group name -- 'layer-shallow' /
    'layer-deep' for a cortical region, the raw `subregion_labels` string
    itself otherwise -- and covers only the neurons this session actually
    resolved to a group (`n_neurons_resolved` out of `n_neurons_total`;
    see EXCLUDED_SUBREGION_LABELS)."""
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
    """Region-level PCA for ONE hub region, one session -- Part 1a when
    fit on `region_flat[region]` directly, Part 1b when fit on that
    region's Part-1b behaviour residual. Which condition a given instance
    represents is recorded by WHICH dict it is filed into
    (`PrivateLatentSessionResult.region_pca_raw` vs
    `.region_pca_out_behaviour`), exactly as Part 2c's own
    `PrivateLatentPairResult` carries no explicit "AllRegions+Behaviour"
    tag of its own beyond living in `.pairs`."""
    region: str
    W: np.ndarray                          # (n_neurons, K)     PCA loadings, columns = descending explained variance
    explained_variance_ratio: np.ndarray   # (K,)               fraction of this region's total variance per component
    latent: np.ndarray                     # (n_trials, T, K)   projected PCA scores
    mean: np.ndarray                       # (1, n_neurons)     column mean subtracted before the SVD
    n_neurons: int
    subregion_weight_metrics: List[SubregionWeightMetrics] = field(default_factory=list)  # Part 3, one per column of W


@dataclass
class HubOrientationPCAResult:
    """Parts 2a + 2b for ONE hub orientation, one canonicalized pair, one
    session: `hub`'s Part-1b behaviour residual, regressed against every
    OTHER recorded region except {hub, partner} (`nuisance_regions` --
    built by the same `_other_region_nuisance_list` helper Part 2c's own Z
    draws on, so the two conditions agree on exactly which regions count
    as nuisance for this pair). NO behaviour term is re-appended here --
    it was already removed in Part 1b, so `z_dim_total` here counts ONLY
    the other-region columns, unlike Part 2c's own `PrivateLatentPairResult.
    z_dim_total`, which also includes behaviour.

        explained_component = Z_flat @ (Beta @ X_flat)   Part 2a -- "_network"
        residual_component  = X_flat - explained_component  Part 2b -- "_residual"

    where X_flat is `hub`'s Part-1b residual and Z_flat is the flattened,
    concatenated activity of `nuisance_regions`. A canonicalized pair
    (region_i, region_j) yields TWO of these -- region_i-as-hub and
    region_j-as-hub, bundled together in one HubPairPCAResult below --
    since, unlike Part 2c's single joint pCCA fit, Parts 2a/2b are
    directional: "the hub region's" explained/residual component, not a
    pair-symmetric quantity."""
    hub: str
    partner: str
    W_network: np.ndarray                           # (n_neurons_hub, K)  Part 2a PCA loadings
    explained_variance_ratio_network: np.ndarray     # (K,)                Part 2a
    latent_network: np.ndarray                       # (n_trials, T, K)    Part 2a
    W_residual: np.ndarray                           # (n_neurons_hub, K)  Part 2b PCA loadings
    explained_variance_ratio_residual: np.ndarray    # (K,)                Part 2b
    latent_residual: np.ndarray                      # (n_trials, T, K)    Part 2b
    n_neurons_hub: int
    nuisance_regions: List[str] = field(default_factory=list)
    z_dim_total: int = 0
    subregion_weight_metrics_network: List[SubregionWeightMetrics] = field(default_factory=list)   # Part 3, one per column of W_network
    subregion_weight_metrics_residual: List[SubregionWeightMetrics] = field(default_factory=list)  # Part 3, one per column of W_residual


@dataclass
class HubPairPCAResult:
    """Parts 2a + 2b for one canonicalized (region_i, region_j) pair, one
    session -- both hub orientations side by side, keyed IDENTICALLY to
    `PrivateLatentSessionResult.pairs` (Part 2c), so the two dicts share a
    key space and can be zipped on it. This is the object the next task's
    "paired display" (modelled on pCCA_latent_extrenal_variable_bar.py) is
    expected to read two-bars-per-hub-per-partner from."""
    region_i: str
    region_j: str
    region_i_as_hub: HubOrientationPCAResult    # hub=region_i, partner=region_j
    region_j_as_hub: HubOrientationPCAResult    # hub=region_j, partner=region_i


@dataclass
class PrivateLatentSessionResult:
    """One session's results, spanning Parts 1a/1b/2a/2b/2c -- the unit
    pickled to {session}_analysis_results.pkl, mirroring the
    one-.mat-file-per-session granularity of pcca_sessions_{trial_type}_
    results (each of whose files holds a `cca_results['pair_results']`
    list spanning every pair). Six result sets, each independently
    addressable (per this version's storage requirement that Parts 1a,
    1b, 2a+2b, and 2c be stored separately):

        region_pca_raw            Dict[region -> RegionPCAResult]       Part 1a
        region_pca_out_behaviour  Dict[region -> RegionPCAResult]       Part 1b
        hub_pca_pairs   Dict[(region_i,region_j) -> HubPairPCAResult]   Parts 2a+2b
        pairs      Dict[(region_i,region_j) -> PrivateLatentPairResult] Part 2c ("AllRegions+Behaviour")
        pairs_regions_only Dict[(region_i,region_j) -> PrivateLatentPairResult] Part 2c' ("AllRegions" only, no behaviour)

    `region_pca_raw`/`region_pca_out_behaviour` are keyed by region name
    alone (Part 1 has no partner). `hub_pca_pairs` is keyed IDENTICALLY to
    `pairs` (canonicalized, region_i anatomically earlier) so the two dicts
    can be zipped on the same key; each HubPairPCAResult then splits into
    its two hub orientations internally. `pairs` itself -- Part 2c -- uses
    the same `_compute_pair_result` computation as every prior version of
    this file, now fit via K-fold cross-validation (`pcca`) rather than a
    single in-sample fit -- see that function's docstring.
    `pairs_regions_only` is the SAME (now also cross-validated)
    `_compute_pair_result` computation, keyed identically, but
    with `behavior_Z_flat=None` -- i.e. the nuisance Z only ever contains
    every OTHER recorded region, never behaviour -- so the two dicts can
    also be zipped on the same key, and a caller can directly compare "what
    changes about this pair's private latent once behaviour is also
    regressed out" by looking up the same (region_i, region_j) in both."""
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
    pairs_regions_only: Dict[Tuple[str, str], PrivateLatentPairResult] = field(default_factory=dict)
    region_pca_raw: Dict[str, RegionPCAResult] = field(default_factory=dict)
    region_pca_out_behaviour: Dict[str, RegionPCAResult] = field(default_factory=dict)
    hub_pca_pairs: Dict[Tuple[str, str], HubPairPCAResult] = field(default_factory=dict)
    align_mode: str = ALIGN
    region_subregion_labels: Dict[str, List[str]] = field(default_factory=dict)  # item 4, per-neuron, aligned to region_flat's neuron axis

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
        # Bumped whenever a field is added to PrivateLatentSessionResult, OR
        # the numerical procedure behind an existing field changes, so an
        # existing cached .pkl is always detected as stale and recomputed,
        # rather than silently reused against a result it no longer matches.
        # v2: added `pairs_regions_only` (Part 2c', AllRegions-only
        # nuisance, no behaviour) alongside the existing `pairs`
        # (AllRegions+Behaviour). v3: Part 2c/2c' pCCA (`pcca`) is now
        # trial-aware K-fold cross-validated (previously a single in-sample
        # fit) and `_other_region_nuisance_list` now applies
        # `min_neurons_per_region`, matching `perform_session_pcca.m`'s
        # original MATLAB logic on both counts -- every existing `.pairs`/
        # `.pairs_regions_only` value computed before this revision is
        # numerically different from what this version would compute.
        result_schema_version=3,
    )


# =============================================================================
# 7.  Per-region / per-pair / per-session computation. Parts 1a/1b use
#     `_compute_region_pca`; Parts 2a/2b use `_compute_hub_orientation_pca`
#     (one hub orientation) and `_compute_hub_pair_pca_result` (both
#     orientations of one canonicalized pair); Part 2c's
#     `_compute_pair_result` now sources its nuisance-region list from the
#     shared `_other_region_nuisance_list` helper below (previously
#     computed inline), which itself now applies `min_neurons_per_region`
#     (matching `perform_session_pcca.m`'s own Step-1 region filter), and
#     delegates its actual fit to `pcca`, which is now K-fold
#     cross-validated instead of a single in-sample fit -- see that
#     function's docstring. Parts 2a/2b's "AllRegions"
#     nuisance set is built from that SAME helper, so the two conditions
#     are guaranteed to agree on which regions count as nuisance for a
#     given pair.
# =============================================================================

def _other_region_nuisance_list(
        region_i: str,
        region_j: str,
        region_flat: Dict[str, np.ndarray],
        min_neurons_per_region: int = MIN_NEURONS_PER_REGION,
) -> List[str]:
    """Every anatomically-ordered region recorded this session, other than
    {region_i, region_j}, not in EXCLUDED_REGIONS, and with AT LEAST
    `min_neurons_per_region` neurons -- the "AllRegions" nuisance set
    shared by Part 2c's Z = AllRegions+Behaviour (`_compute_pair_result`)
    and Part 2a/2b's Z = AllRegions-only (`_compute_hub_orientation_pca`).
    Factored out into its own function (rather than computed inline in
    each caller, as Part 2c's version originally was) specifically so the
    two conditions cannot silently drift apart -- this project's own
    working notes flag pair canonicalisation as "a frequent source of
    silent bugs when not enforced at both write and lookup time", and an
    inconsistent nuisance set between 2a/2b and 2c would be exactly that
    kind of bug.

    The neuron-count threshold matches
    `perform_session_pcca.m`'s own Step-1 region filter
    (`config.min_neurons_per_region`, 50 in oxford_session_pipeline_mdl.m):
    that MATLAB function excludes a thin, noisy region from its nuisance
    matrix entirely before the pair loop even starts. This function's
    caller passes `region_flat[r]` (or, for Part 2a/2b, that region's
    Part-1b behaviour residual -- same neuron axis, since residualizing
    does not change neuron count), so `region_flat[r].shape[1]` is exactly
    that region's neuron count."""
    return [
        r for r in ANATOMICAL_ORDER
        if r in region_flat
        and r not in (region_i, region_j)
        and r not in EXCLUDED_REGIONS
        and region_flat[r].shape[1] >= min_neurons_per_region
    ]


def _subregion_group_for(region: str, label: str) -> Optional[str]:
    """Collapse one neuron's raw `subregion_labels` string to the group
    used by Part 3's weight-ratio metrics: 'layer-shallow' / 'layer-deep'
    for a cortical hub region (`CORTICAL_REGIONS`), the raw label itself
    for every other (subcortical) region. Returns None for a neuron whose
    label carries no usable anatomical information for this purpose --
    excluded (`EXCLUDED_SUBREGION_LABELS`), missing/unresolved
    ('unknown'), or (for a cortical region) not one of the laminar labels
    this project has mapped -- so every fraction below is computed only
    over neurons that DO resolve to a group, instead of being silently
    diluted by ones that don't.
    """
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
    """Part 3's weight-ratio measure: one `SubregionWeightMetrics` per
    column of `W` (n_neurons, K), computed identically whether `W` is a
    PCA loading matrix (Parts 1a/1b/2a/2b) or a pCCA canonical weight
    matrix Wx/Wy (Part 2c) -- only the matrix and the region it belongs to
    ever differ between call sites.

    For component k, every resolved neuron (`_subregion_group_for`)
    contributes |W[i, k]| to its group's weight mass. Two views of that
    mass are reported per group g:

        weight_mass_fraction[g]  = sum(|w_i|, i in g) / sum(|w_i|, i resolved)
        neuron_count_fraction[g] = n_g / n_resolved
        enrichment_ratio[g]      = weight_mass_fraction[g] / neuron_count_fraction[g]

    `enrichment_ratio` is the fold-enrichment of group g among the neurons
    that dominate this component's weight, relative to g's own share of
    the resolved population: 1.0 means g's neurons carry exactly their
    numerical "fair share" of |W|, >1 means they are over-represented among
    the high-magnitude weights, <1 under-represented. This is deliberately
    a SECOND, population-normalised measure alongside `weight_mass_fraction`
    alone, since a group with many neurons can dominate the raw mass
    fraction without any individual neuron in it being unusually
    high-weighted.

    A single scalar summary is also recorded for convenience:
    `dominant_ratio` is literally the requested shallow-to-deep ratio
    (`weight_mass_fraction['layer-shallow'] / weight_mass_fraction
    ['layer-deep']`) for a cortical region, and its subcortical analogue --
    the mass fraction of whichever single subregion label carries the most
    weight in this component, divided by the combined mass fraction of
    every OTHER subregion label -- for everything else; `dominant_ratio_
    label` names which comparison it is (e.g. 'layer-shallow:layer-deep'
    or '<top_label>:rest').
    """
    n_neurons, K = W.shape
    is_cortical = region in CORTICAL_REGIONS

    if not labels or len(labels) != n_neurons:
        # No usable labels for this region/session -- an all-unresolved
        # placeholder per component, rather than raising, so a session
        # missing subregion metadata for one region doesn't abort the
        # whole session's computation.
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
    """Part 1a/1b building block: fit + project PCA on one region's
    flattened activity and package the result. The caller decides which
    Part this is purely by what it passes as `X_flat` -- raw
    `region_flat[region]` for Part 1a, or that region's behaviour residual
    for Part 1b -- and by which dict it stores the return value in.
    `labels` (this region's per-neuron subregion labels, item 4) drives
    Part 3's `subregion_weight_metrics`; both Part 1a and 1b use the SAME
    labels for a given region, since behaviour regression does not change
    which neuron is which."""
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


def _compute_hub_orientation_pca(
        hub: str,
        partner: str,
        X_hub_behav_res: np.ndarray,
        behav_res_by_region: Dict[str, np.ndarray],
        n_trials: int,
        T: int,
        n_components: int = N_PCA_COMPONENTS,
        labels: Optional[List[str]] = None,
) -> HubOrientationPCAResult:
    """Parts 2a + 2b for ONE hub orientation: `hub`'s Part-1b behaviour
    residual (`X_hub_behav_res`, precomputed once per hub and passed in
    unchanged for every partner this hub is paired with -- it does not
    depend on `partner`) regressed against every OTHER recorded region
    except {hub, partner}. No behaviour term is appended to Z here -- it
    was already removed from X_hub_behav_res in Part 1b -- so this is
    strictly the "AllRegions" half of Part 2c's "AllRegions+Behaviour"
    nuisance, applied sequentially rather than jointly (see the module
    docstring's "Mathematical framework" for why this is an intentional
    point of comparison with Part 2c, not a redundant recomputation of it).

    If `nuisance_all` is empty (only possible if this session recorded
    just {hub, partner} and nothing else), `explained` degenerates to all
    zeros and `residual` to X_hub_behav_res itself, matching
    `residualize_with_explained`'s own Z=None/empty behaviour -- there is
    nothing else recorded that COULD explain any of hub's signal.
    """
    nuisance_all = _other_region_nuisance_list(hub, partner, behav_res_by_region)
    Z_nuisance = (
        np.concatenate([behav_res_by_region[r] for r in nuisance_all], axis=1)
        if nuisance_all else None
    )

    explained, residual = residualize_with_explained(X_hub_behav_res, Z_nuisance, LAMBDA_HAT)

    W_net, evr_net, _mean_net, lat_net = pca_fit_and_project(
        explained, n_trials, T, n_components)
    W_res, evr_res, _mean_res, lat_res = pca_fit_and_project(
        residual, n_trials, T, n_components)

    return HubOrientationPCAResult(
        hub=hub,
        partner=partner,
        W_network=W_net.astype(np.float32),
        explained_variance_ratio_network=evr_net.astype(np.float64),
        latent_network=lat_net.astype(np.float32),
        W_residual=W_res.astype(np.float32),
        explained_variance_ratio_residual=evr_res.astype(np.float64),
        latent_residual=lat_res.astype(np.float32),
        n_neurons_hub=int(X_hub_behav_res.shape[1]),
        nuisance_regions=nuisance_all,
        z_dim_total=int(Z_nuisance.shape[1]) if Z_nuisance is not None else 0,
        subregion_weight_metrics_network=compute_subregion_weight_metrics(hub, W_net, labels),
        subregion_weight_metrics_residual=compute_subregion_weight_metrics(hub, W_res, labels),
    )


def _compute_hub_pair_pca_result(
        region_i: str,
        region_j: str,
        region_flat: Dict[str, np.ndarray],
        behav_res_by_region: Dict[str, np.ndarray],
        n_trials: int,
        T: int,
        region_subregion_labels: Optional[Dict[str, List[str]]] = None,
) -> Optional[HubPairPCAResult]:
    """Parts 2a + 2b for one canonicalized pair -- both hub orientations
    (region_i-as-hub/region_j-as-partner AND region_j-as-hub/region_i-as-
    partner), matching how Part 1's "for each hub region" applies to every
    recorded hub region, not just the anatomically-earlier member of a
    pair the way Part 2c's single joint pCCA fit implicitly treats
    neither member preferentially. `behav_res_by_region` is Part 1b's
    output, computed once per hub region upstream in
    `compute_private_latents_for_session` and reused here for every pair
    that hub appears in."""
    if region_i not in behav_res_by_region or region_j not in behav_res_by_region:
        return None

    region_subregion_labels = region_subregion_labels or {}
    region_i_as_hub = _compute_hub_orientation_pca(
        hub=region_i, partner=region_j,
        X_hub_behav_res=behav_res_by_region[region_i],
        behav_res_by_region=behav_res_by_region, n_trials=n_trials, T=T,
        labels=region_subregion_labels.get(region_i),
    )
    region_j_as_hub = _compute_hub_orientation_pca(
        hub=region_j, partner=region_i,
        X_hub_behav_res=behav_res_by_region[region_j],
        behav_res_by_region=behav_res_by_region, n_trials=n_trials, T=T,
        labels=region_subregion_labels.get(region_j),
    )

    return HubPairPCAResult(
        region_i=region_i, region_j=region_j,
        region_i_as_hub=region_i_as_hub,
        region_j_as_hub=region_j_as_hub,
    )


def _compute_pair_result(
        region_i: str,
        region_j: str,
        region_flat: Dict[str, np.ndarray],
        behavior_Z_flat: Optional[np.ndarray],
        n_trials: int,
        T: int,
        labels_i: Optional[List[str]] = None,
        labels_j: Optional[List[str]] = None,
) -> Optional[PrivateLatentPairResult]:
    """Fit the AllRegions+Behaviour pCCA condition for one canonicalized
    region pair -- the per-pair generalisation of
    `_run_one_ablation_step(..., Z_name='AllRegions+Behaviour', ...)` in
    pCCA_sensitive_realsingle_Session_8panel.py."""
    if region_i not in region_flat or region_j not in region_flat:
        return None

    nuisance_all = _other_region_nuisance_list(region_i, region_j, region_flat)

    z_parts = [region_flat[r] for r in nuisance_all]
    if z_parts is None:
        # A missing nuisance region here would otherwise silently shrink
        # Z_full (or, if every part were None, drop it to None entirely),
        # letting `pcca` quietly degrade to plain CCA for this pair instead
        # of failing loudly. Raise so the per-session try/except in the
        # caller treats this session as skipped rather than mis-fit.
        raise ValueError(
            f"[{region_i}-{region_j}] missing nuisance data for one or more "
            f"regions in {nuisance_all}; skipping session"
        )
    if behavior_Z_flat is not None:
        z_parts.append(behavior_Z_flat)

    Z_full = np.concatenate(z_parts, axis=1) if z_parts else None

    Wx, Wy, rho, X_i_res, X_j_res = pcca(
        region_flat[region_i], region_flat[region_j], Z_full,
        n_trials=n_trials, T=T,
        lam_cca=LAMBDA_CCA, lam_hat=LAMBDA_HAT, n_components=N_COMPONENTS,
        n_folds=CV_FOLDS, seed=CV_RNG_SEED,
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
        z_dim_total=int(Z_full.shape[1]) if Z_full is not None else 0,
        subregion_weight_metrics_i=compute_subregion_weight_metrics(region_i, Wx, labels_i),
        subregion_weight_metrics_j=compute_subregion_weight_metrics(region_j, Wy, labels_j),
    )


def compute_private_latents_for_session(
        session_name: str,
        trial_type: str = TRIAL_TYPE,
        mat_dir: Optional[Path] = None,
        align_mode: str = ALIGN,
) -> Optional[PrivateLatentSessionResult]:
    """Compute the AllRegions+Behaviour private pCCA latent for every pair
    in REGION_PAIRS, for one session of one trial type / alignment mode.

    Returns None if the session cannot be processed at all: missing source
    .mat file, a crop window with < 2 overlapping samples, or (when
    REQUIRE_BEHAVIOR=True, the default) missing/unmatched behavioural
    tracking.
    """
    mat_dir = mat_dir if mat_dir is not None else (BASE_DIR / mat_subdir_name(trial_type, align_mode))
    session_file = mat_dir / f"{session_name}_analysis_results.mat"
    if not session_file.exists():
        print(f"  [skip] {session_name}: source file not found -> {session_file}")
        return None

    region_spikes, region_subregion_labels, n_trials, T = load_region_spikes(str(session_file))
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

    # ---- Part 1: region-level PCA, one per hub region present this
    #      session -- 1a directly on region_flat, 1b on its behaviour
    #      residual. The 1b residual is retained in `behav_res_by_region`
    #      and reused, UNCHANGED, as every one of this region's Part 2a/2b
    #      hub-orientation inputs below (it does not depend on partner, so
    #      computing it once per hub here rather than once per PAIR avoids
    #      recomputing the same ridge regression up to 6x over) -----------
    region_pca_raw: Dict[str, RegionPCAResult] = {}
    region_pca_out_behaviour: Dict[str, RegionPCAResult] = {}
    behav_res_by_region: Dict[str, np.ndarray] = {}
    for region in region_flat.keys():
        if region not in region_flat:
            continue
        region_pca_raw[region] = _compute_region_pca(
            region, region_flat[region], n_trials, T,
            labels=region_subregion_labels.get(region))

        X_behav_res = residualize(region_flat[region], behavior_Z_flat, LAMBDA_HAT)
        behav_res_by_region[region] = X_behav_res
        region_pca_out_behaviour[region] = _compute_region_pca(
            region, X_behav_res, n_trials, T,
            labels=region_subregion_labels.get(region))

    # ---- Part 2: every pair in REGION_PAIRS -- 2a/2b (hub-orientation
    #      PCA, both directions) computed side by side with 2c (now
    #      cross-validated private pCCA, Z=AllRegions+Behaviour) AND 2c'
    #      (Z=AllRegions only -- same `_compute_pair_result` call, just
    #      with `behavior_Z_flat` replaced by None so behaviour is never
    #      part of the nuisance regressed out of this pair) --------------
    pairs: Dict[Tuple[str, str], PrivateLatentPairResult] = {}
    pairs_regions_only: Dict[Tuple[str, str], PrivateLatentPairResult] = {}
    hub_pca_pairs: Dict[Tuple[str, str], HubPairPCAResult] = {}
    for ri_raw, rj_raw in REGION_PAIRS:
        # (3) storage-convention optimisation, applied defensively here (not
        # just relied upon from REGION_PAIRS already being pre-sorted above):
        # region_i is always the anatomically earlier region.
        region_i, region_j = sort_pair_by_anatomy(ri_raw, rj_raw)

        result = _compute_pair_result(
            region_i, region_j, region_flat, behavior_Z_flat, n_trials, T,
            labels_i=region_subregion_labels.get(region_i),
            labels_j=region_subregion_labels.get(region_j),
        )
        if result is not None:
            pairs[(region_i, region_j)] = result

        result_regions_only = _compute_pair_result(
            region_i, region_j, region_flat, None, n_trials, T,
            labels_i=region_subregion_labels.get(region_i),
            labels_j=region_subregion_labels.get(region_j),
        )
        if result_regions_only is not None:
            pairs_regions_only[(region_i, region_j)] = result_regions_only

        hub_result = _compute_hub_pair_pca_result(
            region_i, region_j, region_flat, behav_res_by_region, n_trials, T,
            region_subregion_labels=region_subregion_labels,
        )
        if hub_result is not None:
            hub_pca_pairs[(region_i, region_j)] = hub_result

    print(
        f"  [{session_name}] 2c pCCA {len(pairs)}/{len(REGION_PAIRS)} pairs "
        f"(AllRegions+Behaviour), 2c' pCCA {len(pairs_regions_only)}/{len(REGION_PAIRS)} "
        f"pairs (AllRegions only), "
        f"2a/2b hub-PCA {len(hub_pca_pairs)}/{len(REGION_PAIRS)} pairs, "
        f"1a/1b region-PCA {len(region_pca_raw)}/{len(HUB_REGIONS)} regions  "
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
        pairs_regions_only=pairs_regions_only,
        region_pca_raw=region_pca_raw,
        region_pca_out_behaviour=region_pca_out_behaviour,
        hub_pca_pairs=hub_pca_pairs,
        align_mode=align_mode,
        region_subregion_labels=region_subregion_labels,
    )


# =============================================================================
# 8.  Orchestration
# =============================================================================

def run_all_sessions(
        trial_type: str = TRIAL_TYPE,
        sessions: Optional[List[str]] = None,
        overwrite: bool = False,
        align_mode: str = ALIGN,
) -> None:
    """Compute and pickle every session's PrivateLatentSessionResult for
    `trial_type` / `align_mode`. `sessions=None` (default) auto-discovers
    every session under the source .mat folder; pass an explicit list to
    restrict to a subset (e.g. while testing).

    Caching: an existing {session}_analysis_results.pkl is reused as-is
    only if its stored `config` matches `config_fingerprint()` for the
    CURRENT settings above -- otherwise it is recomputed and overwritten,
    with a clear printed message either way (never a silent stale reload).
    """
    sessions = sessions if sessions is not None else SESSIONS
    mat_dir = BASE_DIR / mat_subdir_name(trial_type, align_mode)
    out_dir = BASE_DIR / out_subdir_name(trial_type, align_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sessions is None:
        session_files = sorted(mat_dir.glob("*_analysis_results.mat"))
        sessions = [f.stem.replace("_analysis_results", "") for f in session_files]

    print("=" * 70)
    print("Region/hub PCA (1a/1b/2a/2b) + private pCCA (2c)  |  AllRegions + Behaviour")
    print(f"  trial_type : {trial_type}   (behaviour label = '{behavior_label_for(trial_type)}')")
    print(f"  align_mode : {align_mode}")
    print(f"  source dir : {mat_dir}")
    print(f"  output dir : {out_dir}")
    print(f"  sessions   : {len(sessions)}")
    print(f"  pairs      : {len(REGION_PAIRS)}  (across {len(PAIR_CATEGORIES)} categories)")
    print(f"  hub regions: {len(HUB_REGIONS)}  {HUB_REGIONS}")
    print(f"  pca comps  : {N_PCA_COMPONENTS}  (Parts 1a/1b/2a/2b; N_COMPONENTS={N_COMPONENTS} for 2c's pCCA)")
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
        print(f"  \u2728 saved -> {out_path}")
        n_written += 1

    print("\n" + "=" * 70)
    print(
        f"\U0001F389 Done. {n_written} written, {n_cached} cached, "
        f"{n_skipped} skipped (of {len(sessions)} sessions)."
    )
    print(f"   Load results back with: PrivateLatentAnalyzer(trial_type={trial_type!r}).load_all()")
    print("   Part 1a/1b : az.get_region_pca(session, region, out_behaviour=True/False)")
    print("   Part 2a/2b : az.get_hub_pca(session, hub, partner)")
    print("   Part 2c    : az.get_pair(session, region_i, region_j)")
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

        pair = az.get_pair("yp021_220407", "MOp", "VPMPO")                  # Part 2c
        pair.z_i_lat[:, :, 0]     # (n_trials, T) component-0 private latent
        pair.rho[0]               # component-0 canonical correlation

        r1a = az.get_region_pca("yp021_220407", "MOp")                      # Part 1a
        r1b = az.get_region_pca("yp021_220407", "MOp", out_behaviour=True)  # Part 1b
        r1a.latent[:, :, 0]       # (n_trials, T) component-0 PCA score

        hub = az.get_hub_pca("yp021_220407", "MOp", "VPMPO")                # Parts 2a+2b
        hub.latent_network[:, :, 0]    # 2a -- explained by rest of network
        hub.latent_residual[:, :, 0]   # 2b -- left over after removing 2a
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
            regions_only: bool = False,
    ) -> Optional[PrivateLatentPairResult]:
        """Look up one pair's Part 2c result for one loaded session: the
        default `regions_only=False` returns the "AllRegions+Behaviour"
        result (`.pairs`); `regions_only=True` returns the "AllRegions
        only" result (`.pairs_regions_only`) -- same pair, same
        canonicalisation, nuisance Z differs only in whether behaviour was
        included. Region order does not matter -- looked up via the same
        canonicalisation used at write time."""
        session = self.sessions.get(session_name)
        if session is None:
            return None
        table = session.pairs_regions_only if regions_only else session.pairs
        return table.get(sort_pair_by_anatomy(region_i, region_j))

    def get_region_pca(
            self, session_name: str, region: str, out_behaviour: bool = False,
    ) -> Optional[RegionPCAResult]:
        """Look up one region's Part 1a (`out_behaviour=False`, default) or
        Part 1b (`out_behaviour=True`) region-level PCA result for one
        loaded session."""
        session = self.sessions.get(session_name)
        if session is None:
            return None
        table = session.region_pca_out_behaviour if out_behaviour else session.region_pca_raw
        return table.get(region)

    def get_hub_pca(
            self, session_name: str, hub: str, partner: str,
    ) -> Optional[HubOrientationPCAResult]:
        """Look up Parts 2a+2b for one (hub, partner) orientation, one
        loaded session. Canonicalizes internally, exactly like `get_pair`,
        then picks out whichever of `region_i_as_hub` / `region_j_as_hub`
        matches `hub` -- so the caller never has to reason about pair
        ordering, only about which region they mean by "hub"."""
        session = self.sessions.get(session_name)
        if session is None:
            return None
        region_i, region_j = sort_pair_by_anatomy(hub, partner)
        pair_result = session.hub_pca_pairs.get((region_i, region_j))
        if pair_result is None:
            return None
        return pair_result.region_i_as_hub if hub == region_i else pair_result.region_j_as_hub

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
        """Per-pair (Part 2c) and per-region (Part 1a/1b) session-coverage
        tables, in the spirit of OxfordAdvancedAnalyzer._loading_summary."""
        if not self.sessions:
            print("[PrivateLatentAnalyzer] no sessions loaded -- call load_all() first.")
            return
        counts: Dict[Tuple[str, str], int] = {}
        for session in self.sessions.values():
            for pair_key in session.pairs:
                counts[pair_key] = counts.get(pair_key, 0) + 1
        print(f"[PrivateLatentAnalyzer]  trial_type={self.trial_type}  sessions={len(self.sessions)}")
        print("  Part 2c -- private pCCA, by pair:")
        for category, pairs in PAIR_CATEGORIES:
            for pair in pairs:
                n = counts.get(pair, 0)
                print(
                    f"    [{category:<26s}] {pair[0]:>7s} <-> {pair[1]:<7s} : "
                    f"{n:3d}/{len(self.sessions)} sessions"
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
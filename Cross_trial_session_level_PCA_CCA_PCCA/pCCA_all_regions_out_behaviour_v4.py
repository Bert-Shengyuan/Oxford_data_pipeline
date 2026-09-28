#!/usr/bin/env python3
r"""
pCCA_all_regions_out_behaviour_v4.py
================================================================================

Version 4.0 of `pCCA_all_regions_out_behaviour_v3.py` ("v3"). v4 keeps every
core numerical building block from v3 verbatim -- the ridge/pCCA/CCA math
(`ridge_cca`, `pcca`, `residualize_with_explained`, ...), the k-fold
cross-validation inside `pcca`, and the two-pool neuron-sampling scheme
(`sample_paired_draws` / `sample_nuisance_draws`, `N_SAMPLE_DRAWS` resampled
neuron subsets per region per session) -- but changes WHAT the pipeline is
run on and WHICH nuisance regions enter each computation. It also FIXES and
EXTENDS v3's cumulative-weight top-neuron selection
(`_select_cumulative_weight_neurons`): v3 pooled every PCA/CCA component's
|weight| together into one score per neuron, which was silent-but-wrong the
moment more than one component was requested (only ever exercised with
N_COMPONENTS=1); v4 selects PER COMPONENT instead (cross-draw pooling only,
never cross-component), and now runs that selection on EVERY PCA/CCA weight
in the pipeline -- Groups 1b/2c/3c/4c (as before), PLUS Group 1a (direct
PCA) and Groups 2a/2b/3a/3b/4a/4b (hub-orientation network/residual PCA),
which previously had none at all.

--------------------------------------------------------------------------------
1. What's new: a direct-CCA branch (no residualization at all)
--------------------------------------------------------------------------------
v3 only ever fed `pcca()` a `Z_flat` that was either "all other recorded
regions" (2c') or "all other recorded regions + behaviour" (2c) -- there was
no condition that skipped residualization entirely. v4 adds exactly that:
calling the SAME `pcca()` function with `Z_flat=None`. Since
`_fit_ridge_beta` already returns `None` (i.e. "no nuisance") whenever
`Z_flat` is `None`, `pcca(X, Y, None, ...)` degrades gracefully to a
plain, k-fold cross-validated `ridge_cca` fit -- no code fork needed, no
new math, just a new caller. This is condition `"direct"` below (item 1b
of the analysis structure).

--------------------------------------------------------------------------------
2. What's new: reorganising around region TRIPLETS
--------------------------------------------------------------------------------
v3 ran every one of `REGION_PAIRS` (21 pairs spanning 7 whole-brain
categories) on every session found in `mat_subdir_name`'s directory, with
each pair's "nuisance" design `Z` built from EVERY OTHER recorded region in
that session (subject to `TARGET_SAMPLE_SIZE`).

v4 instead is organised around 3-region TRIPLETS (`TRIPLETS` below), each
with its own list of qualifying sessions (sessions where all three of that
triplet's regions were co-recorded), derived from the hub section of the
threshold-analysis report -- see Section 2b. For a
given triplet {A, B, C}, only the 3 pairs INTERNAL to the triplet are
computed (AB, AC, BC, canonicalised via `sort_pair_by_anatomy`), and each
pair's "third region" nuisance is ONLY the one remaining triplet member --
never the whole rest of the recorded brain. A session can belong to
several triplets, and each
triplet's results for that session are numerically DIFFERENT (a different
third region enters the nuisance design each time), so results are written
to one output directory PER TRIPLET (`out_subdir_name(triplet.slug, ...)`)
-- never merged into one per-session file the way v3's were.

--------------------------------------------------------------------------------
2b. Hub-triplet mode (current default)
--------------------------------------------------------------------------------
`TRIPLETS` is derived at import time from the hub section of
`HUB_REPORT_PATH` (session_triplets_hub_report.txt) by `build_hub_triplets`:
  * report names -> data names via `REGION_MAPPING`; "Striatum" pools
    STR + STRv + PAL into one region "STR" (`pool_region_groups`);
  * a triplet listed under several hubs is analysed once;
  * a session is kept only if all 3 regions have >= TARGET_SAMPLE_SIZE neurons,
    and a triplet only if >= MIN_SESSIONS_PER_TRIPLET sessions remain --
    so changing either threshold changes the triplet set;
  * alternatively, set `SELECTED_TRIPLETS` (e.g. ["MOp-MOs-VALVM"], or using
    report display names as printed in the hub report's headers, e.g.
    ["OFC∩Olf area∩Striatum"] -- data names and display names, and hyphen-
    or ∩-separated, can be mixed freely) to analyse only those triplets: the
    report is searched for each one, sessions are filtered by
    TARGET_SAMPLE_SIZE alone and MIN_SESSIONS_PER_TRIPLET is ignored. With
    the default None, the iterate-everything behaviour applies;
  * outputs go to `pcca_all_regions_out_behaviour_v4_hub_<slug>_...` so
    earlier (non-hub, dorsal-STR-only) v4 result folders are never overwritten.

--------------------------------------------------------------------------------
3. The final, eleven-item analysis structure -- computed for every triplet,
   every one of its sessions, every one of its 3 internal pairs
--------------------------------------------------------------------------------
Group 1 -- no regression at all:
  1a  Direct PCA of one region's own activity (`region_pca_raw`). Changed
      from v3's Part 1a: now sampled like every other group --
      `N_SAMPLE_DRAWS` independent TARGET_SAMPLE_SIZE draws from the FULL
      population (bypassing `selected_neurons`), each with its own PCA
      fit; still no regression, still no CV (PCA has none). Also newly
      carries a per-component cumulative-weight top-neuron selection
      (`RegionPCAResult.selected_neurons`, previously only on 1b-4c).
      `PrivateLatentSessionResult.region_pca_raw`.
  1b  Direct CCA between a triplet-internal pair's two regions, no
      residualization of any kind (new, see (1) above). Sampled/CV'd like
      every other paired condition below. `PrivateLatentPairResult.
      draws["direct"]`.

Group 2 -- nuisance = behaviour ONLY (never the third region):
  2a  PCA of the behaviour-EXPLAINED ("network") component of one region's
      own (raw, full-population, sampled) activity.
  2b  PCA of the behaviour-RESIDUAL component of the same activity/draw.
      `HubPairPCAResult.region_{i,j}_as_hub_draws["behav_only"]` carries
      both 2a (`.W_network`/`.latent_network`) and 2b
      (`.W_residual`/`.latent_residual`) together, one draw at a time --
      see `_compute_hub_orientation_draw`. Both now also get a
      per-component cumulative-weight top-neuron selection
      (`HubPairPCAResult.region_{i,j}_as_hub_selected["behav_only"]
      .network`/`.residual`, previously none at all).
  2c  pCCA between the pair, nuisance Z = behaviour only.
      `PrivateLatentPairResult.draws["behav_only"]`.

Group 3 -- nuisance = the third region ONLY (never behaviour):
  3a  PCA of the third-region-EXPLAINED component of one region's own raw
      activity (hub orientation, exactly v3's Part-2a shape, just with
      "all other regions" narrowed to the one third region).
  3b  PCA of the residual component of the same.
      `HubPairPCAResult.region_{i,j}_as_hub_draws["region_only"]`, with
      the same per-component selection as 2a/2b (see above), keyed
      `["region_only"]`.
  3c  pCCA between the pair, nuisance Z = the third region's sampled
      neurons only (v3's Part 2c' shape, nuisance narrowed to one region).
      `PrivateLatentPairResult.draws["region_only"]`.

Group 4 -- nuisance = the third region AND behaviour together:
  4a  PCA of the third-region-explained component of one region's own
      activity, AFTER that activity has already been behaviour-
      residualized once per region per session (`behav_res_by_region_full`
      -- the SAME sequential composition v3's Part 2a/2b used, just with
      "all other regions" narrowed to the third region; kept sequential
      rather than switched to a joint fit so this condition stays a strict
      narrowing of v3's existing behaviour, not a new modelling choice).
  4b  PCA of the residual component of the same.
      `HubPairPCAResult.region_{i,j}_as_hub_draws["region_behav"]`, with
      the same per-component selection as 2a/2b, keyed `["region_behav"]`.
  4c  pCCA between the pair, nuisance Z = [third region's sampled neurons,
      behaviour] concatenated and residualized JOINTLY in one ridge fit --
      exactly v3's Part 2c shape, nuisance narrowed to one region.
      `PrivateLatentPairResult.draws["region_behav"]`.

Internally, conditions are keyed by short strings rather than by the
1b/2c/3c/4c labels above -- `PAIR_CONDITIONS = ("direct", "behav_only",
"region_only", "region_behav")` for the paired (CCA/pCCA) computations,
`HUB_CONDITIONS = ("behav_only", "region_only", "region_behav")` for the
per-region hub-orientation (PCA network/residual) computations -- see
`PAIR_CONDITION_LABELS` / `HUB_CONDITION_LABELS` for the mapping back to
the 1b/2c/3c/4c/2a/2b/... numbering used above and in conversation.

--------------------------------------------------------------------------------
Everything NOT covered above is unchanged from v3
--------------------------------------------------------------------------------
`_zscore_flat`, `_ridge_inv_sqrt`, `ridge_cca`, `_fit_ridge_beta`,
`_apply_ridge_beta`, `residualize`, `residualize_with_explained`,
`_trial_kfold_indices`, `pcca`, `latent_projections`, `pca_fit_and_project`,
`_make_region_rng`, `sample_paired_draws`, `sample_nuisance_draws`,
`load_region_spikes`, `load_region_spikes_full`, `crop_time_window`,
`load_behavior_regressors`, anatomical canonicalisation, and the
laminar/subregion Part-3 weight-ratio metrics are all copied verbatim from
v3. `_select_cumulative_weight_neurons` is NOT verbatim -- see the top of
this docstring for how its per-component selection and its now-universal
(1a/1b-4c/2a-4b) application differ from v3.

Output goes to its own, triplet-scoped directory tree so this script never
collides with v1/v2/v3's caches (see `out_subdir_name`).

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import itertools
import pickle
import re
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
# 1.  USER-CONFIGURABLE PARAMETERS -- identical to v3, minus the whole-brain
#     REGION_PAIRS/EXCLUDED_REGIONS/SESSIONS knobs (superseded by TRIPLETS,
#     see Section 3), plus the new PAIR_CONDITIONS/HUB_CONDITIONS labels.
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

# ---- CCA / pCCA dimensionality & regularisation (matches v3) -------------
N_COMPONENTS: int = 1
LAMBDA_CCA: float = 1e-4
LAMBDA_HAT: float = 1e-4

# ---- Part *c / cross-validation (matches v3 / oxford_session_pipeline_mdl.m)
CV_FOLDS: int = 10
CV_RNG_SEED: int = 12345

# ---- PCA dimensionality for Parts 1a/2a/2b/3a/3b/4a/4b (matches v3) -------
N_PCA_COMPONENTS: int = 2

# ---- Time windows (matches v3) -------------------------------------------
TIME_RANGE_S: Tuple[float, float] = ALIGNMENT_WINDOWS_S[ALIGN]
BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = ALIGNMENT_WINDOWS_S[ALIGN]

# ---- Regime toggles (matches v3) -----------------------------------------
SUBTRACT_PSTH: bool = False
SHUFFLE_TRIALS: bool = False
REQUIRE_BEHAVIOR: bool = True

# ---- Part 2's neuron-sampling scheme (matches v3 verbatim) ----------------
#      TARGET_SAMPLE_SIZE doubles as the minimum RECORDED (full-population)
#      neuron count per region: a session needs >= this many neurons in every
#      triplet region to be kept, and the triplet's third region needs at
#      least this many to enter a pair's Group-3/4 nuisance design. (Replaces
#      v3's separate `MIN_NEURONS_PER_REGION`, which was always set equal.)



TARGET_SAMPLE_SIZE: int = 40
N_SAMPLE_DRAWS: int = 10
SAMPLE_RNG_SEED: int = 20260916

# ---- Cumulative-weight neuron selection (matches v3 verbatim) -------------
CUMULATIVE_WEIGHT_FRACTION: float = 0.60

# ---- Hub-triplet mode: the triplet list is DERIVED from the hub section of
#      the threshold-analysis report (not hand-curated), then filtered by the
#      two sampling parameters above. See `build_hub_triplets`. --------------
HUB_REPORT_PATH: Path = BASE_DIR / "threshold_analysis_results" / "session_triplets_hub_report.txt"
MIN_SESSIONS_PER_TRIPLET: int = 7
# Explicit triplet list. None (default) -> iterate over every triplet in the
# hub report that passes `MIN_SESSIONS_PER_TRIPLET`. A list -> analyse ONLY
# these triplets, in this order: each is looked up in the report, its sessions
# filtered by `TARGET_SAMPLE_SIZE` neurons/region alone, and
# `MIN_SESSIONS_PER_TRIPLET` is ignored. Each entry is a string such as
# "MOp-MOs-VALVM" (separators '-', '_', ',' or '∩'; region names are
# case-insensitive data names) or a 3-tuple of region names; order within a
# triplet is irrelevant. e.g. ["MOp-MOs-VALVM", "MOp-MOs-VPMPO"]
SELECTED_TRIPLETS: Optional[List[Any]] = ["OFC∩Olf area ∩Striatum"]

# Report/display region name -> region name(s) in the session .mat files.
# A list value means "several recorded regions pooled into one" (neurons
# concatenated); the pooled region is named after the list's FIRST entry.
REGION_MAPPING: Dict[str, Any] = {
    "M1 Ctx":       "MOp",
    "preM Ctx":     "MOs",
    "OFC":          "ORB",
    "mPFC":         "mPFC",
    "motorThal":    "VALVM",
    "sensorThal":   "VPMPO",
    "interThal":    "ILM",
    "MDThal":       "MD",
    "Pulvinar":     "LP",
    "Striatum":     ["STR", "STRv", "PAL"],  # dorsal + ventral combined
    "Hippocampus":  "HIPP",
    "Olf area":     "OLF",
    "Hypothalamus": "HY",
}
REGION_GROUPS: Dict[str, List[str]] = {
    v[0]: list(v) for v in REGION_MAPPING.values() if isinstance(v, (list, tuple))
}

# ---- NEW (v4): the four paired (CCA/pCCA) conditions and three
#      hub-orientation (PCA network/residual) conditions that replace v3's
#      fixed 2c/2c' pair -- see the module docstring, Section 3, for the
#      full mapping to the 1a/1b/2a/2b/2c/3a/3b/3c/4a/4b/4c numbering. ----
PAIR_CONDITIONS: Tuple[str, ...] = ("direct", "behav_only", "region_only", "region_behav")
PAIR_CONDITION_LABELS: Dict[str, str] = {
    "direct":       "1b  direct CCA -- no regression",
    "behav_only":   "2c  pCCA -- behaviour regressed out only",
    "region_only":  "3c  pCCA -- third region regressed out only",
    "region_behav": "4c  pCCA -- third region + behaviour regressed out",
}

HUB_CONDITIONS: Tuple[str, ...] = ("behav_only", "region_only", "region_behav")
HUB_CONDITION_LABELS: Dict[str, str] = {
    "behav_only":   "2a/2b  hub PCA -- behaviour regressed out only",
    "region_only":  "3a/3b  hub PCA -- third region regressed out only",
    "region_behav": "4a/4b  hub PCA -- third region + behaviour regressed out",
}


def mat_subdir_name(trial_type: str, align_mode: str = ALIGN) -> str:
    """MATLAB-pipeline session-region-data source folder (unchanged from
    v3 -- v4 reads the exact same neural .mat inputs)."""
    return f"{trial_type}_{align_mode}_results"


def out_subdir_name(triplet_slug: str, trial_type: str, align_mode: str = ALIGN) -> str:
    """This script's own output folder, one per (triplet, trial_type,
    align_mode) -- deliberately distinct from v1/v2/v3's shared,
    whole-brain output folders, and from every OTHER triplet's, since a
    session appearing in several triplets gets numerically different
    results in each (different third-region nuisance)."""
    return f"pcca_all_regions_out_behaviour_v4_hub_{triplet_slug}_{trial_type}_{align_mode}_results"


def behavior_label_for(trial_type: str) -> str:
    """'cued_hit_long' -> 'cued hit long' (matches v3)."""
    return trial_type.replace("_", " ")


# =============================================================================
# 2.  Anatomical canonicalisation -- copied verbatim from v3.
# =============================================================================

# `ANATOMICAL_ORDER` (Useful_definition) has no "HIPP", which hub triplets
# can contain -- append it locally so its position is deterministic.
ANATOMICAL_ORDER_V4: List[str] = list(ANATOMICAL_ORDER) + [
    r for r in ("HIPP",) if r not in ANATOMICAL_ORDER
]


def get_anatomical_index(region: str) -> int:
    """Get anatomical ordering index for a region."""
    try:
        return ANATOMICAL_ORDER_V4.index(region)
    except ValueError:
        return len(ANATOMICAL_ORDER_V4)


def sort_pair_by_anatomy(region_i: str, region_j: str) -> Tuple[str, str]:
    """Sort a region pair by anatomical order (region_i = earlier, region_j = later)."""
    idx_i = get_anatomical_index(region_i)
    idx_j = get_anatomical_index(region_j)
    if idx_i <= idx_j:
        return (region_i, region_j)
    else:
        return (region_j, region_i)


# =============================================================================
# 3.  Region TRIPLETS -- NEW in v4, replaces v3's whole-brain
#     REGION_PAIRS/PAIR_CATEGORIES/HUB_REGIONS. Each triplet fixes 3
#     regions AND the list of sessions where all 3 are co-recorded with
#     enough neurons (built by `build_hub_triplets` from the hub report).
#     `triplet_pairs_with_third` derives the
#     3 internal (region_i, region_j, third_region) triples for one
#     triplet -- the unit every downstream computation loops over.
# =============================================================================

@dataclass(frozen=True)
class TripletSpec:
    """One 3-region combination this script analyses, plus the session
    list it was scoped to (built by `build_hub_triplets`)."""
    label: str
    regions: Tuple[str, str, str]
    sessions: Tuple[str, ...]
    n_expected: int
    hubs: Tuple[str, ...] = ()     # report hub region(s) this triplet was listed under (data names)

    @property
    def slug(self) -> str:
        """Filesystem-/directory-safe identifier, e.g. 'OLF_ORB_STR'."""
        return "_".join(self.regions)


def _data_region_name(display_name: str) -> str:
    """Report display name (e.g. 'Olf area') -> data region name ('OLF');
    grouped regions (e.g. 'Striatum') map to their pooled name ('STR')."""
    if display_name not in REGION_MAPPING:
        raise KeyError(f"Region {display_name!r} in the hub report has no entry in REGION_MAPPING.")
    v = REGION_MAPPING[display_name]
    return v[0] if isinstance(v, (list, tuple)) else v


def build_hub_triplets(
        report_path: Path = HUB_REPORT_PATH,
        min_neurons: Optional[int] = None,
        min_sessions: int = MIN_SESSIONS_PER_TRIPLET,
        selected: Optional[List[Any]] = None,
) -> List[TripletSpec]:
    """Derive the triplet list from the hub section of the threshold report.

    Each report entry lists a region triplet plus, per session where all 3
    regions were recorded, the 3 neuron counts. A triplet appearing under
    several hubs is de-duplicated (hubs are merged into `TripletSpec.hubs`).
    A session is kept only if EVERY region has >= `min_neurons` neurons
    (default `TARGET_SAMPLE_SIZE` -- enough to be an eligible nuisance
    region and to draw `TARGET_SAMPLE_SIZE` neurons without falling back to
    the whole population), and a triplet is kept
    only if >= `min_sessions` sessions remain. Sorted by session count
    (descending), then label.

    If `selected` is given (see `SELECTED_TRIPLETS`), only those triplets are
    returned, in the order given: `min_sessions` is ignored (only the
    per-region `min_neurons` session filter applies), and a requested triplet
    that is absent from the report or has no qualifying session is skipped
    with a warning. Each item names its 3 regions either by report display
    name (e.g. 'OFC', 'Olf area', 'Striatum' -- as printed in the report's
    headers) or by data name (e.g. 'ORB', 'OLF', 'STR' -- as saved in each
    session's .mat data), case-insensitively and mixable; as a string, use
    '∩' to separate them (matching the report header format, and required
    for any display name containing a space, e.g. 'OFC∩Olf area∩Striatum')
    or hyphens/underscores/commas/spaces for single-word names (e.g.
    'MOp-MOs-VALVM'); a list/tuple of 3 names is also accepted directly."""
    if min_neurons is None:
        min_neurons = TARGET_SAMPLE_SIZE

    header_re = re.compile(r"^\d+\.\s+(.+?)\s+n = \d+")
    row_re = re.compile(r"^\s+(yp\d+_\d+)\s+\((.*)\)\s*$")

    hub: Optional[str] = None
    current: Optional[Tuple[str, ...]] = None
    entries: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    with open(report_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("HUB REGION:"):
                hub = _data_region_name(line.split(":", 1)[1].split("(")[0].strip())
                continue
            m = header_re.match(line)
            if m:
                regions = tuple(sorted(
                    (_data_region_name(n.strip()) for n in m.group(1).split("∩")),
                    key=lambda r: (get_anatomical_index(r), r),
                ))
                current = regions
                entry = entries.setdefault(regions, dict(hubs=[], sessions={}))
                if hub not in entry["hubs"]:
                    entry["hubs"].append(hub)
                continue
            m = row_re.match(line)
            if m and current is not None:
                counts = {
                    _data_region_name(k): int(v)
                    for k, v in (p.rsplit("=", 1) for p in m.group(2).split(", "))
                }
                entries[current]["sessions"][m.group(1)] = counts

    def _spec(regions: Tuple[str, ...], entry: Dict[str, Any]) -> TripletSpec:
        kept = sorted(
            s for s, counts in entry["sessions"].items()
            if all(counts.get(r, 0) >= min_neurons for r in regions)
        )
        return TripletSpec(
            label=" ∩ ".join(regions), regions=regions,
            sessions=tuple(kept), n_expected=len(kept), hubs=tuple(entry["hubs"]),
        )

    if selected is not None:
        by_key = {frozenset(regions): (regions, entry) for regions, entry in entries.items()}
        # Accept both report display names (e.g. "Olf area", "Striatum") and
        # data names (e.g. "OLF", "STR") in `selected`, case-insensitively.
        canon: Dict[str, str] = {}
        for display, mapped in REGION_MAPPING.items():
            data_name = mapped[0] if isinstance(mapped, (list, tuple)) else mapped
            canon.setdefault(display.lower(), data_name)
            canon.setdefault(data_name.lower(), data_name)
        triplets = []
        for item in selected:
            if isinstance(item, str):
                # "∩" is the only separator that can't collide with a
                # multi-word display name (e.g. "Olf area"), so split on it
                # when present; otherwise fall back to splitting single-word
                # (data-name) tokens on hyphens/underscores/commas/spaces.
                names = item.split("∩") if "∩" in item else re.split(r"[-_,\s]+", item.strip())
            else:
                names = list(item)
            names = [n.strip() for n in names if n.strip()]
            try:
                key = frozenset(canon[n.lower()] for n in names)
            except KeyError as exc:
                warnings.warn(f"Selected triplet {item!r}: region {exc.args[0]!r} not in the hub report; skipped.")
                continue
            if len(key) != 3 or key not in by_key:
                warnings.warn(f"Selected triplet {item!r} is not a triplet in the hub report; skipped.")
                continue
            spec = _spec(*by_key[key])
            if not spec.sessions:
                warnings.warn(f"Selected triplet {spec.label!r} has no session with >= {min_neurons} "
                              f"neurons in every region; skipped.")
                continue
            triplets.append(spec)
        return triplets

    triplets = [t for t in (_spec(regions, entry) for regions, entry in entries.items())
                if t.n_expected >= min_sessions]
    triplets.sort(key=lambda t: (-t.n_expected, t.label))
    return triplets


try:
    TRIPLETS: List[TripletSpec] = build_hub_triplets(selected=SELECTED_TRIPLETS)
except FileNotFoundError:
    warnings.warn(f"Hub report not found at {HUB_REPORT_PATH}; TRIPLETS is empty.")
    TRIPLETS = []

TRIPLETS_BY_LABEL: Dict[str, TripletSpec] = {t.label: t for t in TRIPLETS}
TRIPLETS_BY_SLUG: Dict[str, TripletSpec] = {t.slug: t for t in TRIPLETS}


def triplet_pairs_with_third(regions: Tuple[str, str, str]) -> List[Tuple[str, str, str]]:
    """The 3 canonicalized `(region_i, region_j, third_region)` triples
    internal to one triplet's 3 regions -- `region_i`/`region_j` sorted by
    `sort_pair_by_anatomy`, `third_region` whichever of the triplet's 3
    regions is neither."""
    triples: List[Tuple[str, str, str]] = []
    for a, b in itertools.combinations(regions, 2):
        region_i, region_j = sort_pair_by_anatomy(a, b)
        third_region = next(r for r in regions if r not in (region_i, region_j))
        triples.append((region_i, region_j, third_region))
    return triples


# =============================================================================
# 3b. Subregion / laminar-depth classification for Part 3's weight-ratio
#     metrics -- copied verbatim from v3.
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
# 4.  Core PCA / pCCA / CCA primitives -- copied verbatim from v3.
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
    BOTH the explained and residual halves (Parts 2a/3a/4a vs. 2b/3b/4b)."""
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
    """Partition trials into `n_folds` folds, trial-boundary-aware (see
    v1/v3's own docstring for the full rationale -- unchanged here)."""
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
    conditioning on nuisance Z_flat. Copied verbatim from v3 -- including
    the `Z_flat=None` case, which is exactly what v4's new "direct CCA"
    condition (1b) relies on: with no nuisance, `_fit_ridge_beta` returns
    None, every residualization step becomes a no-op copy, and this
    reduces to a plain, k-fold cross-validated `ridge_cca` fit."""
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
    its own loadings. Copied verbatim from v3."""
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
# 4b. Neuron-sampling primitives -- copied verbatim from v3. See the module
#     docstring's Section 2 for how narrowly they are now applied (only the
#     3 regions of one triplet, per session) versus v3 (every recorded
#     region in the whole session).
# =============================================================================

def _make_region_rng(session_name: str, region: str, role: str) -> np.random.Generator:
    """Deterministic RNG for one (session, region, role) key -- `role` is
    "paired" (this region is region_i/region_j of some pair) or "nuisance"
    (this region is the third region of some OTHER pair in the same
    triplet). The same physical region gets two INDEPENDENT draw sets,
    one per role, since every triplet region plays "paired" for the 2
    pairs it belongs to and "nuisance" for the 1 pair it doesn't."""
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
    `m = min(n_draws, n_full // target)` draws are cut, ZERO-overlapping,
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
    `min(target, n_full)`) for a region playing the "nuisance" (third
    region) role -- no sweep/coverage requirement, just `n_draws` separate
    random samples. A short bounded retry avoids an exact-duplicate draw
    where the population is large enough that duplicates are avoidable at
    all; it is not a guarantee when `n_full` is only slightly larger than
    `target`."""
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
# 5.  Data loading -- copied verbatim from v3 (`load_region_spikes` /
#     `load_region_spikes_full` / `crop_time_window` /
#     `load_behavior_regressors`). Each loads EVERY region present in the
#     session's .mat file (not just one triplet's 3), same as v3; the
#     triplet-scoping happens downstream, when deciding which regions'
#     draws/pairs to actually compute.
# =============================================================================

def load_region_spikes(
        session_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], int, int]:
    """Load per-region spike tensors AND per-neuron subregion labels for one
    session, subset to `selected_neurons` -- Part 1a's input pool."""
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
    bypassing `selected_neurons` entirely."""
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


def pool_region_groups(
        region_spikes: Dict[str, np.ndarray],
        region_labels: Dict[str, List[str]],
        groups: Dict[str, List[str]] = REGION_GROUPS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """Merge each group's recorded source regions (e.g. STR + STRv + PAL)
    into ONE region named after the group's key by concatenating neurons
    (axis 1) and subregion labels, in the group's listed order. Absent
    sources are skipped; the sources are removed from the output."""
    spikes = dict(region_spikes)
    labels = dict(region_labels)
    for name, sources in groups.items():
        present = [s for s in sources if s in region_spikes]
        if not present:
            continue
        pooled = np.concatenate([region_spikes[s] for s in present], axis=1)
        pooled_labels = [lab for s in present for lab in region_labels[s]]
        for s in present:
            spikes.pop(s, None)
            labels.pop(s, None)
        spikes[name] = pooled
        labels[name] = pooled_labels
    return spikes, labels


def crop_time_window(
        region_spikes: Dict[str, np.ndarray],
        time_vec_full: np.ndarray,
        window: Tuple[float, float],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Crop the trailing time axis of every region's (n_trials, n, T) tensor
    to the closed interval `window` (copied verbatim from v3)."""
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
    filtered to trials matching `trial_label` (copied verbatim from v3)."""
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
# 6.  Result containers. `SubregionWeightMetrics` copied verbatim from v3.
#     `RegionPCAResult`/`RegionPCADrawResult` (Part 1a) now carry
#     `N_SAMPLE_DRAWS` draws each, like every other group, instead of v3's
#     single fit on `selected_neurons`. `SelectedNeuronSet` now carries a
#     `component` field and is produced ONE PER COMPONENT (cross-draw
#     pooling only, never cross-component -- see
#     `_select_cumulative_weight_neurons`), and that selection is now run
#     on Part 1a (`RegionPCAResult.selected_neurons`) and the hub-
#     orientation PCA (`HubPairPCAResult.region_{i,j}_as_hub_selected`,
#     via the new `HubOrientationSelectedNeurons`) in addition to Part
#     2's paired CCA/pCCA (`PrivateLatentPairResult.selected_neurons_i/
#     _j`), none of which v3 had. Part 2's draw containers also carry a
#     `condition` tag (one of `PAIR_CONDITIONS` / `HUB_CONDITIONS`) and a
#     `third_region` field, replacing v3's fixed "2c vs. 2c'" pair with
#     four/three keyed conditions each.
# =============================================================================

@dataclass
class SubregionWeightMetrics:
    """Part 3's weight-ratio measure for ONE PCA/CCA component of ONE
    region's weight vector (copied verbatim from v3 -- see that file for
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
class SelectedNeuronResidual:
    """ONE neuron (identified by its index into the region's FULL,
    unfiltered neuron axis) that fell in the smallest, highest-weight
    prefix whose pooled |weight| contribution reaches
    `CUMULATIVE_WEIGHT_FRACTION` of the pooled total across this pair's
    `N_SAMPLE_DRAWS` draws, for ONE component of one region side (i or j)
    of one condition -- see `SelectedNeuronSet.component`. `residual` is
    that neuron's residualized, cross-trial (n_trials, T) activity -- the
    SAME per-neuron column of `pcca`'s own `X_i_res`/`X_j_res` used to
    build `z_i_lat`/`z_j_lat`, reshaped to (n_trials, T) -- averaged
    across every draw in which the neuron was sampled. Under
    `condition="direct"`, "residual" is simply that neuron's own sampled
    (unregressed) activity, since there is no residualization for that
    condition."""
    neuron_idx: int
    residual: np.ndarray                 # (n_trials, T) -- averaged across qualifying draws
    n_draws_qualified: int
    qualifying_draw_idx: List[int] = field(default_factory=list)
    qualifying_abs_weight: List[float] = field(default_factory=list)   # per qualifying draw, this component's |weight|


@dataclass
class SelectedNeuronSet:
    """Cumulative-weight neuron selection + saved residualized activity for
    ONE region side of ONE pair, ONE condition, ONE component, one
    session -- `region`'s deduplicated neurons, ranked by pooled |weight|
    contribution to component `component` ONLY, across all
    `N_SAMPLE_DRAWS` draws (never mixed with any other component's
    weights), and kept while their cumulative share of the pooled total
    is still below `cumulative_fraction`. See
    `_select_cumulative_weight_neurons` for exactly how `weight_threshold`
    and `neurons` are derived; one `SelectedNeuronSet` is produced per
    component (component index = position in the enclosing list, e.g.
    `PrivateLatentPairResult.selected_neurons_i[condition][component]`)."""
    region: str
    component: int                       # which PCA/CCA component (0-indexed) this selection is for
    cumulative_fraction: float           # target threshold tau (adjustable hyperparameter)
    weight_threshold: float              # smallest selected neuron's pooled |weight|^2 contribution (NaN if no draws)
    n_total_weight_occurrences: int      # how many (draw, neuron) |weight| entries were pooled for this component
    neurons: List[SelectedNeuronResidual] = field(default_factory=list)


@dataclass
class RegionPCADrawResult:
    """ONE of `N_SAMPLE_DRAWS` independent neuron-sampling draws behind
    Part 1a's direct PCA (no regression at all) of one region's own raw
    activity, one session -- sampled from the FULL recorded population,
    the SAME `sample_paired_draws` TARGET_SAMPLE_SIZE scheme Groups 1b-4c
    use for that region's "paired" role, just under its own independent
    draw set/rng role ("region_pca") since PCA needs no CV. `neuron_idx`
    indexes into the region's FULL (unfiltered) neuron axis -- the same
    axis `PrivateLatentSessionResult.region_subregion_labels_full` is
    aligned to."""
    draw_idx: int
    region: str
    W: np.ndarray
    explained_variance_ratio: np.ndarray
    latent: np.ndarray
    mean: np.ndarray
    n_neurons: int
    neuron_idx: np.ndarray
    subregion_weight_metrics: List[SubregionWeightMetrics] = field(default_factory=list)


@dataclass
class RegionPCAResult:
    """Part 1a for one region, one session: `N_SAMPLE_DRAWS` independent
    TARGET_SAMPLE_SIZE-neuron draws from the FULL recorded population
    (bypassing `selected_neurons` -- the same pool/target Groups 1b-4c
    sample from), each fit with its OWN direct PCA (`draws[k]`). This
    makes 1a's sampling structure consistent with every other group;
    unlike them there is still no regression/nuisance and no CV (PCA has
    none), so each draw is simply an independent PCA fit. `selected_neurons`
    is the SAME cumulative-weight selection Groups 1b-4c run, applied to
    these draws' `W` weights -- a `List[SelectedNeuronSet]` indexed by
    component."""
    region: str
    draws: List[RegionPCADrawResult] = field(default_factory=list)
    selected_neurons: List[SelectedNeuronSet] = field(default_factory=list)


@dataclass
class PrivateLatentPairDrawResult:
    """ONE of `N_SAMPLE_DRAWS` independent neuron-sampling draws behind one
    paired condition (`condition` in `PAIR_CONDITIONS`) of one triplet-
    internal pair, one session. `neuron_idx_i` / `neuron_idx_j` /
    `nuisance_neuron_idx` index into each region's FULL (unfiltered)
    neuron axis -- the same axis `PrivateLatentSessionResult.
    region_subregion_labels_full` is aligned to. For `condition="direct"`
    (1b), `nuisance_regions` is always empty and `z_dim_total` always 0 --
    there is no residualization at all, so `z_i_lat`/`z_j_lat` are simply
    the sampled, z-scored (but otherwise raw) activity projected through
    `Wx`/`Wy`."""
    draw_idx: int
    condition: str                       # one of PAIR_CONDITIONS
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
class PrivateLatentPairResult:
    """Groups 1b/2c/3c/4c for one triplet-internal, canonicalized pair, one
    session: `draws[condition]` / `selected_neurons_i[condition]` /
    `selected_neurons_j[condition]` for each `condition` in
    `PAIR_CONDITIONS` -- see the module docstring, Section 3, for what
    each condition's nuisance design actually is. `selected_neurons_i/_j`
    lists are indexed by component (position 0 = component 0, ...) --
    see `SelectedNeuronSet`."""
    region_i: str
    region_j: str
    third_region: str
    draws: Dict[str, List[PrivateLatentPairDrawResult]] = field(default_factory=dict)
    selected_neurons_i: Dict[str, List[SelectedNeuronSet]] = field(default_factory=dict)
    selected_neurons_j: Dict[str, List[SelectedNeuronSet]] = field(default_factory=dict)


@dataclass
class HubOrientationPCADrawResult:
    """ONE of `N_SAMPLE_DRAWS` draws behind one hub-orientation condition
    (`condition` in `HUB_CONDITIONS`) of one pair -- the network
    (nuisance-explained, Parts 2a/3a/4a) vs. residual (Parts 2b/3b/4b)
    PCA split of `hub`'s own sampled activity."""
    draw_idx: int
    condition: str                       # one of HUB_CONDITIONS
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
class HubOrientationSelectedNeurons:
    """Cumulative-weight neuron selection for ONE hub-orientation condition
    (one of `HUB_CONDITIONS`) of one pair, one session -- `network`/
    `residual` each a `List[SelectedNeuronSet]` indexed by component,
    pooling `HubOrientationPCADrawResult.W_network`/`W_residual` (and
    their matching explained/residual activity) across all
    `N_SAMPLE_DRAWS` draws, per component -- exactly the same
    `_select_cumulative_weight_neurons` selection
    `PrivateLatentPairResult.selected_neurons_i/_j` runs for the paired
    CCA/pCCA weights, just applied to the hub-orientation PCA weights."""
    network: List[SelectedNeuronSet] = field(default_factory=list)
    residual: List[SelectedNeuronSet] = field(default_factory=list)


@dataclass
class HubPairPCAResult:
    """Groups 2a/2b + 3a/3b + 4a/4b for one triplet-internal pair, one
    session -- both hub orientations, each keyed by `condition` in
    `HUB_CONDITIONS`, keyed IDENTICALLY to `.pairs` (same (region_i,
    region_j) key space). `region_{i,j}_as_hub_selected[condition]` is
    that hub orientation's cumulative-weight neuron selection (see
    `HubOrientationSelectedNeurons`)."""
    region_i: str
    region_j: str
    third_region: str
    region_i_as_hub_draws: Dict[str, List[HubOrientationPCADrawResult]] = field(default_factory=dict)
    region_j_as_hub_draws: Dict[str, List[HubOrientationPCADrawResult]] = field(default_factory=dict)
    region_i_as_hub_selected: Dict[str, HubOrientationSelectedNeurons] = field(default_factory=dict)
    region_j_as_hub_selected: Dict[str, HubOrientationSelectedNeurons] = field(default_factory=dict)


@dataclass
class PrivateLatentSessionResult:
    """One (triplet, session)'s v4 results, spanning Groups 1a/1b/2a/2b/2c/
    3a/3b/3c/4a/4b/4c -- the unit pickled to {session}_analysis_results.pkl
    inside this triplet's own output directory. `region_pca_raw` (1a) now
    holds `N_SAMPLE_DRAWS` draws per region too (sampled from the FULL
    population, like every other group -- no longer a single fit on
    `selected_neurons`; see `RegionPCAResult`). `pairs` (1b/2c/3c/4c) /
    `hub_pca_pairs` (2a/2b/3a/3b/4a/4b) each hold exactly the triplet's 3
    internal pairs."""
    session: str
    trial_type: str
    triplet_label: str
    triplet_regions: Tuple[str, str, str]
    time_vec: np.ndarray
    n_trials: int
    T: int
    behavior_available: bool
    behavior_channel_labels: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    region_pca_raw: Dict[str, RegionPCAResult] = field(default_factory=dict)
    pairs: Dict[Tuple[str, str], PrivateLatentPairResult] = field(default_factory=dict)
    hub_pca_pairs: Dict[Tuple[str, str], HubPairPCAResult] = field(default_factory=dict)
    align_mode: str = ALIGN
    region_subregion_labels: Dict[str, List[str]] = field(default_factory=dict)       # selected_neurons labels (provenance only -- 1a no longer fits on this pool)
    region_subregion_labels_full: Dict[str, List[str]] = field(default_factory=dict)  # full-population pool (Part 1a AND Part 2 draws)


def config_fingerprint(triplet: TripletSpec) -> Dict[str, Any]:
    """Snapshot of every parameter that changes the numerical result for
    ONE triplet, stored inside each of its sessions' pickles so a stale
    cache is detected, not silently reused (matches v3's own caching
    contract, now scoped per triplet)."""
    return dict(
        align_mode=ALIGN,
        n_components=N_COMPONENTS,
        n_pca_components=N_PCA_COMPONENTS,
        lambda_cca=LAMBDA_CCA,
        lambda_hat=LAMBDA_HAT,
        anatomical_order=tuple(ANATOMICAL_ORDER_V4),
        region_groups={k: tuple(v) for k, v in REGION_GROUPS.items()},
        triplet_label=triplet.label,
        triplet_regions=tuple(triplet.regions),
        time_range_s=tuple(TIME_RANGE_S),
        behavior_time_range_s=tuple(BEHAVIOR_TIME_RANGE_S),
        subtract_psth=SUBTRACT_PSTH,
        shuffle_trials=SHUFFLE_TRIALS,
        require_behavior=REQUIRE_BEHAVIOR,
        cv_folds=CV_FOLDS,
        cv_rng_seed=CV_RNG_SEED,
        target_sample_size=TARGET_SAMPLE_SIZE,
        n_sample_draws=N_SAMPLE_DRAWS,
        sample_rng_seed=SAMPLE_RNG_SEED,
        cumulative_weight_fraction=CUMULATIVE_WEIGHT_FRACTION,
        pair_conditions=tuple(PAIR_CONDITIONS),
        hub_conditions=tuple(HUB_CONDITIONS),
        result_schema_version=4,   # v4: triplet-scoped reorg + new "direct
        # CCA" (1b) / "behaviour-only" (2a/2b/2c) conditions -- bumped
        # from v3's 3.
    )


# =============================================================================
# 7.  Per-region / per-pair / per-draw / per-session computation.
#     `_compute_region_pca` (Part 1a) is unchanged from v3.
#     `_compute_pair_result_draw` / `_compute_hub_orientation_draw` are
#     unchanged in their MATH from v3 -- both simply gained a `condition`
#     tag to stamp on their result. `_compute_pair_sampled` is the new
#     per-pair orchestrator: for each of `N_SAMPLE_DRAWS` draws it builds
#     all 4 paired-condition Z's and all 3 hub-condition Z's (Section 3 of
#     the module docstring) and calls the two draw-level functions once
#     per condition.
# =============================================================================

def _subregion_group_for(region: str, label: str) -> Optional[str]:
    """Collapse one neuron's raw `subregion_labels` string to Part 3's
    group label (copied verbatim from v3)."""
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
    column of `W` (copied verbatim from v3)."""
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
        X_flat_full: np.ndarray,
        draws_idx: List[np.ndarray],
        n_trials: int,
        T: int,
        n_components: int = N_PCA_COMPONENTS,
        labels_full: Optional[List[str]] = None,
) -> RegionPCAResult:
    """Part 1a: `len(draws_idx)` (`N_SAMPLE_DRAWS`) independent direct PCA
    fits of one region's own raw activity, each on its own
    TARGET_SAMPLE_SIZE-neuron draw of `X_flat_full` (the FULL population,
    z-scored) -- same sampling scheme Groups 1b-4c use, just with no
    nuisance regression and no CV (PCA has none). `selected_neurons` runs
    the SAME cumulative-weight selection Groups 1b-4c run on their pCCA/
    PCA weights, per component, on these draws' `W`; since there is no
    residualization here, the "residual" saved per selected neuron is
    simply its own sampled (raw, z-scored) activity -- the same
    convention `condition="direct"` (1b) uses."""
    draws: List[RegionPCADrawResult] = []
    draw_X_list: List[np.ndarray] = []
    for k, idx in enumerate(draws_idx):
        X_sub = X_flat_full[:, idx]
        labels_sub = [labels_full[i] for i in idx] if labels_full is not None else None
        W, explained_variance_ratio, mean, latent = pca_fit_and_project(
            X_sub, n_trials, T, n_components)
        draws.append(RegionPCADrawResult(
            draw_idx=k,
            region=region,
            W=W.astype(np.float32),
            explained_variance_ratio=explained_variance_ratio.astype(np.float64),
            latent=latent.astype(np.float32),
            mean=mean.astype(np.float32),
            n_neurons=int(X_sub.shape[1]),
            neuron_idx=idx.astype(np.int64),
            subregion_weight_metrics=compute_subregion_weight_metrics(region, W, labels_sub),
        ))
        draw_X_list.append(X_sub)

    selected_neurons = _select_cumulative_weight_neurons(
        region, [d.W for d in draws], [d.neuron_idx for d in draws], draw_X_list, n_trials, T)
    return RegionPCAResult(region=region, draws=draws, selected_neurons=selected_neurons)


def _compute_pair_result_draw(
        draw_idx: int,
        condition: str,
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
    """One paired condition (1b/2c/3c/4c), ONE draw of one pair -- same
    `pcca` call v3's `_compute_pair_result_draw` made, just tagged with
    `condition` and (for 1b) called with `Z_flat=None`. Returns
    `(draw_result, X_i_res, X_j_res)`: the latter two are the SAME
    residualized matrices `z_i_lat`/`z_j_lat` were projected from, handed
    back so the caller can pool them across draws for
    `_select_cumulative_weight_neurons` without refitting `pcca`."""
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
        condition=condition,
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


def _select_cumulative_weight_neurons(
        region: str,
        draw_Wx: List[np.ndarray],           # per draw: (n_sampled, K) -- Wx (region_i) or Wy (region_j)
        draw_neuron_idx: List[np.ndarray],   # per draw: (n_sampled,) -- neuron_idx_i or neuron_idx_j (FULL axis)
        draw_X_res: List[np.ndarray],        # per draw: (T*n_trials, n_sampled) -- X_i_res or X_j_res
        n_trials: int,
        T: int,
        cumulative_fraction: float = CUMULATIVE_WEIGHT_FRACTION,
) -> List[SelectedNeuronSet]:
    """Run the cumulative-weight neuron selection SEPARATELY PER COMPONENT
    -- component `k`'s selection pools every (draw, neuron) |weight| entry
    for component `k` ONLY across all `N_SAMPLE_DRAWS` draws (a neuron's
    weight on one component never contributes to another component's
    energy/ranking), collapses each UNIQUE neuron to ONE pooled
    contribution score `e_i = sum_d w_{d,i}^2`, sorts neurons by `e_i`
    DESCENDING, and keeps the smallest, highest-ranked prefix whose
    cumulative share of the pooled total `sum(e_i)` reaches
    `cumulative_fraction`. Returns one `SelectedNeuronSet` per component,
    in component order (`result[k].component == k`); previously (when
    N_COMPONENTS was assumed to be 1) this pooled every component
    together into a single score, which silently stopped being correct
    the moment more than one component was requested."""
    K = 0
    for Wx in draw_Wx:
        if Wx.size:
            K = max(K, Wx.shape[1])
    if K == 0:
        return []

    results: List[SelectedNeuronSet] = []
    for k in range(K):
        n_entries = 0
        per_neuron_energy: Dict[int, float] = {}
        per_neuron_draw_weight: Dict[int, Dict[int, float]] = {}   # neuron_idx -> {draw_idx: |weight| on component k}
        for d, (Wx, nidx) in enumerate(zip(draw_Wx, draw_neuron_idx)):
            if Wx.size == 0 or nidx.size == 0 or k >= Wx.shape[1]:
                continue
            abs_w = np.abs(Wx[:, k]).astype(np.float64)
            for pos in range(nidx.shape[0]):
                w = float(abs_w[pos])
                nu = int(nidx[pos])
                n_entries += 1
                per_neuron_energy[nu] = per_neuron_energy.get(nu, 0.0) + w * w
                draw_weights = per_neuron_draw_weight.setdefault(nu, {})
                draw_weights[d] = w

        if not per_neuron_energy:
            results.append(SelectedNeuronSet(
                region=region, component=k, cumulative_fraction=cumulative_fraction,
                weight_threshold=float("nan"), n_total_weight_occurrences=0, neurons=[],
            ))
            continue

        total_energy = sum(per_neuron_energy.values())
        ranked = sorted(per_neuron_energy.items(), key=lambda kv: kv[1], reverse=True)

        selected_ids: List[int] = []
        cum_energy = 0.0
        for nu, e in ranked:
            selected_ids.append(nu)
            cum_energy += e
            if total_energy > 0 and (cum_energy / total_energy) >= cumulative_fraction:
                break

        weight_threshold = per_neuron_energy[selected_ids[-1]]

        neurons: List[SelectedNeuronResidual] = []
        for nu in selected_ids:
            draw_weights = per_neuron_draw_weight[nu]
            draw_ids = sorted(draw_weights.keys())

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
                qualifying_abs_weight=[draw_weights[d] for d in draw_ids],
            ))

        neurons.sort(key=lambda r: r.neuron_idx)
        results.append(SelectedNeuronSet(
            region=region, component=k, cumulative_fraction=cumulative_fraction,
            weight_threshold=weight_threshold, n_total_weight_occurrences=n_entries, neurons=neurons,
        ))

    return results


def _compute_hub_orientation_draw(
        draw_idx: int,
        condition: str,
        hub: str,
        partner: str,
        X_hub_sub: np.ndarray,
        idx_hub: np.ndarray,
        Z_nuisance: Optional[np.ndarray],
        nuisance_all: List[str],
        nuisance_idx_k: Dict[str, np.ndarray],
        n_trials: int,
        T: int,
        labels_hub: Optional[List[str]] = None,
) -> Tuple[HubOrientationPCADrawResult, np.ndarray, np.ndarray]:
    """One hub-orientation condition (2a/2b, 3a/3b, or 4a/4b), ONE draw --
    same `residualize_with_explained` call v3's `_compute_hub_orientation_
    draw` made, just tagged with `condition`. `X_hub_sub` is whichever
    pool the CALLER already prepared for this condition: the region's raw
    sampled activity for `"behav_only"`/`"region_only"` (Groups 2/3 --
    behaviour and the third region are never combined for these), or the
    ALREADY behaviour-residualized pool (`behav_res_by_region_full`,
    sliced once per region per session) for `"region_behav"` (Group 4 --
    sequential composition, matching v3's Part-2a/2b precedent). Returns
    `(draw_result, explained, residual)`: the latter two are the SAME
    matrices `latent_network`/`latent_residual` were projected from,
    handed back so the caller can pool them across draws for
    `_select_cumulative_weight_neurons` without recomputing
    `residualize_with_explained`."""
    explained, residual = residualize_with_explained(X_hub_sub, Z_nuisance, LAMBDA_HAT)

    W_net, evr_net, _mean_net, lat_net = pca_fit_and_project(
        explained, n_trials, T, N_PCA_COMPONENTS)
    W_res, evr_res, _mean_res, lat_res = pca_fit_and_project(
        residual, n_trials, T, N_PCA_COMPONENTS)

    draw_result = HubOrientationPCADrawResult(
        draw_idx=draw_idx,
        condition=condition,
        hub=hub,
        partner=partner,
        W_network=W_net.astype(np.float32),
        explained_variance_ratio_network=evr_net.astype(np.float64),
        latent_network=lat_net.astype(np.float32),
        W_residual=W_res.astype(np.float32),
        explained_variance_ratio_residual=evr_res.astype(np.float64),
        latent_residual=lat_res.astype(np.float32),
        n_neurons_hub=int(X_hub_sub.shape[1]),
        neuron_idx_hub=idx_hub.astype(np.int64),
        nuisance_regions=list(nuisance_all),
        nuisance_neuron_idx={r: nuisance_idx_k[r].astype(np.int64) for r in nuisance_all},
        z_dim_total=int(Z_nuisance.shape[1]) if Z_nuisance is not None else 0,
        subregion_weight_metrics_network=compute_subregion_weight_metrics(hub, W_net, labels_hub),
        subregion_weight_metrics_residual=compute_subregion_weight_metrics(hub, W_res, labels_hub),
    )
    return draw_result, explained, residual


def _compute_pair_sampled(
        region_i: str,
        region_j: str,
        third_region: str,
        region_flat_full: Dict[str, np.ndarray],
        behav_res_by_region_full: Dict[str, np.ndarray],
        region_subregion_labels_full: Dict[str, List[str]],
        paired_draws: Dict[str, List[np.ndarray]],
        nuisance_draws: Dict[str, List[np.ndarray]],
        behavior_Z_flat: Optional[np.ndarray],
        n_trials: int,
        T: int,
) -> Tuple[Optional[PrivateLatentPairResult], Optional[HubPairPCAResult]]:
    """Groups 1b/2a/2b/2c/3a/3b/3c/4a/4b/4c for one triplet-internal pair,
    one session -- `N_SAMPLE_DRAWS` independent neuron-sampling draws,
    each producing all 4 paired-condition pCCA/CCA results
    (`PAIR_CONDITIONS`) and all 3 hub-orientation condition results
    (`HUB_CONDITIONS`, x2 orientations) for that draw. `third_region` is
    the ONE remaining region of this pair's triplet -- the only candidate
    nuisance region for Groups 3/4 (never "every other recorded region",
    unlike v3)."""
    if region_i not in region_flat_full or region_j not in region_flat_full:
        return None, None
    if region_i not in paired_draws or region_j not in paired_draws:
        return None, None

    third_eligible = (
        third_region in region_flat_full
        and region_flat_full[third_region].shape[1] >= TARGET_SAMPLE_SIZE
    )
    if not third_eligible:
        warnings.warn(
            f"_compute_pair_sampled: third region {third_region!r} for pair "
            f"({region_i}, {region_j}) has < {TARGET_SAMPLE_SIZE} "
            f"recorded neurons (or is absent this session); Groups 3/4 "
            f"degrade to no region-nuisance for this pair."
        )
    nuisance_all = [third_region] if third_eligible else []

    labels_i_full = region_subregion_labels_full.get(region_i)
    labels_j_full = region_subregion_labels_full.get(region_j)

    draw_idx_i_list: List[np.ndarray] = []
    draw_idx_j_list: List[np.ndarray] = []
    pair_draws: Dict[str, List[PrivateLatentPairDrawResult]] = {c: [] for c in PAIR_CONDITIONS}
    pair_Xi_res: Dict[str, List[np.ndarray]] = {c: [] for c in PAIR_CONDITIONS}
    pair_Xj_res: Dict[str, List[np.ndarray]] = {c: [] for c in PAIR_CONDITIONS}

    hub_draws_i: Dict[str, List[HubOrientationPCADrawResult]] = {c: [] for c in HUB_CONDITIONS}
    hub_draws_j: Dict[str, List[HubOrientationPCADrawResult]] = {c: [] for c in HUB_CONDITIONS}
    hub_net_Xi: Dict[str, List[np.ndarray]] = {c: [] for c in HUB_CONDITIONS}
    hub_res_Xi: Dict[str, List[np.ndarray]] = {c: [] for c in HUB_CONDITIONS}
    hub_net_Xj: Dict[str, List[np.ndarray]] = {c: [] for c in HUB_CONDITIONS}
    hub_res_Xj: Dict[str, List[np.ndarray]] = {c: [] for c in HUB_CONDITIONS}

    for k in range(N_SAMPLE_DRAWS):
        idx_i = paired_draws[region_i][k]
        idx_j = paired_draws[region_j][k]
        idx_third = (
            nuisance_draws[third_region][k]
            if third_eligible else np.array([], dtype=np.int64)
        )
        nuisance_idx_k = {third_region: idx_third} if third_eligible else {}

        X_i_sub = region_flat_full[region_i][:, idx_i]
        X_j_sub = region_flat_full[region_j][:, idx_j]
        X_third_sub = region_flat_full[third_region][:, idx_third] if third_eligible else None

        labels_i_sub = [labels_i_full[i] for i in idx_i] if labels_i_full else None
        labels_j_sub = [labels_j_full[j] for j in idx_j] if labels_j_full else None

        # ---- 1b: direct CCA, no regression at all ---------------------
        draw_1b, Xi_res_1b, Xj_res_1b = _compute_pair_result_draw(
            k, "direct", region_i, region_j, X_i_sub, X_j_sub, idx_i, idx_j, None,
            [], {}, n_trials, T, labels_i_sub, labels_j_sub)
        pair_draws["direct"].append(draw_1b)
        pair_Xi_res["direct"].append(Xi_res_1b)
        pair_Xj_res["direct"].append(Xj_res_1b)

        # ---- 2c: pCCA, nuisance = behaviour only -----------------------
        draw_2c, Xi_res_2c, Xj_res_2c = _compute_pair_result_draw(
            k, "behav_only", region_i, region_j, X_i_sub, X_j_sub, idx_i, idx_j,
            behavior_Z_flat, [], {}, n_trials, T, labels_i_sub, labels_j_sub)
        pair_draws["behav_only"].append(draw_2c)
        pair_Xi_res["behav_only"].append(Xi_res_2c)
        pair_Xj_res["behav_only"].append(Xj_res_2c)

        # ---- 3c: pCCA, nuisance = third region only --------------------
        draw_3c, Xi_res_3c, Xj_res_3c = _compute_pair_result_draw(
            k, "region_only", region_i, region_j, X_i_sub, X_j_sub, idx_i, idx_j,
            X_third_sub, nuisance_all, nuisance_idx_k, n_trials, T, labels_i_sub, labels_j_sub)
        pair_draws["region_only"].append(draw_3c)
        pair_Xi_res["region_only"].append(Xi_res_3c)
        pair_Xj_res["region_only"].append(Xj_res_3c)

        # ---- 4c: pCCA, nuisance = third region + behaviour, JOINT ------
        z_parts = []
        if third_eligible:
            z_parts.append(X_third_sub)
        if behavior_Z_flat is not None:
            z_parts.append(behavior_Z_flat)
        Z_joint = np.concatenate(z_parts, axis=1) if z_parts else None
        draw_4c, Xi_res_4c, Xj_res_4c = _compute_pair_result_draw(
            k, "region_behav", region_i, region_j, X_i_sub, X_j_sub, idx_i, idx_j,
            Z_joint, nuisance_all, nuisance_idx_k, n_trials, T, labels_i_sub, labels_j_sub)
        pair_draws["region_behav"].append(draw_4c)
        pair_Xi_res["region_behav"].append(Xi_res_4c)
        pair_Xj_res["region_behav"].append(Xj_res_4c)

        draw_idx_i_list.append(idx_i)
        draw_idx_j_list.append(idx_j)

        # ---- 2a/2b: hub-orientation PCA, nuisance = behaviour only -----
        draw_2ab_i, net_2ab_i, res_2ab_i = _compute_hub_orientation_draw(
            k, "behav_only", region_i, region_j, X_i_sub, idx_i, behavior_Z_flat,
            [], {}, n_trials, T, labels_i_sub)
        hub_draws_i["behav_only"].append(draw_2ab_i)
        hub_net_Xi["behav_only"].append(net_2ab_i)
        hub_res_Xi["behav_only"].append(res_2ab_i)
        draw_2ab_j, net_2ab_j, res_2ab_j = _compute_hub_orientation_draw(
            k, "behav_only", region_j, region_i, X_j_sub, idx_j, behavior_Z_flat,
            [], {}, n_trials, T, labels_j_sub)
        hub_draws_j["behav_only"].append(draw_2ab_j)
        hub_net_Xj["behav_only"].append(net_2ab_j)
        hub_res_Xj["behav_only"].append(res_2ab_j)

        # ---- 3a/3b: hub-orientation PCA, nuisance = third region only --
        draw_3ab_i, net_3ab_i, res_3ab_i = _compute_hub_orientation_draw(
            k, "region_only", region_i, region_j, X_i_sub, idx_i, X_third_sub,
            nuisance_all, nuisance_idx_k, n_trials, T, labels_i_sub)
        hub_draws_i["region_only"].append(draw_3ab_i)
        hub_net_Xi["region_only"].append(net_3ab_i)
        hub_res_Xi["region_only"].append(res_3ab_i)
        draw_3ab_j, net_3ab_j, res_3ab_j = _compute_hub_orientation_draw(
            k, "region_only", region_j, region_i, X_j_sub, idx_j, X_third_sub,
            nuisance_all, nuisance_idx_k, n_trials, T, labels_j_sub)
        hub_draws_j["region_only"].append(draw_3ab_j)
        hub_net_Xj["region_only"].append(net_3ab_j)
        hub_res_Xj["region_only"].append(res_3ab_j)

        # ---- 4a/4b: hub-orientation PCA, nuisance = third region, on top
        #      of activity ALREADY behaviour-residualized once per region
        #      per session (sequential composition, matches v3's Part
        #      2a/2b precedent) -------------------------------------------
        X_i_behav_res_sub = behav_res_by_region_full[region_i][:, idx_i]
        X_j_behav_res_sub = behav_res_by_region_full[region_j][:, idx_j]
        draw_4ab_i, net_4ab_i, res_4ab_i = _compute_hub_orientation_draw(
            k, "region_behav", region_i, region_j, X_i_behav_res_sub, idx_i, X_third_sub,
            nuisance_all, nuisance_idx_k, n_trials, T, labels_i_sub)
        hub_draws_i["region_behav"].append(draw_4ab_i)
        hub_net_Xi["region_behav"].append(net_4ab_i)
        hub_res_Xi["region_behav"].append(res_4ab_i)
        draw_4ab_j, net_4ab_j, res_4ab_j = _compute_hub_orientation_draw(
            k, "region_behav", region_j, region_i, X_j_behav_res_sub, idx_j, X_third_sub,
            nuisance_all, nuisance_idx_k, n_trials, T, labels_j_sub)
        hub_draws_j["region_behav"].append(draw_4ab_j)
        hub_net_Xj["region_behav"].append(net_4ab_j)
        hub_res_Xj["region_behav"].append(res_4ab_j)

    # ---- Cumulative-weight neuron selection + saved residualized activity,
    #      pooled across all N_SAMPLE_DRAWS draws, PER COMPONENT, separately
    #      for region_i, region_j, and each of the 4 paired conditions. ----
    selected_i: Dict[str, List[SelectedNeuronSet]] = {}
    selected_j: Dict[str, List[SelectedNeuronSet]] = {}
    for cond in PAIR_CONDITIONS:
        selected_i[cond] = _select_cumulative_weight_neurons(
            region_i, [d.Wx for d in pair_draws[cond]], draw_idx_i_list, pair_Xi_res[cond], n_trials, T)
        selected_j[cond] = _select_cumulative_weight_neurons(
            region_j, [d.Wy for d in pair_draws[cond]], draw_idx_j_list, pair_Xj_res[cond], n_trials, T)

    # ---- Same selection, applied to the hub-orientation PCA weights
    #      (2a/2b/3a/3b/4a/4b) -- network and residual each selected
    #      independently, per component, per condition, per orientation. --
    hub_selected_i: Dict[str, HubOrientationSelectedNeurons] = {}
    hub_selected_j: Dict[str, HubOrientationSelectedNeurons] = {}
    for cond in HUB_CONDITIONS:
        hub_selected_i[cond] = HubOrientationSelectedNeurons(
            network=_select_cumulative_weight_neurons(
                region_i, [d.W_network for d in hub_draws_i[cond]], draw_idx_i_list,
                hub_net_Xi[cond], n_trials, T),
            residual=_select_cumulative_weight_neurons(
                region_i, [d.W_residual for d in hub_draws_i[cond]], draw_idx_i_list,
                hub_res_Xi[cond], n_trials, T),
        )
        hub_selected_j[cond] = HubOrientationSelectedNeurons(
            network=_select_cumulative_weight_neurons(
                region_j, [d.W_network for d in hub_draws_j[cond]], draw_idx_j_list,
                hub_net_Xj[cond], n_trials, T),
            residual=_select_cumulative_weight_neurons(
                region_j, [d.W_residual for d in hub_draws_j[cond]], draw_idx_j_list,
                hub_res_Xj[cond], n_trials, T),
        )

    pair_result = PrivateLatentPairResult(
        region_i=region_i, region_j=region_j, third_region=third_region,
        draws=pair_draws, selected_neurons_i=selected_i, selected_neurons_j=selected_j)
    hub_pair_result = HubPairPCAResult(
        region_i=region_i, region_j=region_j, third_region=third_region,
        region_i_as_hub_draws=hub_draws_i, region_j_as_hub_draws=hub_draws_j,
        region_i_as_hub_selected=hub_selected_i, region_j_as_hub_selected=hub_selected_j)
    return pair_result, hub_pair_result


def compute_private_latents_for_session(
        session_name: str,
        triplet: TripletSpec,
        trial_type: str = TRIAL_TYPE,
        mat_dir: Optional[Path] = None,
        align_mode: str = ALIGN,
) -> Optional[PrivateLatentSessionResult]:
    """Compute Groups 1a/1b/2a/2b/2c/3a/3b/3c/4a/4b/4c for one session,
    scoped to ONE triplet's 3 regions and 3 internal pairs. Returns None
    if the session cannot be processed at all (missing source .mat file, a
    crop window with < 2 overlapping samples, or -- when
    REQUIRE_BEHAVIOR=True, the default -- missing behavioural tracking) --
    same failure contract as v3.
    """
    mat_dir = mat_dir if mat_dir is not None else (BASE_DIR / mat_subdir_name(trial_type, align_mode))
    session_file = mat_dir / f"{session_name}_analysis_results.mat"
    if not session_file.exists():
        print(f"  [skip] {session_name}: source file not found -> {session_file}")
        return None

    # ---- Part 1a pool: selected_neurons-filtered (unchanged from v3) ----
    region_spikes_sel, region_subregion_labels_sel, n_trials, T = load_region_spikes(
        str(session_file))
    if not region_spikes_sel:
        print(f"  [skip] {session_name}: no regions loaded")
        return None

    # ---- Groups 1b-4c pool: full/unfiltered population -------------------
    region_spikes_full, region_subregion_labels_full, n_trials_full, T_full = (
        load_region_spikes_full(str(session_file)))
    if not region_spikes_full:
        print(f"  [skip] {session_name}: no regions loaded (full pool)")
        return None
    if n_trials_full != n_trials or T_full != T:
        warnings.warn(
            f"[{session_name}] Part-1a/Part-2 pools disagree on n_trials/T "
            f"({n_trials}/{T} vs {n_trials_full}/{T_full}) from the same "
            f"source file; proceeding with the Part-1a (selected_neurons) "
            f"pool's n_trials/T for both."
        )

    # ---- Pool grouped regions (STR + STRv + PAL -> "STR"), both pools ----
    region_spikes_sel, region_subregion_labels_sel = pool_region_groups(
        region_spikes_sel, region_subregion_labels_sel)
    region_spikes_full, region_subregion_labels_full = pool_region_groups(
        region_spikes_full, region_subregion_labels_full)

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
            f"with behaviour-dependent conditions degraded to no nuisance, "
            f"since REQUIRE_BEHAVIOR=False."
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

    # ---- Flatten + z-score the full pool once, reused across every pair
    #      AND Group 1a below -- `region_spikes_sel` (`selected_neurons`)
    #      is no longer flattened/used for PCA input; it still supplies
    #      this session's canonical n_trials/T and `region_subregion_labels`
    #      (kept for provenance -- see `PrivateLatentSessionResult`). ------
    region_flat_full: Dict[str, np.ndarray] = {
        r: _zscore_flat(X, subtract_psth=SUBTRACT_PSTH, shuffle_trials=SHUFFLE_TRIALS)
        for r, X in region_spikes_full.items()
    }
    behavior_Z_flat: Optional[np.ndarray] = None
    if behav_combined_raw is not None:
        behavior_Z_flat = _zscore_flat(behav_combined_raw, subtract_psth=SUBTRACT_PSTH)

    # ---- Group 1a: direct PCA, no regression, this triplet's 3 regions --
    #      `N_SAMPLE_DRAWS` TARGET_SAMPLE_SIZE draws from the FULL
    #      population (bypassing `selected_neurons`), same pool/target
    #      Groups 1b-4c sample from, just under their own independent
    #      "region_pca" draw role so they don't share a draw set with
    #      that region's "paired"/"nuisance" role in the pairs below. -----
    region_pca_raw: Dict[str, RegionPCAResult] = {}
    for region in triplet.regions:
        if region not in region_flat_full:
            continue
        n_full_region = region_flat_full[region].shape[1]
        region_pca_draws_idx = sample_paired_draws(
            n_full_region, _make_region_rng(session_name, region, "region_pca"))
        region_pca_raw[region] = _compute_region_pca(
            region, region_flat_full[region], region_pca_draws_idx, n_trials, T,
            labels_full=region_subregion_labels_full.get(region))

    # ---- Groups 1b-4c prep: behaviour-residualize the FULL pool ONCE per
    #      triplet region (needed for Groups 4a/4b's sequential
    #      composition -- see module docstring, Section 3, Group 4). -----
    behav_res_by_region_full: Dict[str, np.ndarray] = {
        r: residualize(region_flat_full[r], behavior_Z_flat, LAMBDA_HAT)
        for r in triplet.regions if r in region_flat_full
    }

    # ---- Groups 1b-4c sampling: precompute each triplet region's draws
    #      ONCE per session -- a "paired" draw set (used whenever this
    #      region is region_i/region_j of one of the triplet's 2 pairs
    #      containing it) and a "nuisance" draw set (used for the 1 pair
    #      this region is the THIRD region of). -----------------------
    paired_draws: Dict[str, List[np.ndarray]] = {}
    nuisance_draws: Dict[str, List[np.ndarray]] = {}
    for region in triplet.regions:
        if region not in region_flat_full:
            continue
        n_full = region_flat_full[region].shape[1]
        paired_draws[region] = sample_paired_draws(
            n_full, _make_region_rng(session_name, region, "paired"))
        nuisance_draws[region] = sample_nuisance_draws(
            n_full, _make_region_rng(session_name, region, "nuisance"))

    # ---- Groups 1b-4c: the triplet's 3 internal pairs -------------------
    pairs: Dict[Tuple[str, str], PrivateLatentPairResult] = {}
    hub_pca_pairs: Dict[Tuple[str, str], HubPairPCAResult] = {}
    for region_i, region_j, third_region in triplet_pairs_with_third(triplet.regions):
        pair_result, hub_result = _compute_pair_sampled(
            region_i, region_j, third_region,
            region_flat_full, behav_res_by_region_full, region_subregion_labels_full,
            paired_draws, nuisance_draws, behavior_Z_flat, n_trials, T,
        )
        if pair_result is not None:
            pairs[(region_i, region_j)] = pair_result
        if hub_result is not None:
            hub_pca_pairs[(region_i, region_j)] = hub_result

    print(
        f"  [{session_name}] triplet {triplet.label}: "
        f"{len(pairs)}/3 pairs computed (x{N_SAMPLE_DRAWS} draws, "
        f"{len(PAIR_CONDITIONS)} paired conditions + {len(HUB_CONDITIONS)} "
        f"hub conditions each), 1a region-PCA {len(region_pca_raw)}/3 regions "
        f"(x{N_SAMPLE_DRAWS} draws each) "
        f"(n_trials={n_trials}, T={T}, behaviour="
        f"{'yes' if behav_combined_raw is not None else 'no'})"
    )

    return PrivateLatentSessionResult(
        session=session_name,
        trial_type=trial_type,
        triplet_label=triplet.label,
        triplet_regions=tuple(triplet.regions),
        time_vec=time_vec.astype(np.float64),
        n_trials=n_trials,
        T=T,
        behavior_available=behav_combined_raw is not None,
        behavior_channel_labels=behavior_channel_labels,
        config=config_fingerprint(triplet),
        region_pca_raw=region_pca_raw,
        pairs=pairs,
        hub_pca_pairs=hub_pca_pairs,
        align_mode=align_mode,
        region_subregion_labels=region_subregion_labels_sel,
        region_subregion_labels_full=region_subregion_labels_full,
    )


# =============================================================================
# 8.  Orchestration -- reorganised around TRIPLETS (item 2 of the request):
#     `run_triplet` processes exactly one triplet's own session list;
#     `run_all_triplets` loops over every triplet in `TRIPLETS` (or a
#     caller-provided subset).
# =============================================================================

def run_triplet(
        triplet: TripletSpec,
        trial_type: str = TRIAL_TYPE,
        sessions: Optional[List[str]] = None,
        overwrite: bool = False,
        align_mode: str = ALIGN,
) -> None:
    """Compute and pickle every session's PrivateLatentSessionResult for
    ONE triplet. `sessions` defaults to `triplet.sessions` (the
    list this triplet was scoped to); pass a subset for a quick test run.
    Caching contract matches v3: an existing {session}_analysis_results.pkl
    is reused only if its stored `config` matches the CURRENT
    `config_fingerprint(triplet)`."""
    sessions = list(sessions) if sessions is not None else list(triplet.sessions)
    mat_dir = BASE_DIR / mat_subdir_name(trial_type, align_mode)
    out_dir = BASE_DIR / out_subdir_name(triplet.slug, trial_type, align_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"v4.0  |  triplet {triplet.label}  ({', '.join(triplet.regions)})")
    print(f"  trial_type : {trial_type}   (behaviour label = '{behavior_label_for(trial_type)}')")
    print(f"  align_mode : {align_mode}")
    print(f"  source dir : {mat_dir}")
    print(f"  output dir : {out_dir}")
    print(f"  sessions   : {len(sessions)}  (n_expected={triplet.n_expected}, "
          f"hubs={list(triplet.hubs)})")
    print(f"  pca comps  : {N_PCA_COMPONENTS}  (Groups 1a/2a/2b/3a/3b/4a/4b; "
          f"N_COMPONENTS={N_COMPONENTS} for the pCCA/CCA *c groups)")
    print(f"  sampling   : target={TARGET_SAMPLE_SIZE} neurons/region, "
          f"{N_SAMPLE_DRAWS} draws/pair/session")
    print(f"  paired cond: {list(PAIR_CONDITIONS)}")
    print(f"  hub cond   : {list(HUB_CONDITIONS)}")
    print("=" * 70)

    if len(sessions) != triplet.n_expected:
        warnings.warn(
            f"run_triplet({triplet.label!r}): expected {triplet.n_expected} "
            f"sessions, got {len(sessions)}."
        )

    current_config = config_fingerprint(triplet)
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
                session_name, triplet, trial_type, mat_dir, align_mode=align_mode)
        except Exception as exc:
            print(f"  \U0001F4A5 [ERROR] {session_name}: {exc}")
            n_skipped += 1
            continue

        has_any_result = result is not None and (
            result.pairs or result.hub_pca_pairs or result.region_pca_raw
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
        f"\U0001F389 Done triplet {triplet.label}. {n_written} written, "
        f"{n_cached} cached, {n_skipped} skipped (of {len(sessions)} sessions)."
    )
    print("=" * 70)


def run_all_triplets(
        trial_type: str = TRIAL_TYPE,
        triplets: List[TripletSpec] = TRIPLETS,
        overwrite: bool = False,
        align_mode: str = ALIGN,
) -> None:
    """Run `run_triplet` for every triplet in `triplets` (default: every
    hub-derived triplet in `TRIPLETS`), in order."""
    for triplet in triplets:
        run_triplet(triplet, trial_type=trial_type, overwrite=overwrite, align_mode=align_mode)


def main() -> None:
    run_all_triplets()


# =============================================================================
# 9.  Analyzer -- the Python-native, .pkl-loading counterpart of v3's
#     `PrivateLatentAnalyzer`, scoped to one triplet's own output
#     directory and adapted for the 4 paired / 3 hub conditions.
# =============================================================================

class PrivateLatentAnalyzer:
    """
    Indexes the `*_analysis_results.pkl` files written by `run_triplet` for
    ONE triplet.

    Typical use
    -----------
        triplet = TRIPLETS_BY_LABEL["ORB ∩ OLF ∩ STR"]
        az = PrivateLatentAnalyzer(triplet, trial_type="cued_hit_long")
        az.load_all()
        az.summary()

        # 1b: direct CCA, no regression
        draws = az.get_pair_draws("yp010_220209", "OLF", "STR", condition="direct")

        # 4c: pCCA, nuisance = third region + behaviour
        draws = az.get_pair_draws("yp010_220209", "OLF", "STR", condition="region_behav")
        draws[0].z_i_lat[:, :, 0]      # draw 0's component-0 private latent
        draws[0].neuron_idx_i          # draw 0's sampled region_i neuron indices

        r1a = az.get_region_pca("yp010_220209", "OLF")                      # 1a
        r1a.draws[0].latent[:, :, 0]   # draw 0's component-0 direct-PCA latent
        r1a.draws[0].neuron_idx        # draw 0's sampled OLF neuron indices

        # 3a/3b: hub-orientation PCA, nuisance = third region only
        hub_draws = az.get_hub_pca_draws("yp010_220209", "OLF", "STR", condition="region_only")
        hub_draws[0].latent_network[:, :, 0]

        sel = az.get_selected_neurons(                                     # component 0 by default
            "yp010_220209", "OLF", "STR", side="i", condition="region_behav")
        sel.weight_threshold          # smallest selected neuron's pooled |weight|^2 contribution
        sel.neurons[0].neuron_idx     # a cumulative-weight-selected OLF neuron's FULL-population index
        sel.neurons[0].residual       # (n_trials, T) its residualized activity, averaged
                                       # across every draw in which it was sampled

        # same selection, but on the hub-orientation PCA (2a/2b/3a/3b/4a/4b)
        # and on 1a's direct region PCA -- both per-component too:
        hub_sel = az.get_hub_selected_neurons(
            "yp010_220209", "OLF", "STR", orientation="network", condition="region_only")
        r1a.selected_neurons[0].neurons[0].neuron_idx   # component 0's top 1a neuron
    """

    def __init__(
            self,
            triplet: TripletSpec,
            base_dir: Path = BASE_DIR,
            trial_type: str = TRIAL_TYPE,
            align_mode: str = ALIGN,
    ) -> None:
        self.triplet = triplet
        self.base_dir = Path(base_dir)
        self.trial_type = trial_type
        self.align_mode = align_mode
        self.results_dir = self.base_dir / out_subdir_name(triplet.slug, trial_type, align_mode)
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
            f"[PrivateLatentAnalyzer v4] triplet={self.triplet.label!r} "
            f"loaded {len(self.sessions)} session(s) from {self.results_dir}"
        )
        return self.sessions

    def get_pair_draws(
            self, session_name: str, region_i: str, region_j: str,
            condition: str = "region_behav",
    ) -> List[PrivateLatentPairDrawResult]:
        """All `N_SAMPLE_DRAWS` draws for one pair, one condition (one of
        `PAIR_CONDITIONS`: "direct"=1b, "behav_only"=2c,
        "region_only"=3c, "region_behav"=4c), one loaded session. Empty
        list if the pair/session/condition isn't present."""
        if condition not in PAIR_CONDITIONS:
            raise ValueError(f"condition must be one of {PAIR_CONDITIONS}, got {condition!r}")
        session = self.sessions.get(session_name)
        if session is None:
            return []
        pair_result = session.pairs.get(sort_pair_by_anatomy(region_i, region_j))
        if pair_result is None:
            return []
        return pair_result.draws.get(condition, [])

    def get_pair_draw(
            self, session_name: str, region_i: str, region_j: str,
            draw_idx: int, condition: str = "region_behav",
    ) -> Optional[PrivateLatentPairDrawResult]:
        """One specific draw (0-indexed) of one pair, one condition."""
        draws = self.get_pair_draws(session_name, region_i, region_j, condition)
        for d in draws:
            if d.draw_idx == draw_idx:
                return d
        return None

    def get_selected_neurons(
            self, session_name: str, region_i: str, region_j: str, side: str,
            condition: str = "region_behav", component: int = 0,
    ) -> Optional[SelectedNeuronSet]:
        """Top-pCCA-weight neuron selection (+ saved residualized activity)
        for one region side of one pair, one condition, ONE component,
        one loaded session -- selection is per-component (a neuron's
        weight on another component never affects this one's ranking),
        so pass `component` explicitly once N_COMPONENTS > 1. `side="i"`
        for region_i (the anatomically earlier of the pair, NOT
        necessarily whichever region name you passed as `region_i`) or
        `side="j"` for region_j. `None` if the pair/session/condition/
        component isn't present."""
        if side not in ("i", "j"):
            raise ValueError(f"side must be 'i' or 'j', got {side!r}")
        if condition not in PAIR_CONDITIONS:
            raise ValueError(f"condition must be one of {PAIR_CONDITIONS}, got {condition!r}")
        session = self.sessions.get(session_name)
        if session is None:
            return None
        pair_result = session.pairs.get(sort_pair_by_anatomy(region_i, region_j))
        if pair_result is None:
            return None
        table = pair_result.selected_neurons_i if side == "i" else pair_result.selected_neurons_j
        sets = table.get(condition, [])
        return sets[component] if 0 <= component < len(sets) else None

    def get_hub_selected_neurons(
            self, session_name: str, hub: str, partner: str, orientation: str,
            condition: str = "region_behav", component: int = 0,
    ) -> Optional[SelectedNeuronSet]:
        """Top-PCA-weight neuron selection (+ saved activity) for one hub
        orientation's network (`orientation="network"`, Parts 2a/3a/4a) or
        residual (`orientation="residual"`, Parts 2b/3b/4b) PCA weights,
        one condition, ONE component, one loaded session -- same
        per-component selection as `get_selected_neurons`, applied to the
        hub-orientation PCA instead of the paired CCA/pCCA. `None` if the
        pair/session/condition/component isn't present."""
        if orientation not in ("network", "residual"):
            raise ValueError(f"orientation must be 'network' or 'residual', got {orientation!r}")
        if condition not in HUB_CONDITIONS:
            raise ValueError(f"condition must be one of {HUB_CONDITIONS}, got {condition!r}")
        session = self.sessions.get(session_name)
        if session is None:
            return None
        region_i, region_j = sort_pair_by_anatomy(hub, partner)
        pair_result = session.hub_pca_pairs.get((region_i, region_j))
        if pair_result is None:
            return None
        table = pair_result.region_i_as_hub_selected if hub == region_i else pair_result.region_j_as_hub_selected
        bundle = table.get(condition)
        if bundle is None:
            return None
        sets = bundle.network if orientation == "network" else bundle.residual
        return sets[component] if 0 <= component < len(sets) else None

    def get_region_pca(self, session_name: str, region: str) -> Optional[RegionPCAResult]:
        """Group 1a: direct PCA, no regression -- `.draws` holds all
        `N_SAMPLE_DRAWS` draws, sampled from the FULL population like
        every other group; `.selected_neurons[component]` is that
        component's cumulative-weight neuron selection (see
        `RegionPCAResult`)."""
        session = self.sessions.get(session_name)
        if session is None:
            return None
        return session.region_pca_raw.get(region)

    def get_hub_pca_draws(
            self, session_name: str, hub: str, partner: str,
            condition: str = "region_behav",
    ) -> List[HubOrientationPCADrawResult]:
        """All `N_SAMPLE_DRAWS` draws for one (hub, partner) orientation,
        one condition (one of `HUB_CONDITIONS`: "behav_only"=2a/2b,
        "region_only"=3a/3b, "region_behav"=4a/4b), one loaded session.
        Canonicalizes internally, so the caller never has to reason about
        pair ordering."""
        if condition not in HUB_CONDITIONS:
            raise ValueError(f"condition must be one of {HUB_CONDITIONS}, got {condition!r}")
        session = self.sessions.get(session_name)
        if session is None:
            return []
        region_i, region_j = sort_pair_by_anatomy(hub, partner)
        pair_result = session.hub_pca_pairs.get((region_i, region_j))
        if pair_result is None:
            return []
        table = pair_result.region_i_as_hub_draws if hub == region_i else pair_result.region_j_as_hub_draws
        return table.get(condition, [])

    def get_hub_pca_draw(
            self, session_name: str, hub: str, partner: str, draw_idx: int,
            condition: str = "region_behav",
    ) -> Optional[HubOrientationPCADrawResult]:
        """One specific draw (0-indexed) of one hub orientation/condition."""
        draws = self.get_hub_pca_draws(session_name, hub, partner, condition)
        for d in draws:
            if d.draw_idx == draw_idx:
                return d
        return None

    def iter_pair_across_sessions(self, region_i: str, region_j: str):
        """Yield (session_name, PrivateLatentPairResult) for every loaded
        session that has this pair -- `.draws[condition]` holds all
        N_SAMPLE_DRAWS draws for that session/condition."""
        pair_key = sort_pair_by_anatomy(region_i, region_j)
        for session_name, session in self.sessions.items():
            if pair_key in session.pairs:
                yield session_name, session.pairs[pair_key]

    def summary(self) -> None:
        """Per-pair (all 4 conditions), per-region (1a) session-coverage
        tables, in the spirit of v3's own `summary`."""
        if not self.sessions:
            print("[PrivateLatentAnalyzer v4] no sessions loaded -- call load_all() first.")
            return
        pairs = triplet_pairs_with_third(self.triplet.regions)
        print(
            f"[PrivateLatentAnalyzer v4]  triplet={self.triplet.label}  "
            f"trial_type={self.trial_type}  sessions={len(self.sessions)}"
        )
        print("  Paired conditions (sessions covered / total draws):")
        for region_i, region_j, third_region in pairs:
            for cond in PAIR_CONDITIONS:
                n = 0
                nd = 0
                for session in self.sessions.values():
                    pr = session.pairs.get((region_i, region_j))
                    if pr is None:
                        continue
                    draws = pr.draws.get(cond, [])
                    if draws:
                        n += 1
                        nd += len(draws)
                print(
                    f"    {region_i:>7s} <-> {region_j:<7s} (third={third_region:<7s}) "
                    f"[{PAIR_CONDITION_LABELS[cond]:<45s}] : "
                    f"{n:3d}/{len(self.sessions)} sessions, {nd:4d} draws"
                )
        region_counts: Dict[str, int] = {}
        for session in self.sessions.values():
            for region in session.region_pca_raw:
                region_counts[region] = region_counts.get(region, 0) + 1
        print("  Group 1a -- region-level PCA, by region:")
        for region in self.triplet.regions:
            n = region_counts.get(region, 0)
            print(f"    {region:>7s} : {n:3d}/{len(self.sessions)} sessions")


if __name__ == "__main__":
    main()

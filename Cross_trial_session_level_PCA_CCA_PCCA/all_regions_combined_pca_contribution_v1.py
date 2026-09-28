#!/usr/bin/env python3
r"""
all_regions_combined_pca_contribution_v1.py
================================================================================

Standalone processing script (no plotting) computing, for every session of
one trial type, a single PCA fit on ALL recorded regions' neurons pooled
together into one population -- as opposed to `pCCA_all_regions_out_
behaviour_v4.py` ("v4"), which only ever runs PCA/CCA PER REGION or PER
PAIR, scoped to one region triplet at a time. Coding style and several
primitives (`_zscore_flat`, `pca_fit_and_project`, `load_region_spikes_full`,
`pool_region_groups`, `sample_paired_draws`, `_make_region_rng`,
`ANATOMICAL_ORDER`-based canonicalisation, the per-session pickle +
config-fingerprint caching contract) are copied verbatim or near-verbatim
from v4 -- this project's convention (see `PCA_hub_variance_gini_boxplots.
py`'s own docstring) is "primitives copied, not imported", so this script
stays independently auditable and never depends on v4's triplet-scoped
machinery. Everything specific to residualisation/CCA/pCCA (ridge fits,
k-fold CV, nuisance regions, behaviour regressors) is absent here on
purpose -- there is no regression of any kind in this script, only PCA.

--------------------------------------------------------------------------------
Part 1 -- Input all neurons (no subsampling, no plotting)
--------------------------------------------------------------------------------
For each session: pool EVERY recorded region's neurons (after the usual
STR + STRv + PAL -> "STR" grouping) into one big (T * n_trials, n_neurons_
total) matrix and fit ONE PCA on it, keeping the top `N_PCA_COMPONENTS` (3)
components -- `compute_combined_pca_for_session`. For each component
independently (never mixed across components), each region's contribution
is its neurons' share of that component's total squared-weight ("energy")
mass: `energy_fraction[region] = sum(W[region_neurons, k]**2) /
sum(W[:, k]**2)` -- see `_combined_pca_region_contributions`. Results are
pickled per session to `all_regions_combined_pca_part1_allneurons_
{trial_type}_{align_mode}_results/`.

--------------------------------------------------------------------------------
Part 2 -- Neuron-count-controlled subpool (10 draws, still no plotting)
--------------------------------------------------------------------------------
Same combined-PCA-and-contribution computation as Part 1, except each
region first has its neuron count controlled down to `TARGET_SAMPLE_SIZE`
(matching v4's own `TARGET_SAMPLE_SIZE`/`sample_paired_draws` scheme, so a
region with more recorded neurons than another can't dominate the combined
pool purely by numbers), and the whole thing is repeated for
`N_SAMPLE_DRAWS` (10) independent random subpools per session --
`compute_combined_pca_sampled_for_session`. Each component's region-
contribution energy is pooled ACROSS the 10 draws, per neuron, before the
per-region share is taken -- `_pool_region_contributions_across_draws` --
exactly the cross-draw-but-never-cross-component pooling convention v4's
own `_select_cumulative_weight_neurons` uses, just without that function's
additional cumulative-threshold cutoff (see "Design decisions" below).
Results are pickled per session to `all_regions_combined_pca_part2_
sampled_{trial_type}_{align_mode}_results/`.

--------------------------------------------------------------------------------
Design decisions confirmed with the user
--------------------------------------------------------------------------------
* Region-contribution metric: each region's RAW share of total squared-
  weight energy for a component, over every neuron entering that fit --
  NOT a `CUMULATIVE_WEIGHT_FRACTION`-thresholded top-neuron subset. (v4's
  `CUMULATIVE_WEIGHT_FRACTION`/cumulative-cutoff selection is therefore not
  used anywhere in this script.)
* No plotting in either part -- both are pure data-processing/pickling
  passes.
* Session discovery: every `*_analysis_results.mat` file found under
  `BASE_DIR / mat_subdir_name(TRIAL_TYPE, ALIGN)` (same glob convention
  `Useful_definition.py`'s `OxfordAdvancedAnalyzer` uses), not a triplet-
  scoped or hand-curated session list -- a session that fails to load, or
  ends up with fewer than `MIN_REGIONS_FOR_COMBINED` regions after
  pooling, is skipped with a printed reason.

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

# `ANATOMICAL_ORDER` / `safe_array` copied from `Useful_definition.py` rather
# than imported -- that module also pulls in seaborn/sklearn/pandas (unused
# here; this script does no plotting or classification), and per this
# project's own "primitives copied, not imported" convention (see
# `PCA_hub_variance_gini_boxplots.py`'s docstring) a standalone script keeps
# its own copy of the handful of primitives it actually needs.
ANATOMICAL_ORDER: List[str] = [
    "mPFC", "ORB", "MOp", "MOs", "OLF",
    "STR", "STRv",
    "MD", "LP", "VALVM", "VPMPO", "ILM",
    "HY",
]


def safe_array(x) -> Optional[np.ndarray]:
    """Cast a MATLAB-loaded object to a numpy array, returning None on failure."""
    try:
        if x is None:
            return None
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS.
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

# ---- PCA dimensionality -- explicitly 3 per this script's own spec (v4's
#      own N_PCA_COMPONENTS default is 2; not reused here). -----------------
N_PCA_COMPONENTS: int = 3

TIME_RANGE_S: Tuple[float, float] = ALIGNMENT_WINDOWS_S[ALIGN]

# ---- Regime toggles (match v4's own defaults) ------------------------------
SUBTRACT_PSTH: bool = False
SHUFFLE_TRIALS: bool = False

# ---- Part 2's neuron-count control -- same target/draw-count vocabulary as
#      v4's own TARGET_SAMPLE_SIZE / N_SAMPLE_DRAWS, sampled with v4's own
#      `sample_paired_draws` (copied below). -------------------------------
TARGET_SAMPLE_SIZE: int = 40
N_SAMPLE_DRAWS: int = 10
SAMPLE_RNG_SEED: int = 20260928

# ---- A session needs at least this many recorded regions (after STR
#      pooling) for "all regions combined" to mean anything. ---------------
MIN_REGIONS_FOR_COMBINED: int = 2


def mat_subdir_name(trial_type: str, align_mode: str = ALIGN) -> str:
    """MATLAB-pipeline session-region-data source folder (matches v4)."""
    return f"{trial_type}_{align_mode}_results"


def out_subdir_name_part1(trial_type: str, align_mode: str = ALIGN) -> str:
    return f"all_regions_combined_pca_part1_allneurons_{trial_type}_{align_mode}_results"


def out_subdir_name_part2(trial_type: str, align_mode: str = ALIGN) -> str:
    return f"all_regions_combined_pca_part2_sampled_{trial_type}_{align_mode}_results"


# =============================================================================
# 2.  Anatomical canonicalisation -- copied from v4 (adds "HIPP", absent from
#     Useful_definition.ANATOMICAL_ORDER, so triplet-derived hub sessions'
#     region stay orderable here too).
# =============================================================================

ANATOMICAL_ORDER_LOCAL: List[str] = list(ANATOMICAL_ORDER) + [
    r for r in ("HIPP",) if r not in ANATOMICAL_ORDER
]


def get_anatomical_index(region: str) -> int:
    try:
        return ANATOMICAL_ORDER_LOCAL.index(region)
    except ValueError:
        return len(ANATOMICAL_ORDER_LOCAL)


# ---- Report/display region name -> region name(s) in the session .mat
#      files -- copied verbatim from v4 (only the pooling side is needed
#      here; the report-display-name side of v4's REGION_MAPPING is not,
#      since this script has no hub-report-derived triplet list). ----------
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


# =============================================================================
# 3.  Core PCA primitive + neuron-sampling primitives -- copied verbatim
#     from v4. No ridge/CCA/pCCA machinery here: this script never
#     regresses anything out.
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


def latent_projections(X_flat: np.ndarray, W: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """Project flattened activity onto every column of a loading matrix at
    once -> (n_trials, T, K)."""
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
    its own loadings. Copied verbatim from v4."""
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


def _make_region_rng(session_name: str, region: str, role: str) -> np.random.Generator:
    """Deterministic RNG for one (session, region, role) key -- copied from
    v4. `role="combined_pca"` here, a role name v4 never uses, so draws
    never collide with v4's own "paired"/"nuisance"/"region_pca" roles even
    if SAMPLE_RNG_SEED happened to match."""
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
    values in `[0, n_full)`) -- copied verbatim from v4. The first
    `m = min(n_draws, n_full // target)` draws are cut, zero-overlapping,
    from one random permutation; the rest are independent uniform draws.
    If `n_full < target`, every draw uses the full (capped) population."""
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


# =============================================================================
# 4.  Data loading -- `load_region_spikes_full` / `pool_region_groups`
#     copied verbatim from v4. `load_region_spikes` (the `selected_neurons`-
#     filtered variant) is NOT needed here -- both parts of this script draw
#     from the FULL recorded population, same as v4's Part 2 pool.
# =============================================================================

def load_region_spikes_full(
        session_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], int, int]:
    """Load per-region spike tensors AND per-neuron subregion labels for
    EVERY recorded neuron of one session -- bypasses `selected_neurons`
    entirely (copied verbatim from v4)."""
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
    into ONE region named after the group's key (copied verbatim from v4)."""
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


def discover_sessions(trial_type: str = TRIAL_TYPE, align_mode: str = ALIGN) -> List[str]:
    """Every session with a source .mat file under this trial type's
    directory, sorted -- same glob convention `Useful_definition.py`'s
    `OxfordAdvancedAnalyzer.load_all` uses."""
    mat_dir = BASE_DIR / mat_subdir_name(trial_type, align_mode)
    files = sorted(mat_dir.glob("*_analysis_results.mat"))
    return [f.stem.replace("_analysis_results", "") for f in files]


# =============================================================================
# 5.  Result containers.
# =============================================================================

@dataclass
class RegionContribution:
    """One region's share of one PCA component's weight energy, for one
    combined-PCA fit (Part 1's single fit, or one of Part 2's draws / its
    cross-draw pooled summary). `energy_fraction` is
    `sum(|W[region_neurons, k]|**2) / sum(|W[:, k]|**2)` -- each
    component computed independently, never mixed with any other
    component. `weight_mass_fraction` is the same ratio using `|W|`
    (unsquared) instead of `|W|**2`, kept alongside for interpretability."""
    region: str
    n_neurons: int
    energy_fraction: float
    weight_mass_fraction: float


@dataclass
class CombinedPCAResult:
    """ONE all-regions-combined PCA fit: Part 1's single full-population
    fit (`draw_idx=0`), or one of Part 2's `N_SAMPLE_DRAWS` neuron-count-
    controlled subpool fits."""
    draw_idx: int
    regions: List[str]                                    # concatenation order (anatomical)
    region_n_neurons: Dict[str, int]                       # neurons per region IN THIS FIT
    region_neuron_idx: Dict[str, np.ndarray]                # per region: indices into that region's FULL neuron axis
    W: np.ndarray                                          # (n_neurons_total, K)
    explained_variance_ratio: np.ndarray                    # (K,)
    latent: np.ndarray                                      # (n_trials, T, K)
    mean: np.ndarray                                        # (1, n_neurons_total)
    region_contributions: List[List[RegionContribution]] = field(default_factory=list)  # [component][region]


@dataclass
class CombinedPCASessionResult:
    """Part 1 (all recorded neurons, single fit) result for one session."""
    session: str
    trial_type: str
    time_vec: np.ndarray
    n_trials: int
    T: int
    align_mode: str
    config: Dict[str, Any]
    regions: List[str]
    region_subregion_labels_full: Dict[str, List[str]]
    combined_pca: CombinedPCAResult


@dataclass
class CombinedPCASampledSessionResult:
    """Part 2 (neuron-count-controlled, `N_SAMPLE_DRAWS` draws) result for
    one session. `region_contributions_pooled` is the SAME per-component
    region-contribution metric as `CombinedPCAResult.region_contributions`,
    just with each neuron's squared-weight energy pooled ACROSS every draw
    that sampled it (per component, independently) before regions' shares
    are taken -- see `_pool_region_contributions_across_draws`."""
    session: str
    trial_type: str
    time_vec: np.ndarray
    n_trials: int
    T: int
    align_mode: str
    config: Dict[str, Any]
    regions: List[str]
    region_subregion_labels_full: Dict[str, List[str]]
    draws: List[CombinedPCAResult] = field(default_factory=list)
    region_contributions_pooled: List[List[RegionContribution]] = field(default_factory=list)


def config_fingerprint_part1() -> Dict[str, Any]:
    """Snapshot of every parameter that changes Part 1's numerical result,
    stored in each session's pickle so a stale cache is detected, not
    silently reused (same caching contract as v4)."""
    return dict(
        part="part1_all_neurons",
        align_mode=ALIGN,
        n_pca_components=N_PCA_COMPONENTS,
        anatomical_order=tuple(ANATOMICAL_ORDER_LOCAL),
        region_groups={k: tuple(v) for k, v in REGION_GROUPS.items()},
        time_range_s=tuple(TIME_RANGE_S),
        subtract_psth=SUBTRACT_PSTH,
        shuffle_trials=SHUFFLE_TRIALS,
        min_regions_for_combined=MIN_REGIONS_FOR_COMBINED,
        schema_version=1,
    )


def config_fingerprint_part2() -> Dict[str, Any]:
    d = config_fingerprint_part1()
    d.update(
        part="part2_sampled_controlled",
        target_sample_size=TARGET_SAMPLE_SIZE,
        n_sample_draws=N_SAMPLE_DRAWS,
        sample_rng_seed=SAMPLE_RNG_SEED,
    )
    return d


# =============================================================================
# 6.  Region-contribution computation.
# =============================================================================

def _build_region_bounds(
        regions: List[str],
        region_n_neurons: Dict[str, int],
) -> List[Tuple[str, int, int]]:
    """`regions`' concatenation order -> `[(region, start, end), ...]`
    column-index ranges into the combined `W`/`X_combined` matrix."""
    bounds: List[Tuple[str, int, int]] = []
    start = 0
    for r in regions:
        n_r = region_n_neurons[r]
        bounds.append((r, start, start + n_r))
        start += n_r
    return bounds


def _region_contribution_from_energy(
        region_energy: Dict[str, float],
        region_mass: Dict[str, float],
        region_n: Dict[str, int],
) -> List[RegionContribution]:
    total_energy = sum(region_energy.values())
    total_mass = sum(region_mass.values())
    out = [
        RegionContribution(
            region=r,
            n_neurons=region_n[r],
            energy_fraction=(region_energy[r] / total_energy) if total_energy > 0 else float("nan"),
            weight_mass_fraction=(region_mass[r] / total_mass) if total_mass > 0 else float("nan"),
        )
        for r in region_energy
    ]
    out.sort(key=lambda rc: get_anatomical_index(rc.region))
    return out


def _combined_pca_region_contributions(
        W: np.ndarray,
        region_bounds: List[Tuple[str, int, int]],
) -> List[List[RegionContribution]]:
    """Per component (independently, never mixed across components), each
    region's share of that component's squared-weight energy, computed
    over every neuron in `W` -- no cumulative-threshold cutoff (see the
    module docstring's "Design decisions")."""
    K = W.shape[1]
    result: List[List[RegionContribution]] = []
    for k in range(K):
        w_col = W[:, k].astype(np.float64)
        region_energy: Dict[str, float] = {}
        region_mass: Dict[str, float] = {}
        region_n: Dict[str, int] = {}
        for region, start, end in region_bounds:
            seg = w_col[start:end]
            region_energy[region] = float(np.sum(seg ** 2))
            region_mass[region] = float(np.sum(np.abs(seg)))
            region_n[region] = int(end - start)
        result.append(_region_contribution_from_energy(region_energy, region_mass, region_n))
    return result


def _pool_region_contributions_across_draws(
        draws: List[CombinedPCAResult],
) -> List[List[RegionContribution]]:
    """Part 2's cross-draw pooling: for each component (independently),
    pool every (draw, neuron) squared-weight entry by `(region,
    full-population neuron index)` -- a neuron sampled in several draws has
    its energy SUMMED across them -- then take each region's share of the
    pooled total. `n_neurons` in the returned `RegionContribution` is the
    count of DISTINCT neurons from that region that appeared in at least
    one draw (not a per-draw count, since draws sample different neurons).
    Mirrors v4's `_select_cumulative_weight_neurons` cross-draw-but-never-
    cross-component pooling, minus that function's cumulative cutoff."""
    if not draws:
        return []
    K = draws[0].W.shape[1]

    result: List[List[RegionContribution]] = []
    for k in range(K):
        per_neuron_energy: Dict[Tuple[str, int], float] = {}
        per_neuron_mass: Dict[Tuple[str, int], float] = {}
        for d in draws:
            bounds = _build_region_bounds(d.regions, d.region_n_neurons)
            w_col = d.W[:, k].astype(np.float64)
            for region, start, end in bounds:
                idx_full = d.region_neuron_idx[region]
                seg = w_col[start:end]
                for pos in range(seg.shape[0]):
                    key = (region, int(idx_full[pos]))
                    w = float(seg[pos])
                    per_neuron_energy[key] = per_neuron_energy.get(key, 0.0) + w * w
                    per_neuron_mass[key] = per_neuron_mass.get(key, 0.0) + abs(w)

        regions_all = sorted({d_r for d in draws for d_r in d.regions}, key=get_anatomical_index)
        region_energy = {r: 0.0 for r in regions_all}
        region_mass = {r: 0.0 for r in regions_all}
        region_n_distinct = {r: 0 for r in regions_all}
        distinct_neurons: Dict[str, set] = {r: set() for r in regions_all}
        for (region, nu), e in per_neuron_energy.items():
            region_energy[region] += e
            distinct_neurons[region].add(nu)
        for (region, nu), m in per_neuron_mass.items():
            region_mass[region] += m
        for r in regions_all:
            region_n_distinct[r] = len(distinct_neurons[r])

        result.append(_region_contribution_from_energy(region_energy, region_mass, region_n_distinct))
    return result


def _fit_combined_pca(
        regions: List[str],
        region_X: Dict[str, np.ndarray],
        region_idx: Dict[str, np.ndarray],
        n_trials: int,
        T: int,
        draw_idx: int,
        n_components: int = N_PCA_COMPONENTS,
) -> CombinedPCAResult:
    """Concatenate every region's (already flattened + z-scored, already
    subsampled if Part 2) neuron columns in `regions`' order, fit ONE PCA
    on the combined pool, and compute the per-component region-
    contribution breakdown. Shared by both Part 1 (called once, with
    `region_idx` = each region's full identity range) and Part 2 (called
    once per draw, with `region_idx` = that draw's sampled indices)."""
    X_combined = np.concatenate([region_X[r] for r in regions], axis=1)
    W, evr, mean, latent = pca_fit_and_project(X_combined, n_trials, T, n_components)

    region_n_neurons = {r: region_X[r].shape[1] for r in regions}
    bounds = _build_region_bounds(regions, region_n_neurons)
    contributions = _combined_pca_region_contributions(W, bounds)

    return CombinedPCAResult(
        draw_idx=draw_idx,
        regions=list(regions),
        region_n_neurons=region_n_neurons,
        region_neuron_idx={r: region_idx[r].astype(np.int64) for r in regions},
        W=W.astype(np.float32),
        explained_variance_ratio=evr.astype(np.float64),
        latent=latent.astype(np.float32),
        mean=mean.astype(np.float32),
        region_contributions=contributions,
    )


# =============================================================================
# 7.  Per-session computation -- Part 1 and Part 2.
# =============================================================================

def _load_and_prepare_session(
        session_name: str,
        mat_dir: Path,
) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, List[str]], int, int, np.ndarray, List[str]]]:
    """Shared loading/pooling/z-scoring step for both parts. Returns
    `(region_flat_full, region_subregion_labels_full, n_trials, T,
    time_vec, regions)` or `None` if the session can't be processed.
    `region_flat_full[r]` is `r`'s FULL recorded population, flattened and
    z-scored -- `(T * n_trials, n_neurons_r)`. `regions` is anatomically
    sorted."""
    session_file = mat_dir / f"{session_name}_analysis_results.mat"
    if not session_file.exists():
        print(f"  [skip] {session_name}: source file not found -> {session_file}")
        return None

    region_spikes_full, region_labels_full, n_trials, T = load_region_spikes_full(str(session_file))
    if not region_spikes_full:
        print(f"  [skip] {session_name}: no regions loaded")
        return None

    region_spikes_full, region_labels_full = pool_region_groups(region_spikes_full, region_labels_full)

    regions = sorted(region_spikes_full.keys(), key=get_anatomical_index)
    if len(regions) < MIN_REGIONS_FOR_COMBINED:
        print(
            f"  [skip] {session_name}: only {len(regions)} region(s) recorded "
            f"(< {MIN_REGIONS_FOR_COMBINED}) -- nothing to combine"
        )
        return None

    time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)

    region_flat_full = {
        r: _zscore_flat(region_spikes_full[r], subtract_psth=SUBTRACT_PSTH, shuffle_trials=SHUFFLE_TRIALS)
        for r in regions
    }
    return region_flat_full, region_labels_full, n_trials, T, time_vec, regions


def compute_combined_pca_for_session(
        session_name: str,
        trial_type: str = TRIAL_TYPE,
        mat_dir: Optional[Path] = None,
        align_mode: str = ALIGN,
) -> Optional[CombinedPCASessionResult]:
    """Part 1: one all-regions-combined PCA fit on this session's FULL
    recorded population (no subsampling)."""
    mat_dir = mat_dir if mat_dir is not None else (BASE_DIR / mat_subdir_name(trial_type, align_mode))
    prepared = _load_and_prepare_session(session_name, mat_dir)
    if prepared is None:
        return None
    region_flat_full, region_labels_full, n_trials, T, time_vec, regions = prepared

    region_idx_full = {
        r: np.arange(region_flat_full[r].shape[1], dtype=np.int64) for r in regions
    }
    combined = _fit_combined_pca(regions, region_flat_full, region_idx_full, n_trials, T, draw_idx=0)

    print(
        f"  [{session_name}] Part 1 combined PCA: {len(regions)} regions "
        f"({', '.join(regions)}), {combined.W.shape[0]} neurons total, "
        f"K={combined.W.shape[1]} components (n_trials={n_trials}, T={T})"
    )

    return CombinedPCASessionResult(
        session=session_name, trial_type=trial_type, time_vec=time_vec.astype(np.float64),
        n_trials=n_trials, T=T, align_mode=align_mode,
        config=config_fingerprint_part1(), regions=regions,
        region_subregion_labels_full=region_labels_full,
        combined_pca=combined,
    )


def compute_combined_pca_sampled_for_session(
        session_name: str,
        trial_type: str = TRIAL_TYPE,
        mat_dir: Optional[Path] = None,
        align_mode: str = ALIGN,
) -> Optional[CombinedPCASampledSessionResult]:
    """Part 2: `N_SAMPLE_DRAWS` all-regions-combined PCA fits, each on a
    subpool where every region contributes at most `TARGET_SAMPLE_SIZE`
    neurons (controlling for unequal recorded neuron counts across
    regions), plus the cross-draw pooled region-contribution summary."""
    mat_dir = mat_dir if mat_dir is not None else (BASE_DIR / mat_subdir_name(trial_type, align_mode))
    prepared = _load_and_prepare_session(session_name, mat_dir)
    if prepared is None:
        return None
    region_flat_full, region_labels_full, n_trials, T, time_vec, regions = prepared

    region_draws_idx: Dict[str, List[np.ndarray]] = {
        r: sample_paired_draws(
            region_flat_full[r].shape[1],
            _make_region_rng(session_name, r, "combined_pca"),
            target=TARGET_SAMPLE_SIZE, n_draws=N_SAMPLE_DRAWS,
        )
        for r in regions
    }

    draws: List[CombinedPCAResult] = []
    for k in range(N_SAMPLE_DRAWS):
        region_X_k = {r: region_flat_full[r][:, region_draws_idx[r][k]] for r in regions}
        region_idx_k = {r: region_draws_idx[r][k] for r in regions}
        draws.append(_fit_combined_pca(regions, region_X_k, region_idx_k, n_trials, T, draw_idx=k))

    region_contributions_pooled = _pool_region_contributions_across_draws(draws)

    print(
        f"  [{session_name}] Part 2 combined PCA: {len(regions)} regions "
        f"({', '.join(regions)}), target={TARGET_SAMPLE_SIZE} neurons/region, "
        f"{N_SAMPLE_DRAWS} draws, K={N_PCA_COMPONENTS} components "
        f"(n_trials={n_trials}, T={T})"
    )

    return CombinedPCASampledSessionResult(
        session=session_name, trial_type=trial_type, time_vec=time_vec.astype(np.float64),
        n_trials=n_trials, T=T, align_mode=align_mode,
        config=config_fingerprint_part2(), regions=regions,
        region_subregion_labels_full=region_labels_full,
        draws=draws,
        region_contributions_pooled=region_contributions_pooled,
    )


# =============================================================================
# 8.  Orchestration -- one pickle per session per part, same
#     config-fingerprint caching contract as v4's `run_triplet`.
# =============================================================================

def run_part1(
        trial_type: str = TRIAL_TYPE,
        sessions: Optional[List[str]] = None,
        overwrite: bool = False,
        align_mode: str = ALIGN,
) -> None:
    sessions = list(sessions) if sessions is not None else discover_sessions(trial_type, align_mode)
    mat_dir = BASE_DIR / mat_subdir_name(trial_type, align_mode)
    out_dir = BASE_DIR / out_subdir_name_part1(trial_type, align_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Part 1  |  all-regions-combined PCA -- ALL recorded neurons, no subsampling")
    print(f"  trial_type : {trial_type}")
    print(f"  align_mode : {align_mode}")
    print(f"  source dir : {mat_dir}")
    print(f"  output dir : {out_dir}")
    print(f"  sessions   : {len(sessions)}")
    print(f"  pca comps  : {N_PCA_COMPONENTS}")
    print("=" * 70)

    current_config = config_fingerprint_part1()
    n_written = n_cached = n_skipped = 0

    for idx, session_name in enumerate(sessions, 1):
        print(f"\n[{idx}/{len(sessions)}] {session_name}")
        out_path = out_dir / f"{session_name}_analysis_results.pkl"

        if out_path.exists() and not overwrite:
            try:
                with open(out_path, "rb") as fh:
                    cached: CombinedPCASessionResult = pickle.load(fh)
                if cached.config == current_config:
                    print("  cached, config unchanged -> skip")
                    n_cached += 1
                    continue
                print("  cached copy has a stale config -> recomputing")
            except Exception as exc:
                print(f"  cached copy unreadable ({exc}) -> recomputing")

        try:
            result = compute_combined_pca_for_session(session_name, trial_type, mat_dir, align_mode=align_mode)
        except Exception as exc:
            print(f"  [ERROR] {session_name}: {exc}")
            n_skipped += 1
            continue

        if result is None:
            n_skipped += 1
            continue

        with open(out_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  saved -> {out_path}")
        n_written += 1

    print("\n" + "=" * 70)
    print(f"Done Part 1. {n_written} written, {n_cached} cached, {n_skipped} skipped (of {len(sessions)} sessions).")
    print("=" * 70)


def run_part2(
        trial_type: str = TRIAL_TYPE,
        sessions: Optional[List[str]] = None,
        overwrite: bool = False,
        align_mode: str = ALIGN,
) -> None:
    sessions = list(sessions) if sessions is not None else discover_sessions(trial_type, align_mode)
    mat_dir = BASE_DIR / mat_subdir_name(trial_type, align_mode)
    out_dir = BASE_DIR / out_subdir_name_part2(trial_type, align_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Part 2  |  all-regions-combined PCA -- neuron-count-controlled subpool")
    print(f"  trial_type : {trial_type}")
    print(f"  align_mode : {align_mode}")
    print(f"  source dir : {mat_dir}")
    print(f"  output dir : {out_dir}")
    print(f"  sessions   : {len(sessions)}")
    print(f"  pca comps  : {N_PCA_COMPONENTS}")
    print(f"  sampling   : target={TARGET_SAMPLE_SIZE} neurons/region, {N_SAMPLE_DRAWS} draws/session")
    print("=" * 70)

    current_config = config_fingerprint_part2()
    n_written = n_cached = n_skipped = 0

    for idx, session_name in enumerate(sessions, 1):
        print(f"\n[{idx}/{len(sessions)}] {session_name}")
        out_path = out_dir / f"{session_name}_analysis_results.pkl"

        if out_path.exists() and not overwrite:
            try:
                with open(out_path, "rb") as fh:
                    cached: CombinedPCASampledSessionResult = pickle.load(fh)
                if cached.config == current_config:
                    print("  cached, config unchanged -> skip")
                    n_cached += 1
                    continue
                print("  cached copy has a stale config -> recomputing")
            except Exception as exc:
                print(f"  cached copy unreadable ({exc}) -> recomputing")

        try:
            result = compute_combined_pca_sampled_for_session(
                session_name, trial_type, mat_dir, align_mode=align_mode)
        except Exception as exc:
            print(f"  [ERROR] {session_name}: {exc}")
            n_skipped += 1
            continue

        if result is None:
            n_skipped += 1
            continue

        with open(out_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  saved -> {out_path}")
        n_written += 1

    print("\n" + "=" * 70)
    print(f"Done Part 2. {n_written} written, {n_cached} cached, {n_skipped} skipped (of {len(sessions)} sessions).")
    print("=" * 70)


def main() -> None:
    sessions = discover_sessions(TRIAL_TYPE, ALIGN)
    run_part1(TRIAL_TYPE, sessions=sessions, align_mode=ALIGN)
    run_part2(TRIAL_TYPE, sessions=sessions, align_mode=ALIGN)


if __name__ == "__main__":
    main()

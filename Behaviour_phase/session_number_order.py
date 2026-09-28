"""
filter_sessions_by_region_pairs.py
====================================
Scans all Oxford dataset session files across every condition subdirectory
and, for the full set of regions discovered in the data, enumerates every
possible pairwise (r = 2) and triplet (r = 3) region combination. For each
combination it computes the number of sessions in which *all* regions in
that combination are co-recorded, then ranks the combinations from most
to fewest qualifying sessions.

A session qualifies for a combination if *all* regions in that combination
appear as keys in ``pca_results`` within its .mat file, which is the
authoritative per-region record.

Output
------
- Console summary: ranked table (region combination, session count) for
  pairs and for triplets, ordered top (most sessions) to bottom (fewest).
- ``session_region_pairs_report.txt``  –  machine-readable flat text report,
  including full session-name listings, written to the same directory as
  this script.
- ``session_triplets_usable_barplots.png`` – for each minimum
  neurons-per-region setting in ``MIN_NEURON_THRESHOLDS``, a bar chart of
  which region triplets give how many "usable" sessions (i.e. every region
  in the triplet has at least that many neurons in that session).

Usage
-----
    python filter_sessions_by_region_pairs.py
"""

from __future__ import annotations

import itertools
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import mat73  # pip install mat73
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# ── Configuration  ────────────────────────────────────────────────────────────
# =============================================================================

BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")

# Condition label  →  subdirectory name
RESULTS_SUBDIRS: Dict[str, str] = {
    "cued_hit_long":   "cued_hit_long_default_move_onset_results",
    # "spont_hit_long":  "sessions_spont_hit_long_results",
    # "spont_miss_long": "sessions_spont_miss_long_results",
}

# Combination sizes to enumerate (2 = pairs, 3 = triplets). Extend freely,
# e.g. add 4 for quadruplets — the machinery below is size-agnostic.
COMBINATION_SIZES: Tuple[int, ...] = (3,4)

# Optional: regions that MUST appear in every enumerated combination.
# Leave empty for a fully unconstrained scan. Example: ("preM Ctx",)
# restricts every pair/triplet to those containing preM Ctx (raw code
# "MOs"), replicating the old script's implicit convention.
REQUIRED_REGIONS: Tuple[str, ...] = ()

# Optional: regions to drop from the discovered region set before
# enumeration (mirrors the EXCLUDED_REGIONS blacklist convention used in
# the pCCA pivot-ablation scripts). Leave empty to include everything found.
EXCLUDED_REGIONS: Tuple[str, ...] = (["analysis_timestamp","config","session_name","fiber","other"])

# Raw ``pca_results`` region codes -> readable region labels, matching the
# naming used by ``tcell_cellmetrics_struct.mat`` (``brainRegion_names2``,
# see ``tcell_session_triplets.py``) so both scripts refer to regions the
# same way. 'Striatum' merges the dorsal ('STR') and ventral ('STRv')
# raw codes into a single combined region.
REGION_MAPPING: Dict[str, str | List[str]] = {
    'M1 Ctx':       'MOp',
    'preM Ctx':     'MOs',
    'OFC':          'ORB',
    'mPFC':         'mPFC',
    'motorThal':    'VALVM',
    'sensorThal':   'VPMPO',
    'interThal':    'ILM',
    'MDThal':       'MD',
    'Pulvinar':     'LP',
    'Striatum':     ['STR', 'STRv', 'PAL'], # dorsal + ventral combined
    'Hippocampus':  'HIPP',
    'Olf area':     'OLF',
    'Hypothalamus': 'HY',
}

# Inverse of REGION_MAPPING: raw region code -> readable label.
RAW_TO_NICE_REGION: Dict[str, str] = {
    raw: nice
    for nice, raw_or_list in REGION_MAPPING.items()
    for raw in (raw_or_list if isinstance(raw_or_list, list) else [raw_or_list])
}


def _merge_region_count(a: int, b: int) -> int:
    """Combine two neuron counts for regions collapsed onto the same
    readable label (e.g. STR + STRv -> Striatum). -1 means "unknown"."""
    if a < 0 and b < 0:
        return -1
    return max(a, 0) + max(b, 0)


def apply_region_mapping(raw_counts: Dict[str, int]) -> Dict[str, int]:
    """Translate raw pca_results region codes to readable labels, merging
    counts for any raw codes that collapse onto the same label."""
    mapped: Dict[str, int] = {}
    for region, n in raw_counts.items():
        nice = RAW_TO_NICE_REGION.get(region, region)
        if nice in mapped:
            mapped[nice] = _merge_region_count(mapped[nice], n)
        else:
            mapped[nice] = n
    return mapped

# Minimum neuron count required in EVERY region of a combination for a
# session to be listed as available in the console table / text report
# (e.g. 50 -> a triplet only lists sessions where all three regions have
# >= 50 neurons, and triplets with no such session are hidden).
# ``None`` = no neuron filter, only require the regions to be co-recorded.
MIN_NEURONS_PER_REGION: int | None = 40

# All generated figures and text reports are written to this folder.
OUTPUT_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset/threshold_analysis_results")

# Output report path
REPORT_PATH = OUTPUT_DIR / "session_region_pairs_report.txt"




# Minimum neurons-per-region settings to sweep for the "usable sessions"
# triplet bar plots.
MIN_NEURON_THRESHOLDS: Tuple[int, ...] = (40,50)

# Only the top-N triplets (by usable-session count) are drawn per subplot,
# to keep the bar charts readable.
TOP_N_TRIPLETS_PLOT = 30

BARPLOT_PATH = OUTPUT_DIR / "session_triplets_usable_barplots.png"

# Hub-region view: one column per hub region, each showing the top-N
# triplets that contain that hub, ranked by number of usable sessions.
# Use the readable labels from REGION_MAPPING (keys), e.g. 'M1 Ctx' = MOp,
# 'preM Ctx' = MOs, 'motorThal' = VALVM, 'sensorThal' = VPMPO.
HUB_REGIONS: Tuple[str, ...] = ("Striatum","M1 Ctx")
TOP_N_HUB_TRIPLETS = 30
# Minimum neurons in EVERY region of the triplet (None = co-recorded only).
HUB_MIN_NEURONS: int | None = MIN_NEURONS_PER_REGION
HUB_BARPLOT_PATH = OUTPUT_DIR / "session_triplets_hub_barplots.png"
HUB_REPORT_PATH = OUTPUT_DIR / "session_triplets_hub_report.txt"


# =============================================================================
# ── Region extraction  ────────────────────────────────────────────────────────
# =============================================================================

def _coerce_int(value: object) -> int | None:
    """Best-effort conversion of a scalar (possibly a 0-d numpy array) to int."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _region_data_neuron_counts(data: dict) -> Dict[str, int]:
    """
    Pull true per-region neuron counts from ``region_data.regions[region]
    .n_neurons`` (the raw recording count, unaffected by any PCA/CCA
    subsampling to a fixed number of target neurons).
    """
    region_data = data.get("region_data")
    if not isinstance(region_data, dict):
        return {}
    regions = region_data.get("regions")
    if not isinstance(regions, dict):
        return {}

    counts: Dict[str, int] = {}
    for region, region_struct in regions.items():
        if isinstance(region_struct, dict):
            n_neurons = _coerce_int(region_struct.get("n_neurons"))
            if n_neurons is not None:
                counts[region] = n_neurons
    return counts


def extract_region_neuron_counts(mat_path: Path) -> Dict[str, int]:
    """
    Return a mapping of region label -> neuron count for a single session
    .mat file. Region *membership* (which regions qualify a session for a
    combination) is taken from ``pca_results`` keys when present (falling
    back to ``region_data.regions`` keys for files without PCA/CCA results);
    the neuron *count* itself is taken from
    ``region_data.regions[region].n_neurons`` — the true recorded neuron
    count, since ``pca_results[region].n_neurons`` is capped at the fixed
    number of neurons subsampled for PCA/CCA (e.g. 50).
    """
    try:
        data = mat73.loadmat(str(mat_path))
    except Exception as exc:
        print(f"  [WARN] Could not load {mat_path.name}: {exc}", file=sys.stderr)
        return {}

    true_counts = _region_data_neuron_counts(data)

    # ── Primary source of region membership: pca_results keys ───────────────
    pca = data.get("pca_results")
    if isinstance(pca, dict) and pca:
        raw_counts = {region: true_counts.get(region, -1) for region in pca.keys()}
        return apply_region_mapping(raw_counts)

    # ── Newer pipeline files (v4.0+) have no pca_results / cca_results: the
    #    region_data.regions keys are the membership record ───────────────────
    if true_counts:
        return apply_region_mapping(true_counts)

    # ── Fallback: harvest membership from cca pair entries ───────────────────
    regions: Set[str] = set()
    try:
        cca = data.get("cca_results")
        if not isinstance(cca, dict):
            return {}
        pair_results = cca.get("pair_results", [])
        if not isinstance(pair_results, (list, tuple)):
            return {}
        for pr in pair_results:
            if not isinstance(pr, dict):
                continue
            for field in ("region_i", "region_j"):
                raw = pr.get(field)
                if isinstance(raw, str) and raw:
                    regions.add(raw)
                elif isinstance(raw, (list, tuple)) and raw:
                    regions.add(str(raw[0]))
    except Exception as exc:
        print(f"  [WARN] CCA fallback failed for {mat_path.name}: {exc}",
              file=sys.stderr)

    raw_counts = {region: true_counts.get(region, -1) for region in regions}
    return apply_region_mapping(raw_counts)


# =============================================================================
# ── Session catalogue  ────────────────────────────────────────────────────────
# =============================================================================

def build_session_catalogue(
    base_dir: Path,
    results_subdirs: Dict[str, str],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Walk every condition subdirectory and build a three-level catalogue:
    condition -> session_name -> {region: n_neurons}.
    """
    catalogue: Dict[str, Dict[str, Dict[str, int]]] = {}

    for cond, subdir_name in results_subdirs.items():
        cond_dir = base_dir / subdir_name
        catalogue[cond] = {}

        if not cond_dir.exists():
            print(
                f"[WARN] Condition directory not found, skipping: {cond_dir}",
                file=sys.stderr,
            )
            continue

        mat_files = sorted(cond_dir.glob("*_analysis_results.mat"))
        print(f"[{cond}]  {len(mat_files)} session file(s) found in {cond_dir}")

        for mat_path in mat_files:
            session_name = mat_path.stem.replace("_analysis_results", "")
            catalogue[cond][session_name] = extract_region_neuron_counts(mat_path)

    return catalogue


# =============================================================================
# ── Region discovery & combination enumeration  ─────────────────────────────
# =============================================================================

def discover_all_regions(
    catalogue: Dict[str, Dict[str, Dict[str, int]]],
    excluded: Tuple[str, ...] = (),
) -> List[str]:
    """Union of every region label observed across all sessions/conditions."""
    all_regions: Set[str] = set()
    for sessions in catalogue.values():
        for region_counts in sessions.values():
            all_regions.update(region_counts.keys())
    all_regions -= set(excluded)
    return sorted(all_regions)


def generate_combinations(
    regions: List[str],
    r: int,
    required: Tuple[str, ...] = (),
) -> List[Tuple[str, ...]]:
    """
    Enumerate all size-r combinations of ``regions``, optionally forcing
    every combination to contain the ``required`` subset.
    """
    required = tuple(required)
    r_remaining = r - len(required)
    if r_remaining < 0:
        raise ValueError(
            f"len(required)={len(required)} exceeds combination size r={r}"
        )
    remaining_pool = [x for x in regions if x not in required]

    combos: List[Tuple[str, ...]] = []
    for c in itertools.combinations(remaining_pool, r_remaining):
        combos.append(tuple(sorted(required + c)))
    return combos


# =============================================================================
# ── Filtering  ────────────────────────────────────────────────────────────────
# =============================================================================

def filter_sessions(
    catalogue: Dict[str, Dict[str, Dict[str, int]]],
    target_groups: List[Tuple[str, ...]],
    min_neurons: int | None = None,
) -> List[Dict[str, List[str]]]:
    """
    For each target group (r_1, r_2, ...), return a dict mapping each condition
    label to the list of session names where ALL regions in the group are present
    and, if ``min_neurons`` is set, each has at least that many neurons.
    """
    results: List[Dict[str, List[str]]] = []

    for group in target_groups:
        group_result: Dict[str, List[str]] = defaultdict(list)
        for cond, sessions in catalogue.items():
            for session_name, region_counts in sessions.items():
                if not all(r in region_counts for r in group):
                    continue
                if min_neurons is not None and not all(
                    region_counts[r] >= min_neurons for r in group
                ):
                    continue
                group_result[cond].append(session_name)
        for cond in group_result:
            group_result[cond].sort()
        results.append(dict(group_result))

    return results


def _format_neuron_counts(
    session_name: str,
    cond: str,
    group: Tuple[str, ...],
    catalogue: Dict[str, Dict[str, Dict[str, int]]],
) -> str:
    """Render 'session_name   REGION=n, REGION=n, ...' for a group's regions."""
    region_counts = catalogue.get(cond, {}).get(session_name, {})
    parts = []
    for r in group:
        n = region_counts.get(r, -1)
        parts.append(f"{r}={n}" if n >= 0 else f"{r}=?")
    return f"{session_name}   ({', '.join(parts)})"


def _union_sessions(per_condition: Dict[str, List[str]]) -> List[str]:
    """Return the sorted union of session names across all conditions."""
    all_sessions: Set[str] = set()
    for sessions in per_condition.values():
        all_sessions.update(sessions)
    return sorted(all_sessions)


# =============================================================================
# ── Ranking  ──────────────────────────────────────────────────────────────────
# =============================================================================

RankedEntry = Tuple[Tuple[str, ...], int, List[str], Dict[str, List[str]]]


def rank_by_count(
    results: List[Dict[str, List[str]]],
    groups: List[Tuple[str, ...]],
) -> List[RankedEntry]:
    """
    Pair each group with its union session count and sort descending
    (ties broken alphabetically by region-combination label for
    reproducibility).
    """
    ranked: List[RankedEntry] = []
    for group, per_condition in zip(groups, results):
        union = _union_sessions(per_condition)
        ranked.append((group, len(union), union, per_condition))

    ranked.sort(key=lambda entry: (-entry[1], " ∩ ".join(entry[0])))
    return ranked


def _label(group: Tuple[str, ...]) -> str:
    return " ∩ ".join(group)


def usable_session_counts(
    catalogue: Dict[str, Dict[str, Dict[str, int]]],
    groups: List[Tuple[str, ...]],
    min_neurons: int,
) -> List[RankedEntry]:
    """
    Like ``rank_by_count``, but a session only counts as "usable" for a
    group if every region in the group has at least ``min_neurons``
    neurons in that session (instead of merely being present).
    """
    ranked: List[RankedEntry] = []
    for group in groups:
        per_condition: Dict[str, List[str]] = defaultdict(list)
        for cond, sessions in catalogue.items():
            for session_name, region_counts in sessions.items():
                if all(region_counts.get(r, -1) >= min_neurons for r in group):
                    per_condition[cond].append(session_name)
        for cond in per_condition:
            per_condition[cond].sort()
        union = _union_sessions(per_condition)
        ranked.append((group, len(union), union, dict(per_condition)))

    ranked.sort(key=lambda entry: (-entry[1], _label(entry[0])))
    return ranked


# =============================================================================
# ── Plotting  ─────────────────────────────────────────────────────────────────
# =============================================================================

def plot_usable_sessions_by_threshold(
    catalogue: Dict[str, Dict[str, Dict[str, int]]],
    groups: List[Tuple[str, ...]],
    thresholds: Tuple[int, ...],
    out_path: Path,
    top_n: int = TOP_N_TRIPLETS_PLOT,
) -> None:
    """
    One figure, one subplot per minimum-neurons-per-region threshold:
    horizontal bar chart of the top-N triplets ranked by number of usable
    sessions at that threshold. Top/right spines are hidden on every panel.
    """
    n_thresholds = len(thresholds)
    n_cols = 4
    n_rows = -(-n_thresholds // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 0.4 * top_n + 2))
    axes = np.atleast_1d(axes).flatten()

    for ax, min_neurons in zip(axes, thresholds):
        ranked = usable_session_counts(catalogue, groups, min_neurons)
        top = [entry for entry in ranked if entry[1] > 0][:top_n]

        ax.set_title(f"min neurons/region = {min_neurons}")
        ax.set_xlabel("usable sessions")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if not top:
            ax.text(0.5, 0.5, "no usable sessions", ha="center", va="center",
                     transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        names = [_label(group) for group, _count, _sessions, _pc in top][::-1]
        counts = [count for _group, count, _sessions, _pc in top][::-1]
        ax.barh(names, counts, color="#4C72B0")
        ax.tick_params(axis="y", labelsize=7)
        for y, c in enumerate(counts):
            ax.text(c, y, f" {c}", va="center", fontsize=7)

    for ax in axes[n_thresholds:]:
        ax.axis("off")

    fig.suptitle("Old Oxford usable sessions per region triplet", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Wrote: {out_path}")


def plot_hub_triplets(
    catalogue: Dict[str, Dict[str, Dict[str, int]]],
    all_regions: List[str],
    hubs: Tuple[str, ...],
    min_neurons: int | None,
    out_path: Path,
    top_n: int = TOP_N_HUB_TRIPLETS,
) -> None:
    """
    One figure, one column per hub region: horizontal bar chart of the
    top-N triplets containing that hub, ranked by number of usable
    sessions (every region >= ``min_neurons`` neurons). Bars are labelled
    with the two partner regions; the hub is the column title.
    """
    fig, axes = plt.subplots(
        1, len(hubs), figsize=(5.5 * len(hubs), 0.4 * top_n + 2), squeeze=False
    )
    thr_note = f", >= {min_neurons} neurons/region" if min_neurons is not None else ""

    for ax, hub in zip(axes[0], hubs):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("usable sessions")

        if hub not in all_regions:
            ax.set_title(f"hub: {hub}\n(region not found)")
            ax.text(0.5, 0.5, "hub region not in data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        groups = generate_combinations(all_regions, 3, required=(hub,))
        results = filter_sessions(catalogue, groups, min_neurons)
        ranked = [e for e in rank_by_count(results, groups) if e[1] > 0][:top_n]

        ax.set_title(f"hub: {hub}")
        if not ranked:
            ax.text(0.5, 0.5, "no usable sessions", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        names = [
            " ∩ ".join(r for r in group if r != hub) for group, *_ in ranked
        ][::-1]
        counts = [count for _g, count, _s, _pc in ranked][::-1]
        ax.barh(names, counts, color="#4C72B0")
        ax.tick_params(axis="y", labelsize=8)
        for y, c in enumerate(counts):
            ax.text(c, y, f" {c}", va="center", fontsize=8)

    fig.suptitle(f"Top {top_n} triplets per hub region (usable sessions{thr_note})",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Wrote: {out_path}")


# =============================================================================
# ── Reporting  ────────────────────────────────────────────────────────────────
# =============================================================================

def print_ranked_table(
    ranked: List[RankedEntry],
    title: str,
    catalogue: Dict[str, Dict[str, Dict[str, int]]],
) -> None:
    """Print a compact ranked (rank, combination, count) table to stdout,
    with each qualifying session's per-region neuron counts listed beneath."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    width = max((len(_label(g)) for g, *_ in ranked), default=20)
    for rank, (group, count, _union, per_condition) in enumerate(ranked, start=1):
        print(f"  {rank:>4}.  {_label(group):<{width}}   n = {count}")
        for cond, sess_list in per_condition.items():
            for s in sess_list:
                print(f"          {_format_neuron_counts(s, cond, group, catalogue)}")

    print(sep)


def write_ranked_report(
    ranked_by_size: Dict[int, List[RankedEntry]],
    conditions: List[str],
    catalogue: Dict[str, Dict[str, Dict[str, int]]],
    report_path: Path,
) -> None:
    """Write full ranked report (counts + session names + neuron counts) to disk."""
    lines: List[str] = []
    sep = "=" * 72

    size_word = {2: "PAIRWISE", 3: "TRIPLET"}

    for r, ranked in ranked_by_size.items():
        thr_note = (
            f", >= {MIN_NEURONS_PER_REGION} neurons/region"
            if MIN_NEURONS_PER_REGION is not None else ""
        )
        title = f"{size_word.get(r, f'{r}-TUPLE')} REGION CO-RECORDING RANKING (r = {r}{thr_note})"
        lines.append(sep)
        lines.append(title)
        lines.append(sep)

        for rank, (group, count, _union, per_condition) in enumerate(ranked, start=1):
            lines.append("")
            lines.append(f"{rank}. {_label(group)}   n = {count}")
            for cond in conditions:
                sess_list = per_condition.get(cond, [])
                lines.append(f"    [{cond}]  {len(sess_list)} session(s)")
                for s in sess_list:
                    lines.append(f"      {_format_neuron_counts(s, cond, group, catalogue)}")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Full ranked report written to: {report_path}")


def write_hub_report(
    catalogue: Dict[str, Dict[str, Dict[str, int]]],
    all_regions: List[str],
    hubs: Tuple[str, ...],
    conditions: List[str],
    min_neurons: int | None,
    report_path: Path,
) -> None:
    """
    Text report, one section per hub region: every triplet containing the
    hub that has at least one usable session, ranked by session count, with
    each session's per-region neuron counts (same layout as the main report).
    """
    lines: List[str] = []
    sep = "=" * 72
    thr_note = f", >= {min_neurons} neurons/region" if min_neurons is not None else ""

    for hub in hubs:
        lines.append(sep)
        lines.append(f"HUB REGION: {hub}   (triplets, r = 3{thr_note})")
        lines.append(sep)

        if hub not in all_regions:
            lines.append("")
            lines.append("  (hub region not found in data)")
            lines.append("")
            continue

        groups = generate_combinations(all_regions, 3, required=(hub,))
        results = filter_sessions(catalogue, groups, min_neurons)
        ranked = [e for e in rank_by_count(results, groups) if e[1] > 0]
        if not ranked:
            lines.append("")
            lines.append("  (no usable sessions)")
        for rank, (group, count, _union, per_condition) in enumerate(ranked, start=1):
            lines.append("")
            lines.append(f"{rank}. {_label(group)}   n = {count}")
            for cond in conditions:
                sess_list = per_condition.get(cond, [])
                lines.append(f"    [{cond}]  {len(sess_list)} session(s)")
                for s in sess_list:
                    lines.append(f"      {_format_neuron_counts(s, cond, group, catalogue)}")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Hub report written to: {report_path}")


# =============================================================================
# ── Entry point  ──────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    print(f"[INFO] Base directory      : {BASE_DIR}")
    print(f"[INFO] Conditions          : {list(RESULTS_SUBDIRS.keys())}")
    print(f"[INFO] Combination sizes   : {COMBINATION_SIZES}")
    print(f"[INFO] Required regions    : {REQUIRED_REGIONS or '(none)'}")
    print(f"[INFO] Excluded regions    : {EXCLUDED_REGIONS or '(none)'}")
    print(f"[INFO] Min neurons/region  : {MIN_NEURONS_PER_REGION or '(no filter)'}\n")

    # 1. Build the full session × region catalogue
    catalogue = build_session_catalogue(BASE_DIR, RESULTS_SUBDIRS)

    # 2. Discover the region vocabulary present in the data
    all_regions = discover_all_regions(catalogue, excluded=EXCLUDED_REGIONS)
    print(f"[INFO] Discovered {len(all_regions)} region(s): {all_regions}\n")

    conditions = list(RESULTS_SUBDIRS.keys())

    # 3. For each requested combination size, enumerate → filter → rank
    ranked_by_size: Dict[int, List[RankedEntry]] = {}
    groups_by_size: Dict[int, List[Tuple[str, ...]]] = {}
    for r in COMBINATION_SIZES:
        groups = generate_combinations(all_regions, r, required=REQUIRED_REGIONS)
        groups_by_size[r] = groups
        results = filter_sessions(catalogue, groups, MIN_NEURONS_PER_REGION)
        ranked = rank_by_count(results, groups)
        if MIN_NEURONS_PER_REGION is not None:
            ranked = [entry for entry in ranked if entry[1] > 0]
        ranked_by_size[r] = ranked

        size_word = {2: "PAIRWISE", 3: "TRIPLET"}.get(r, f"{r}-TUPLE")
        thr_note = (
            f", >= {MIN_NEURONS_PER_REGION} neurons/region"
            if MIN_NEURONS_PER_REGION is not None else ""
        )
        print_ranked_table(
            ranked,
            f"{size_word} REGION CO-RECORDING RANKING (r = {r}{thr_note})",
            catalogue,
        )

    # 4. Write the full detailed report (counts + session names + neuron counts)
    write_ranked_report(ranked_by_size, conditions, catalogue, REPORT_PATH)

    # 5. Plot, per minimum-neurons-per-region setting, which triplets give
    #    how many usable sessions.
    triplet_groups = groups_by_size.get(
        3, generate_combinations(all_regions, 3, required=REQUIRED_REGIONS)
    )
    plot_usable_sessions_by_threshold(
        catalogue, triplet_groups, MIN_NEURON_THRESHOLDS, BARPLOT_PATH
    )

    # 6. Hub-region view: top-N triplets per hub region.
    plot_hub_triplets(
        catalogue, all_regions, HUB_REGIONS, HUB_MIN_NEURONS, HUB_BARPLOT_PATH
    )
    write_hub_report(
        catalogue, all_regions, HUB_REGIONS, conditions, HUB_MIN_NEURONS,
        HUB_REPORT_PATH,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
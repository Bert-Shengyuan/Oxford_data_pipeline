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
from typing import Dict, FrozenSet, List, Set, Tuple

import mat73  # pip install mat73

# =============================================================================
# ── Configuration  ────────────────────────────────────────────────────────────
# =============================================================================

BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")

# Condition label  →  subdirectory name
RESULTS_SUBDIRS: Dict[str, str] = {
    "cued_hit_long":   "sessions_cued_hit_long_results",
    # "spont_hit_long":  "sessions_spont_hit_long_results",
    # "spont_miss_long": "sessions_spont_miss_long_results",
}

# Combination sizes to enumerate (2 = pairs, 3 = triplets). Extend freely,
# e.g. add 4 for quadruplets — the machinery below is size-agnostic.
COMBINATION_SIZES: Tuple[int, ...] = (2, 3)

# Optional: regions that MUST appear in every enumerated combination.
# Leave empty for a fully unconstrained scan. Example: ("MOs",) restricts
# every pair/triplet to those containing MOs, replicating the old script's
# implicit convention.
REQUIRED_REGIONS: Tuple[str, ...] = ()

# Optional: regions to drop from the discovered region set before
# enumeration (mirrors the EXCLUDED_REGIONS blacklist convention used in
# the pCCA pivot-ablation scripts). Leave empty to include everything found.
EXCLUDED_REGIONS: Tuple[str, ...] = (["analysis_timestamp","config","session_name"])

# Output report path
REPORT_PATH = Path(__file__).parent / "session_region_pairs_report.txt"


# =============================================================================
# ── Region extraction  ────────────────────────────────────────────────────────
# =============================================================================

def extract_recorded_regions(mat_path: Path) -> Set[str]:
    """
    Return the set of region labels present in ``pca_results`` for a single
    session .mat file.
    """
    try:
        data = mat73.loadmat(str(mat_path))
    except Exception as exc:
        print(f"  [WARN] Could not load {mat_path.name}: {exc}", file=sys.stderr)
        return set()

    # ── Primary source: pca_results keys ────────────────────────────────────
    pca = data.get("pca_results")
    if isinstance(pca, dict) and pca:
        return set(pca.keys())

    # ── Fallback: harvest from cca pair entries ──────────────────────────────
    regions: Set[str] = set()
    try:
        cca = data.get("cca_results")
        if not isinstance(cca, dict):
            return regions
        pair_results = cca.get("pair_results", [])
        if not isinstance(pair_results, (list, tuple)):
            return regions
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

    return regions


# =============================================================================
# ── Session catalogue  ────────────────────────────────────────────────────────
# =============================================================================

def build_session_catalogue(
    base_dir: Path,
    results_subdirs: Dict[str, str],
) -> Dict[str, Dict[str, FrozenSet[str]]]:
    """Walk every condition subdirectory and build a two-level catalogue."""
    catalogue: Dict[str, Dict[str, FrozenSet[str]]] = {}

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
            regions = extract_recorded_regions(mat_path)
            catalogue[cond][session_name] = frozenset(regions)

    return catalogue


# =============================================================================
# ── Region discovery & combination enumeration  ─────────────────────────────
# =============================================================================

def discover_all_regions(
    catalogue: Dict[str, Dict[str, FrozenSet[str]]],
    excluded: Tuple[str, ...] = (),
) -> List[str]:
    """Union of every region label observed across all sessions/conditions."""
    all_regions: Set[str] = set()
    for sessions in catalogue.values():
        for regions in sessions.values():
            all_regions.update(regions)
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
    catalogue: Dict[str, Dict[str, FrozenSet[str]]],
    target_groups: List[Tuple[str, ...]],
) -> List[Dict[str, List[str]]]:
    """
    For each target group (r_1, r_2, ...), return a dict mapping each condition
    label to the list of session names where ALL regions in the group are present.
    """
    results: List[Dict[str, List[str]]] = []

    for group in target_groups:
        group_result: Dict[str, List[str]] = defaultdict(list)
        for cond, sessions in catalogue.items():
            for session_name, regions in sessions.items():
                if all(r in regions for r in group):
                    group_result[cond].append(session_name)
        for cond in group_result:
            group_result[cond].sort()
        results.append(dict(group_result))

    return results


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


# =============================================================================
# ── Reporting  ────────────────────────────────────────────────────────────────
# =============================================================================

def print_ranked_table(ranked: List[RankedEntry], title: str) -> None:
    """Print a compact ranked (rank, combination, count) table to stdout."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    width = max((len(_label(g)) for g, *_ in ranked), default=20)
    for rank, (group, count, _union, _per_cond) in enumerate(ranked, start=1):
        print(f"  {rank:>4}.  {_label(group):<{width}}   n = {count}")

    print(sep)


def write_ranked_report(
    pair_ranked: List[RankedEntry],
    triple_ranked: List[RankedEntry],
    conditions: List[str],
    report_path: Path,
) -> None:
    """Write full ranked report (counts + session names) to disk."""
    lines: List[str] = []
    sep = "=" * 72

    for title, ranked in (
        ("PAIRWISE REGION CO-RECORDING RANKING (r = 2)", pair_ranked),
        ("TRIPLET REGION CO-RECORDING RANKING (r = 3)", triple_ranked),
    ):
        lines.append(sep)
        lines.append(title)
        lines.append(sep)

        for rank, (group, count, union, per_condition) in enumerate(ranked, start=1):
            lines.append("")
            lines.append(f"{rank}. {_label(group)}   n = {count}")
            for cond in conditions:
                sess_list = per_condition.get(cond, [])
                lines.append(f"    [{cond}]  {len(sess_list)} session(s)")
                for s in sess_list:
                    lines.append(f"      {s}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Full ranked report written to: {report_path}")


# =============================================================================
# ── Entry point  ──────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    print(f"[INFO] Base directory      : {BASE_DIR}")
    print(f"[INFO] Conditions          : {list(RESULTS_SUBDIRS.keys())}")
    print(f"[INFO] Combination sizes   : {COMBINATION_SIZES}")
    print(f"[INFO] Required regions    : {REQUIRED_REGIONS or '(none)'}")
    print(f"[INFO] Excluded regions    : {EXCLUDED_REGIONS or '(none)'}\n")

    # 1. Build the full session × region catalogue
    catalogue = build_session_catalogue(BASE_DIR, RESULTS_SUBDIRS)

    # 2. Discover the region vocabulary present in the data
    all_regions = discover_all_regions(catalogue, excluded=EXCLUDED_REGIONS)
    print(f"[INFO] Discovered {len(all_regions)} region(s): {all_regions}\n")

    conditions = list(RESULTS_SUBDIRS.keys())

    # 3. For each requested combination size, enumerate → filter → rank
    ranked_by_size: Dict[int, List[RankedEntry]] = {}
    for r in COMBINATION_SIZES:
        groups = generate_combinations(all_regions, r, required=REQUIRED_REGIONS)
        results = filter_sessions(catalogue, groups)
        ranked = rank_by_count(results, groups)
        ranked_by_size[r] = ranked

        size_word = {2: "PAIRWISE", 3: "TRIPLET"}.get(r, f"{r}-TUPLE")
        print_ranked_table(ranked, f"{size_word} REGION CO-RECORDING RANKING (r = {r})")

    # 4. Write the full detailed report (counts + session names) to disk
    write_ranked_report(
        ranked_by_size.get(2, []),
        ranked_by_size.get(3, []),
        conditions,
        REPORT_PATH,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
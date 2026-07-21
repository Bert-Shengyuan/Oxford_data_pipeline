"""
filter_sessions_by_region_pairs.py
====================================
Scans all Oxford dataset session files across every condition subdirectory
and partitions sessions into six lists based on which target regions
are co-recorded:

    List 1  –  MOs  ∩  VPMPO   (+ any other regions)
    List 2  –  MOs  ∩  MOp     (+ any other regions)
    List 3  –  MOs  ∩  VALVM   (+ any other regions)
    List 4  –  MOs  ∩  VPMPO  ∩  ORB     (+ any other regions)
    List 5  –  MOs  ∩  VPMPO  ∩  MOp     (+ any other regions)
    List 6  –  MOs  ∩  VPMPO  ∩  VALVM   (+ any other regions)

A session qualifies for a list if *all* target regions appear as keys in
``pca_results`` within its .mat file, which is the authoritative per-region
record.

Output
------
- Console summary table: per-condition breakdown and union totals.
- ``session_region_pairs_report.txt``  –  machine-readable flat text report
  written to the same directory as this script.

Usage
-----
    python filter_sessions_by_region_pairs.py
"""

from __future__ import annotations

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

# Target region combinations defining the six lists (支持 2 个或 3 个元素的 tuple)
TARGET_GROUPS: List[Tuple[str, ...]] = [
    ("MOs", "VPMPO"),          # List 1
    ("MOs", "MOp"),            # List 2
    ("MOs", "VALVM"),          # List 3
    ("MOs", "VPMPO", "ORB"),   # List 4
    ("MOs", "VPMPO", "MOp"),   # List 5
    ("MOs", "VPMPO", "VALVM"),
    ("MOs", "ORB", "VALVM"),# List 6
    ("MOp","mPFC"),
    ("MOs", "VPMPO", "LP"),
]

LIST_LABELS: List[str] = [
    "List 1 : MOs ∩ VPMPO",
    "List 2 : MOs ∩ MOp",
    "List 3 : MOs ∩ VALVM",
    "List 4 : MOs ∩ VPMPO ∩ ORB",
    "List 5 : MOs ∩ VPMPO ∩ MOp",
    "List 6 : MOs ∩ VPMPO ∩ VALVM",
    "List 7 : MOs ∩ ORB ∩ VALVM",
    "List 8 : MOp ∩ mPFC",
    "List 9 : MOs ∩ VPMPO ∩ LP",
]

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
                # 检查当前 session 是否包含了该组合中的所有脑区
                if all(r in regions for r in group):
                    group_result[cond].append(session_name)
        # Sort for reproducibility
        for cond in group_result:
            group_result[cond].sort()
        results.append(dict(group_result))

    return results


# =============================================================================
# ── Reporting  ────────────────────────────────────────────────────────────────
# =============================================================================

def _union_sessions(per_condition: Dict[str, List[str]]) -> List[str]:
    """Return the sorted union of session names across all conditions."""
    all_sessions: Set[str] = set()
    for sessions in per_condition.values():
        all_sessions.update(sessions)
    return sorted(all_sessions)


def print_report(
    results: List[Dict[str, List[str]]],
    list_labels: List[str],
    conditions: List[str],
) -> None:
    """Print a structured summary to stdout."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  SESSION FILTER REPORT  –  region co-recording inventory")
    print(sep)

    for idx, (label, per_condition) in enumerate(zip(list_labels, results)):
        union = _union_sessions(per_condition)
        print(f"\n{'─' * 72}")
        print(f"  {label}")
        print(f"  Union across all conditions: {len(union)} unique session(s)")
        print(f"{'─' * 72}")

        for cond in conditions:
            sess_list = per_condition.get(cond, [])
            print(f"\n  [{cond}]  →  {len(sess_list)} session(s)")
            if sess_list:
                for s in sess_list:
                    print(f"      • {s}")
            else:
                print("      (none)")

    print(f"\n{sep}\n")


def write_report(
    results: List[Dict[str, List[str]]],
    list_labels: List[str],
    conditions: List[str],
    report_path: Path,
) -> None:
    """Write the same report to a plain-text file for downstream use."""
    lines: List[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("SESSION FILTER REPORT  –  region co-recording inventory")
    lines.append(sep)

    for label, per_condition in zip(list_labels, results):
        union = _union_sessions(per_condition)
        lines.append("")
        lines.append(f"{label}")
        lines.append(f"Union across all conditions: {len(union)} unique session(s)")
        lines.append(sep)

        for cond in conditions:
            sess_list = per_condition.get(cond, [])
            lines.append(f"  [{cond}]  {len(sess_list)} session(s)")
            for s in sess_list:
                lines.append(f"    {s}")

    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Report written to: {report_path}")


# =============================================================================
# ── Convenience accessors  ────────────────────────────────────────────────────
# =============================================================================

def get_union_list(
    results: List[Dict[str, List[str]]],
    list_index: int,
) -> List[str]:
    """Return the union (across conditions) of qualifying session names."""
    return _union_sessions(results[list_index])


# =============================================================================
# ── Entry point  ──────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    print(f"[INFO] Base directory : {BASE_DIR}")
    print(f"[INFO] Conditions     : {list(RESULTS_SUBDIRS.keys())}\n")

    # 1. Build the full session × region catalogue
    catalogue = build_session_catalogue(BASE_DIR, RESULTS_SUBDIRS)

    # 2. Apply the region-group filters
    results = filter_sessions(catalogue, TARGET_GROUPS)

    # 3. Print structured report to stdout
    print_report(results, LIST_LABELS, list(RESULTS_SUBDIRS.keys()))

    # 4. Write the same report to disk
    write_report(results, LIST_LABELS, list(RESULTS_SUBDIRS.keys()), REPORT_PATH)

    # ── Expose the six lists as named Python objects ────────────────────────
    list_MOs_VPMPO       = get_union_list(results, 0)   # List 1
    list_MOs_MOp         = get_union_list(results, 1)   # List 2
    list_MOs_VALVM       = get_union_list(results, 2)   # List 3
    list_MOs_VPMPO_ORB   = get_union_list(results, 3)   # List 4
    list_MOs_VPMPO_MOp   = get_union_list(results, 4)   # List 5
    list_MOs_VPMPO_VALVM = get_union_list(results, 5)   # List 6

    print("Sessions recording MOs + VPMPO       :", list_MOs_VPMPO)
    print("Sessions recording MOs + MOp         :", list_MOs_MOp)
    print("Sessions recording MOs + VALVM       :", list_MOs_VALVM)
    print("Sessions recording MOs + VPMPO + ORB :", list_MOs_VPMPO_ORB)
    print("Sessions recording MOs + VPMPO + MOp :", list_MOs_VPMPO_MOp)
    print("Sessions recording MOs + VPMPO + VALVM:", list_MOs_VPMPO_VALVM)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
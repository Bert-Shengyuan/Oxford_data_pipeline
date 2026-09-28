"""
tcell_session_triplets.py
==========================
Neuron-level counterpart to ``session_number_order.py``.

Instead of scanning per-session ``*_analysis_results.mat`` files, this
script reads the single flat neuron table stored in
``tcell_cellmetrics_struct.mat`` (``tcell_struct``, one row per neuron,
44479 rows) and, for every brain-region triplet discovered in the data,
finds the sessions in which all three regions were recorded and lists
every individual neuron that session/triplet contains.

Field mapping (per user spec)
------------------------------
- ``tcell_struct.bc_bc_unitType``      -> unit-quality label ('GOOD', 'MUA', 'NOISE', 'NON-SOMA')
- ``tcell_struct.brainRegion_names2``  -> region field
- ``tcell_struct.sessiondate``         -> session field

Two output files are written (one line per qualifying session, with the
per-region neuron counts for that triplet):
- ``session_triplets_good.txt``      – unit-quality label == "Good" only
- ``session_triplets_good_mua.txt``  – unit-quality label in {"Good", "MUA"}

It also plots, for each minimum-neurons-per-region setting in
``MIN_NEURON_THRESHOLDS``, how many sessions are "usable" per triplet
(i.e. every region in the triplet has at least that many neurons in that
session), as grouped bar charts:
- ``session_triplets_barplots_good.png``
- ``session_triplets_barplots_good_mua.png``

Note on loading: ``tcell_cellmetrics_struct.mat`` is a MATLAB v7.3
(HDF5) file, the same format ``mat73`` reads. It is ~630MB, and a full
``mat73.loadmat`` (which materializes all 261 fields) takes several
minutes. This script instead reads only the handful of fields it needs
directly via ``h5py`` (the library ``mat73`` itself is built on), which
takes on the order of seconds.

Usage
-----
    python tcell_session_triplets.py
"""

from __future__ import annotations

import itertools
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# ── Configuration  ────────────────────────────────────────────────────────────
# =============================================================================

MAT_PATH = Path("/Users/shengyuancai/Downloads/Oxford_dataset/tcell_cellmetrics_struct.mat")
STRUCT_NAME = "tcell_struct"

TRIPLET_SIZE = 3

# Region labels to drop before enumerating triplets (not real brain regions).
EXCLUDED_REGIONS: Tuple[str, ...] = ("other",)

OUTPUT_GOOD = Path(__file__).parent / "session_triplets_good.txt"
OUTPUT_GOOD_MUA = Path(__file__).parent / "session_triplets_good_mua.txt"

# Minimum neurons-per-region settings to sweep for the "usable sessions" bar plots.
MIN_NEURON_THRESHOLDS: Tuple[int, ...] = (50, 60, 70, 80, 90, 100, 200)

# Only the top-N triplets (by usable-session count) are drawn per subplot, to
# keep the bar charts readable (there are 286 triplets in total; the full
# ranking for every triplet is in the .txt report).
TOP_N_TRIPLETS_PLOT = 15

OUTPUT_BARPLOT_GOOD = Path(__file__).parent / "session_triplets_barplots_good.png"
OUTPUT_BARPLOT_GOOD_MUA = Path(__file__).parent / "session_triplets_barplots_good_mua.png"

NeuronRecord = Tuple[int, int, str]  # (UID, cluID, unitType)


# =============================================================================
# ── Loading  ──────────────────────────────────────────────────────────────────
# =============================================================================

def _decode_h5_string(f: h5py.File, ref: h5py.Reference) -> str:
    """Decode a single HDF5-referenced char array (one MATLAB cell entry) to str."""
    arr = f[ref][()]
    if arr.dtype.kind in ("u", "i"):
        return "".join(chr(c) for c in np.asarray(arr).flatten())
    return str(arr)


def _decode_cell_column(f: h5py.File, ds: h5py.Dataset) -> List[str]:
    """Decode a 1xN MATLAB cell array of strings (stored as HDF5 object refs)."""
    return [_decode_h5_string(f, ref) for ref in ds[0, :]]


def load_neuron_table(mat_path: Path) -> Dict[str, np.ndarray]:
    """Read only the fields needed from tcell_struct: unit type, region, session, ids."""
    with h5py.File(mat_path, "r") as f:
        g = f[STRUCT_NAME]
        unit_type = _decode_cell_column(f, g["bc_bc_unitType"])
        region = _decode_cell_column(f, g["brainRegion_names2"])
        session = _decode_cell_column(f, g["sessiondate"])
        uid = g["UID"][0, :].astype(np.int64)
        clu_id = g["cluID"][0, :].astype(np.int64)

    n = len(unit_type)
    if not (len(region) == n and len(session) == n and len(uid) == n and len(clu_id) == n):
        raise ValueError("Field length mismatch in tcell_struct.")

    return {
        "unit_type": np.array(unit_type),
        "region": np.array(region),
        "session": np.array(session),
        "uid": uid,
        "clu_id": clu_id,
    }


# =============================================================================
# ── Catalogue: session -> region -> neurons  ─────────────────────────────────
# =============================================================================

def build_catalogue(
    table: Dict[str, np.ndarray],
    allowed_unit_types: Set[str],
    excluded_regions: Tuple[str, ...],
) -> Dict[str, Dict[str, List[NeuronRecord]]]:
    catalogue: Dict[str, Dict[str, List[NeuronRecord]]] = defaultdict(lambda: defaultdict(list))

    n = len(table["unit_type"])
    for i in range(n):
        ut = table["unit_type"][i]
        if ut not in allowed_unit_types:
            continue
        region = table["region"][i]
        if region in excluded_regions:
            continue
        session = table["session"][i]
        catalogue[session][region].append(
            (int(table["uid"][i]), int(table["clu_id"][i]), ut)
        )

    return {s: dict(region_map) for s, region_map in catalogue.items()}


def discover_regions(catalogue: Dict[str, Dict[str, List[NeuronRecord]]]) -> List[str]:
    regions: Set[str] = set()
    for region_map in catalogue.values():
        regions.update(region_map.keys())
    return sorted(regions)


# =============================================================================
# ── Triplet enumeration & ranking  ────────────────────────────────────────────
# =============================================================================

def generate_triplets(regions: List[str]) -> List[Tuple[str, ...]]:
    return [tuple(sorted(c)) for c in itertools.combinations(regions, TRIPLET_SIZE)]


RankedEntry = Tuple[Tuple[str, ...], int, List[str]]


def rank_triplets(
    catalogue: Dict[str, Dict[str, List[NeuronRecord]]],
    triplets: List[Tuple[str, ...]],
) -> List[RankedEntry]:
    ranked: List[RankedEntry] = []
    for triplet in triplets:
        qualifying_sessions = sorted(
            session
            for session, region_map in catalogue.items()
            if all(region in region_map for region in triplet)
        )
        ranked.append((triplet, len(qualifying_sessions), qualifying_sessions))

    ranked.sort(key=lambda entry: (-entry[1], " ∩ ".join(entry[0])))
    return ranked


def usable_session_counts(
    catalogue: Dict[str, Dict[str, List[NeuronRecord]]],
    triplets: List[Tuple[str, ...]],
    min_neurons: int,
) -> List[RankedEntry]:
    """
    Like ``rank_triplets``, but a session only counts as "usable" for a
    triplet if every region in the triplet has at least ``min_neurons``
    neurons in that session (instead of merely being present).
    """
    ranked: List[RankedEntry] = []
    for triplet in triplets:
        qualifying_sessions = sorted(
            session
            for session, region_map in catalogue.items()
            if all(len(region_map.get(region, [])) >= min_neurons for region in triplet)
        )
        ranked.append((triplet, len(qualifying_sessions), qualifying_sessions))

    ranked.sort(key=lambda entry: (-entry[1], " ∩ ".join(entry[0])))
    return ranked


# =============================================================================
# ── Plotting  ─────────────────────────────────────────────────────────────────
# =============================================================================

def plot_usable_sessions_by_threshold(
    catalogue: Dict[str, Dict[str, List[NeuronRecord]]],
    triplets: List[Tuple[str, ...]],
    thresholds: Tuple[int, ...],
    label: str,
    out_path: Path,
    top_n: int = TOP_N_TRIPLETS_PLOT,
) -> None:
    """
    One figure, one subplot per minimum-neurons-per-region threshold:
    horizontal bar chart of the top-N triplets ranked by number of usable
    sessions at that threshold.
    """
    n_thresholds = len(thresholds)
    n_cols = 4
    n_rows = -(-n_thresholds // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 0.4 * top_n + 2))
    axes = np.atleast_1d(axes).flatten()

    for ax, min_neurons in zip(axes, thresholds):
        ranked = usable_session_counts(catalogue, triplets, min_neurons)
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

        names = [" ∩ ".join(triplet) for triplet, _count, _sessions in top][::-1]
        counts = [count for _triplet, count, _sessions in top][::-1]
        ax.barh(names, counts, color="#4C72B0")
        ax.tick_params(axis="y", labelsize=7)
        for y, c in enumerate(counts):
            ax.text(c, y, f" {c}", va="center", fontsize=7)

    for ax in axes[n_thresholds:]:
        ax.axis("off")

    fig.suptitle(f"Usable sessions per region triplet — {label}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Wrote: {out_path}")


# =============================================================================
# ── Reporting  ────────────────────────────────────────────────────────────────
# =============================================================================

def write_report(
    ranked: List[RankedEntry],
    catalogue: Dict[str, Dict[str, List[NeuronRecord]]],
    label: str,
    out_path: Path,
) -> None:
    lines: List[str] = []
    sep = "=" * 80
    lines.append(sep)
    lines.append(f"TRIPLET REGION CO-RECORDING RANKING — {label} (r = {TRIPLET_SIZE})")
    lines.append(sep)

    for rank, (triplet, n_sessions, sessions) in enumerate(ranked, start=1):
        lines.append("")
        lines.append(f"{rank}. {' ∩ '.join(triplet)}   n_sessions = {n_sessions}")
        for session in sessions:
            region_map = catalogue[session]
            per_region_counts = ", ".join(
                f"{r}={len(region_map[r])}" for r in triplet
            )
            lines.append(f"    [session {session}]  ({per_region_counts})")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote: {out_path}  ({len(ranked)} triplet(s))")


# =============================================================================
# ── Entry point  ──────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    print(f"[INFO] Input file        : {MAT_PATH}")
    print(f"[INFO] Excluded regions  : {EXCLUDED_REGIONS}")

    t0 = time.time()
    table = load_neuron_table(MAT_PATH)
    print(f"[INFO] Loaded {len(table['unit_type'])} neurons in {time.time() - t0:.1f}s\n")

    jobs = (
        ({"GOOD"}, "Good units only", OUTPUT_GOOD, OUTPUT_BARPLOT_GOOD),
        ({"GOOD", "MUA"}, "Good + MUA units", OUTPUT_GOOD_MUA, OUTPUT_BARPLOT_GOOD_MUA),
    )

    for allowed_unit_types, label, out_path, plot_path in jobs:
        catalogue = build_catalogue(table, allowed_unit_types, EXCLUDED_REGIONS)
        regions = discover_regions(catalogue)
        n_sessions = len(catalogue)
        print(f"[INFO] [{label}] {n_sessions} session(s), {len(regions)} region(s): {regions}")

        triplets = generate_triplets(regions)
        ranked = rank_triplets(catalogue, triplets)
        write_report(ranked, catalogue, label, out_path)

        plot_usable_sessions_by_threshold(
            catalogue, triplets, MIN_NEURON_THRESHOLDS, label, plot_path
        )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

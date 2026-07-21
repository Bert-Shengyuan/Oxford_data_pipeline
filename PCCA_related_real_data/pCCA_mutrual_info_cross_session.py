"""
pcca_cross_session_mi_bar.py
=============================

Cross-session mutual-information (MI) summary for a FIXED set of 21
anatomically-motivated region pairs, organised into 7 categories
(thalamic-thalamic, cortico-cortical, cortico-motor-thalamic,
cortico-sensory-thalamic, to-HY, to-STR, other), pooled across sessions.

Follows the exact "full joint ablation" MI construction used by
``create_session_mi_bar`` in the single-session script:

    Z   = every candidate region except {region_i, region_j}
    MI  = -sum_k log(1 - rho_k^2)

with rho_k the canonical correlations of pCCA(region_i, region_j | Z).

Standalone by design (§0, "primitive copying over importing" — this
project's established convention): every core primitive (zscore /
flatten, ridge CCA, residualisation, pCCA, spike-tensor loading) is
duplicated verbatim from pcca_single_session.py / pcca_sequential_
ablation.py, so the per-session MI values computed here are
byte-identical to what create_session_mi_bar would produce for the
same session.

Pipeline
--------
1.  For every session (either the explicit ``SESSION_NAMES`` list, or,
    if that list is left empty, every session file auto-discovered in
    ``SESSION_SUBDIR``), load region_spikes and compute
    MI(region_i, region_j) for each of the 21 fixed pairs in
    ``PAIR_CATEGORIES`` — exactly as create_session_mi_bar does, one
    session at a time. A pair is skipped for a session if either of
    its two regions is absent from that session's candidate set.
2.  Pool the resulting (region_i, region_j) -> MI records across
    sessions.  A pair is retained only if it was observed in
    > MIN_SESSIONS sessions (data-availability filter; not annotated
    on the figure).
3.  Plot a single narrow, tall horizontal bar chart: one bar per
    retained pair, grouped and colour-coded by anatomical CATEGORY
    (not by anchor region), height = cross-session mean, error bar =
    SEM, individual per-session values overlaid as jittered dots.
    VALVM is displayed as "motor Thal" and VPMPO as "sens Thal"
    everywhere in the figure and in console/summary output; the raw
    region names are preserved in the per-session records CSV for
    auditability.
"""

from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import zscore

try:
    import mat73
except ImportError as exc:
    raise SystemExit("mat73 is required: pip install mat73") from exc

from Useful_definition import ANATOMICAL_ORDER, safe_array


# =============================================================================
# 0.  Core CCA / pCCA primitives — copied verbatim from pcca_single_session.py
#     §0b (itself copied from pcca_sequential_ablation.py), per this
#     project's "primitive copying over importing" convention, so every
#     pCCA fit in this script follows a byte-identical data path (same
#     ridge whitening, same residualisation, same per-region neuron-
#     selection masking) to the single-session pipeline.
# =============================================================================

LAMBDA_CCA:   float = 1e-4   # ridge coefficient added to Cxx / Cyy in whitening
LAMBDA_HAT:   float = 1e-4   # ridge on Z'Z in the nuisance hat matrix
N_COMPONENTS: int   = 5      # canonical dimensions retained per pCCA fit

# Preprocessing regime — held at "raw" (no PSTH subtraction, no trial
# shuffling), matching create_session_mi_bar's own defaults.
SUBTRACT_PSTH:  bool = False
SHUFFLE_TRIALS: bool = False


def _zscore_flat(
    X: np.ndarray,
    *,
    subtract_psth: bool = False,
    shuffle_trials: bool = False,
    rng: Optional[np.random.Generator] = None,
    perm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-neuron z-score, optional PSTH subtraction / trial shuffle, then
    flatten (n_trials, n, T) -> (T * n_trials, n)."""
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
    return flat.T  # (T * n_trials, n)


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


def residualize(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_hat: float = LAMBDA_HAT,
) -> np.ndarray:
    if Z_flat is None or Z_flat.ndim < 2 or Z_flat.shape[1] == 0:
        return X_flat.copy()
    n, m  = Z_flat.shape
    ZtZ   = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    Beta  = np.linalg.solve(ZtZ, Z_flat.T)
    return X_flat - Z_flat @ (Beta @ X_flat)


def pcca(
        X_flat: np.ndarray,
        Y_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_cca: float = LAMBDA_CCA,
        lam_hat: float = LAMBDA_HAT,
        n_components: int = N_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_res = residualize(X_flat, Z_flat, lam_hat)
    Y_res = residualize(Y_flat, Z_flat, lam_hat)
    Wx, Wy, rho = ridge_cca(X_res, Y_res, lam_cca, n_components)
    return Wx, Wy, rho, X_res, Y_res


def load_region_spikes(
        session_path: str,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Every region's raw (n_trials, n_neurons, T) spike tensor, with each
    region's own 'selected_neurons' mask already applied."""
    data = mat73.loadmat(session_path)
    rd   = data.get("region_data", {})
    regs = rd.get("regions", {})

    region_spikes: Dict[str, np.ndarray] = {}
    n_trials_out = T_out = None

    for rname, info in regs.items():
        if not isinstance(info, dict):
            continue
        sd = safe_array(info.get("spike_data"))
        if sd is None or sd.ndim != 3:
            continue
        sel = safe_array(info.get("selected_neurons"))
        if sel is not None and sel.size > 0:
            sd = sd[:, sel.ravel().astype(int) - 1, :]
        region_spikes[rname] = sd.astype(np.float32)
        if n_trials_out is None:
            n_trials_out, _, T_out = sd.shape

    print(
        f"  [load_region_spikes]  {len(region_spikes)} regions loaded  "
        f"| n_trials={n_trials_out}  T={T_out}"
    )
    return region_spikes, int(n_trials_out), int(T_out)


# =============================================================================
# 1.  Configuration
# =============================================================================

BASE_DIR       = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
SESSION_SUBDIR = "pcca_sessions_cued_hit_long_results"

# Session selection: leave SESSION_NAMES EMPTY (the default) to
# auto-discover and loop over every session file present in
# BASE_DIR / SESSION_SUBDIR (matched as "<session_name>_analysis_results.mat").
# Populate this list explicitly to restrict the run to specific sessions —
# doing so always takes precedence over auto-discovery.
SESSION_NAMES: List[str] = []

# ---------------------------------------------------------------------------
# Fixed anatomical region pairs of interest, grouped into 7 categories.
# This REPLACES the old FIXED_REGIONS x EXCLUDED_REGIONS combinatorial
# construction: the 21 pairs below (and only these) are computed, in this
# fixed grouping/order, regardless of what other regions happen to be
# present in a given session.
# ---------------------------------------------------------------------------
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

# Flattened (region_i, region_j, category) view of PAIR_CATEGORIES, used
# directly by the per-session MI computation.
REGION_PAIRS: List[Tuple[str, str, str]] = [
    (region_i, region_j, category)
    for category, pairs in PAIR_CATEGORIES
    for (region_i, region_j) in pairs
]

# Cosmetic display-name remapping applied only at print/plot time; the raw
# region_spikes / .mat keys ("VALVM", "VPMPO") are used everywhere else so
# that the underlying data path is unaffected.
DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
}


def _display_name(region: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(region, region)


# User-configurable region blacklist (case-insensitive match), identical
# in spirit and content to pcca_single_session.py's EXCLUDED_REGIONS.
# This still governs which regions are eligible as NUISANCE regressors
# (the Z in "every candidate region except {region_i, region_j}"); it no
# longer has any bearing on which pairs are tested, since that is now
# controlled entirely by PAIR_CATEGORIES / REGION_PAIRS above.
EXCLUDED_REGIONS: List[str] = ["STRv", "LP", "OLF"]




# Keep a (region_i, region_j) pair only if it was observed in
# MORE than this many sessions (i.e. n > MIN_SESSIONS).
MIN_SESSIONS: int = 3
MI_VALUE_CAP: float = 1
OUTPUT_DIR = BASE_DIR / "Paper_output" / "figures_pcca_cross_session"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _is_fiber_region(region_name: str) -> bool:
    return "fiber" in region_name.lower()


def _is_excluded_region(region_name: str) -> bool:
    if _is_fiber_region(region_name):
        return True
    excluded_lower = {r.lower() for r in EXCLUDED_REGIONS}
    return region_name.lower() in excluded_lower

def _load_records_csv(path: Path) -> List[dict]:
    """Inverse of _write_records_csv — reload a previously-saved
    per-session MI records CSV instead of recomputing it from the .mat
    files."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        records = [
            dict(
                session=row["session"],
                region_i=row["region_i"],
                region_j=row["region_j"],
                category=row["category"],
                mi=float(row["mi"]),
            )
            for row in reader
        ]
    print(f"loaded cached records: {path}  ({len(records)} rows)")
    return records


def _write_records_csv(all_records: List[dict], path: Path) -> None:
    if not all_records:
        return

def _candidate_regions(region_spikes: Dict[str, np.ndarray]) -> List[str]:
    """Regions eligible as a nuisance regressor for a given session:
    everything in region_spikes, ordered by ANATOMICAL_ORDER (session
    regions absent from ANATOMICAL_ORDER appended alphabetically as a
    defensive fallback), excluding fiber-tract catch-alls and
    EXCLUDED_REGIONS. (Note: a region can serve as a PAIR member — i.e.
    region_i / region_j — even if EXCLUDED_REGIONS would have blocked it
    as a nuisance regressor; the two roles are independent.)"""
    present = set(region_spikes.keys())
    ordered = [r for r in ANATOMICAL_ORDER if r in present]
    extra   = sorted(present - set(ordered))
    return [r for r in ordered + extra if not _is_excluded_region(r)]


def _resolve_session_names() -> List[str]:
    """SESSION_NAMES verbatim if the user has populated it explicitly;
    otherwise auto-discover every session file present in SESSION_SUBDIR."""
    if SESSION_NAMES:
        print(f"[session selection] using explicit SESSION_NAMES ({len(SESSION_NAMES)} sessions)")
        return list(SESSION_NAMES)

    suffix = "_analysis_results.mat"
    session_dir = BASE_DIR / SESSION_SUBDIR
    if not session_dir.exists():
        warnings.warn(f"session directory not found: {session_dir}")
        return []

    discovered = sorted(
        p.name[: -len(suffix)]
        for p in session_dir.glob(f"*{suffix}")
        if p.name.endswith(suffix)
    )
    print(f"[session selection] SESSION_NAMES empty -> auto-discovered "
          f"{len(discovered)} session(s) in {session_dir}")
    return discovered


# =============================================================================
# 2.  Per-session MI computation
#     ("full joint ablation": Z = every candidate region except
#     {region_i, region_j}) — mirrors create_session_mi_bar exactly,
#     one session at a time, restricted to the 21 fixed REGION_PAIRS.
# =============================================================================

def compute_session_mi(
    session_path: Path,
    session_name: str,
    region_pairs: List[Tuple[str, str, str]],
) -> List[dict]:
    try:
        region_spikes, _, _ = load_region_spikes(str(session_path))
    except Exception as exc:
        warnings.warn(f"[{session_name}] load_region_spikes failed: {exc}")
        return []

    candidates = _candidate_regions(region_spikes)
    if len(candidates) < 2:
        warnings.warn(
            f"[{session_name}] fewer than 2 candidate regions; skipping."
        )
        return []
    candidate_set = set(candidates)

    # Pre-flatten every candidate region once per session (O(R), not O(R^2)
    # from recomputing _zscore_flat inside the nested pair/nuisance loop).
    X_flat: Dict[str, np.ndarray] = {
        r: _zscore_flat(region_spikes[r],
                        subtract_psth=SUBTRACT_PSTH,
                        shuffle_trials=SHUFFLE_TRIALS)
        for r in candidates
    }

    records: List[dict] = []
    for region_i, region_j, category in region_pairs:
        if region_i not in candidate_set or region_j not in candidate_set:
            print(
                f"  [{session_name}] pair '{region_i}-{region_j}' "
                f"unavailable (one or both regions absent); skipping."
            )
            continue

        nuisance_regions = [
            r for r in candidates if r not in (region_i, region_j)
        ]
        Z_full = (
            np.concatenate([X_flat[r] for r in nuisance_regions], axis=1)
            if nuisance_regions else None
        )
        try:
            _, _, rho, _, _ = pcca(X_flat[region_i], X_flat[region_j], Z_full)
        except Exception as exc:
            warnings.warn(
                f"[{session_name}] pCCA({region_i}, {region_j} | "
                f"all others) failed: {exc}"
            )
            continue

        rho_sq = np.clip(rho.astype(float), 0.0, 1.0 - 1e-6) ** 2
        mi_val = float(-1/2*np.sum(np.log1p(-rho_sq)))
        if not np.isfinite(mi_val):
            continue

        records.append(dict(
            region_i=region_i,
            region_j=region_j,
            category=category,
            session=session_name,
            mi=mi_val,
        ))

    return records


# =============================================================================
# 3.  Cross-session pooling
# =============================================================================

def aggregate_mi(all_records: List[dict]) -> Dict[Tuple[str, str], dict]:
    """Pool per-session MI records by (region_i, region_j) and retain
    only pairs observed in > MIN_SESSIONS sessions."""
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    category_of: Dict[Tuple[str, str], str] = {}
    for r in all_records:
        if r["mi"] > MI_VALUE_CAP:
            warnings.warn(
                f"[{r['session']}] {r['region_i']}-{r['region_j']}: "
                f"MI={r['mi']:.3f} > {MI_VALUE_CAP}, dropping observation"
            )
            continue
        key = (r["region_i"], r["region_j"])
        grouped[key].append(r["mi"])
        category_of[key] = r["category"]

    summary: Dict[Tuple[str, str], dict] = {}
    for key, vals in grouped.items():
        n = len(vals)
        if n <= MIN_SESSIONS:
            continue
        vals_arr = np.asarray(vals, dtype=float)
        summary[key] = dict(
            n=n,
            mean=float(vals_arr.mean()),
            sem=float(vals_arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            values=vals_arr,
            category=category_of[key],
        )
    return summary


# =============================================================================
# 4.  Figure — narrow, tall, poster-scale horizontal bar chart
# =============================================================================

# One colour per anatomical category (in PAIR_CATEGORIES order); bars are
# grouped and coloured by category, NOT by anchor region.
CATEGORY_COLORS: Dict[str, str] = {
    "thalamic-thalamic":          "#4C72B0",   # blue
    "cortico-cortical":           "#DD8452",   # orange
    "cortico-motor thalamic":     "#55A868",   # green
    "cortico-sensory thalamic":   "#C44E52",   # red
    "to HY":                      "#8172B2",   # purple
    "to STR":                     "#937860",   # brown
    "other":                      "#64B5CD",   # cyan
}
DOT_COLOR   = "#262626"
BAR_HEIGHT  = 0.62
GROUP_GAP   = 1.35        # extra vertical spacing inserted between groups
DOT_JITTER  = 0.22


def plot_cross_session_mi(
    summary: Dict[Tuple[str, str], dict],
    pair_categories: List[Tuple[str, List[Tuple[str, str]]]],
    save_path: Path,
) -> Optional[plt.Figure]:
    if not summary:
        print("No region pairs survived the MIN_SESSIONS filter; nothing to plot.")
        return None

    # Group by category (in PAIR_CATEGORIES order), sort each group
    # descending by cross-session mean MI — same convention as the
    # single-session MI bar panels.
    groups: List[Tuple[str, List[Tuple[Tuple[str, str], dict]]]] = []
    for category, pair_list in pair_categories:
        pairs = [
            ((region_i, region_j), summary[(region_i, region_j)])
            for (region_i, region_j) in pair_list
            if (region_i, region_j) in summary
        ]
        pairs.sort(key=lambda kv: kv[1]["mean"], reverse=True)
        if pairs:
            groups.append((category, pairs))

    if not groups:
        print("No non-empty category groups after filtering; nothing to plot.")
        return None

    rng = np.random.default_rng(0)  # locally-scoped RNG, cosmetic dot jitter only

    labels: List[str] = []
    means:  List[float] = []
    sems:   List[float] = []
    colors: List[str] = []
    y_positions: List[float] = []
    dot_x: List[float] = []
    dot_y: List[float] = []
    group_spans: List[Tuple[str, float, float]] = []

    y = 0.0
    for category, pairs in groups:
        y_start = y
        for (region_i, region_j), stats in pairs:
            labels.append(f"{_display_name(region_i)}\u2194{_display_name(region_j)}")
            means.append(stats["mean"])
            sems.append(stats["sem"])
            colors.append(CATEGORY_COLORS.get(category, "#888888"))
            y_positions.append(y)

            jitter = rng.uniform(-DOT_JITTER, DOT_JITTER, size=stats["values"].size)
            dot_x.extend(stats["values"].tolist())
            dot_y.extend((y + jitter).tolist())

            y += 1.0
        group_spans.append((category, y_start, y - 1.0))
        y += GROUP_GAP - 1.0

    n_bars = len(labels)
    fig_h  = max(7.0, 0.55 * n_bars + 1.8)   # narrow and tall
    fig, ax = plt.subplots(figsize=(9, fig_h))

    # Faint per-group background bands for visual grouping
    for category, y_lo, y_hi in group_spans:
        ax.axhspan(
            y_lo - BAR_HEIGHT / 2 - 0.25, y_hi + BAR_HEIGHT / 2 + 0.25,
            color=CATEGORY_COLORS.get(category, "#888888"), alpha=0.07, zorder=0,
        )

    ax.barh(
        y_positions, means, height=BAR_HEIGHT, color=colors,
        edgecolor="white", linewidth=0.6, zorder=2,
    )
    ax.errorbar(
        means, y_positions, xerr=sems, fmt="none",
        ecolor="black", elinewidth=1.8, capsize=4, capthick=1.8, zorder=3,
    )
    ax.scatter(
        dot_x, dot_y, s=22, color=DOT_COLOR, alpha=0.55,
        linewidths=0, zorder=4,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=18)
    ax.margins(y=0.015)
    ax.invert_yaxis()

    ax.set_xlim(left=0,right=0.5)
    #ax.set_xlabel(r"Mutual information   $-\frac{1}{2}\sum_k \log(1-\rho_k^2)$", fontsize=18)
    ax.tick_params(axis="x", labelsize=18)
    # ax.set_title(
    #     "pCCA mutual information\ncross-session mean $\\pm$ SEM",
    #     fontsize=21, fontweight="bold", pad=14,
    # )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS[category], label=category, alpha=0.9)
        for category, _ in pair_categories if category in CATEGORY_COLORS
    ]
    # ax.legend(
    #     handles=legend_handles, fontsize=17, frameon=False,
    #     loc="lower right", title="Region-pair category", title_fontsize=17,
    # )

    fig.tight_layout()
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"saved: {save_path}")
    return fig


# =============================================================================
# 5.  Driver
# =============================================================================

def _write_records_csv(all_records: List[dict], path: Path) -> None:
    if not all_records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["session", "region_i", "region_j", "category", "mi"]
        )
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)
    print(f"saved: {path}")


def _write_summary_csv(summary: Dict[Tuple[str, str], dict], path: Path) -> None:
    if not summary:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "region_i", "region_j", "display_pair", "category",
            "n_sessions", "mean_mi", "sem_mi",
        ])
        for (region_i, region_j), stats in sorted(summary.items()):
            display_pair = f"{_display_name(region_i)}<->{_display_name(region_j)}"
            writer.writerow([
                region_i, region_j, display_pair, stats["category"],
                stats["n"], f"{stats['mean']:.6f}", f"{stats['sem']:.6f}",
            ])
    print(f"saved: {path}")

def main() -> None:
    records_path = OUTPUT_DIR / "cross_session_mi_records.csv"

    if records_path.exists():
        print(f"found cached records at {records_path}; skipping session recomputation.")
        all_records = _load_records_csv(records_path)
    else:
        session_names = _resolve_session_names()
        if not session_names:
            print("No sessions to process; aborting.")
            return

        all_records = []

        for session_name in session_names:
            session_path = BASE_DIR / SESSION_SUBDIR / f"{session_name}_analysis_results.mat"
            if not session_path.exists():
                warnings.warn(f"session file not found, skipping: {session_path}")
                continue

            print(f"[{session_name}] computing per-session MI ...")
            recs = compute_session_mi(session_path, session_name, REGION_PAIRS)
            print(f"  -> {len(recs)} region-pair MI values")
            all_records.extend(recs)

    if not all_records:
        print("No MI records computed across any session; aborting.")
        return

    summary = aggregate_mi(all_records)
    if not records_path.exists():
        _write_records_csv(all_records, records_path)
    _write_summary_csv(summary, OUTPUT_DIR / "cross_session_mi_summary.csv")

    print(f"\n{len(summary)} region pairs retained (n > {MIN_SESSIONS} sessions):")
    for (region_i, region_j), stats in sorted(summary.items()):
        print(
            f"  {_display_name(region_i)}<->{_display_name(region_j)} "
            f"[{stats['category']}]: n={stats['n']}  "
            f"mean={stats['mean']:.3f}  sem={stats['sem']:.3f}"
        )

    _write_records_csv(all_records, OUTPUT_DIR / "cross_session_mi_records.csv")
    _write_summary_csv(summary, OUTPUT_DIR / "cross_session_mi_summary.csv")

    save_path = OUTPUT_DIR / "cross_session_mi_bar_by_category_0_6.png"
    fig = plot_cross_session_mi(summary, PAIR_CATEGORIES, save_path)
    if fig is not None:
        plt.close(fig)


if __name__ == "__main__":
    main()
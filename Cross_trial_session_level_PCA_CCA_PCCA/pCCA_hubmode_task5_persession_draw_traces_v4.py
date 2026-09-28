#!/usr/bin/env python3
r"""
pCCA_hubmode_task5_persession_draw_traces_v4.py
================================================================================

v4 counterpart of ``pCCA_hubmode_task5_persession_draw_traces_v3.py``. The
plotting style is unchanged: every contributing SESSION gets its own row
(panel); within a row, the `N_SAMPLE_DRAWS` per-draw trial-averaged latent
traces are thin light lines, the session's pooled (draws x trials) mean is
the bold line, and a shaded band is its SEM. Draws are sign-aligned within a
session (`align_signs_spectral`) before pooling, exactly as in v3; the
session panels are then sign-aligned to one another (per column, on the
pooled session mean; `ALIGN_ACROSS_SESSIONS`).

What changes is the data layer and what drives the loop. Neural results now
come from ``pCCA_all_regions_out_behaviour_v4.py``'s TRIPLET-scoped pickles
(`pcca_all_regions_out_behaviour_v4_hub_<A>_<B>_<C>_<trial_type>_<align>_results`),
and the set of triplets is driven by ``SELECTED_TRIPLETS`` (Section 1),
which defaults to whatever that pipeline script has set:

  * ``SELECTED_TRIPLETS = None`` (default mode -- the previous behaviour):
    every folder matching the pattern above is found on disk and, for each
    triplet containing `HUB_REGION`, one figure per (hub, partner) pair is
    drawn -- one row per session, hub-side latent of the pCCA condition
    picked by `PCCA_VARIANT`. This is the v3 Task-5 figure, once per triplet.

  * ``SELECTED_TRIPLETS = ["MOp-MOs-VALVM", ...]`` (triplet mode): only
    those triplets are analysed (looked up with the pipeline's own
    `build_hub_triplets`), and each gets one figure section per analysis
    part, 1a through 4c, each figure spanning ALL of the triplet's sessions
    (rows) -- see `PART_SPECS`:

        1a  direct PCA of each region                    (N_SAMPLE_DRAWS draws,
                                                          sampled like every
                                                          other part below)
        1b  direct CCA (no regression)                   both sides of a pair
        2a/2b  hub PCA, behaviour explained / residual   one hub orientation
        2c  pCCA, behaviour regressed out                both sides of a pair
        3a/3b  hub PCA, third region explained / residual
        3c  pCCA, third region regressed out
        4a/4b  hub PCA, third region + behaviour explained / residual
        4c  pCCA, third region + behaviour regressed out

    Pair parts (1b/2c/3c/4c) draw one figure per internal pair with two
    columns (region_i | region_j); hub parts draw one figure per
    (hub, partner) orientation (6 per part, since the third region differs).

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pCCA_all_regions_out_behaviour_v4 as v4  # noqa: E402
from pCCA_all_regions_out_behaviour_v4 import (  # noqa: E402
    PrivateLatentAnalyzer,
    PrivateLatentSessionResult,
    PrivateLatentPairResult,
    PrivateLatentPairDrawResult,
    RegionPCAResult,
    RegionPCADrawResult,
    HubOrientationPCADrawResult,
    HubOrientationSelectedNeurons,
    HubPairPCAResult,
    SubregionWeightMetrics,
    SelectedNeuronSet,
    SelectedNeuronResidual,
    TripletSpec,
    PAIR_CONDITIONS,
    PAIR_CONDITION_LABELS,
    HUB_CONDITION_LABELS,
    BASE_DIR,
    build_hub_triplets,
    get_anatomical_index,
    out_subdir_name,
    sort_pair_by_anatomy,
    triplet_pairs_with_third,
)

# The v4 pipeline is normally *run* directly, which bakes '__main__' into
# every pickled dataclass's module reference; unpickling from THIS script's
# '__main__' needs the same classes reachable under it.
for _cls in (
        PrivateLatentSessionResult, PrivateLatentPairResult,
        PrivateLatentPairDrawResult, RegionPCAResult, RegionPCADrawResult,
        HubOrientationPCADrawResult, HubOrientationSelectedNeurons, HubPairPCAResult,
        SubregionWeightMetrics, SelectedNeuronSet, SelectedNeuronResidual,
):
    setattr(sys.modules['__main__'], _cls.__name__, _cls)

# Same Z2 spectral-sync sign-alignment used across sessions elsewhere --
# reused to align the N_SAMPLE_DRAWS independent draws WITHIN one session.
from cross_trial_type_cca_analysis import align_signs_spectral  # noqa: E402

# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS
# =============================================================================

# ---- Triplet selection -----------------------------------------------------
# None -> default mode (scan every matching result folder on disk).
# A list -> triplet mode, e.g. ["MOp-MOs-VALVM", "MOp-MOs-VPMPO"]. Follows the
# pipeline's own hyperparameter unless overridden here.
SELECTED_TRIPLETS: Optional[List[Any]] = v4.SELECTED_TRIPLETS

TRIAL_TYPE: str = v4.TRIAL_TYPE
ALIGN_MODE: str = v4.ALIGN
Align_type_value = ALIGN_MODE.replace("_", " ")
if ALIGN_MODE == 'default_move_onset':
    Align_type_value = 'Move onset'

COMPONENT_INDEX: int = 0                       # which CC / PC to plot

# Sign-align every column's session panels to one another (on top of the
# within-session draw alignment), so the orientation is consistent down a
# column. Purely a display choice: flips the plotted traces only.
ALIGN_ACROSS_SESSIONS: bool = True

# ---- Default-mode-only (SELECTED_TRIPLETS is None) ------------------------
HUB_REGION: str = 'MOp'
ROI_REGIONS: Optional[List[str]] = None        # None -> every other region of the triplet
PCCA_VARIANT: str = 'regions_behavior'         # 'regions_only' | 'regions_behavior'
_VARIANT_TO_CONDITION: Dict[str, str] = {
    'regions_only':     'region_only',         # 3c
    'regions_behavior': 'region_behav',        # 4c
}

# ---- Triplet-mode-only: which analysis parts to draw ----------------------
ALL_PARTS: Tuple[str, ...] = ('1a', '1b', '2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c')
PARTS_TO_PLOT: List[str] = list(ALL_PARTS)

# ---- Plot styling (unchanged from v3) --------------------------------------
ROW_HEIGHT: float = 1.6
FIG_WIDTH: float = 6.5                         # per column
DRAW_LINE_ALPHA: float = 0.5
DRAW_LINE_WIDTH: float = 0.7
MEAN_LINE_WIDTH: float = 2.0
SEM_BAND_ALPHA: float = 0.22
TICK_FONTSIZE: int = 18
SAVE_DPI: int = 400

DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
}
HUB_MODE_BAND_COLORS: Dict[str, str] = {
    'MOs':   '#DD8452',
    'MOp':   '#55A868',
    'VALVM': '#4C72B0',
    'VPMPO': '#C44E52',
}
FALLBACK_PALETTE: List[str] = [
    "#8172B2", "#937860", "#64B5CD", "#CCB974",
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
]

OUTPUT_DIR: Path = (BASE_DIR / "Paper_output"
                    / f"pcca_all_regions_hubmode_v4__cumulative_sessions_{TRIAL_TYPE}_{ALIGN_MODE}"
                    / "task5_persession_draw_traces")


def _display_name(region: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(region, region)


def _lighten(hex_color: str, amount: float = 0.45) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _region_color(region: str) -> str:
    if region in HUB_MODE_BAND_COLORS:
        return HUB_MODE_BAND_COLORS[region]
    return FALLBACK_PALETTE[get_anatomical_index(region) % len(FALLBACK_PALETTE)]


# =============================================================================
# 2.  Data gathering -- one row of data per session per column: that
#     session's per-draw trial-averaged traces (light lines) plus its pooled
#     (draws x trials) mean/SEM (bold line + band). Same formulas as v3.
#
#     Draws are INDEPENDENT fits, so their arbitrary component signs are
#     aligned with `align_signs_spectral` BEFORE pooling (see v3). Only one
#     side is available per column, so the same stack is passed as both the u
#     and v argument and only `u_flip` is used.
# =============================================================================

def _sign_align_draws(draws_data: List[np.ndarray]) -> List[np.ndarray]:
    """Sign-align a list of per-draw (n_trials, T, K) latent arrays (all
    from the SAME region side) per component, before any pooling."""
    if len(draws_data) < 2:
        return draws_data

    draw_means = np.stack([arr.mean(axis=0) for arr in draws_data], axis=0)  # (n_draws, T, K)
    T = draw_means.shape[1]
    _, _, flip_decisions = align_signs_spectral(draw_means, draw_means, epoch=(0, T))

    signed: List[np.ndarray] = []
    for i, arr in enumerate(draws_data):
        arr = arr.copy()
        for comp_idx, decision in flip_decisions[i].items():
            if decision['u_flip']:
                arr[:, :, comp_idx] *= -1.0
        signed.append(arr)
    return signed


def _align_rows_across_sessions(rows: Dict[str, dict], sessions: List[str]) -> Dict[str, dict]:
    """Flip whole session rows (draw lines, mean; SEM is sign-free) so their
    trial-averaged traces share one orientation across `sessions`, via the same
    Z2 spectral sync applied to the per-session mean traces. Runs after the
    within-session draw alignment, on each session's pooled mean. Sessions
    whose time axis differs in length are left untouched."""
    names = [s for s in sessions if s in rows]
    if len(names) < 2:
        return rows
    T = len(rows[names[0]]['mean'])
    if any(len(rows[s]['mean']) != T for s in names):
        warnings.warn("Session time axes differ in length; skipping cross-session sign alignment.")
        return rows

    stack = np.stack([rows[s]['mean'] for s in names], axis=0)[:, :, None]   # (n_sessions, T, 1)
    _, _, flips = align_signs_spectral(stack, stack, epoch=(0, T))
    for i, name in enumerate(names):
        if flips[i][0]['u_flip']:
            row = rows[name]
            row['mean'] = -row['mean']
            row['draw_traces'] = [-t for t in row['draw_traces']]
    return rows


# A fetcher maps one loaded session to a list of (n_trials, T, K) latent
# arrays, one per draw (including 1a, which is now draws-based too).
Fetch = Callable[[PrivateLatentSessionResult], Optional[List[np.ndarray]]]


@dataclass
class FigureColumn:
    title: str
    color: str
    fetch: Fetch
    draw_based: bool = True


@dataclass
class FigureSpec:
    stem: str                    # file name without extension
    suptitle: str
    columns: List[FigureColumn]


def gather_column_rows(
        analyzer: PrivateLatentAnalyzer,
        sessions: List[str],
        column: FigureColumn,
        component_index: int = COMPONENT_INDEX,
) -> Dict[str, dict]:
    """``{session: {time_vec, draw_traces, mean, sem, n_trials_total,
    n_draws}}`` for every session that has data in this column."""
    rows: Dict[str, dict] = {}
    for session_name in sessions:
        session_result = analyzer.sessions.get(session_name)
        if session_result is None:
            continue
        arrs = column.fetch(session_result)
        if not arrs:
            continue
        arrs = [a for a in arrs if component_index < a.shape[2]]  # each (n_trials, T, K)
        if not arrs:
            continue
        if column.draw_based:
            arrs = _sign_align_draws(arrs)

        blocks = [a[:, :, component_index] for a in arrs]          # each (n_trials, T)
        pooled = np.concatenate(blocks, axis=0)                    # (n_draws * n_trials, T)
        rows[session_name] = dict(
            time_vec=session_result.time_vec,
            draw_traces=[b.mean(axis=0) for b in blocks] if column.draw_based else [],
            mean=pooled.mean(axis=0),
            sem=pooled.std(axis=0) / np.sqrt(pooled.shape[0]),
            n_trials_total=pooled.shape[0],
            n_draws=len(blocks) if column.draw_based else 0,
        )
    return rows


# =============================================================================
# 3.  Figure specs -- which columns make up each figure.
# =============================================================================

# Part -> (kind, condition, hub-latent attribute). See the module docstring.
PART_SPECS: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {
    '1a': ('region', None,           None),
    '1b': ('pair',   'direct',       None),
    '2a': ('hub',    'behav_only',   'latent_network'),
    '2b': ('hub',    'behav_only',   'latent_residual'),
    '2c': ('pair',   'behav_only',   None),
    '3a': ('hub',    'region_only',  'latent_network'),
    '3b': ('hub',    'region_only',  'latent_residual'),
    '3c': ('pair',   'region_only',  None),
    '4a': ('hub',    'region_behav', 'latent_network'),
    '4b': ('hub',    'region_behav', 'latent_residual'),
    '4c': ('pair',   'region_behav', None),
}


def _part_title(part: str) -> str:
    kind, cond, attr = PART_SPECS[part]
    if kind == 'region':
        return "1a  direct PCA -- no regression"
    if kind == 'pair':
        return PAIR_CONDITION_LABELS[cond]
    nuisance = {'behav_only': "behaviour", 'region_only': "third region",
                'region_behav': "third region + behaviour"}[cond]
    what = "explained" if attr == 'latent_network' else "residual"
    return f"{part}  hub PCA -- {nuisance} {what} component"


def _pair_side_fetch(pair_key: Tuple[str, str], cond: str, side: str) -> Fetch:
    attr = 'z_i_lat' if side == 'i' else 'z_j_lat'

    def fetch(sr: PrivateLatentSessionResult) -> Optional[List[np.ndarray]]:
        pr = sr.pairs.get(pair_key)
        if pr is None:
            return None
        return [getattr(d, attr) for d in pr.draws.get(cond, [])]
    return fetch


def _hub_fetch(pair_key: Tuple[str, str], hub: str, cond: str, attr: str) -> Fetch:
    def fetch(sr: PrivateLatentSessionResult) -> Optional[List[np.ndarray]]:
        hp = sr.hub_pca_pairs.get(pair_key)
        if hp is None:
            return None
        table = hp.region_i_as_hub_draws if hub == pair_key[0] else hp.region_j_as_hub_draws
        return [getattr(d, attr) for d in table.get(cond, [])]
    return fetch


def _region_fetch(region: str) -> Fetch:
    def fetch(sr: PrivateLatentSessionResult) -> Optional[List[np.ndarray]]:
        r = sr.region_pca_raw.get(region)
        if r is None:
            return None
        return [d.latent for d in r.draws]
    return fetch


def build_part_figures(triplet: TripletSpec, part: str) -> List[FigureSpec]:
    """Every figure of one analysis part for one triplet."""
    kind, cond, attr = PART_SPECS[part]
    title = _part_title(part)
    specs: List[FigureSpec] = []

    if kind == 'region':
        for region in triplet.regions:
            specs.append(FigureSpec(
                stem=f"{part}_{region}",
                suptitle=f"{title}\n{_display_name(region)}",
                columns=[FigureColumn(_display_name(region), _region_color(region),
                                      _region_fetch(region))],
            ))
        return specs

    for region_i, region_j, third in triplet_pairs_with_third(triplet.regions):
        pair_key = (region_i, region_j)
        if kind == 'pair':
            specs.append(FigureSpec(
                stem=f"{part}_{region_i}_{region_j}",
                suptitle=f"{title}\n{_display_name(region_i)} ↔ {_display_name(region_j)}"
                         f"  (third: {_display_name(third)})",
                columns=[
                    FigureColumn(_display_name(region_i), _region_color(region_i),
                                 _pair_side_fetch(pair_key, cond, 'i')),
                    FigureColumn(_display_name(region_j), _region_color(region_j),
                                 _pair_side_fetch(pair_key, cond, 'j')),
                ],
            ))
        else:  # hub orientation: each region of the pair as the hub
            for hub, partner in ((region_i, region_j), (region_j, region_i)):
                specs.append(FigureSpec(
                    stem=f"{part}_{hub}_hub_{partner}_third_{third}",
                    suptitle=f"{title}\n{_display_name(hub)} as hub, partner "
                             f"{_display_name(partner)}  (third: {_display_name(third)})",
                    columns=[FigureColumn(_display_name(hub), _region_color(hub),
                                          _hub_fetch(pair_key, hub, cond, attr))],
                ))
    return specs


def build_default_figures(triplet: TripletSpec) -> List[FigureSpec]:
    """Default mode: v3's Task-5 figure -- hub-side latent, one figure per
    (HUB_REGION, partner) pair of this triplet."""
    if HUB_REGION not in triplet.regions:
        return []
    cond = _VARIANT_TO_CONDITION.get(PCCA_VARIANT, PCCA_VARIANT)
    if cond not in PAIR_CONDITIONS:
        raise ValueError(f"PCCA_VARIANT={PCCA_VARIANT!r} does not map to one of {PAIR_CONDITIONS}")
    specs: List[FigureSpec] = []
    for partner in triplet.regions:
        if partner == HUB_REGION or (ROI_REGIONS is not None and partner not in ROI_REGIONS):
            continue
        pair_key = sort_pair_by_anatomy(HUB_REGION, partner)
        side = 'i' if pair_key[0] == HUB_REGION else 'j'
        specs.append(FigureSpec(
            stem=f"task5_persession_draws_{PCCA_VARIANT}_{HUB_REGION}_{partner}_comp{COMPONENT_INDEX}",
            suptitle=f"{_display_name(HUB_REGION)} ↔ {_display_name(partner)}  "
                     f"(comp {COMPONENT_INDEX}, {PCCA_VARIANT})",
            columns=[FigureColumn(_display_name(HUB_REGION), _region_color(HUB_REGION),
                                  _pair_side_fetch(pair_key, cond, side))],
        ))
    return specs


# =============================================================================
# 4.  Plotting -- N_session rows x n_columns, one figure per FigureSpec.
# =============================================================================

def plot_figure(
        spec: FigureSpec,
        sessions: List[str],
        rows_by_col: List[Dict[str, dict]],
        triplet: TripletSpec,
        save_path: Path,
        row_height: float = ROW_HEIGHT,
        fig_width: float = FIG_WIDTH,
        dpi: int = SAVE_DPI,
) -> None:
    keep = [s for s in sessions if all(s in rows for rows in rows_by_col)]
    if not keep:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return

    n_rows, n_cols = len(keep), len(spec.columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width * n_cols, row_height * n_rows),
                             sharex=True, sharey='col', squeeze=False)

    for c, (column, rows) in enumerate(zip(spec.columns, rows_by_col)):
        color = column.color
        light_color = _lighten(color, 0.55)
        for r, session_name in enumerate(keep):
            ax, row = axes[r, c], rows[session_name]
            for trace in row['draw_traces']:
                ax.plot(row['time_vec'], trace, color=light_color,
                        linewidth=DRAW_LINE_WIDTH, alpha=DRAW_LINE_ALPHA, zorder=1)

            mean_trace, sem_trace = row['mean'], row['sem']
            ax.fill_between(row['time_vec'], mean_trace - sem_trace, mean_trace + sem_trace,
                            color=color, alpha=SEM_BAND_ALPHA, zorder=2)
            ax.plot(row['time_vec'], mean_trace, color=color,
                    linewidth=MEAN_LINE_WIDTH, alpha=0.95, zorder=3)

            ax.axvline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
            if row['n_draws']:
                info = f"n={row['n_draws']} draws, {row['n_trials_total']} draws×trials"
            else:
                info = f"{row['n_trials_total']} trials"
            ax.text(0.01, 0.90, f"{session_name}  ({info})" if c == 0 else f"({info})",
                    transform=ax.transAxes, fontsize=TICK_FONTSIZE - 7, va='top', ha='left')
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)
        if n_cols > 1:
            axes[0, c].set_title(column.title, fontsize=TICK_FONTSIZE - 4, color=color)
        axes[-1, c].set_xlabel("Time from reach (s)", fontsize=TICK_FONTSIZE - 2)
        axes[-1, c].tick_params(axis='x', labelsize=TICK_FONTSIZE - 2)

    fig.suptitle(f"{spec.suptitle}  (comp {COMPONENT_INDEX})\n"
                 f"{triplet.label} · {Align_type_value}", fontsize=TICK_FONTSIZE)
    fig.tight_layout(h_pad=0.15, rect=(0.0, 0.0, 1.0, 0.94))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)


def render_specs(
        analyzer: PrivateLatentAnalyzer,
        triplet: TripletSpec,
        sessions: List[str],
        specs: List[FigureSpec],
        out_dir: Path,
) -> None:
    for spec in specs:
        rows_by_col = [gather_column_rows(analyzer, sessions, col) for col in spec.columns]
        shown = [s for s in sessions if all(s in rows for rows in rows_by_col)]
        if ALIGN_ACROSS_SESSIONS:
            rows_by_col = [_align_rows_across_sessions(rows, shown) for rows in rows_by_col]
        n = len(shown)
        print(f"  [{spec.stem}] {n} session(s) with data")
        plot_figure(spec, sessions, rows_by_col, triplet, out_dir / f"{spec.stem}.png")


# =============================================================================
# 5.  Triplet discovery + driver
# =============================================================================

def discover_triplets_on_disk(
        base_dir: Path = BASE_DIR,
        trial_type: str = TRIAL_TYPE,
        align_mode: str = ALIGN_MODE,
) -> List[TripletSpec]:
    """Default mode: every ``pcca_all_regions_out_behaviour_v4_hub_<A>_<B>_<C>_
    <trial_type>_<align_mode>_results`` folder under `base_dir`."""
    prefix = "pcca_all_regions_out_behaviour_v4_hub_"
    suffix = f"_{trial_type}_{align_mode}_results"
    triplets: List[TripletSpec] = []
    for folder in sorted(Path(base_dir).glob(f"{prefix}*{suffix}")):
        if not folder.is_dir():
            continue
        regions = tuple(folder.name[len(prefix):-len(suffix)].split("_"))
        if len(regions) != 3:
            warnings.warn(f"Skipping {folder.name}: could not parse 3 regions from its name.")
            continue
        triplets.append(TripletSpec(label=" ∩ ".join(regions), regions=regions,
                                    sessions=(), n_expected=0))
    return triplets


def _load_triplet(triplet: TripletSpec) -> Optional[PrivateLatentAnalyzer]:
    analyzer = PrivateLatentAnalyzer(triplet, base_dir=BASE_DIR, trial_type=TRIAL_TYPE,
                                     align_mode=ALIGN_MODE)
    if not analyzer.available_sessions():
        warnings.warn(f"No result pickles for {triplet.label!r} in {analyzer.results_dir}; "
                      f"run pCCA_all_regions_out_behaviour_v4.py for it first. Skipped.")
        return None
    analyzer.load_all()
    return analyzer


def main() -> None:
    triplet_mode = SELECTED_TRIPLETS is not None
    print("=" * 70)
    print("HUB-MODE v4 Task 5 (per-session, per-draw trace) figures")
    print("=" * 70)
    print(f"  mode               : {'triplet (parts 1a-4c)' if triplet_mode else 'default (all folders on disk)'}")
    print(f"  trial type / align : {TRIAL_TYPE} / {ALIGN_MODE}")
    print(f"  component index    : {COMPONENT_INDEX}")
    if triplet_mode:
        print(f"  selected triplets  : {SELECTED_TRIPLETS}")
        print(f"  parts              : {PARTS_TO_PLOT}")
    else:
        print(f"  hub region         : {HUB_REGION}   (ROI: {ROI_REGIONS or 'all partners in triplet'})")
        print(f"  pcca variant       : {PCCA_VARIANT}")
    print(f"  output directory   : {OUTPUT_DIR}")
    print("=" * 70)

    triplets = (build_hub_triplets(selected=SELECTED_TRIPLETS) if triplet_mode
                else discover_triplets_on_disk())
    if not triplets:
        print("  no triplets to process.")
        return

    for triplet in triplets:
        print(f"\n### triplet {triplet.label}")
        analyzer = _load_triplet(triplet)
        if analyzer is None:
            continue
        sessions = sorted(analyzer.sessions)
        if triplet_mode:
            missing = [s for s in triplet.sessions if s not in analyzer.sessions]
            if missing:
                warnings.warn(f"{triplet.label}: {len(missing)} expected session(s) have no "
                              f"pickle yet: {missing}")
            for part in PARTS_TO_PLOT:
                print(f" -- part {part}: {_part_title(part)}")
                render_specs(analyzer, triplet, sessions, build_part_figures(triplet, part),
                             OUTPUT_DIR / "selected_triplets" / triplet.slug / f"part_{part}")
        else:
            specs = build_default_figures(triplet)
            if not specs:
                print(f"  {HUB_REGION!r} not in this triplet; skipped.")
                continue
            render_specs(analyzer, triplet, sessions, specs,
                         OUTPUT_DIR / "all_triplets" / triplet.slug)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
PCA_hub_variance_gini_boxplots.py
================================================================================

Cross-session box-plot visualisations of two per-session, per-region/per-hub
quantities that ``pcca_all_regions_out_behaviour.py`` already computes and
pickles as auxiliary PCA output -- never refit here, only read back:

  * ``explained_variance_ratio`` / ``explained_variance_ratio_network`` /
    ``explained_variance_ratio_residual`` -- the fraction of that region's
    (or that hub-orientation's network/residual component's) own variance
    captured by one PCA component.
  * ``W`` / ``W_network`` / ``W_residual`` -- the PCA loading matrix that
    component was fit with, one column per component, one row per neuron --
    reduced here to the Gini coefficient of |W[:, component_idx]|, a
    standard sparsity/inequality measure: 0 means every neuron contributes
    equally to that component, 1 means a single neuron accounts for all of
    it.

Both quantities are read via ``PrivateLatentAnalyzer.get_region_pca`` (Parts
1a/1b) and ``.get_hub_pca`` (Parts 2a/2b), exactly the same accessors
``PCA_latent_extrenal_variable_part.py`` uses -- this script follows that
file's hub-region display framework (a small, explicit HUB_REGIONS list,
each paired against every region in a separately-configurable PAIRED_REGIONS
list, anatomically ordered) but does not import from it: only
``pCCA_all_regions_out_behaviour.py``'s small canonicalisation/analyzer
primitives are reused (project convention: primitives copied, not imported,
so this script stays independently auditable and does not depend on a
sibling script that happens to share a naming typo). Unlike that file, this
script never touches raw spike tensors, behaviour regressors, or the
CrossSession*Analyzer sign-alignment machinery -- every value plotted here
is a single per-session scalar (an explained-variance fraction or a Gini
coefficient), so there is no cross-session sign or trace-alignment step to
perform; "cross-session" here means nothing more than "one box, one dot per
session, pooled across every loaded session".

--------------------------------------------------------------------------------
Figure layout
--------------------------------------------------------------------------------
For EACH hub region in HUB_REGIONS, a 1x3 box-plot figure:

    Panel 1  Stage 1a ("raw") vs. Stage 1b ("behaviour regressed out")
             explained variance / Gini, two boxes, for the hub region
             itself.
    Panel 2  Stage 2a ("network"-explained component) explained variance /
             Gini, one box per OTHER region in PAIRED_REGIONS the hub is
             paired with, in anatomical order.
    Panel 3  Stage 2b ("residual" component) explained variance / Gini,
             same partners, same anatomical order.

When HUB_REGIONS has exactly four entries, an ADDITIONAL combined 4x3
figure is produced (one row per hub, same three columns) -- see
``plot_hub_grid_boxplots``, which is not hardcoded to four rows and works
for any HUB_REGIONS length if called directly.

Box style: white-filled box + black median/whiskers (standard box plot),
with every session's own value drawn as a small jittered, category-coloured
dot on top -- the box summarises the distribution, the dots show every
session that went into it, matching the reference figure this script's
style was requested to follow.

Two independent driver passes -- ``_run_variance_boxplots`` (explained
variance, Requirement 2) and ``_run_gini_boxplots`` (Gini coefficient,
Requirement 3) -- share the SAME layout code (``build_hub_panel_data`` /
``plot_hub_grid_boxplots``) but never share plotted data with each other.

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import csv
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')

# =============================================================================
# 0.  Imports -- only pCCA_all_regions_out_behaviour.py's own analyzer +
#     canonicalisation primitives are reused (imported, not copied, since
#     they are the actual read path for the pickled auxiliary output this
#     script visualises); everything display-specific (region display
#     names, colours, box/dot drawing) is written fresh here, per this
#     project's "primitives copied, not imported" convention.
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pCCA_all_regions_out_behaviour import (  # noqa: E402
    PrivateLatentAnalyzer,
    PrivateLatentSessionResult,
    PrivateLatentPairResult,
    RegionPCAResult,
    HubOrientationPCAResult,
    HubPairPCAResult,
    HUB_REGIONS as ALL_REGIONS_OF_INTEREST,
    N_PCA_COMPONENTS,
    get_anatomical_index,
)

# Same fix as PCA_latent_extrenal_variable_part.py / pCCA_latent_extrenal_
# variable_bar.py: pcca_all_regions_out_behaviour.py is normally *run*
# directly (as '__main__'), so pickle bakes that module name into every
# saved dataclass instance; unpickling from a DIFFERENT '__main__' (this
# script) requires the same classes reachable under '__main__' here too.
sys.modules['__main__'].PrivateLatentSessionResult = PrivateLatentSessionResult
sys.modules['__main__'].PrivateLatentPairResult = PrivateLatentPairResult
sys.modules['__main__'].RegionPCAResult = RegionPCAResult
sys.modules['__main__'].HubOrientationPCAResult = HubOrientationPCAResult
sys.modules['__main__'].HubPairPCAResult = HubPairPCAResult


# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS
# =============================================================================

# ---- Which pickle folder to read (one trial type -- this script only
#      ever visualises ONE condition's own auxiliary PCA output at a time).
TRIAL_TYPE: str = 'cued_hit_long'

# ---- Paths ------------------------------------------------------------------
BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
OUTPUT_DIR = BASE_DIR / "Paper_output" / f"pca_hub_variance_gini_boxplots_{TRIAL_TYPE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Sessions (same list PCA_latent_extrenal_variable_part.py uses; a
#      session missing from the loaded pickles is simply skipped). ---------
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

# ---- (Requirement 1) which PCA component to plot -- 0-indexed, default the
#      first component; freely change to inspect any other fitted component.
COMPONENT_INDEX: int = 0

# ---- (Requirement 1) hub regions and their paired regions, each its own
#      independent, freely-extendable/reducible hyperparameter -- mirrors
#      HUB_MODE_HUB_REGIONS / HUB_MODE_ROI_REGIONS in PCA_latent_extrenal_
#      variable_part.py's hub-region framework, but declared fresh here
#      rather than imported, so editing one script's lists never affects
#      the other's.
HUB_REGIONS: List[str] = ['MOs', 'MOp', 'VALVM', 'VPMPO']       # MOs, MOp, motor thalamus, sensory thalamus
PAIRED_REGIONS: List[str] = list(ALL_REGIONS_OF_INTEREST)       # every region a hub may be paired with

# ---- Minimum sessions a box needs to be drawn at all (mirrors this
#      project's MIN_SESSIONS_THRESHOLD convention; declared locally rather
#      than imported so this script does not need cross_trial_type_pca_
#      analysis.py's heavier import surface for a single int constant).
MIN_SESSIONS: int = 3

# ---- Fixed y-limits, INDEPENDENT per column -- [Stage 1a/1b, Network (2a),
#      Residual (2b)] -- None in a slot autoscales that column from its own
#      data. Gini is bounded in [0, 1] by construction, so a fixed axis is
#      the sensible default for all three of its columns; explained
#      variance has no natural fixed ceiling across arbitrary datasets, so
#      it autoscales unless overridden per column.
EXPLAINED_VAR_YLIM: List[Optional[Tuple[float, float]]] = [(0.0, 10), (0.0, 10), (0.0, 40)]
GINI_YLIM: List[Optional[Tuple[float, float]]] = [(0.25, 0.8), (0.25, 0.8), (0.25, 0.8)]

# ---- Relative column widths [Stage 1a/1b, Network (2a), Residual (2b)] --
#      the Stage 1a/1b panel only ever has two boxes, vs. one box per
#      paired region (typically more) in the other two, so it defaults
#      narrower than them.
PANEL_WIDTH_RATIOS: Tuple[float, float, float] = (0.8, 2.0, 2.0)

# ---- Display names -- copied from PCA_latent_extrenal_variable_part.py's
#      own DISPLAY_NAME_OVERRIDES, so axis labels read identically across
#      both scripts' figures.
DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
}


def _display_name(region: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(region, region)


# ---- Category colours -- one fixed colour per region code (panels 2/3),
#      plus a separate two-colour pair for panel 1's "1a vs 1b" stages.
#      Boxes themselves stay white/black (see _draw_box_dot_panel); only
#      the overlaid per-session dots are coloured, matching the reference
#      figure's box+coloured-swarm style.
REGION_COLOR_PALETTE: Dict[str, str] = {
    'ORB':   '#20A39E',   # teal
    'MOp':   '#3D5A80',   # slate blue
    'MOs':   '#F2AE30',   # amber
    'STR':   '#4F9D69',   # green
    'VALVM': '#E07A5F',   # motor Thal -- orange
    'VPMPO': '#9B5DE5',   # sens Thal -- purple
    'HY':    '#F15BB5',   # pink
}
STAGE_COLORS: Dict[str, str] = {
    'Raw':                  '#4C72B0',
    'Behav regressed':  '#DD8452',
}

# ---- Style knobs -------------------------------------------------------------
BOX_WIDTH: float = 0.55
DOT_JITTER_FRAC: float = 0.55
DOT_SIZE: float = 26.0
TICK_FONTSIZE: int = 16
SAVE_DPI: int = 400


def _stage_color(stage: str) -> str:
    return STAGE_COLORS.get(stage, '#888888')


def _region_color(region: str) -> str:
    return REGION_COLOR_PALETTE.get(region, '#888888')


# =============================================================================
# 2.  Gini coefficient -- standard inequality measure applied to |W[:,
#     component_idx]|. A PCA loading is signed (arbitrary up to a per-
#     component flip, per pCCA_all_regions_out_behaviour.py's own "Note on
#     scope"), so the coefficient is computed on MAGNITUDES: "how unevenly
#     is this component's variance-capturing weight distributed across
#     neurons", not "how unevenly signed". 0 = every neuron contributes
#     equally; 1 = a single neuron accounts for the entire component.
# =============================================================================
def gini_coefficient(x: np.ndarray) -> float:
    x = np.abs(np.asarray(x, dtype=np.float64))
    total = x.sum()
    if total <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x_sorted) - (n + 1) * total) / (n * total))


# =============================================================================
# 3.  Per-session metric extractors -- explained variance (Requirement 2)
#     and Gini-of-weights (Requirement 3), each reading straight off the
#     dataclass fields named in this file's own docstring.
# =============================================================================
def _explained_variance_pct(evr: np.ndarray, component_idx: int) -> Optional[float]:
    """`evr` : (K,) explained_variance_ratio / _network / _residual, a
    fraction in [0, 1] -- returned here as a percentage."""
    if component_idx >= evr.shape[0]:
        return None
    return float(evr[component_idx]) * 100.0


def _gini_of_component(W: np.ndarray, component_idx: int) -> Optional[float]:
    """`W` : (n_neurons, K) PCA loading matrix / _network / _residual."""
    if component_idx >= W.shape[1]:
        return None
    return gini_coefficient(W[:, component_idx])


# =============================================================================
# 4.  Data gathering -- builds the three panels' {category: (n_sessions,)
#     array} dicts for ONE hub region, given a metric-extraction callback
#     per stage (so the SAME function serves both Requirement 2 and
#     Requirement 3 -- only the callbacks differ).
# =============================================================================
def _sessions_present(analyzer: PrivateLatentAnalyzer, sessions: List[str]) -> List[str]:
    return [s for s in sessions if s in analyzer.sessions]


def build_hub_panel_data(
        hub: str,
        paired_regions: List[str],
        analyzer: PrivateLatentAnalyzer,
        sessions: List[str],
        region_metric_fn: Callable[[RegionPCAResult], Optional[float]],
        hub_network_metric_fn: Callable[[HubOrientationPCAResult], Optional[float]],
        hub_residual_metric_fn: Callable[[HubOrientationPCAResult], Optional[float]],
        min_sessions: int = MIN_SESSIONS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Returns (panel1, panel2, panel3):
      panel1 = {'1a (raw)': array, '1b (behaviour regressed)': array}   -- Part 1a/1b, hub region only
      panel2 = {partner: array, ...}   (anatomical order)                -- Part 2a ("network"), hub vs. every partner
      panel3 = {partner: array, ...}   (anatomical order)                -- Part 2b ("residual"), same partners
    Categories with fewer than `min_sessions` values are dropped."""
    panel1_raw: Dict[str, List[float]] = {'Raw': [], 'Behav regressed': []}
    for session in sessions:
        r1a = analyzer.get_region_pca(session, hub, out_behaviour=False)
        if r1a is not None:
            v = region_metric_fn(r1a)
            if v is not None:
                panel1_raw['Raw'].append(v)
        r1b = analyzer.get_region_pca(session, hub, out_behaviour=True)
        if r1b is not None:
            v = region_metric_fn(r1b)
            if v is not None:
                panel1_raw['Behav regressed'].append(v)

    ordered_partners = sorted(
        {r for r in paired_regions if r != hub}, key=get_anatomical_index)
    panel2_raw: Dict[str, List[float]] = {p: [] for p in ordered_partners}
    panel3_raw: Dict[str, List[float]] = {p: [] for p in ordered_partners}
    for partner in ordered_partners:
        for session in sessions:
            hpca = analyzer.get_hub_pca(session, hub, partner)
            if hpca is None:
                continue
            vr = hub_residual_metric_fn(hpca)
            if vr is not None:
                panel2_raw[partner].append(vr)
            vn = hub_network_metric_fn(hpca)
            if vn is not None:
                panel3_raw[partner].append(vn)


    def _finalize(raw: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        out = {}
        for cat, vals in raw.items():
            if len(vals) < min_sessions:
                continue
            out[cat] = np.asarray(vals, dtype=np.float64)
        return out

    return _finalize(panel1_raw), _finalize(panel2_raw), _finalize(panel3_raw)


# =============================================================================
# 5.  Plotting -- one box-plus-jittered-dots panel, and the n_hub x 3 grid
#     of panels built from it.
# =============================================================================
def _draw_box_dot_panel(
        ax: plt.Axes,
        data_by_category: Dict[str, np.ndarray],
        color_fn: Callable[[str], str],
        display_fn: Callable[[str], str] = _display_name,
        ylabel: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
) -> None:
    """White box + black median/whiskers (standard box plot), one jittered,
    category-coloured dot per session on top."""
    categories = list(data_by_category.keys())
    if not categories:
        ax.axis('off')
        return

    positions = np.arange(len(categories))
    values = [data_by_category[c] for c in categories]

    ax.boxplot(
        values, positions=positions, widths=BOX_WIDTH, patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.8),
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.3),
        whiskerprops=dict(color='black', linewidth=1.3),
        capprops=dict(color='black', linewidth=1.3),
        zorder=2,
    )

    rng = np.random.default_rng(0)
    for pos, cat, vals in zip(positions, categories, values):
        color = color_fn(cat)
        jitter = rng.uniform(-BOX_WIDTH / 2 * DOT_JITTER_FRAC,
                             BOX_WIDTH / 2 * DOT_JITTER_FRAC, size=vals.size)
        ax.scatter(pos + jitter, vals, s=DOT_SIZE, color=color, alpha=0.9,
                  linewidths=0, zorder=3)

    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([display_fn(c) for c in categories], fontsize=TICK_FONTSIZE - 2,
                       rotation=35, ha='right', rotation_mode='anchor')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=TICK_FONTSIZE)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title, fontsize=TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 2)


PanelData = Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]
PANEL_TITLES: Tuple[str, str, str] = ("Single region", "Residual", "Shared part")


def plot_hub_grid_boxplots(
        hub_list: List[str],
        panel_data_by_hub: Dict[str, PanelData],
        value_label: str,
        save_path: Path,
        column_ylims: Tuple[Optional[Tuple[float, float]], ...] = (None, None, None),
        width_ratios: Tuple[float, float, float] = PANEL_WIDTH_RATIOS,
        panel_width: float = 2.6,
        row_height: float = 3.8,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    """(2a) n_hub_list x 3 grid of box-plot panels -- ONE row per hub in
    `hub_list`, columns = [Stage 1a/1b, Network (2a), Residual (2b)], the
    first narrower than the other two by `width_ratios` (default 1:2:2,
    see PANEL_WIDTH_RATIOS). `column_ylims` sets each column's y-axis
    independently (None = autoscale that column). Not hardcoded to any
    particular `hub_list` length: called once per hub (n=1) for each hub's
    own figure, and once more with all four hubs (n=4) for the combined
    figure when HUB_REGIONS has exactly four entries -- see (2b) and
    `_run_variance_boxplots` / `_run_gini_boxplots`."""
    n_hubs = len(hub_list)
    if n_hubs == 0:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    fig_width = panel_width * sum(width_ratios)
    fig, axes = plt.subplots(
        n_hubs, 3, figsize=(fig_width, row_height * n_hubs), squeeze=False,
        gridspec_kw={'width_ratios': list(width_ratios)},
    )

    for row, hub in enumerate(hub_list):
        panel1, panel2, panel3 = panel_data_by_hub[hub]
        for col, data in enumerate((panel1, panel2, panel3)):
            ax = axes[row][col]
            color_fn = _stage_color if col == 0 else _region_color
            title = PANEL_TITLES[col] if row == 0 else None
            _draw_box_dot_panel(
                ax, data, color_fn, _display_name,
                ylabel=(value_label if col == 0 else None),
                ylim=column_ylims[col], title=title,
            )
        axes[row][0].text(
            -0.55, 0.5, _display_name(hub), transform=axes[row][0].transAxes,
            fontsize=TICK_FONTSIZE + 2, fontweight='bold', va='center', ha='right', rotation=90,
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 6.  Pairwise statistical testing -- flattens every hub's three panels
#     into one pool of (hub, panel, category, values) boxes and runs a
#     two-sided Mann-Whitney U test (nonparametric, unpaired -- boxes being
#     compared do not in general share the same session set, so no attempt
#     is made to pair by session identity) between EVERY pair drawn from
#     that pool. Since the pool already contains every box in every hub's
#     figure (both Stage 1a/1b boxes plus every paired-region network/
#     residual box), this single all-pairs pass covers both halves of the
#     request at once: pairs sharing a hub are the "within each hub region"
#     comparisons, pairs from different hubs are the "across all hub
#     regions" comparisons -- distinguished in the output by the `scope`
#     column rather than being two separate computations.
# =============================================================================
def _flatten_panel_entries(
        panel_data_by_hub: Dict[str, PanelData],
) -> List[Tuple[str, str, str, np.ndarray]]:
    """Every hub's three panels, flattened into one list of (hub,
    panel_name, category, values) boxes -- the pool `run_pairwise_
    significance` draws every comparison from."""
    panel_names = ('stage_1a_1b', 'network', 'residual')
    entries: List[Tuple[str, str, str, np.ndarray]] = []
    for hub, panels in panel_data_by_hub.items():
        for panel_name, data in zip(panel_names, panels):
            for category, values in data.items():
                entries.append((hub, panel_name, category, values))
    return entries


def run_pairwise_significance(
        panel_data_by_hub: Dict[str, PanelData],
        metric_name: str,
        save_path: Path,
        min_n: int = 2,
) -> None:
    """(Requirement 4) Pairwise Mann-Whitney U test between every two boxes
    in `panel_data_by_hub`, written to `save_path` as one row per pair. A
    box with fewer than `min_n` sessions is still listed against every
    other box but gets an empty p_value/u_statistic (too few points to
    test) rather than being silently dropped from the CSV."""
    entries = _flatten_panel_entries(panel_data_by_hub)
    rows: List[dict] = []
    for (hub_a, panel_a, cat_a, vals_a), (hub_b, panel_b, cat_b, vals_b) in combinations(entries, 2):
        scope = 'within_hub' if hub_a == hub_b else 'across_hub'
        if vals_a.size < min_n or vals_b.size < min_n:
            u_stat, p_val = '', ''
        else:
            u_stat, p_val = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            u_stat, p_val = float(u_stat), float(p_val)
        rows.append(dict(
            metric=metric_name, scope=scope,
            hub_a=hub_a, panel_a=panel_a, category_a=cat_a, n_a=vals_a.size, mean_a=float(vals_a.mean()),
            hub_b=hub_b, panel_b=panel_b, category_b=cat_b, n_b=vals_b.size, mean_b=float(vals_b.mean()),
            u_statistic=u_stat, p_value=p_val,
        ))

    if not rows:
        print(f"  [stats] nothing to test for {save_path.name}; skipping.")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  [stats] {len(rows)} pairwise tests -> {save_path}")


# =============================================================================
# 7.  Drivers -- Requirement 2 (explained variance) and Requirement 3 (Gini
#     of weights) are independent passes over the SAME hub list, sharing
#     only the layout/stats code above; neither pass reads the other's data.
# =============================================================================
def _run_variance_boxplots(analyzer: PrivateLatentAnalyzer, sessions: List[str]) -> None:
    print("\n" + "#" * 70)
    print(f"# Requirement 2 -- explained variance, component {COMPONENT_INDEX}")
    print("#" * 70)

    panel_data_by_hub: Dict[str, PanelData] = {}
    for hub in HUB_REGIONS:
        panel_data_by_hub[hub] = build_hub_panel_data(
            hub, PAIRED_REGIONS, analyzer, sessions,
            region_metric_fn=lambda r: _explained_variance_pct(r.explained_variance_ratio, COMPONENT_INDEX),
            hub_residual_metric_fn=lambda h: _explained_variance_pct(h.explained_variance_ratio_residual,COMPONENT_INDEX),
            hub_network_metric_fn=lambda h: _explained_variance_pct(h.explained_variance_ratio_network, COMPONENT_INDEX),

        )
        plot_hub_grid_boxplots(
            [hub], {hub: panel_data_by_hub[hub]}, "Explained variance (%)",
            OUTPUT_DIR / f"explained_variance_comp{COMPONENT_INDEX}_{hub}.png",
            column_ylims=EXPLAINED_VAR_YLIM,
        )

    if len(HUB_REGIONS) == 4:
        plot_hub_grid_boxplots(
            HUB_REGIONS, panel_data_by_hub, "Explained variance (%)",
            OUTPUT_DIR / f"explained_variance_comp{COMPONENT_INDEX}_combined.png",
            column_ylims=EXPLAINED_VAR_YLIM,
        )

    run_pairwise_significance(
        panel_data_by_hub, "explained_variance",
        OUTPUT_DIR / f"explained_variance_comp{COMPONENT_INDEX}_pairwise_pvalues.csv",
    )


def _run_gini_boxplots(analyzer: PrivateLatentAnalyzer, sessions: List[str]) -> None:
    print("\n" + "#" * 70)
    print(f"# Requirement 3 -- Gini coefficient of W, component {COMPONENT_INDEX}")
    print("#" * 70)

    panel_data_by_hub: Dict[str, PanelData] = {}
    for hub in HUB_REGIONS:
        panel_data_by_hub[hub] = build_hub_panel_data(
            hub, PAIRED_REGIONS, analyzer, sessions,
            region_metric_fn=lambda r: _gini_of_component(r.W, COMPONENT_INDEX),
            hub_residual_metric_fn=lambda h: _gini_of_component(h.W_residual, COMPONENT_INDEX),
            hub_network_metric_fn=lambda h: _gini_of_component(h.W_network, COMPONENT_INDEX),
        )
        plot_hub_grid_boxplots(
            [hub], {hub: panel_data_by_hub[hub]}, "Gini coefficient",
            OUTPUT_DIR / f"weight_gini_comp{COMPONENT_INDEX}_{hub}.png",
            column_ylims=GINI_YLIM,
        )

    if len(HUB_REGIONS) == 4:
        plot_hub_grid_boxplots(
            HUB_REGIONS, panel_data_by_hub, "Gini coefficient",
            OUTPUT_DIR / f"weight_gini_comp{COMPONENT_INDEX}_combined.png",
            column_ylims=GINI_YLIM,
        )

    run_pairwise_significance(
        panel_data_by_hub, "gini_coefficient",
        OUTPUT_DIR / f"weight_gini_comp{COMPONENT_INDEX}_pairwise_pvalues.csv",
    )


def main() -> None:
    print("=" * 70)
    print("HUB-REGION PCA WEIGHT / EXPLAINED-VARIANCE BOX PLOTS")
    print("(reads pcca_all_regions_out_behaviour.py's own pickled auxiliary")
    print(" output -- W / W_network / W_residual and explained_variance_ratio")
    print(" / _network / _residual -- never refits anything)")
    print("=" * 70)
    print(f"  trial type       : {TRIAL_TYPE}")
    print(f"  component index  : {COMPONENT_INDEX}  (of {N_PCA_COMPONENTS} fit)")
    print(f"  hub regions      : {HUB_REGIONS}")
    print(f"  paired regions   : {PAIRED_REGIONS}")
    print(f"  min sessions/box : {MIN_SESSIONS}")
    print(f"  output directory : {OUTPUT_DIR}")
    print("=" * 70)

    analyzer = PrivateLatentAnalyzer(base_dir=BASE_DIR, trial_type=TRIAL_TYPE)
    analyzer.load_all()
    sessions = _sessions_present(analyzer, SESSIONS)
    print(f"  sessions loaded  : {len(sessions)} (of {len(SESSIONS)} configured)")

    _run_variance_boxplots(analyzer, sessions)
    _run_gini_boxplots(analyzer, sessions)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

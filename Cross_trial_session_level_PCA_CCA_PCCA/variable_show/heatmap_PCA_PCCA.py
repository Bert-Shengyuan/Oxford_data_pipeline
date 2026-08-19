#!/usr/bin/env python3
r"""
cross_trial_type_region_variable_heatmap.py
================================================================================

Fig. 4d-style (Peng et al., bioRxiv 2024.11.04.621878v2) heatmaps of
behavioural-GLM R^2, built directly on top of the two existing peak/delta/
variance scripts in this pipeline:

  * PCA_latent_extrenal_variable_bar.py   (pca_cross_session_peak_delta_
                                            variance.py)  -- single-region PCA
  * pCCA_latent_extrenal_variable_bar.py  (cross_trial_type_peak_delta_
                                            reward_variance.py) -- pairwise
                                            (p)CCA, KERNEL_MODE = 'pcca'/'cca'

Both parent scripts already compute and cache exactly the R^2 values this
script needs, in behavior_variance_records.csv (tasks 3-4 of each parent
script). This script does no new numerical work -- it is a pure re-slicing
and re-plotting layer on top of already-cached results:

  parent scripts' layout : one figure, rows = external variable
                            (position/speed/reward...), columns = region
                            (or region pair)
  THIS script's layout   : one figure PER hub region, rows = (analysis,
                            hub, partner) triplet, columns = external
                            variable

Row construction, per hub region h in REGIONS_OF_INTEREST
--------------------------------------------------------------------------
    row 0        PCA(h)
    row 1..N     PCCA(h, partner) for every OTHER region "partner" in
                 REGIONS_OF_INTEREST, in anatomical order

Every entry in REGIONS_OF_INTEREST is therefore simultaneously (a) a hub --
it gets its own figure -- and (b) a partner -- it gets one row in every
OTHER hub's figure. Lengthening REGIONS_OF_INTEREST is the only edit needed
to bring a new region into the whole analysis, provided that region's
PCA/PCCA records already exist in the three CSVs below.

Component 0 (this pipeline's own leading-dimension convention -- see
COMPONENT_INDICES in both parent scripts) is drawn by default. List
additional 0-indexed components in COMPONENTS_TO_PLOT to draw them as
side-by-side subplots, exactly like the PC1/PC2/PC3 panels of Fig. 4d.

--------------------------------------------------------------------------
TWO ASSUMPTIONS WORTH VERIFYING against your actual CSVs -- both are
checked loudly by _validate_tables() below (a clear warning/error rather
than a silently wrong number) if either turns out to be mistaken:

  1. Folder naming. Your message named the CCA results as living under
     ...pcca... and the PCCA results under ...cca.... Both parent scripts
     build
         OUTPUT_DIR = BASE_DIR / "Paper_output" / f"cross_trial_type_{KERNEL_MODE}_{REFERENCE_TYPE}_peak_delta_variance"
     with KERNEL_MODE literally 'pcca' or 'cca' -- i.e. the folder name IS
     the content. I have assumed the natural pairing (pcca folder <-> pCCA
     records, cca folder <-> plain-CCA records) below; swap the two paths
     in CSV_PATHS if that is wrong.
  2. region_role. The pCCA/CCA task3_4_variance_summary.csv carries a
     region_role column ('region_i' / 'region_j') that appears to tag which
     of the two "left-region" / "right-region" bar-chart orientations a row
     belongs to, layered on top of the already-canonical region_i/region_j
     pair. I have assumed this duplicates the SAME R^2 value once per
     orientation and pinned the lookup to region_role == 'region_i' so a
     session-mean below can't silently average over duplicate copies of one
     number. If behavior_variance_records.csv does not carry this column at
     all, _pcca_row_mask degrades gracefully and ignores it.
--------------------------------------------------------------------------

Flexibility
--------------------------------------------------------------------------
Everything you're likely to want to change lives in the GLOBAL CONFIG block
below:
  * REGIONS_OF_INTEREST : which regions get their own figure / are
                           available as PCCA partners (see row-construction
                           note above)
  * COMPONENTS_TO_PLOT   : [0] by default; e.g. [0, 1, 2] for three
                           side-by-side panels per figure
  * TRIAL_TYPE_TO_PLOT   : 'cued_hit_long' (task 3, reference-condition R^2)
                           by default; 'spont_hit_long' / 'spont_miss_long'
                           for task 4 (projected-condition R^2)
  * DISPLAY_NAME_OVERRIDES, HEATMAP_CMAP / HEATMAP_VMAX, font sizes, ...

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1.  GLOBAL CONFIG
#     Everything that should change without touching the logic below lives
#     here. Add a region to REGIONS_OF_INTEREST and it automatically becomes
#     both a hub (its own figure) and a partner (a new row in every other
#     hub's figure) -- no other code changes required.
# =============================================================================

# ---- paths ------------------------------------------------------------
BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
PAPER_OUTPUT_DIR = BASE_DIR / "Paper_output"

REFERENCE_TYPE = "cued_hit_long"        # folder-name suffix; matches REFERENCE_TYPE
                                         # in both parent scripts

# See docstring point (1) above about the cca/pcca folder-naming swap.
CSV_PATHS: Dict[str, Path] = {
    "pca":  PAPER_OUTPUT_DIR / f"cross_trial_type_pca_{REFERENCE_TYPE}_peak_delta_variance"  / "behavior_variance_records.csv",
    "cca":  PAPER_OUTPUT_DIR / f"cross_trial_type_cca_{REFERENCE_TYPE}_peak_delta_variance"  / "behavior_variance_records.csv",
    "pcca": PAPER_OUTPUT_DIR / f"cross_trial_type_pcca_{REFERENCE_TYPE}_peak_delta_variance" / "behavior_variance_records.csv",
}

TRIAL_TYPE_TO_PLOT = "cued_hit_long"    # 'cued_hit_long'   -> task 3 (reference condition)
                                         # 'spont_hit_long'  -> task 4 (projected)
                                         # 'spont_miss_long' -> task 4 (projected)

# ---- regions --------------------------------------------------------------
REGIONS_OF_INTEREST: List[str] = ["ORB", "MOp", "MOs", "STR", "VALVM", "VPMPO", "HY"]

# Local copy of Useful_definition.ANATOMICAL_ORDER (kept local, not
# imported, per this project's "primitive copying" convention -- update by
# hand if the master list changes). Unrecognised regions sort last with a
# warning rather than raising, so REGIONS_OF_INTEREST can grow ahead of this
# list without crashing.
ANATOMICAL_ORDER: List[str] = [
    "mPFC", "ORB", "MOp", "MOs", "OLF",
    "STR", "STRv",
    "MD", "LP", "VALVM", "VPMPO", "ILM",
    "HY",
]

# ---- external variables (columns) ------------------------------------------
EXTERNAL_VARIABLES: List[Tuple[str, str]] = [
    ("position", "Paw position"),
    ("speed", "Speed"),
    ("reward_presence", "Reward presence"),
    ("reward_consumption", "Reward consumption"),
]

# ---- components -------------------------------------------------------
COMPONENTS_TO_PLOT: List[int] = [0]     # e.g. [0, 1, 2] for three side-by-side panels

# ---- display names ----------------------------------------------------
DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
    "ORB": "OFC",
    "MOp": "M1 Ctx",
    "MOs": "preM Ctx",
}

# ---- heatmap appearance -------------------------------------------------
HEATMAP_CMAP = "Reds"
HEATMAP_VMIN = 0.0
HEATMAP_VMAX: Optional[float] = 0.2    # None -> shared max across every plotted cell
ANNOTATE_CELLS = False                  # True to print R^2 values inside cells
MISSING_CELL_COLOR = "#d9d9d9"          # a missing record renders distinctly from R^2 = 0

# ---- typography ---------------------------------------------------------
# Matches the magnitude used directly by TICK_FONTSIZE / LEGEND_FONTSIZE in
# both parent scripts; neither of them scales these by a separate
# POSTER_SCALE multiplier, so none is invented here either.
TITLE_FONTSIZE = 18
TICK_FONTSIZE = 18
LABEL_FONTSIZE = 18

# ---- output -----------------------------------------------------------
FIGURE_OUTPUT_DIR = PAPER_OUTPUT_DIR / "figures_region_PCA_PCCA_variable_heatmap"
SAVE_FORMATS: Tuple[str, ...] = ("png",'png')
SAVE_DPI = 400                           # matches SAVE_DPI in both parent scripts
SHOW_FIGURES = False
PRINT_SUMMARY_TABLES = True              # console echo of each figure's matrix


# =============================================================================
# 2.  SMALL PRIMITIVES (copied, not imported -- see Useful_definition.py /
#     cross_trial_type_pca_analysis.py / cross_trial_type_cca_analysis.py)
# =============================================================================

def get_anatomical_index(region: str) -> int:
    try:
        return ANATOMICAL_ORDER.index(region)
    except ValueError:
        print(f"[warn] '{region}' not found in the local ANATOMICAL_ORDER copy; "
              f"sorting it last. Add it to ANATOMICAL_ORDER above if that's unexpected.")
        return len(ANATOMICAL_ORDER)


def anatomical_sort(regions: Sequence[str]) -> List[str]:
    return sorted(regions, key=get_anatomical_index)


def _display_name(region: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(region, region)


def sort_pair_by_anatomy(region_i: str, region_j: str) -> Tuple[str, str]:
    """Anatomical canonicalisation, matching cross_trial_type_cca_analysis.
    sort_pair_by_anatomy (copied here per the "primitive copying"
    convention): the anatomically-earlier region is always region_i."""
    a, b = sorted((region_i, region_j), key=get_anatomical_index)
    return a, b


# =============================================================================
# 3.  ROW SPECIFICATION
# =============================================================================

@dataclass(frozen=True)
class RowSpec:
    analysis: str                   # "pca" or "pcca"
    hub: str                        # raw region code (this figure's region)
    partner: Optional[str] = None   # raw region code; None for the PCA row

    @property
    def label(self) -> str:
        if self.analysis == "pca":
            return "PCA"
        return f"with {_display_name(self.partner)}"


def build_row_specs(hub: str) -> List[RowSpec]:
    """Row 0 = PCA(hub); rows 1..N = PCCA(hub, partner) for every other
    region currently in REGIONS_OF_INTEREST, anatomically ordered."""
    specs: List[RowSpec] = [RowSpec(analysis="pca", hub=hub)]
    partners = anatomical_sort([r for r in REGIONS_OF_INTEREST if r != hub])
    specs += [RowSpec(analysis="pcca", hub=hub, partner=p) for p in partners]
    return specs


# =============================================================================
# 4.  DATA LOADING
# =============================================================================

REQUIRED_COMMON_COLUMNS = {"session", "component", "predictor", "r2", "trial_type"}


def _load_csv(analysis: str) -> pd.DataFrame:
    path = CSV_PATHS[analysis]
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the {analysis.upper()} behavior-variance CSV at:\n    {path}\n"
            f"Check BASE_DIR / CSV_PATHS at the top of the script, and the docstring note "
            f"about a possible cca/pcca folder swap."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COMMON_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} ({analysis}) is missing expected column(s): "
                          f"{sorted(missing)}. Columns found: {sorted(df.columns)}")
    has_pair_cols = {"region_i", "region_j"}.issubset(df.columns)
    has_single_col = "region" in df.columns
    if not (has_pair_cols or has_single_col):
        raise ValueError(f"{path.name} ({analysis}) has neither 'region' nor "
                          f"'region_i'/'region_j' -- can't tell which region(s) each row is for.")
    return df


def load_all_tables() -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for analysis in ("pca", "pcca"):
        tables[analysis] = _load_csv(analysis)
    try:
        tables["cca"] = _load_csv("cca")
    except FileNotFoundError as exc:
        print(f"[info] {exc}\n[info] continuing without it -- the default row spec "
              f"(PCA self + PCCA pairs) doesn't need it.")
    return tables


def _validate_tables(tables: Dict[str, pd.DataFrame]) -> None:
    """Loud, cheap sanity checks -- pair-canonicalisation / naming mismatches
    have silently dropped data in this pipeline before, so surface anything
    suspicious up front rather than let it show up as an unexplained blank
    row in a poster figure."""
    wanted_predictors = {key for key, _label in EXTERNAL_VARIABLES}
    for analysis, df in tables.items():
        found_predictors = set(df["predictor"].unique())
        missing_predictors = wanted_predictors - found_predictors
        if missing_predictors:
            print(f"[warn] {analysis.upper()}: predictor(s) {sorted(missing_predictors)} "
                  f"not present; available: {sorted(found_predictors)}")

        found_trial_types = set(df["trial_type"].unique())
        if TRIAL_TYPE_TO_PLOT not in found_trial_types:
            print(f"[warn] {analysis.upper()}: TRIAL_TYPE_TO_PLOT='{TRIAL_TYPE_TO_PLOT}' "
                  f"not present; available: {sorted(found_trial_types)}")

        if "region" in df.columns:
            found_regions = set(df["region"].unique())
        else:
            found_regions = set(df["region_i"].unique()) | set(df["region_j"].unique())
        missing_regions = set(REGIONS_OF_INTEREST) - found_regions
        if missing_regions:
            print(f"[warn] {analysis.upper()}: region(s) {sorted(missing_regions)} from "
                  f"REGIONS_OF_INTEREST not present in this table.")


# =============================================================================
# 5.  R^2 LOOKUP
# =============================================================================

def _pca_row_mask(df: pd.DataFrame, hub: str) -> pd.Series:
    if "region" in df.columns:
        return df["region"] == hub
    # Unified-schema fallback: a PCA row would carry the region in region_i
    # with region_j empty.
    region_j_empty = df["region_j"].isna() if "region_j" in df.columns else True
    return (df["region_i"] == hub) & region_j_empty


def _pcca_row_mask(df: pd.DataFrame, hub: str, partner: str) -> pd.Series:
    region_i, region_j = sort_pair_by_anatomy(hub, partner)
    # test = f"('{region_i}', '{region_j}')"
    # test2 = df["pair"][0]
    mask = (df["pair"] == f"('{region_i}', '{region_j}')")& (df["region"] == hub)
    # if "region_role" in df.columns:
    #     mask == (df["region"] == hub)
    return mask


def lookup_r2(spec: RowSpec, predictor: str, component: int, tables: Dict[str, pd.DataFrame]) -> float:
    df = tables.get(spec.analysis)
    if df is None:
        return np.nan

    mask = ((df["predictor"] == predictor)
            & (df["component"] == component)
            & (df["trial_type"] == TRIAL_TYPE_TO_PLOT))
    if spec.analysis == "pca":
        mask &= _pca_row_mask(df, spec.hub)
    else:
        mask &= _pcca_row_mask(df, spec.hub, spec.partner)

    values = df.loc[mask, "r2"]
    if values.empty:
        partner_txt = "" if spec.partner is None else f" \u00d7 {_display_name(spec.partner)}"
        print(f"[warn] no rows for {spec.analysis.upper()} {_display_name(spec.hub)}{partner_txt}, "
              f"predictor='{predictor}', component={component}, trial_type='{TRIAL_TYPE_TO_PLOT}'.")
        return np.nan
    return float(values.mean())     # mean R^2 across sessions


# =============================================================================
# 6.  MATRIX CONSTRUCTION
# =============================================================================

def build_matrix(hub: str, component: int, tables: Dict[str, pd.DataFrame]) -> Tuple[List[RowSpec], np.ndarray]:
    row_specs = build_row_specs(hub)
    matrix = np.full((len(row_specs), len(EXTERNAL_VARIABLES)), np.nan)
    for i, spec in enumerate(row_specs):
        for j, (predictor_key, _label) in enumerate(EXTERNAL_VARIABLES):
            matrix[i, j] = lookup_r2(spec, predictor_key, component, tables)
    return row_specs, matrix


# =============================================================================
# 7.  PLOTTING
# =============================================================================

def _draw_heatmap(ax, matrix: np.ndarray, row_labels: List[str], vmax: float, show_row_labels: bool):
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color=MISSING_CELL_COLOR)

    im = ax.imshow(masked, cmap=cmap, vmin=HEATMAP_VMIN, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(EXTERNAL_VARIABLES)))
    ax.set_xticklabels([label for _key, label in EXTERNAL_VARIABLES],
                        fontsize=TICK_FONTSIZE, rotation=40, ha="right", rotation_mode="anchor")

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels if show_row_labels else [""] * len(row_labels), fontsize=TICK_FONTSIZE)

    ax.set_xticks(np.arange(-0.5, len(EXTERNAL_VARIABLES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)

    if ANNOTATE_CELLS:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                             fontsize=max(TICK_FONTSIZE - 6, 8))

    return im


def plot_region_figure(
    hub: str,
    cache: Dict[Tuple[str, int], Tuple[List[RowSpec], np.ndarray]],
    vmax: float,
) -> plt.Figure:
    n_panels = len(COMPONENTS_TO_PLOT)
    fig_width = 3.4 * n_panels + 2.2
    # constrained_layout (rather than tight_layout, which the parent scripts
    # use) because it is colorbar-aware -- a colorbar spanning several
    # heatmap panels via fig.colorbar(im, ax=axes.tolist(), ...) is exactly
    # the case tight_layout warns it can get wrong.
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4.6), squeeze=False,
                              constrained_layout=True)
    axes = axes[0]

    im = None
    for idx, (ax, component) in enumerate(zip(axes, COMPONENTS_TO_PLOT)):
        row_specs, matrix = cache[(hub, component)]
        row_labels = [s.label for s in row_specs]
        im = _draw_heatmap(ax, matrix, row_labels, vmax, show_row_labels=(idx == 0))
        if n_panels > 1:
            ax.set_title(f"Component {component + 1}", fontsize=TITLE_FONTSIZE - 4)

    fig.suptitle(_display_name(hub), fontsize=TITLE_FONTSIZE, fontweight="normal")

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.05, pad=0.02)
    vmin, vmax = im.get_clim()
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_label("Mean $R^2$ of GLM", fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=max(TICK_FONTSIZE - 4, 8))

    return fig


def _print_summary(hub: str, cache: Dict[Tuple[str, int], Tuple[List[RowSpec], np.ndarray]]) -> None:
    for component in COMPONENTS_TO_PLOT:
        row_specs, matrix = cache[(hub, component)]
        print(f"\n--- {_display_name(hub)} | component {component} ---")
        header = "".join(f"{label:>20s}" for _key, label in EXTERNAL_VARIABLES)
        print(f"{'':<20s}{header}")
        for spec, row in zip(row_specs, matrix):
            row_txt = "".join(f"{v:20.3f}" if not np.isnan(v) else f"{'--':>20s}" for v in row)
            print(f"{spec.label:<20s}{row_txt}")


# =============================================================================
# 8.  MAIN
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("REGION x ANALYSIS-TYPE  vs  EXTERNAL-VARIABLE  HEATMAPS")
    print("=" * 70)
    print(f"  regions of interest : {REGIONS_OF_INTEREST}")
    print(f"  components to plot  : {COMPONENTS_TO_PLOT}")
    print(f"  trial type          : {TRIAL_TYPE_TO_PLOT}")
    print(f"  external variables  : {[k for k, _ in EXTERNAL_VARIABLES]}")
    print(f"  output directory    : {FIGURE_OUTPUT_DIR}")
    print("=" * 70)

    tables = load_all_tables()
    _validate_tables(tables)

    hubs = anatomical_sort(REGIONS_OF_INTEREST)

    # Single pass: build every (hub, component) matrix once, reused both for
    # the shared colour ceiling and for plotting.
    cache: Dict[Tuple[str, int], Tuple[List[RowSpec], np.ndarray]] = {
        (hub, component): build_matrix(hub, component, tables)
        for hub in hubs
        for component in COMPONENTS_TO_PLOT
    }

    if HEATMAP_VMAX is not None:
        vmax = HEATMAP_VMAX
    else:
        stacked = np.concatenate([m.ravel() for _specs, m in cache.values()])
        vmax = 1.0 if np.all(np.isnan(stacked)) else float(np.nanmax(stacked))
    print(f"\n[info] shared colour-scale ceiling (mean R^2): {vmax:.4f}")

    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for hub in hubs:
        if PRINT_SUMMARY_TABLES:
            _print_summary(hub, cache)
        fig = plot_region_figure(hub, cache, vmax)
        stem = f"region_variable_heatmap_{hub}"
        for fmt in SAVE_FORMATS:
            out_path = FIGURE_OUTPUT_DIR / f"{stem}.{fmt}"
            fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
            print(f"[info] saved {out_path}")
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
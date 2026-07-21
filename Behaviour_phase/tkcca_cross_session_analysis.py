"""
tkcca_analysis.py
=================
Extensions to the OxfordAdvancedAnalyzer framework for temporal-kernel CCA.

Two top-level additions are provided:

  1.  ``TkCCAPair``  – dataclass mirroring the new tkCCA-specific fields that
      ``perform_session_tkcca.m`` appends to every ``pair_results`` struct.

  2.  ``TkCCAExtractor``  – mixin / standalone helper that walks the same
      ``cca_results → pair_results`` tree already handled by
      ``OxfordAdvancedAnalyzer._extract_cca_weights_and_trials``, harvesting
      only the tkCCA fields, and populating a per-condition store
      ``self.tkcca``.

  3.  Visualization suite (Section 3 of the figure plan):

      plot_dominant_lag_heatmap   –  upper-triangle heatmap of  τ*(pair)
      plot_significance_heatmap   –  upper-triangle heatmap of  mean max ρ
      plot_canonical_correlogram  –  ρ(τ) traces for selected pairs

Mathematical conventions
------------------------
  Let K = number of lag bins,  n_sig = number of significant components.

  canonical_correlogram  ∈  ℝ^{K × n_sig}
      correlogram[k, c]  =  Corr( X_{τ_k} w_x(τ_k, c),  Y w_y(c) )
      where  τ_k  is the k-th element of  tkcca_lags_bins.

  Dominant lag (per pair, per session):
      τ*(session, pair)  =  tkcca_lags_seconds[ argmax_k |correlogram[k, 0]| ]
      (first significant component only; sign encodes directionality).

  Aggregated dominant lag (across sessions):
      τ̄(pair)  =  median_sessions  τ*(session, pair)

  Positive τ̄  ⇒  region_i leads region_j.
  Negative τ̄  ⇒  region_j leads region_i.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

# Re-use utilities from the existing framework
from Useful_definition import (
    ANATOMICAL_ORDER,
    OxfordAdvancedAnalyzer,
    safe_array,
    first_present,
)


# ---------------------------------------------------------------------------
# 1.  New dataclass  –  TkCCAPair
# ---------------------------------------------------------------------------

@dataclass
class TkCCAPair:
    """
    Holds the tkCCA-specific outputs for one region pair from one session.

    All shape annotations follow the MATLAB storage convention confirmed in the
    debugger:

        canonical_correlogram  :  (K, n_sig)
        tkcca_lags_bins        :  (K,)          integer lag indices
        tkcca_lags_seconds     :  (K,)          lag values in seconds
        wx_temporal            :  (n_i, K, n_sig)   spatio-temporal filter
        wy_stationary          :  (n_j, n_sig)       stationary filter
        mean_cv_rho            :  (n_components,)    cross-validated ρ per comp
        significant_components :  list[int]          1-indexed (MATLAB convention)
    """
    session: str
    region_i: str
    region_j: str

    # Core lag-structure fields
    canonical_correlogram: np.ndarray           # (K, n_sig)
    tkcca_lags_bins: np.ndarray                 # (K,)
    tkcca_lags_seconds: np.ndarray              # (K,)

    # Spatial filters (kept for downstream weight-map figures)
    wx_temporal: np.ndarray                     # (n_i, K, n_sig)
    wy_stationary: np.ndarray                   # (n_j, n_sig)

    # Summary statistics
    mean_cv_rho: np.ndarray                     # (n_components,)
    significant_components: List[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived convenience properties
    # ------------------------------------------------------------------

    @property
    def n_sig(self) -> int:
        return self.canonical_correlogram.shape[1]

    @property
    def dominant_lag_s(self) -> float:
        """
        τ*(s) for the first significant component.

        τ*(s)  =  tkcca_lags_seconds[ argmax_k |correlogram[k, 0]| ]

        The sign of correlogram at the dominant lag is preserved so that
        positive τ ⇒ region_i leads region_j.
        """
        if self.n_sig == 0:
            return np.nan
        corr_c0 = self.canonical_correlogram[:, 0]          # (K,)
        k_star = int(np.argmax(np.abs(corr_c0)))
        lag_s = float(self.tkcca_lags_seconds[k_star])
        # Preserve sign of the dominant correlation to encode directionality
        return lag_s * np.sign(corr_c0[k_star])

    @property
    def max_rho(self) -> float:
        """Maximum cross-validated canonical correlation across all components."""
        return float(np.nanmax(self.mean_cv_rho)) if self.mean_cv_rho.size > 0 else np.nan


# ---------------------------------------------------------------------------
# 2.  Extractor mixin  –  TkCCAExtractor
# ---------------------------------------------------------------------------

class TkCCAExtractor:
    """
    Mixin for OxfordAdvancedAnalyzer that harvests the tkCCA-specific fields.

    Usage
    -----
    Either subclass both:

        class MyAnalyzer(TkCCAExtractor, OxfordAdvancedAnalyzer): ...

    Or instantiate standalone and call  .extract_from_analyzer(analyzer).

    After extraction, the store
        self.tkcca[cond][(r_i, r_j)]  ->  List[TkCCAPair]
    mirrors the shape of  OxfordAdvancedAnalyzer.single_trial.
    """

    def _init_tkcca_store(self, conditions: List[str]) -> None:
        self.tkcca: Dict[str, Dict[Tuple[str, str], List[TkCCAPair]]] = {
            c: {} for c in conditions
        }

    # ------------------------------------------------------------------
    # Public entry point when used as a mixin inside OxfordAdvancedAnalyzer
    # ------------------------------------------------------------------

    def load_all(self) -> None:                          # type: ignore[override]
        """Override load_all to also harvest tkCCA fields."""
        self._init_tkcca_store(self.conditions)
        super().load_all()                               # runs the original loading

    def _load_session(self, file_path: Path, session: str, cond: str) -> None:  # type: ignore[override]
        super()._load_session(file_path, session, cond)  # existing extraction
        # Now harvest the tkCCA-specific block from the same file
        try:
            import mat73
            data = mat73.loadmat(str(file_path))
            self._extract_tkcca_fields(data, session, cond)
        except Exception as exc:
            print(f"    [{session}] tkCCA field extraction failed: {exc}")

    # ------------------------------------------------------------------
    # Standalone extraction (when used without inheritance)
    # ------------------------------------------------------------------

    def extract_from_analyzer(
        self,
        analyzer: OxfordAdvancedAnalyzer,
        raw_data_cache: Optional[Dict[str, dict]] = None,
    ) -> None:
        """
        Re-walk the already-loaded raw MATLAB dictionaries.

        Parameters
        ----------
        analyzer :
            A fully loaded OxfordAdvancedAnalyzer instance.
        raw_data_cache :
            Optional dict mapping session_path_str -> mat73 dict.  If None,
            files are reloaded from disk (slightly slower but always correct).
        """
        import mat73
        self._init_tkcca_store(analyzer.conditions)
        for cond, path in analyzer.condition_dirs.items():
            for f in sorted(path.glob("*_analysis_results.mat")):
                session = f.stem.replace("_analysis_results", "")
                try:
                    data = (raw_data_cache or {}).get(str(f)) or mat73.loadmat(str(f))
                    self._extract_tkcca_fields(data, session, cond)
                except Exception as exc:
                    print(f"    [{session}] reloading failed: {exc}")

    # ------------------------------------------------------------------
    # Core extraction logic
    # ------------------------------------------------------------------

    def _extract_tkcca_fields(
        self, data: dict, session: str, cond: str
    ) -> None:
        """
        Navigate:
            data['cca_results']['pair_results'][i]
                -> 'canonical_correlogram'   (K, n_sig)
                -> 'tkcca_lags_bins'         (1, K) or (K,)
                -> 'tkcca_lags_seconds'      (1, K) or (K,)
                -> 'wx_temporal'             (n_i, K, n_sig)
                -> 'wy_stationary'           (n_j, n_sig)
                -> 'mean_cv_R2'  (via cv_results)
                -> 'significant_components'
        """
        cca = data.get("cca_results")
        if not isinstance(cca, dict):
            return
        pair_results = cca.get("pair_results")
        if isinstance(pair_results, np.ndarray):
            pair_results = pair_results.tolist()
        if not isinstance(pair_results, (list, tuple)):
            return

        for pr in pair_results:
            if not isinstance(pr, dict):
                continue

            r_i = self._as_str_static(pr.get("region_i"))
            r_j = self._as_str_static(pr.get("region_j"))
            if not r_i or not r_j:
                continue

            # ── canonical correlogram ──────────────────────────────────────
            corr_gram = safe_array(pr.get("canonical_correlogram"))
            if corr_gram is None:
                # This pair_result was produced by standard CCA, not tkCCA;
                # skip silently.
                continue

            # MATLAB stores (K, n_sig) — verify & coerce to 2-D
            if corr_gram.ndim == 1:
                corr_gram = corr_gram[:, np.newaxis]

            # ── lag vectors ───────────────────────────────────────────────
            lag_bins = safe_array(pr.get("tkcca_lags_bins"))
            lag_secs = safe_array(pr.get("tkcca_lags_seconds"))
            if lag_bins is None or lag_secs is None:
                continue
            lag_bins = lag_bins.ravel().astype(int)
            lag_secs = lag_secs.ravel()

            # ── spatial filters ───────────────────────────────────────────
            wx = safe_array(pr.get("wx_temporal"))       # (n_i, K, n_sig)
            wy = safe_array(pr.get("wy_stationary"))     # (n_j, n_sig)
            if wx is None or wy is None:
                continue

            # ── summary statistics ────────────────────────────────────────
            mean_cv_rho = None
            cv_res = pr.get("cv_results")
            if isinstance(cv_res, dict):
                mean_cv_rho = safe_array(
                    first_present(cv_res, ["mean_cv_R2", "mean_cv_rho"])
                )
            if mean_cv_rho is None:
                mean_cv_rho = safe_array(pr.get("mean_cv_R2")) or np.array([])
            mean_cv_rho = mean_cv_rho.ravel()

            sig_raw = pr.get("significant_components")
            if sig_raw is None:
                sig_comps: List[int] = []
            elif isinstance(sig_raw, (int, float)):
                sig_comps = [int(sig_raw)]
            else:
                sig_comps = [int(s) for s in np.asarray(sig_raw).ravel()]

            tk_pair = TkCCAPair(
                session=session,
                region_i=r_i,
                region_j=r_j,
                canonical_correlogram=corr_gram,
                tkcca_lags_bins=lag_bins,
                tkcca_lags_seconds=lag_secs,
                wx_temporal=wx,
                wy_stationary=wy,
                mean_cv_rho=mean_cv_rho,
                significant_components=sig_comps,
            )

            self.tkcca[cond].setdefault((r_i, r_j), []).append(tk_pair)

    @staticmethod
    def _as_str_static(v) -> Optional[str]:
        try:
            if v is None:
                return None
            if isinstance(v, str):
                return v
            if isinstance(v, np.ndarray):
                return str(v.item()) if v.size > 0 else None
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return str(v[0])
            return str(v)
        except Exception:
            return None

# ---------------------------------------------------------------------------
# 3.  Convenience subclass  –  combines both parents
# ---------------------------------------------------------------------------

class OxfordTkCCAAnalyzer(TkCCAExtractor, OxfordAdvancedAnalyzer):
    """
    # Drop-in replacement for OxfordAdvancedAnalyzer that additionally
    # populates  self.tkcca  after  .load_all().
    #
    # Example
    # -------
    # >>> analyzer = OxfordTkCCAAnalyzer(
    # ...     base_results_dir="/data/Oxford_dataset",
    # ...     results_subdirs={"spont_miss_long": "tkcca_sessions_spont_miss_long_results"},
    # ...     n_components=5,
    # ... )
    # >>> analyzer.load_all()
    # >>> fig = plot_dominant_lag_heatmap(analyzer.tkcca, cond="spont_miss_long")
    # """
    # pass


# ---------------------------------------------------------------------------
# 4.  Visualization – figure 1: dominant lag heatmap + significance heatmap
# ---------------------------------------------------------------------------

def _build_pair_matrices(
    tkcca_store: Dict[Tuple[str, str], List[TkCCAPair]],
    region_order: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate TkCCAPair records into region-pair matrices.

    Returns
    -------
    lag_matrix   :  (R, R)  median dominant lag τ̄(pair) in seconds
    rho_matrix   :  (R, R)  median max ρ across sessions
    count_matrix :  (R, R)  number of sessions contributing to each pair
    """
    R = len(region_order)
    idx = {r: i for i, r in enumerate(region_order)}

    lag_matrix   = np.full((R, R), np.nan)
    rho_matrix   = np.full((R, R), np.nan)
    count_matrix = np.zeros((R, R), dtype=int)

    for (r_i, r_j), pairs in tkcca_store.items():
        if r_i not in idx or r_j not in idx:
            continue
        i, j = idx[r_i], idx[r_j]

        dom_lags = np.array([p.dominant_lag_s for p in pairs])
        max_rhos = np.array([p.max_rho       for p in pairs])

        valid = np.isfinite(dom_lags)
        if valid.sum() == 0:
            continue

        # Store in upper triangle by anatomical order convention
        ri, rj = (i, j) if i < j else (j, i)
        lag_matrix[ri, rj]   = float(np.nanmedian(dom_lags))
        rho_matrix[ri, rj]   = float(np.nanmedian(max_rhos))
        count_matrix[ri, rj] = int(valid.sum())

    return lag_matrix, rho_matrix, count_matrix



def plot_dominant_lag_heatmap(
        tkcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[TkCCAPair]]],
        cond: str,
        region_order: Optional[List[str]] = None,
        lag_range_s: float = 0.15,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
) -> plt.Figure:
    if region_order is None:
        region_order = ANATOMICAL_ORDER

    store = tkcca_store_by_cond.get(cond, {})
    lag_mat, rho_mat, cnt_mat = _build_pair_matrices(store, region_order)

    present = [
        r for r in region_order
        if any(r in (ri, rj) for (ri, rj) in store)
    ]
    if not present:
        raise ValueError(f"No tkCCA pairs found for condition '{cond}'")

    pidx = [region_order.index(r) for r in present]
    lag_sub = lag_mat[np.ix_(pidx, pidx)]
    cnt_sub = cnt_mat[np.ix_(pidx, pidx)]
    R = len(present)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, R * 0.9), max(5, R * 0.85)),
                               constrained_layout=True)
    else:
        fig = ax.figure

    cmap = mpl.colormaps.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-lag_range_s, vcenter=0.0, vmax=lag_range_s)

    # --- UPDATED MASK LOGIC ---
    # Mask if: 1) It's the lower triangle OR 2) Session count is < 3
    mask_tril = np.tril(np.ones_like(lag_sub, dtype=bool), k=-1)
    mask_low_n = cnt_sub < 3
    combined_mask = mask_tril | mask_low_n

    display = np.where(combined_mask, np.nan, lag_sub)
    # --------------------------

    img = ax.imshow(display, cmap=cmap, norm=norm, aspect="equal",
                    interpolation="nearest")

    for i in range(R):
        for j in range(i + 1, R):
            n = cnt_sub[i, j]
            # Text only shown if cell isn't masked
            if n >= 3:
                lag_val = lag_sub[i, j]
                txt_col = "white" if abs(lag_val) > 0.5 * lag_range_s else "black"
                ax.text(j, i, f"n={n}", ha="center", va="center",
                        fontsize=7, color=txt_col, fontweight="normal")

    ax.set_xticks(range(R))
    ax.set_yticks(range(R))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(present, fontsize=9)
    ax.set_title(
        rf"Dominant lag $\tau^*$ (s)  —  {cond}"
        f"\n(n \u2265 3) | Red: row leads; Blue: col leads",
        fontsize=10, pad=8,
    )

    cbar = fig.colorbar(img, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(r"Median dominant lag $\bar{\tau}$ (s)", fontsize=9)
    cbar.ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_significance_heatmap(
        tkcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[TkCCAPair]]],
        cond: str,
        region_order: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
) -> plt.Figure:
    if region_order is None:
        region_order = ANATOMICAL_ORDER

    store = tkcca_store_by_cond.get(cond, {})
    _, rho_mat, cnt_mat = _build_pair_matrices(store, region_order)

    present = [
        r for r in region_order
        if any(r in (ri, rj) for (ri, rj) in store)
    ]
    if not present:
        raise ValueError(f"No tkCCA pairs found for condition '{cond}'")

    pidx = [region_order.index(r) for r in present]
    rho_sub = rho_mat[np.ix_(pidx, pidx)]
    cnt_sub = cnt_mat[np.ix_(pidx, pidx)]
    R = len(present)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, R * 0.9), max(5, R * 0.85)),
                               constrained_layout=True)
    else:
        fig = ax.figure

    # --- UPDATED MASK LOGIC ---
    mask_tril = np.tril(np.ones_like(rho_sub, dtype=bool), k=-1)
    mask_low_n = cnt_sub < 3
    combined_mask = mask_tril | mask_low_n

    display = np.where(combined_mask, np.nan, rho_sub)
    # --------------------------

    cmap_sig = mpl.colormaps.get_cmap("YlOrRd")
    img = ax.imshow(display, cmap=cmap_sig, vmin=0.0, vmax=1.0,
                    aspect="equal", interpolation="nearest")

    for i in range(R):
        for j in range(i + 1, R):
            n = cnt_sub[i, j]
            rho_val = rho_sub[i, j]
            # Only annotate if we are showing the cell
            if n >= 3 and np.isfinite(rho_val):
                txt_col = "white" if rho_val > 0.65 else "black"
                ax.text(j, i,
                        f"{rho_val:.2f}\nn={n}",
                        ha="center", va="center",
                        fontsize=6.5, color=txt_col)

    ax.set_xticks(range(R))
    ax.set_yticks(range(R))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(present, fontsize=9)
    ax.set_title(
        rf"Median max $\bar{{\rho}}$  —  {cond}"
        "\n(n \u2265 3 cross-validated canonical correlation)",
        fontsize=10, pad=8,
    )
    cbar = fig.colorbar(img, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(r"Median $\max_c\,\bar{\rho}_c$", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5.  Visualization – figure 2: canonical correlogram curves  ρ(τ)
# ---------------------------------------------------------------------------

def plot_canonical_correlogram(
    tkcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[TkCCAPair]]],
    cond: str,
    pairs: Optional[List[Tuple[str, str]]] = None,
    n_components: int = 1,
    region_order: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the canonical correlogram  ρ(τ_k, c)  for selected region pairs.

    Each panel shows the mean ± SEM across sessions of the cross-validated
    canonical correlation as a function of temporal lag τ (seconds).

    The peak location and its sign directly encode the inter-regional
    communication direction:

        peak at τ* > 0  ⇒  region_i(t - τ*)  drives  region_j(t)
        peak at τ* < 0  ⇒  region_j(t)       drives  region_i(t + |τ*|)

    Parameters
    ----------
    pairs :
        List of (region_i, region_j) tuples to plot.  If None, all available
        pairs are plotted (potentially many subplots).
    n_components :
        How many significant components to overlay per subplot (default: 1).
    """
    if region_order is None:
        region_order = ANATOMICAL_ORDER

    store = tkcca_store_by_cond.get(cond, {})
    if not store:
        raise ValueError(f"No tkCCA data for condition '{cond}'")

    # Select pairs to plot and enforce anatomical ordering within each
    if pairs is None:
        selected = sorted(store.keys(), key=lambda p: (
            region_order.index(p[0]) if p[0] in region_order else 999,
            region_order.index(p[1]) if p[1] in region_order else 999,
        ))
    else:
        selected = [p for p in pairs if p in store]
        if not selected:
            raise ValueError("None of the requested pairs were found in the tkCCA store.")

    n_panels = len(selected)
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    if figsize is None:
        figsize = (ncols * 3.8, nrows * 3.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()

    # Colour cycle for components
    comp_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for panel_idx, pair_key in enumerate(selected):
        ax = axes_flat[panel_idx]
        r_i, r_j = pair_key
        pair_list = store[pair_key]

        if not pair_list:
            ax.set_visible(False)
            continue

        # Verify all sessions share the same lag vector
        lag_secs = pair_list[0].tkcca_lags_seconds          # (K,)
        K = len(lag_secs)

        # Stack correlograms:  (n_sessions, K, n_sig_max)
        max_comps = min(n_components, min(p.n_sig for p in pair_list))
        if max_comps == 0:
            ax.text(0.5, 0.5, "no sig. components", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            ax.set_title(f"{r_i} → {r_j}", fontsize=9)
            continue

        for c_idx in range(max_comps):
            # Collect ρ(τ, c) across sessions that have at least c_idx+1 sig comps
            session_traces = []
            for p in pair_list:
                if p.n_sig > c_idx:
                    # Align lag vector (defensive: some sessions may differ by ±1 bin)
                    if p.canonical_correlogram.shape[0] == K:
                        session_traces.append(p.canonical_correlogram[:, c_idx])

            if not session_traces:
                continue

            traces = np.vstack(session_traces)          # (n_sessions, K)
            mean_trace = np.nanmean(traces, axis=0)
            sem_trace  = stats.sem(traces, axis=0, nan_policy="omit")

            color = comp_colors[c_idx % len(comp_colors)]
            ax.plot(lag_secs, mean_trace, color=color, linewidth=1.8,
                    label=f"comp {c_idx + 1}")
            ax.fill_between(lag_secs,
                            mean_trace - sem_trace,
                            mean_trace + sem_trace,
                            color=color, alpha=0.2)

        # Decorations
        ax.axvline(0.0, color="k", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.axhline(0.0, color="k", linewidth=0.4, linestyle=":", alpha=0.4)
        ax.set_xlabel(r"Lag $\tau$ (s)", fontsize=8)
        ax.set_ylabel(r"$\rho(\tau)$", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        n_sess = len(pair_list)
        ax.set_title(f"{r_i} → {r_j}  (n={n_sess})", fontsize=9)
        if max_comps > 1:
            ax.legend(fontsize=7, loc="upper left")

    # Hide unused panels
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        rf"Canonical correlograms $\rho(\tau)$  —  {cond}",
        fontsize=12, fontweight="bold",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 6.  Compound figure  –  lag heatmap + significance heatmap side by side
# ---------------------------------------------------------------------------

def plot_tkcca_summary(
    tkcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[TkCCAPair]]],
    cond: str,
    region_order: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel compound figure:  [lag heatmap | significance heatmap].

    This is the canonical summary figure for a single condition. It answers
    simultaneously:

        (a) *Which* pairs show reliable cross-regional temporal coupling?
            →  significance panel (colour = median max ρ)

        (b) *When* does that coupling peak, and in which direction?
            →  lag panel (colour = median dominant lag τ̄)

    Intended for inclusion as a manuscript figure panel.
    """
    fig, (ax_lag, ax_sig) = plt.subplots(
        1, 2,
        figsize=(16, 7),
        constrained_layout=True,
    )

    plot_dominant_lag_heatmap(tkcca_store_by_cond, cond,
                              region_order=region_order, ax=ax_lag)
    plot_significance_heatmap(tkcca_store_by_cond, cond,
                              region_order=region_order, ax=ax_sig)

    fig.suptitle(
        f"tkCCA inter-regional temporal coupling summary  —  {cond}",
        fontsize=13, fontweight="bold",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 7.  Example driver
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir   = "/Users/shengyuancai/Downloads/Oxford_dataset"
    output_dir = Path(base_dir) / "Paper_output" / "figures_tkcca"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = OxfordTkCCAAnalyzer(
        base_results_dir=base_dir,
        results_subdirs={
            "cued_hit_long": "tkcca_sessions_cued_hit_long_results",
        },
        n_components=5,
    )
    analyzer.load_all()

    cond = "cued_hit_long"

    # ── Figure 1a/1b: compound summary heatmap ─────────────────────────────
    plot_tkcca_summary(
        analyzer.tkcca, cond,
        save_path=str(output_dir / f"tkcca_summary_{cond}.png"),
    )

    # ── Figure 2: canonical correlograms for all pairs ─────────────────────
    plot_canonical_correlogram(
        analyzer.tkcca, cond,
        n_components=2,
        save_path=str(output_dir / f"tkcca_correlograms_{cond}.png"),
    )

    plt.show()


if __name__ == "__main__":
    main()
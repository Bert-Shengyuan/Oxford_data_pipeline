"""
pcca_cross_session_analysis.py
===============================
Cross-session analysis and visualisation for partial CCA outputs produced by
``perform_session_pcca.m``.

Parallel to ``tkcca_cross_session_analysis.py``, this module provides:

  1.  ``PCCAPair``          – dataclass for the pCCA-specific fields from each
                              ``pair_results`` struct entry.

  2.  ``PCCAExtractor``     – mixin / standalone extractor that walks
                              ``cca_results → pair_results`` and populates
                              ``self.pcca[cond][(r_i, r_j)]  ->  List[PCCAPair]``.

  3.  ``OxfordPCCAAnalyzer`` – convenience combined subclass.

  4.  Visualisation suite:

      plot_rho_heatmap            Upper-triangular matrix of median dominant_rho.
                                  Replaces tkCCA lag heatmap as the primary
                                  inter-regional coupling summary.

      plot_subspace_dim_heatmap   Upper-triangular matrix of median subspace_dim
                                  (number of significant pCCA dimensions).

      plot_mi_heatmap             Upper-triangular matrix of median mutual
                                  information MI = -sum_k log(1-rho_k^2).

      plot_gini_distribution      Per-region box-strip panel of Gini coefficients
                                  across all sessions and pairs.  Analogous to
                                  Gonzalez et al. (2026) Fig. 1f: monotonically
                                  decreasing Gini along the circuit reveals
                                  progressive subspace expansion.

      plot_variance_retained      Per-pair strip-box of variance_X/Y_retained
                                  across sessions; diagnostic for how strongly
                                  Z confounded each pair.

      plot_pcca_summary           Compound three-panel figure [rho | dim | MI].

      plot_condition_rho_comparison
                                  Overlaid cumulative-density (or boxplot)
                                  comparison of dominant_rho across the three
                                  trial conditions for a user-specified set of
                                  region pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from Useful_definition import (
    ANATOMICAL_ORDER,
    OxfordAdvancedAnalyzer,
    safe_array,
    first_present,
)


# ---------------------------------------------------------------------------
# 1.  Dataclass  –  PCCAPair
# ---------------------------------------------------------------------------

@dataclass
class PCCAPair:
    """
    Holds all pCCA-specific outputs for one region pair from one session.

    All scalar fields are Python floats; array shapes follow MATLAB storage.
    """
    session:  str
    region_i: str
    region_j: str

    # Weight matrices for the dominant dimension (column 0)
    Wx: np.ndarray        # (n_i, n_components)
    Wy: np.ndarray        # (n_j, n_components)

    # Cross-validated canonical correlations
    mean_cv_rho: np.ndarray       # (n_components,)
    significant_components: List[int] = field(default_factory=list)

    # pCCA-specific scalars
    dominant_rho:        float = np.nan
    subspace_dim:        int   = 0
    mutual_information:  float = np.nan
    gini_i:              float = np.nan   # Gini of |Wx[:,0]|
    gini_j:              float = np.nan   # Gini of |Wy[:,0]|
    var_X_retained:      float = np.nan
    var_Y_retained:      float = np.nan

    nuisance_regions:    List[str] = field(default_factory=list)
    nuisance_n_neurons:  int       = 0
    is_partial:          bool      = False

    # ------------------------------------------------------------------
    @property
    def n_sig(self) -> int:
        return len(self.significant_components)

    @property
    def max_rho(self) -> float:
        return float(np.nanmax(self.mean_cv_rho)) if self.mean_cv_rho.size > 0 else np.nan


# ---------------------------------------------------------------------------
# 2.  Extractor mixin  –  PCCAExtractor
# ---------------------------------------------------------------------------

class PCCAExtractor:
    """
    Mixin for OxfordAdvancedAnalyzer that harvests the pCCA-specific fields.

    Usage (as mixin)::

        class MyAnalyzer(PCCAExtractor, OxfordAdvancedAnalyzer): ...

    Or standalone::

        extractor = PCCAExtractor()
        extractor.extract_from_analyzer(analyzer)

    After extraction::

        extractor.pcca[cond][(r_i, r_j)]  ->  List[PCCAPair]
    """

    def _init_pcca_store(self, conditions: List[str]) -> None:
        self.pcca: Dict[str, Dict[Tuple[str, str], List[PCCAPair]]] = {
            c: {} for c in conditions
        }

    # ------------------------------------------------------------------
    # Mixin overrides
    # ------------------------------------------------------------------

    def load_all(self) -> None:   # type: ignore[override]
        self._init_pcca_store(self.conditions)
        super().load_all()

    def _load_session(self, file_path: Path, session: str, cond: str) -> None:  # type: ignore[override]
        super()._load_session(file_path, session, cond)
        try:
            import mat73
            data = mat73.loadmat(str(file_path))
            self._extract_pcca_fields(data, session, cond)
        except Exception as exc:
            print(f"    [{session}] pCCA field extraction failed: {exc}")

    # ------------------------------------------------------------------
    # Standalone extraction
    # ------------------------------------------------------------------

    def extract_from_analyzer(
        self,
        analyzer: OxfordAdvancedAnalyzer,
        raw_data_cache: Optional[Dict[str, dict]] = None,
    ) -> None:
        import mat73
        self._init_pcca_store(analyzer.conditions)
        for cond, path in analyzer.condition_dirs.items():
            for f in sorted(path.glob("*_analysis_results.mat")):
                session = f.stem.replace("_analysis_results", "")
                try:
                    data = (raw_data_cache or {}).get(str(f)) or mat73.loadmat(str(f))
                    self._extract_pcca_fields(data, session, cond)
                except Exception as exc:
                    print(f"    [{session}] reload failed: {exc}")

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def _extract_pcca_fields(self, data: dict, session: str, cond: str) -> None:
        """
        Walk ``data['cca_results']['pair_results']`` and harvest every
        pCCA-compatible entry (i.e., those that contain 'dominant_rho').
        Entries produced by plain CCA or tkCCA are silently skipped.
        """
        cca = data.get("cca_results")
        if not isinstance(cca, dict):
            return
        pr_list = cca.get("pair_results")
        if isinstance(pr_list, np.ndarray):
            pr_list = pr_list.tolist()
        if not isinstance(pr_list, (list, tuple)):
            return

        for pr in pr_list:
            if not isinstance(pr, dict):
                continue

            r_i = self._as_str_static(pr.get("region_i"))
            r_j = self._as_str_static(pr.get("region_j"))
            if not r_i or not r_j:
                continue

            # Gate: only process pCCA results (must have dominant_rho)
            if pr.get("dominant_rho") is None:
                continue

            Wx = safe_array(pr.get("mean_A_matrix"))
            Wy = safe_array(pr.get("mean_B_matrix"))
            if Wx is None or Wy is None:
                continue
            if Wx.ndim == 1:
                Wx = Wx[:, np.newaxis]
            if Wy.ndim == 1:
                Wy = Wy[:, np.newaxis]

            # cv_results
            mean_cv_rho = np.array([])
            cv_res = pr.get("cv_results")
            if isinstance(cv_res, dict):
                r = safe_array(first_present(cv_res, ["mean_cv_R2", "mean_cv_rho"]))
                if r is not None:
                    mean_cv_rho = r.ravel()

            # significant_components
            sig_raw = pr.get("significant_components")
            if sig_raw is None:
                sig_comps: List[int] = []
            elif isinstance(sig_raw, (int, float)):
                sig_comps = [int(sig_raw)]
            else:
                sig_comps = [int(s) for s in np.asarray(sig_raw).ravel()]

            def _sc(key: str, default: float = np.nan) -> float:
                v = pr.get(key)
                if v is None:
                    return default
                try:
                    return float(np.asarray(v).ravel()[0])
                except Exception:
                    return default

            def _si(key: str, default: int = 0) -> int:
                return int(_sc(key, default))

            def _sb(key: str) -> bool:
                v = pr.get(key)
                if v is None:
                    return False
                try:
                    return bool(np.asarray(v).ravel()[0])
                except Exception:
                    return False

            def _sl(key: str) -> List[str]:
                v = pr.get(key)
                if v is None:
                    return []
                if isinstance(v, str):
                    return [v]
                if isinstance(v, (list, tuple)):
                    return [str(s) for s in v if s is not None]
                return []

            pair = PCCAPair(
                session=session,
                region_i=r_i,
                region_j=r_j,
                Wx=Wx,
                Wy=Wy,
                mean_cv_rho=mean_cv_rho,
                significant_components=sig_comps,
                dominant_rho=_sc("dominant_rho"),
                subspace_dim=_si("subspace_dim"),
                mutual_information=_sc("mutual_information"),
                gini_i=_sc("gini_weights_i"),
                gini_j=_sc("gini_weights_j"),
                var_X_retained=_sc("variance_X_retained"),
                var_Y_retained=_sc("variance_Y_retained"),
                nuisance_regions=_sl("nuisance_regions"),
                nuisance_n_neurons=_si("nuisance_n_neurons"),
                is_partial=_sb("is_partial"),
            )
            self.pcca[cond].setdefault((r_i, r_j), []).append(pair)

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
# 3.  Combined convenience class
# ---------------------------------------------------------------------------

class OxfordPCCAAnalyzer(PCCAExtractor, OxfordAdvancedAnalyzer):
    """
    Drop-in replacement for OxfordAdvancedAnalyzer that additionally
    populates ``self.pcca`` after ``.load_all()``.

    Example
    -------
    >>> analyzer = OxfordPCCAAnalyzer(
    ...     base_results_dir="/data/Oxford_dataset",
    ...     results_subdirs={"cued_hit_long": "pcca_sessions_cued_hit_long_results"},
    ...     n_components=5,
    ... )
    >>> analyzer.load_all()
    >>> fig = plot_rho_heatmap(analyzer.pcca, cond="cued_hit_long")
    """
    pass


# ---------------------------------------------------------------------------
# 4.  Internal aggregate helpers
# ---------------------------------------------------------------------------

def _build_scalar_matrices(
    pcca_store: Dict[Tuple[str, str], List[PCCAPair]],
    region_order: List[str],
    attribute: str,
    aggregator=np.nanmedian,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an (R × R) matrix and a count matrix from a PCCAPair scalar
    attribute.

    Returns
    -------
    mat   : (R, R)  aggregated (median by default) per pair
    cnt   : (R, R)  number of sessions contributing
    """
    R   = len(region_order)
    idx = {r: i for i, r in enumerate(region_order)}

    mat = np.full((R, R), np.nan)
    cnt = np.zeros((R, R), dtype=int)

    for (r_i, r_j), pairs in pcca_store.items():
        if r_i not in idx or r_j not in idx:
            continue
        i, j = idx[r_i], idx[r_j]
        vals = np.array([getattr(p, attribute) for p in pairs], dtype=float)
        valid = np.isfinite(vals)
        if not valid.any():
            continue
        ri, rj = (i, j) if i < j else (j, i)
        mat[ri, rj] = float(aggregator(vals[valid]))
        cnt[ri, rj] = int(valid.sum())

    return mat, cnt


def _trim_to_present(
    mat: np.ndarray,
    cnt: np.ndarray,
    region_order: List[str],
    pcca_store: Dict[Tuple[str, str], List[PCCAPair]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Crop full (R×R) matrices to rows/columns that actually appear in data."""
    present = [
        r for r in region_order
        if any(r in (ri, rj) for (ri, rj) in pcca_store)
    ]
    pidx = [region_order.index(r) for r in present]
    return mat[np.ix_(pidx, pidx)], cnt[np.ix_(pidx, pidx)], present


def _draw_upper_triangle_heatmap(
    ax: plt.Axes,
    fig: plt.Figure,
    mat: np.ndarray,
    cnt: np.ndarray,
    labels: List[str],
    title: str,
    cbar_label: str,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = 1.0,
    min_n: int = 3,
    fmt_str: str = "{:.2f}\nn={:d}",
    diverging: bool = False,
) -> None:
    """
    Shared renderer for all upper-triangular heatmaps in this module.

    Cells with fewer than ``min_n`` contributing sessions are masked.
    """
    R = len(labels)

    mask_lo_n = cnt < min_n
    mask_tril = np.tril(np.ones((R, R), dtype=bool), k=-1)
    display = np.where(mask_lo_n | mask_tril, np.nan, mat)

    if diverging:
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = ax.imshow(display, cmap=cmap, norm=norm, aspect="equal",
                       interpolation="nearest")
    else:
        im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest")

    for i in range(R):
        for j in range(i + 1, R):
            n   = cnt[i, j]
            val = mat[i, j]
            if n >= min_n and np.isfinite(val):
                tc = "white" if (val - vmin) / (vmax - vmin + 1e-9) > 0.6 else "black"
                ax.text(j, i, fmt_str.format(val, n),
                        ha="center", va="center", fontsize=6.5, color=tc)

    ax.set_xticks(range(R)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(R)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label(cbar_label, fontsize=9)


# ---------------------------------------------------------------------------
# 5.  Visualisation — dominant rho heatmap
# ---------------------------------------------------------------------------

def plot_rho_heatmap(
    pcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[PCCAPair]]],
    cond: str,
    region_order: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    min_n: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Upper-triangular heatmap of median dominant_rho across sessions.

    Dominant rho = canonical correlation of the first significant pCCA
    dimension.  This is the primary inter-regional coupling strength metric
    and is the pCCA analogue of the tkCCA significance heatmap.
    """
    if region_order is None:
        region_order = ANATOMICAL_ORDER

    store = pcca_store_by_cond.get(cond, {})
    if not store:
        raise ValueError(f"No pCCA data for condition '{cond}'")

    mat, cnt = _build_scalar_matrices(store, region_order, "dominant_rho")
    mat_sub, cnt_sub, present = _trim_to_present(mat, cnt, region_order, store)
    R = len(present)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6, R * 0.9), max(5, R * 0.85)),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    _draw_upper_triangle_heatmap(
        ax, fig, mat_sub, cnt_sub, present,
        title=fr"Median dominant $\rho$  —  {cond}  (n≥{min_n})",
        cbar_label=r"Median dominant $\rho$ (dim 1)",
        cmap="YlOrRd", vmin=0.0, vmax=1.0, min_n=min_n,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 6.  Visualisation — subspace dimensionality heatmap
# ---------------------------------------------------------------------------

def plot_subspace_dim_heatmap(
    pcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[PCCAPair]]],
    cond: str,
    region_order: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    min_n: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Upper-triangular heatmap of median subspace dimensionality (number of
    significant pCCA dimensions) across sessions.

    Dimensionality quantifies the complexity of the communication subspace.
    High dimensionality means many independent patterns of coordinated
    activity link the two regions.
    """
    if region_order is None:
        region_order = ANATOMICAL_ORDER

    store = pcca_store_by_cond.get(cond, {})
    mat, cnt = _build_scalar_matrices(store, region_order, "subspace_dim")
    mat_sub, cnt_sub, present = _trim_to_present(mat, cnt, region_order, store)
    R = len(present)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6, R * 0.9), max(5, R * 0.85)),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    vmax = max(float(np.nanmax(mat_sub)), 1.0) if np.isfinite(mat_sub).any() else 5.0

    _draw_upper_triangle_heatmap(
        ax, fig, mat_sub, cnt_sub, present,
        title=f"Median subspace dimensionality  —  {cond}  (n≥{min_n})",
        cbar_label="Median # significant pCCA dims",
        cmap="PuBuGn", vmin=0.0, vmax=vmax, min_n=min_n,
        fmt_str="{:.1f}\nn={:d}",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 7.  Visualisation — mutual information heatmap
# ---------------------------------------------------------------------------

def plot_mi_heatmap(
    pcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[PCCAPair]]],
    cond: str,
    region_order: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    min_n: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Upper-triangular heatmap of median mutual information across sessions.

    MI = -sum_k log(1 - rho_k^2) integrates across all pCCA dimensions.
    It is the information-theoretic complement of the rho heatmap: pairs
    with moderate rho but high dimensionality can transmit more information
    than pairs with very high rho on a single dimension.
    """
    if region_order is None:
        region_order = ANATOMICAL_ORDER

    store = pcca_store_by_cond.get(cond, {})
    mat, cnt = _build_scalar_matrices(store, region_order, "mutual_information")
    mat_sub, cnt_sub, present = _trim_to_present(mat, cnt, region_order, store)
    R = len(present)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6, R * 0.9), max(5, R * 0.85)),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    vmax = max(float(np.nanpercentile(mat_sub[np.isfinite(mat_sub)], 95)), 0.1) \
        if np.isfinite(mat_sub).any() else 1.0

    _draw_upper_triangle_heatmap(
        ax, fig, mat_sub, cnt_sub, present,
        title=r"Median MI  $-\sum_k\log(1-\rho_k^2)$" + f"  —  {cond}  (n≥{min_n})",
        cbar_label=r"Median mutual information (nats)",
        cmap="plasma", vmin=0.0, vmax=vmax, min_n=min_n,
        fmt_str="{:.2f}\nn={:d}",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 8.  Visualisation — Gini coefficient distribution  (Fig. 1f analog)
# ---------------------------------------------------------------------------

def plot_gini_distribution(
    pcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[PCCAPair]]],
    cond: str,
    region_order: Optional[List[str]] = None,
    min_n_sessions: int = 3,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Per-region box-strip panel of |W| Gini coefficients (dominant pCCA dim),
    pooled across all sessions and all pairs in which the region participates.

    Analogous to Gonzalez et al. (2026) Fig. 1f: a monotonically decreasing
    Gini from upstream to downstream regions indicates progressive subspace
    expansion (more neurons recruited at later processing stages).

    Both the region_i Gini (gini_i, Wx[:,0]) and region_j Gini (gini_j,
    Wy[:,0]) are collected for each region, pooled across all pairs.

    A Page's L test for a linear trend along region_order is annotated.
    """
    if region_order is None:
        region_order = ANATOMICAL_ORDER

    store = pcca_store_by_cond.get(cond, {})
    gini_by_region: Dict[str, List[float]] = {}

    for (r_i, r_j), pairs in store.items():
        for p in pairs:
            if np.isfinite(p.gini_i):
                gini_by_region.setdefault(r_i, []).append(p.gini_i)
            if np.isfinite(p.gini_j):
                gini_by_region.setdefault(r_j, []).append(p.gini_j)

    present = [r for r in region_order
               if r in gini_by_region and len(gini_by_region[r]) >= min_n_sessions]

    if not present:
        raise ValueError(f"No Gini data for condition '{cond}' (n>={min_n_sessions})")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(5.0, 0.9 * len(present)), 4.5),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    rng      = np.random.default_rng(seed=42)
    medians  = []
    for xi, rname in enumerate(present):
        vals = np.array(gini_by_region[rname])
        medians.append(float(np.median(vals)))
        jitter = rng.uniform(-0.14, 0.14, size=vals.size)
        ax.scatter(xi + jitter, vals, s=18, alpha=0.55, color="C2",
                   zorder=3, linewidths=0)
        ax.boxplot(
            vals, positions=[xi], widths=0.38,
            patch_artist=True, notch=False,
            boxprops=dict(facecolor="C2", alpha=0.35),
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
            flierprops=dict(marker=""), zorder=2,
        )

    # Overlay median trend line
    ax.plot(range(len(present)), medians, color="k", lw=1.2,
            ls="--", zorder=4, label="Median trend")

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Gini coefficient of |W| (dominant dim)", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # Page's L statistic for monotone trend (direction: decreasing along list)
    # Reformulate as testing for *monotone decrease* → flip medians
    try:
        from scipy.stats import page_trend_test
        result_dec = page_trend_test(
            np.array([gini_by_region[r] for r in present], dtype=object),
            predicted_ranks=list(range(len(present), 0, -1)),  # decreasing
        )
        p_str = f"Page's L (decrease): p = {result_dec.pvalue:.3f}"
    except Exception:
        # scipy < 1.8 may not have page_trend_test
        p_str = ""

    ax.set_title(
        f"Gini coefficient distribution across regions  —  {cond}\n"
        + (p_str if p_str else "Higher Gini = sparser subspace coding"),
        fontsize=10,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 9.  Visualisation — variance retained after partialling
# ---------------------------------------------------------------------------

def plot_variance_retained(
    pcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[PCCAPair]]],
    cond: str,
    region_order: Optional[List[str]] = None,
    selected_pairs: Optional[List[Tuple[str, str]]] = None,
    min_n: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Per-pair strip-box plot of variance_X_retained and variance_Y_retained
    across sessions.

    Values close to 1 indicate that the nuisance matrix Z explained little
    shared variance, so pCCA ≈ plain CCA.  Low values indicate strong
    confounding by Z, making partial conditioning essential.

    Pairs are sorted in anatomical order and displayed on the y-axis.  The
    two quantities (X, Y) for each pair are shown as two overlapping strips
    with distinct colours.
    """
    if region_order is None:
        region_order = ANATOMICAL_ORDER

    store = pcca_store_by_cond.get(cond, {})
    pair_wl = (
        set(map(tuple, selected_pairs)) if selected_pairs is not None else None
    )

    def _key(p):
        ki = region_order.index(p[0]) if p[0] in region_order else 999
        kj = region_order.index(p[1]) if p[1] in region_order else 999
        return (ki, kj)

    data_x: Dict[Tuple[str, str], List[float]] = {}
    data_y: Dict[Tuple[str, str], List[float]] = {}
    for (r_i, r_j), pairs in store.items():
        if pair_wl is not None and (r_i, r_j) not in pair_wl:
            continue
        if len(pairs) < min_n:
            continue
        vx = [p.var_X_retained for p in pairs if np.isfinite(p.var_X_retained)]
        vy = [p.var_Y_retained for p in pairs if np.isfinite(p.var_Y_retained)]
        if vx:
            data_x[(r_i, r_j)] = vx
        if vy:
            data_y[(r_i, r_j)] = vy

    pair_keys = sorted(set(data_x) | set(data_y), key=_key)
    if not pair_keys:
        raise ValueError(f"No variance-retained data for condition '{cond}'")

    P = len(pair_keys)
    fig, ax = plt.subplots(
        figsize=(7, max(3.0, 0.4 * P + 1.5)),
        constrained_layout=True,
    )
    rng = np.random.default_rng(seed=0)

    for yi, pk in enumerate(pair_keys):
        for offset, data_dict, col, lbl in [
            (-0.18, data_x, "C3", "X"),
            (+0.18, data_y, "C0", "Y"),
        ]:
            vals = np.array(data_dict.get(pk, []))
            if not vals.size:
                continue
            jitter = rng.uniform(-0.06, 0.06, size=vals.size)
            ax.scatter(vals, yi + offset + jitter, s=20, alpha=0.55,
                       color=col, zorder=3, linewidths=0,
                       label=lbl if yi == 0 else "_nolegend_")
            ax.plot([np.nanmedian(vals)], [yi + offset],
                    marker="D", ms=6, color=col, zorder=4)

    ax.axvline(1.0, color="k", lw=0.7, ls="--", alpha=0.4)
    ax.set_yticks(range(P))
    ax.set_yticklabels(
        [f"{r_i}→{r_j}" for r_i, r_j in pair_keys], fontsize=8
    )
    ax.set_xlabel("Fraction of variance retained after Z removal", fontsize=10)
    ax.set_xlim(-0.02, 1.08)
    ax.legend(
        [plt.Line2D([0], [0], marker="o", color="w", mfc="C3", ms=7),
         plt.Line2D([0], [0], marker="o", color="w", mfc="C0", ms=7)],
        ["region_i (X)", "region_j (Y)"],
        fontsize=8, frameon=False, loc="lower right",
    )
    ax.set_title(
        f"Nuisance-conditioning efficiency  —  {cond}\n"
        "Low value = Z strongly confounded the raw pair",
        fontsize=10,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 10. Visualisation — condition comparison of dominant rho
# ---------------------------------------------------------------------------

def plot_condition_rho_comparison(
    pcca_by_cond: Dict[str, Dict[Tuple[str, str], List[PCCAPair]]],
    pairs: List[Tuple[str, str]],
    conditions: Optional[List[str]] = None,
    region_order: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlaid cumulative-density comparison of dominant_rho across the three
    trial conditions for a set of user-specified region pairs.

    Each subplot = one pair.  Each condition = one CDF curve.  This allows
    direct assessment of whether voluntary (spontaneous hit) vs. cued vs.
    miss trials differ in inter-regional coupling strength — the central
    question of your dataset.

    Parameters
    ----------
    pairs : list of (region_i, region_j) to show.
    conditions : subset of condition labels to include (default: all).
    """
    if region_order is None:
        region_order = ANATOMICAL_ORDER
    if conditions is None:
        conditions = list(pcca_by_cond.keys())

    cond_colors = {c: f"C{i}" for i, c in enumerate(conditions)}

    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = int(np.ceil(n_pairs / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.2, nrows * 3.5),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for pi, (r_i, r_j) in enumerate(pairs):
        ax = axes_flat[pi]
        any_drawn = False

        for cond in conditions:
            store = pcca_by_cond.get(cond, {})
            pair_data = store.get((r_i, r_j), [])
            rhos = np.array([p.dominant_rho for p in pair_data
                             if np.isfinite(p.dominant_rho)])
            if rhos.size == 0:
                continue
            sorted_rhos = np.sort(rhos)
            cdf = np.arange(1, rhos.size + 1) / rhos.size
            ax.step(sorted_rhos, cdf, color=cond_colors[cond],
                    lw=1.8, label=f"{cond} (n={rhos.size})", where="post")
            any_drawn = True

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"Dominant $\rho$", fontsize=9)
        ax.set_ylabel("CDF", fontsize=9)
        ax.set_title(f"{r_i} → {r_j}", fontsize=10)
        if any_drawn:
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        r"pCCA dominant $\rho$ — condition comparison",
        fontsize=13, fontweight="bold",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 11. Compound summary — three-panel [rho | dim | MI]
# ---------------------------------------------------------------------------

def plot_pcca_summary(
    pcca_store_by_cond: Dict[str, Dict[Tuple[str, str], List[PCCAPair]]],
    cond: str,
    region_order: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compound three-panel figure: [dominant rho | subspace dim | MI].

    Provides a comprehensive session-level summary in a single manuscript
    figure.  The three panels answer:
        (a) How strong is the inter-regional coupling?          (rho)
        (b) How complex (multi-dimensional) is the subspace?   (dim)
        (c) How much information is transferred in total?       (MI)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(22, 8),
        constrained_layout=True,
    )
    plot_rho_heatmap(pcca_store_by_cond, cond, region_order=region_order, ax=ax1)
    plot_subspace_dim_heatmap(pcca_store_by_cond, cond, region_order=region_order, ax=ax2)
    plot_mi_heatmap(pcca_store_by_cond, cond, region_order=region_order, ax=ax3)

    fig.suptitle(
        f"pCCA inter-regional subspace summary  —  {cond}",
        fontsize=14, fontweight="bold",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 12.  Example driver
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir   = "/Users/shengyuancai/Downloads/Oxford_dataset"
    output_dir = Path(base_dir) / "Paper_output" / "figures_pcca"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = OxfordPCCAAnalyzer(
        base_results_dir=base_dir,
        results_subdirs={
            "cued_hit_long":   "pcca_sessions_cued_hit_long_results",
        },
        n_components=5,
    # "spont_hit_long":  "pcca_sessions_spont_hit_long_results",
    # "spont_miss_long": "pcca_sessions_spont_miss_long_results",
    )
    analyzer.load_all()

    cond = "cued_hit_long"

    # ── Compound summary ──────────────────────────────────────────────────
    plot_pcca_summary(
        analyzer.pcca, cond,
        save_path=str(output_dir / f"pcca_summary_{cond}.png"),
    )

    # ── Gini distribution (Fig. 1f analog) ────────────────────────────────
    plot_gini_distribution(
        analyzer.pcca, cond,
        save_path=str(output_dir / f"pcca_gini_{cond}.png"),
    )

    # ── Variance retained diagnostic ──────────────────────────────────────
    plot_variance_retained(
        analyzer.pcca, cond,
        selected_pairs=[
            ("ORB", "STR"), ("mPFC", "STR"),
            ("MOp", "STR"), ("MOs", "STR"),
        ],
        save_path=str(output_dir / f"pcca_variance_retained_{cond}.png"),
    )

    # ── Condition comparison (cued vs. spont hit vs. miss) ────────────────
    plot_condition_rho_comparison(
        analyzer.pcca,
        pairs=[
            ("ORB", "STR"), ("mPFC", "STR"),
            ("MOp", "STR"), ("MOs", "STR"),
            ("MOp", "MOs"),
        ],
        save_path=str(output_dir / "pcca_condition_rho_comparison.png"),
    )

    plt.show()


if __name__ == "__main__":
    main()
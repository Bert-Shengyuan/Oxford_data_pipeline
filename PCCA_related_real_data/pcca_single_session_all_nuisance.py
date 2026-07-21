"""
pcca_single_session.py
======================
Single-session visualisation for partial CCA outputs produced by
``perform_session_pcca.m``.

Unlike the tkCCA visualiser this file replaces, there is no temporal-lag
structure (no ``wx_temporal``, no ``canonical_correlogram``).  The key
objects are instead the standard canonical weight matrices
``mean_A_matrix`` / ``mean_B_matrix`` (Wx, Wy) and the pCCA-specific
diagnostic fields appended by ``perform_session_pcca.m``.

Per-pair figures (one set per region pair, produced by
``create_pcca_pair_figures``):

  Figure 1  ``*_1_rastermap.png``
      Two rows (region_i, region_j).  Each row: Rastermap-ordered,
      trial-averaged PSTH (RdBu_r, real time axis) beside a horizontal
      pCCA weight barh (Wx[:,0] and Wy[:,0], dominant dimension).
      Mirrors the tkCCA rastermap figure; remains the visual anchor for
      interpreting which neurons drive the subspace.

  Figure 2  ``*_2_projections.png``
      2 × n_sig grid: region_i latent z_i(t) and region_j latent z_j(t)
      for each significant pCCA dimension.  Individual trials shown as
      thin translucent lines; trial mean ± SEM overlaid.

  Figure 3  ``*_3_canonical_spectrum.png``
      Two panels side-by-side:
        Left:  rho_c bar chart (canonical correlations per component,
               significant bars highlighted in C3/C0).
        Right: variance-retained diagnostic (scalar values
               variance_X_retained, variance_Y_retained after nuisance
               partialling), shown as a horizontal stacked bar so the
               user can judge how strongly Z confounded the pair.
      Analogous to the "subspace dimensionality" panels in Gonzalez
      et al. (2026) Fig. 1c/f.

  Figure 4  ``*_4_weight_scatter.png``
      Scatter of |Wx(:,0)| vs |Wy(:,0)| for neurons in CA1 (the pivot
      region, when the pair includes it).  Captures the Gonzalez et al.
      (2026) Fig. 1d observation that neurons with large weights in one
      subspace tend to have large weights in others.  Generalised here to
      any pair: both axes are plotted for whichever region is common.

Session-wide figures (produced by the corresponding methods):

  Figure 5  ``*_rho_heatmap.png``
      Upper-triangular colour matrix of dominant_rho (the rho of the first
      significant pCCA dimension) across all valid region pairs.  Replaces
      the tkCCA correlogram heatmap as the session-level summary.

  Figure 6  ``*_mi_bar.png``
      Horizontal bar chart of mutual information (MI) per region pair,
      sorted descending.  MI = -sum_k log(1-rho_k^2) (Gaussian
      approximation; Gonzalez et al. 2026 Methods).  Offers a single
      scalar that integrates across all pCCA dimensions.

  Figure 7  ``*_gini_panel.png``
      Per-region box-strip panel of |Wx| Gini coefficients (dominant
      dimension) across all pairs in which that region participates.
      Analogous to Gonzalez et al. (2026) Fig. 1f; monotonically
      decreasing Gini from early to late areas would indicate progressive
      subspace expansion along the circuit.

  Figure 8  ``*_subspace_angles.png``
      Pairwise principal-angle heatmap between pCCA subspaces that share
      a common pivot region.  Subspaces computed from mean_A_matrix and
      mean_B_matrix; angles reveal whether distinct partners recruit the
      same (small angle) or orthogonal (large angle) subsets of pivot
      neurons.  Analogous to Gonzalez et al. (2026) Fig. 1g.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import zscore

try:
    import mat73
except ImportError as exc:
    raise SystemExit("mat73 is required: pip install mat73") from exc

try:
    from rastermap import Rastermap
    RASTERMAP_AVAILABLE = True
except ImportError:
    RASTERMAP_AVAILABLE = False
    warnings.warn("rastermap not available; falling back to identity ordering.")

from Useful_definition import ANATOMICAL_ORDER, safe_array, first_present


# =============================================================================
# 1.  Lightweight pair container
# =============================================================================

@dataclass
class _SessionPCCAPair:
    """
    Parsed view of one ``pair_results`` entry from a pCCA session file.

    All arrays follow MATLAB storage conventions (confirmed from debugger):
        mean_A_matrix  : (n_i, n_components)
        mean_B_matrix  : (n_j, n_components)
        mean_cv_R2     : (n_components,)
        sig_comps      : list of int (1-indexed MATLAB indices)
    """
    region_i: str
    region_j: str

    X_i: np.ndarray           # (n_trials, n_i, T)
    X_j: np.ndarray           # (n_trials, n_j, T)

    Wx: np.ndarray            # (n_i, n_components)  mean canonical weights
    Wy: np.ndarray            # (n_j, n_components)

    mean_cv_rho: np.ndarray   # (n_components,)
    sig_comps: List[int]      # 1-indexed significant component indices

    # pCCA-specific diagnostics (scalar floats)
    dominant_rho: float
    subspace_dim: int
    mutual_information: float
    gini_i: float             # Gini of |Wx[:,0]|
    gini_j: float             # Gini of |Wy[:,0]|
    var_X_retained: float
    var_Y_retained: float
    nuisance_regions: List[str]
    is_partial: bool

    @property
    def n_sig(self) -> int:
        return len(self.sig_comps)

    @property
    def pair_label(self) -> str:
        return f"{self.region_i} \u2194 {self.region_j}"


# =============================================================================
# 2.  Session visualiser
# =============================================================================

class OxfordPCCASessionVisualizer:
    """
    Single ``*_analysis_results.mat`` file in; per-pair + session-wide
    figures out.

    Typical call sequence::

        viz = OxfordPCCASessionVisualizer(
            session_results_path="…/yp020_220331_analysis_results.mat",
            session_name="yp020_220331",
            output_dir="…/figures_pcca/yp020_220331",
        )
        viz.load_session()
        viz.compute_global_rastermap()
        viz.run_all_pairs(component_idx=0)
    """

    def __init__(
        self,
        session_results_path: str,
        session_name: str,
        output_dir: str,
        time_range_s: Tuple[float, float] = (-1.5, 3.0),
    ) -> None:
        self.session_path = Path(session_results_path)
        self.session_name = session_name
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.time_range_s = time_range_s

        self.region_data: Optional[dict] = None
        self.cca_results: Optional[dict] = None
        self.pair_results: List[dict]    = []
        self.pair_index: Dict[Tuple[str, str], int] = {}
        self.valid_regions: List[str]    = []
        self.T:   Optional[int]          = None
        self.time_vec: Optional[np.ndarray] = None

        self.global_sort: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_session(self) -> bool:
        try:
            data = mat73.loadmat(str(self.session_path))
        except Exception as exc:
            print(f"[{self.session_name}] load failed: {exc}")
            return False

        self.region_data = data.get("region_data", {})
        self.cca_results = data.get("cca_results", {})

        if not isinstance(self.region_data, dict) or "regions" not in self.region_data:
            print(f"[{self.session_name}] missing region_data.regions")
            return False

        pr_raw = self.cca_results.get("pair_results", []) \
            if isinstance(self.cca_results, dict) else []
        if isinstance(pr_raw, np.ndarray):
            pr_raw = pr_raw.tolist()
        if not isinstance(pr_raw, (list, tuple)):
            pr_raw = []
        self.pair_results = list(pr_raw)

        for idx, pr in enumerate(self.pair_results):
            if not isinstance(pr, dict):
                continue
            r_i = self._as_str(pr.get("region_i"))
            r_j = self._as_str(pr.get("region_j"))
            if r_i and r_j:
                self.pair_index[(r_i, r_j)] = idx

        regions_dict = self.region_data["regions"]
        self.valid_regions = [
            r for r, info in regions_dict.items()
            if isinstance(info, dict) and "spike_data" in info
        ]

        if self.valid_regions:
            sd0 = safe_array(regions_dict[self.valid_regions[0]]["spike_data"])
            if sd0 is not None and sd0.ndim == 3:
                self.T = int(sd0.shape[2])
                self.time_vec = np.linspace(
                    self.time_range_s[0], self.time_range_s[1], self.T
                )

        print(f"[{self.session_name}] loaded: {len(self.valid_regions)} regions, "
              f"{len(self.pair_index)} pairs, T={self.T}")
        return True

    # ------------------------------------------------------------------
    # Global Rastermap
    # ------------------------------------------------------------------

    def compute_global_rastermap(
        self,
        n_pcs: int = 200,
        locality: float = 0.0,
        grid_upsample: int = 5,
    ) -> None:
        """Pool all regions into one matrix, fit Rastermap once, store per-region orderings."""
        if self.region_data is None:
            raise RuntimeError("Call load_session() first.")

        regions_dict = self.region_data["regions"]
        pooled, offsets = [], {}
        cursor = 0

        for rname in self.valid_regions:
            sd = safe_array(regions_dict[rname].get("spike_data"))
            if sd is None or sd.ndim != 3:
                continue
            sel = safe_array(regions_dict[rname].get("selected_neurons"))
            if sel is not None:
                sd = sd[:, sel.ravel().astype(int) - 1, :]
            n_trials, n_neurons, T = sd.shape
            mat = sd.transpose(1, 2, 0).reshape(n_neurons, T * n_trials)
            pooled.append(mat)
            offsets[rname] = (cursor, cursor + n_neurons)
            cursor += n_neurons

        if not pooled:
            print(f"[{self.session_name}] no spike_data for Rastermap")
            return

        pooled_z = zscore(np.vstack(pooled), axis=1, nan_policy="omit")
        pooled_z = np.nan_to_num(pooled_z, nan=0.0)

        if not RASTERMAP_AVAILABLE:
            for rname, (lo, hi) in offsets.items():
                self.global_sort[rname] = np.arange(hi - lo)
            return

        model = Rastermap(
            n_PCs=min(n_pcs, pooled_z.shape[0]),
            locality=locality,
            grid_upsample=grid_upsample,
        )
        model.fit(pooled_z)
        g = model.isort
        for rname, (lo, hi) in offsets.items():
            keep = (g >= lo) & (g < hi)
            self.global_sort[rname] = g[keep] - lo

        print(f"[{self.session_name}] Rastermap fitted "
              f"({pooled_z.shape[0]} neurons, {len(offsets)} regions)")

    # ------------------------------------------------------------------
    # Per-pair extraction
    # ------------------------------------------------------------------

    def _build_session_pair(self, region_i: str, region_j: str) -> Optional[_SessionPCCAPair]:
        """Parse one ``pair_results`` entry into ``_SessionPCCAPair``."""
        key = (region_i, region_j)
        if key not in self.pair_index:
            print(f"  ({region_i}, {region_j}) not in pair_results")
            return None

        pr = self.pair_results[self.pair_index[key]]
        Wx = safe_array(pr.get("mean_A_matrix"))
        Wy = safe_array(pr.get("mean_B_matrix"))
        if Wx is None or Wy is None:
            print(f"  ({region_i}, {region_j}): missing mean_A/B_matrix")
            return None

        if Wx.ndim == 1:
            Wx = Wx[:, np.newaxis]
        if Wy.ndim == 1:
            Wy = Wy[:, np.newaxis]

        # Cross-validated canonical correlations
        cvr = pr.get("cv_results", {})
        mean_cv_rho = np.array([])
        if isinstance(cvr, dict):
            r = safe_array(first_present(cvr, ["mean_cv_R2", "mean_cv_rho"]))
            if r is not None:
                mean_cv_rho = r.ravel()

        # Significant components (MATLAB 1-indexed)
        sig_raw = pr.get("significant_components")
        if sig_raw is None:
            sig_comps: List[int] = []
        elif isinstance(sig_raw, (int, float)):
            sig_comps = [int(sig_raw)]
        else:
            sig_comps = [int(s) for s in np.asarray(sig_raw).ravel()]

        # pCCA-specific scalars (with defensive fallbacks)
        def _scalar(key_: str, default: float = np.nan) -> float:
            v = pr.get(key_)
            if v is None:
                return default
            try:
                return float(np.asarray(v).ravel()[0])
            except Exception:
                return default

        def _bool_field(key_: str) -> bool:
            v = pr.get(key_)
            if v is None:
                return False
            try:
                return bool(np.asarray(v).ravel()[0])
            except Exception:
                return False

        def _strlist(key_: str) -> List[str]:
            v = pr.get(key_)
            if v is None:
                return []
            if isinstance(v, str):
                return [v]
            if isinstance(v, (list, tuple)):
                return [str(s) for s in v if s is not None]
            return []

        # Spike data (apply neuron selection from pair_result)
        regions_dict = self.region_data["regions"]
        X_i = safe_array(regions_dict.get(region_i, {}).get("spike_data"))
        X_j = safe_array(regions_dict.get(region_j, {}).get("spike_data"))
        sel_i = safe_array(pr.get("selected_neurons_i"))
        sel_j = safe_array(pr.get("selected_neurons_j"))
        if X_i is not None and sel_i is not None:
            X_i = X_i[:, sel_i.ravel().astype(int) - 1, :]
        if X_j is not None and sel_j is not None:
            X_j = X_j[:, sel_j.ravel().astype(int) - 1, :]

        if X_i is None or X_j is None:
            print(f"  ({region_i}, {region_j}): spike_data unavailable")
            return None

        return _SessionPCCAPair(
            region_i=region_i,
            region_j=region_j,
            X_i=X_i,
            X_j=X_j,
            Wx=Wx,
            Wy=Wy,
            mean_cv_rho=mean_cv_rho,
            sig_comps=sig_comps,
            dominant_rho=_scalar("dominant_rho"),
            subspace_dim=int(_scalar("subspace_dim", 0)),
            mutual_information=_scalar("mutual_information"),
            gini_i=_scalar("gini_weights_i"),
            gini_j=_scalar("gini_weights_j"),
            var_X_retained=_scalar("variance_X_retained"),
            var_Y_retained=_scalar("variance_Y_retained"),
            nuisance_regions=_strlist("nuisance_regions"),
            is_partial=_bool_field("is_partial"),
        )

    # ------------------------------------------------------------------
    # Projection utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore_per_neuron(X: np.ndarray) -> np.ndarray:
        """
        Z-score each neuron across all (trial × time) samples.
        Mirrors the MATLAB convention: zscore(mat, 0, 1).

        X : (n_trials, n_neurons, T)
        Returns: (n_trials, n_neurons, T)
        """
        n_trials, n, T = X.shape
        flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
        flat = zscore(flat, axis=1, nan_policy="omit")
        flat = np.nan_to_num(flat, nan=0.0)
        return flat.reshape(n, T, n_trials).transpose(2, 0, 1)

    @staticmethod
    def _project(X_z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        z(r, t) = X_z(r, :, t) @ w    for every trial r and time t.
        X_z : (n_trials, n_neurons, T)
        w   : (n_neurons,)
        Returns: (n_trials, T)
        """
        return np.einsum("rnt,n->rt", X_z, w)

    @staticmethod
    def _plot_with_trials(
        ax: plt.Axes,
        time_vec: np.ndarray,
        trials: np.ndarray,
        color: str,
        label: Optional[str] = None,
        alpha_trial: float = 0.05,
        lw_trial: float = 0.25,
    ) -> None:
        """
        Individual trial traces (thin, translucent) with mean ± SEM overlay.
        trials: (n_trials, T)
        """
        for tr in trials:
            ax.plot(time_vec, tr, color=color, lw=lw_trial,
                    alpha=alpha_trial, rasterized=True)
        mean = np.nanmean(trials, axis=0)
        sem  = np.nanstd(trials, axis=0) / np.sqrt(trials.shape[0])
        ax.plot(time_vec, mean, color=color, lw=2.0, label=label, zorder=3)
        ax.fill_between(time_vec, mean - sem, mean + sem,
                        color=color, alpha=0.25, zorder=2)
        total_min = np.nanmin(trials)
        total_max = np.nanmax(trials)

        ax.set_ylim(bottom=total_min * 0.5, top=total_max * 0.5)

    # ------------------------------------------------------------------
    # Figure 1 — Rastermap + pCCA weight bar
    # ------------------------------------------------------------------

    def _figure_rastermap(
        self,
        pair: _SessionPCCAPair,
        c: int,
        n_show: int,
        save: bool,
    ) -> Optional[plt.Figure]:
        """
        Two-row figure (region_i top, region_j bottom).
        Each row: Rastermap-ordered PSTH (RdBu_r) beside a horizontal barh
        of the pCCA weight vector (Wx[:,c] or Wy[:,c]).

        Nuisance regions that were conditioned out are listed in the suptitle
        so the reader knows what has been partialled.
        """
        X_i_z = self._zscore_per_neuron(pair.X_i)
        X_j_z = self._zscore_per_neuron(pair.X_j)

        fig, axes = plt.subplots(
            2, 2,
            figsize=(12, 11),
            gridspec_kw={"width_ratios": [4.5, 1.0], "hspace": 0.45, "wspace": 0.08},
        )

        for row, (X_z, weight, rname, bar_lbl) in enumerate([
            (X_i_z, pair.Wx[:, c], pair.region_i, r"$W_x$ (dominant dim)"),
            (X_j_z, pair.Wy[:, c], pair.region_j, r"$W_y$ (dominant dim)"),
        ]):
            self._draw_rastermap_panel(
                fig,
                ax_raster=axes[row, 0],
                ax_bar=axes[row, 1],
                X_z=X_z,
                region_name=rname,
                sort_idx=self.global_sort.get(rname, np.arange(X_z.shape[1])),
                weight=weight,
                n_show=n_show,
                time_vec=self.time_vec,
                bar_label=bar_lbl,
            )

        partial_note = (
            f"Conditioned on: {', '.join(pair.nuisance_regions)}"
            if pair.is_partial and pair.nuisance_regions
            else "Standard CCA (no nuisance regions)"
        )
        fig.suptitle(
            f"{self.session_name}   |   {pair.pair_label}   "
            f"(dim 1,  ρ = {pair.dominant_rho:.3f})\n{partial_note}",
            fontsize=12, fontweight="bold",
        )

        if save:
            out = self.output_dir / (
                f"1_rastermap_{self.session_name}_{pair.region_i}_{pair.region_j}.png"
            )
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    @staticmethod
    def _draw_rastermap_panel(
        fig: plt.Figure,
        ax_raster: plt.Axes,
        ax_bar: plt.Axes,
        X_z: np.ndarray,
        region_name: str,
        sort_idx: np.ndarray,
        weight: np.ndarray,
        n_show: int,
        time_vec: np.ndarray,
        bar_label: str,
    ) -> None:
        """One Rastermap row: PSTH imshow (left) + pCCA weight barh (right)."""
        n_neurons = X_z.shape[1]
        if sort_idx.size != n_neurons:
            sort_idx = np.arange(n_neurons)

        step = max(1, n_neurons // n_show)
        sel  = sort_idx[::step][:n_show]
        n_sel = len(sel)

        psth = X_z.mean(axis=0)[sel]   # (n_sel, T)
        vmax = max(float(np.nanpercentile(np.abs(psth), 99)), 0.5)

        im = ax_raster.imshow(
            psth, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            extent=[time_vec[0], time_vec[-1], n_sel, 0], origin="upper",
        )
        ax_raster.axvline(0, color="red", linestyle="--", lw=1.5, alpha=0.9,
                          label="Reach onset")
        ax_raster.legend(loc="upper right", fontsize=8, framealpha=0.4)
        ax_raster.set_xlabel("Time (s)", fontsize=10)
        ax_raster.set_ylabel("Neurons (global Rastermap order)", fontsize=10)
        ax_raster.set_title(f"{region_name} — pCCA-ordered activity", fontsize=11)
        cb = fig.colorbar(im, ax=ax_raster, fraction=0.03, pad=0.03)
        cb.set_label("Z-scored firing rate", fontsize=8)

        pos = ax_bar.get_position()
        ax_bar.set_position([pos.x0 + 0.1, pos.y0, pos.width, pos.height])

        ypos  = np.arange(n_sel) + 0.5
        w_sel = weight[sel].ravel()
        colors = ["C3" if w > 0 else "C0" for w in w_sel]
        ax_bar.barh(ypos, w_sel, height=0.8, color=colors, alpha=0.85)
        ax_bar.axvline(0, color="k", lw=0.8, alpha=0.6)
        ax_bar.set_ylim(n_sel, 0)
        ax_bar.set_xlabel("pCCA weight", fontsize=8)
        ax_bar.set_title(bar_label, fontsize=8)
        ax_bar.tick_params(axis="both", labelsize=7)
        for sp in ("top", "right", "left"):
            ax_bar.spines[sp].set_visible(False)
        plt.setp(ax_bar.get_yticklabels(), visible=False)

    # ------------------------------------------------------------------
    # Figure 2 — pCCA projections across significant dimensions
    # ------------------------------------------------------------------

    def _figure_projections(
        self,
        pr: dict,
        pair: _SessionPCCAPair,
        save: bool,
    ) -> Optional[plt.Figure]:
        """
        2 × n_sig grid of the pCCA residual projections z_i(t) and z_j(t),
        stored by ``calculate_pcca_projections`` in MATLAB.

        Parsing follows the same mat73 convention as the tkCCA visualiser:
            projections['components'][k]     -> list-of-one-dict or dict
            projections['components'][k][0]  -> {'region_i_trials', 'region_j_trials', 'R2'}
        """
        proj = pr.get("projections")
        if not isinstance(proj, dict):
            print(f"  no projections for ({pair.region_i}, {pair.region_j})")
            return None

        components = proj.get("components")
        if not isinstance(components, (list, tuple)) or not components:
            return None

        comp_data = []
        for comp in components:
            cd = comp[0] if isinstance(comp, (list, tuple)) and comp else comp
            if not isinstance(cd, dict):
                continue
            zi = safe_array(cd.get("region_i_trials"))
            zj = safe_array(cd.get("region_j_trials"))
            if zi is None or zj is None or zi.ndim != 2 or zj.ndim != 2:
                continue
            R2_raw = cd.get("R2", 0.0)
            R2 = float(np.asarray(R2_raw).ravel()[0])
            comp_data.append({"zi": zi, "zj": zj, "R2": R2})

        if not comp_data:
            return None

        n_comps = len(comp_data)
        T_proj  = comp_data[0]["zi"].shape[1]
        time_raw = safe_array(first_present(proj, ["time_axis"]))
        time_vec = (
            time_raw.ravel() if time_raw is not None and time_raw.size == T_proj
            else np.linspace(self.time_range_s[0], self.time_range_s[1], T_proj)
        )

        fig, axes = plt.subplots(
            2, n_comps,
            figsize=(max(10.0, 3.5 * n_comps), 7.0),
            gridspec_kw={"hspace": 0.38, "wspace": 0.25},
            squeeze=False,
        )

        for ci, cd in enumerate(comp_data):
            ax_i, ax_j = axes[0, ci], axes[1, ci]

            self._plot_with_trials(ax_i, time_vec, cd["zi"], color="C3")
            ax_i.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax_i.set_title(fr"Dim {ci + 1}   $\rho = {cd['R2']:.3f}$", fontsize=9)
            ax_i.tick_params(labelsize=7)
            if ci == 0:
                ax_i.set_ylabel(fr"{pair.region_i}  $z_i(t)$", fontsize=10)
            for sp in ("top", "right"):
                ax_i.spines[sp].set_visible(False)

            self._plot_with_trials(ax_j, time_vec, cd["zj"], color="C0")
            ax_j.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax_j.set_xlabel("Time (s)", fontsize=9)
            ax_j.tick_params(labelsize=7)
            if ci == 0:
                ax_j.set_ylabel(fr"{pair.region_j}  $z_j(t)$", fontsize=10)
            for sp in ("top", "right"):
                ax_j.spines[sp].set_visible(False)

        fig.suptitle(
            f"{self.session_name}   |   {pair.pair_label}   "
            "— pCCA projections (residuals after Z partialling)",
            fontsize=12,
        )
        if save:
            out = self.output_dir / (
                f"2_projections_{self.session_name}_{pair.region_i}_{pair.region_j}.png"
            )
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Figure 3 — Canonical correlation spectrum + nuisance diagnostic
    # ------------------------------------------------------------------

    def _figure_canonical_spectrum(
        self,
        pair: _SessionPCCAPair,
        save: bool,
    ) -> Optional[plt.Figure]:
        """
        Left:  Bar chart of rho_c across all components.
               Significant components (sig_comps) highlighted.
        Right: Stacked horizontal bar of variance retained in X and Y
               after Z partialling — a direct readout of how much the
               nuisance regions were confounding the raw pair.

        Analogous to Gonzalez et al. (2026) subspace dimensionality panels.
        """
        rho = pair.mean_cv_rho
        if rho.size == 0:
            return None

        n_comp = rho.size
        sig_set = set(pair.sig_comps)        # 1-indexed

        fig, (ax_rho, ax_var) = plt.subplots(
            1, 2,
            figsize=(12, 4.5),
            gridspec_kw={"width_ratios": [3.0, 1.5], "wspace": 0.35},
            constrained_layout=True,
        )

        # ── Left: rho spectrum ──────────────────────────────────────────
        x = np.arange(1, n_comp + 1)
        bar_colors = [
            "C3" if i in sig_set else "#BBBBBB"
            for i in x
        ]
        ax_rho.bar(x, rho, color=bar_colors, edgecolor="none", alpha=0.85)
        ax_rho.axhline(0, color="k", lw=0.6)
        ax_rho.set_xlabel("pCCA dimension", fontsize=10)
        ax_rho.set_ylabel(r"Cross-validated $\rho$", fontsize=10)
        ax_rho.set_xlim(0.3, n_comp + 0.7)
        ax_rho.set_title(
            fr"Canonical correlation spectrum"
            f"\n{pair.pair_label}  |  "
            fr"$\rho_{{1}} = {pair.dominant_rho:.3f}$, "
            fr"dim = {pair.subspace_dim}",
            fontsize=10,
        )

        # Patch legend: significant vs not
        from matplotlib.patches import Patch
        ax_rho.legend(
            handles=[
                Patch(facecolor="C3", label="Significant"),
                Patch(facecolor="#BBBBBB", label="Not significant"),
            ],
            fontsize=8, loc="upper right", frameon=False,
        )
        for sp in ("top", "right"):
            ax_rho.spines[sp].set_visible(False)

        # MI annotation
        mi_str = (f"MI = {pair.mutual_information:.3f}"
                  if np.isfinite(pair.mutual_information) else "MI = n/a")
        ax_rho.text(
            0.02, 0.96, mi_str,
            transform=ax_rho.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        )

        # ── Right: variance-retained stacked bar ────────────────────────
        var_x = pair.var_X_retained if np.isfinite(pair.var_X_retained) else 0.0
        var_y = pair.var_Y_retained if np.isfinite(pair.var_Y_retained) else 0.0

        labels  = [fr"{pair.region_i}  $X_{{res}}$",
                   fr"{pair.region_j}  $Y_{{res}}$"]
        vals    = [var_x, var_y]
        ypos    = np.array([0.0, 1.0])
        ax_var.barh(ypos, vals, height=0.4, color=["C3", "C0"], alpha=0.85)
        ax_var.barh(ypos, [1 - v for v in vals], height=0.4,
                    left=vals, color=["#FFCDD2", "#BBDEFB"], alpha=0.6)
        ax_var.axvline(1, color="k", lw=0.6, ls="--", alpha=0.5)
        ax_var.set_xlim(0, 1.05)
        ax_var.set_yticks(ypos)
        ax_var.set_yticklabels(labels, fontsize=9)
        ax_var.set_xlabel("Fraction of variance\nretained after Z removal", fontsize=9)
        ax_var.set_title(
            "Nuisance-conditioning\ndiagnostic",
            fontsize=10,
        )
        for sp in ("top", "right"):
            ax_var.spines[sp].set_visible(False)

        nuisance_note = (
            "Partialled: " + ", ".join(pair.nuisance_regions)
            if pair.is_partial else "No partialling"
        )
        ax_var.text(
            0.02, -0.18, nuisance_note,
            transform=ax_var.transAxes, fontsize=7, color="gray",
        )

        if save:
            out = self.output_dir / (
                f"3_canonical_spectrum.png_{self.session_name}_{pair.region_i}_{pair.region_j}"
            )
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Figure 4 — Weight scatter |Wx| vs |Wy| per neuron (pivot-region)
    # ------------------------------------------------------------------

    def _figure_weight_scatter(
        self,
        pair: _SessionPCCAPair,
        save: bool,
    ) -> Optional[plt.Figure]:
        """
        Scatter plot of absolute pCCA weight magnitudes for the dominant
        dimension.

        Each point is a neuron in the region indicated on the x-axis; its
        x-coordinate is |Wx[n, 0]| (contribution to subspace defined with
        region_j as partner) and y-coordinate is |Wy[n, 0]| if the same
        neuron is shared, or vice versa.

        In the two-region case this is a direct point cloud (|Wx| vs |Wy|)
        with a Pearson correlation annotated — directly analogous to
        Gonzalez et al. (2026) Fig. 1e, which shows that neurons with large
        weights in one subspace tend to have large weights in others.

        For the single-pair case plotted here, the x-axis is |Wx| for
        region_i and the y-axis is |Wy| for region_j.  Because these are
        different populations the scatter displays weight-magnitude
        *distributions*, not neuron-by-neuron correspondence.  We therefore
        show both marginal histograms as small insets.
        """
        wx_abs = np.abs(pair.Wx[:, 0])
        wy_abs = np.abs(pair.Wy[:, 0])

        fig = plt.figure(figsize=(8, 5.5), constrained_layout=True)
        gs  = fig.add_gridspec(2, 2, width_ratios=[3.5, 1], height_ratios=[1, 3.5],
                               hspace=0.05, wspace=0.05)
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        # Use min(n_i, n_j) points; pad with NaN if sizes differ
        n_min = min(len(wx_abs), len(wy_abs))
        wx_plot = wx_abs[:n_min]
        wy_plot = wy_abs[:n_min]

        # Gini-sized markers: larger = higher Gini (sparser coding)
        gini_i = pair.gini_i if np.isfinite(pair.gini_i) else 0.5
        gini_j = pair.gini_j if np.isfinite(pair.gini_j) else 0.5
        msize  = 12 + 40 * np.clip((gini_i + gini_j) / 2, 0, 1)

        ax_main.scatter(wx_plot, wy_plot, s=msize, color="steelblue",
                        alpha=0.55, linewidths=0, rasterized=True)
        ax_main.set_xlabel(fr"|$W_x$|  ({pair.region_i})", fontsize=10)
        ax_main.set_ylabel(fr"|$W_y$|  ({pair.region_j})", fontsize=10)
        for sp in ("top", "right"):
            ax_main.spines[sp].set_visible(False)

        # Annotation: Gini coefficients
        gini_txt = (
            f"Gini {pair.region_i}: {gini_i:.3f}\n"
            f"Gini {pair.region_j}: {gini_j:.3f}"
        )
        ax_main.text(
            0.97, 0.03, gini_txt,
            transform=ax_main.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        )

        # Marginal histograms
        ax_top.hist(wx_plot, bins=25, color="C3", alpha=0.7, density=True)
        ax_top.set_ylabel("Density", fontsize=7)
        ax_top.tick_params(labelbottom=False, labelsize=7)
        for sp in ("top", "right"):
            ax_top.spines[sp].set_visible(False)

        ax_right.hist(wy_plot, bins=25, color="C0", alpha=0.7, density=True,
                      orientation="horizontal")
        ax_right.set_xlabel("Density", fontsize=7)
        ax_right.tick_params(labelleft=False, labelsize=7)
        for sp in ("top", "right"):
            ax_right.spines[sp].set_visible(False)

        fig.suptitle(
            f"{self.session_name}   |   {pair.pair_label}   "
            "— Dominant-dim weight magnitudes",
            fontsize=11, fontweight="bold",
        )

        if save:
            out = self.output_dir / (
                f"4_weight_scatter_{self.session_name}_{pair.region_i}_{pair.region_j}.png"
            )
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Session-wide figure 5 — dominant rho heatmap
    # ------------------------------------------------------------------

    def create_session_rho_heatmap(
        self,
        component_idx: int = 0,
        region_order: Optional[List[str]] = None,
        selected_pairs: Optional[List[Tuple[str, str]]] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Upper-triangular colour matrix of dominant_rho (ρ of the first
        significant pCCA dimension) across all region pairs.

        Replaces the tkCCA correlogram heatmap as the canonical session-level
        summary panel.
        """
        if region_order is None:
            region_order = ANATOMICAL_ORDER

        pair_wl = (
            set(map(tuple, selected_pairs)) if selected_pairs is not None else None
        )
        rows = []
        for (r_i, r_j), idx in self.pair_index.items():
            if pair_wl is not None and (r_i, r_j) not in pair_wl:
                continue
            pr  = self.pair_results[idx]
            rho = pr.get("dominant_rho")
            if rho is None:
                continue
            try:
                rho_val = float(np.asarray(rho).ravel()[0])
            except Exception:
                continue
            mi_raw = pr.get("mutual_information")
            mi_val = float(np.asarray(mi_raw).ravel()[0]) if mi_raw is not None else np.nan
            rows.append((r_i, r_j, rho_val, mi_val))

        if not rows:
            print(f"[{self.session_name}] no pCCA dominant_rho to plot")
            return None

        # Build upper-triangular matrix in anatomical order
        present = [r for r in region_order
                   if any(r in (ri, rj) for ri, rj, *_ in rows)]
        R    = len(present)
        ridx = {r: i for i, r in enumerate(present)}
        mat  = np.full((R, R), np.nan)
        mi_mat = np.full((R, R), np.nan)

        for r_i, r_j, rho_val, mi_val in rows:
            if r_i in ridx and r_j in ridx:
                i, j = ridx[r_i], ridx[r_j]
                ii, jj = (i, j) if i < j else (j, i)
                mat[ii, jj]    = rho_val
                mi_mat[ii, jj] = mi_val

        mask = np.tril(np.ones((R, R), dtype=bool))
        display = np.where(mask, np.nan, mat)

        fig, ax = plt.subplots(
            figsize=(max(6, R * 0.9), max(5, R * 0.85)),
            constrained_layout=True,
        )
        im = ax.imshow(
            display, aspect="equal", cmap="YlOrRd",
            vmin=0.0, vmax=float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0,
            interpolation="nearest",
        )
        for i in range(R):
            for j in range(i + 1, R):
                v = mat[i, j]
                if np.isfinite(v):
                    txt_col = "white" if v > 0.5 else "black"
                    mi_str  = f"\nMI={mi_mat[i,j]:.2f}" if np.isfinite(mi_mat[i, j]) else ""
                    ax.text(j, i, f"{v:.2f}{mi_str}",
                            ha="center", va="center", fontsize=6.5, color=txt_col)

        ax.set_xticks(range(R)); ax.set_xticklabels(present, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(R)); ax.set_yticklabels(present, fontsize=9)
        ax.set_title(
            fr"{self.session_name}   —   pCCA dominant $\rho$ (dim 1)",
            fontsize=11,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cb.set_label(r"Dominant $\rho$", fontsize=9)

        if save:
            out = self.output_dir / f"{self.session_name}_rho_heatmap.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Session-wide figure 6 — MI bar chart
    # ------------------------------------------------------------------

    def create_session_mi_bar(
        self,
        region_order: Optional[List[str]] = None,
        selected_pairs: Optional[List[Tuple[str, str]]] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Horizontal bar of mutual information per region pair, sorted
        descending.  MI = -sum_k log(1-rho_k^2) integrates over all pCCA
        dimensions into a single scalar.
        """
        if region_order is None:
            region_order = ANATOMICAL_ORDER

        pair_wl = (
            set(map(tuple, selected_pairs)) if selected_pairs is not None else None
        )
        records = []
        for (r_i, r_j), idx in self.pair_index.items():
            if pair_wl is not None and (r_i, r_j) not in pair_wl:
                continue
            pr  = self.pair_results[idx]
            mi_raw = pr.get("mutual_information")
            if mi_raw is None:
                continue
            try:
                mi_val = float(np.asarray(mi_raw).ravel()[0])
            except Exception:
                continue
            if not np.isfinite(mi_val):
                continue
            records.append((f"{r_i}↔{r_j}", mi_val))

        if not records:
            print(f"[{self.session_name}] no MI values to plot")
            return None

        records.sort(key=lambda x: x[1], reverse=True)
        labels, vals = zip(*records)

        fig, ax = plt.subplots(
            figsize=(7, max(3.0, 0.4 * len(records))),
            constrained_layout=True,
        )
        ypos = np.arange(len(records))
        ax.barh(ypos, vals, height=0.6, color="steelblue", alpha=0.85)
        ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(r"Mutual information  $-\sum_k\log(1-\rho_k^2)$", fontsize=10)
        ax.set_title(
            f"{self.session_name}  —  pCCA mutual information per pair",
            fontsize=11,
        )
        ax.invert_yaxis()
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        if save:
            out = self.output_dir / f"{self.session_name}_mi_bar.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Session-wide figure 7 — Gini coefficient panel
    # ------------------------------------------------------------------

    def create_session_gini_panel(
        self,
        region_order: Optional[List[str]] = None,
        selected_pairs: Optional[List[Tuple[str, str]]] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Box-strip plot of |W| Gini coefficients per region across all pairs.

        Analogous to Gonzalez et al. (2026) Fig. 1f: monotonically
        decreasing Gini from upstream to downstream regions would indicate
        progressive expansion of the subspace (more neurons recruited at
        successive processing stages).
        """
        if region_order is None:
            region_order = ANATOMICAL_ORDER

        pair_wl = (
            set(map(tuple, selected_pairs)) if selected_pairs is not None else None
        )
        gini_by_region: Dict[str, List[float]] = {}

        for (r_i, r_j), idx in self.pair_index.items():
            if pair_wl is not None and (r_i, r_j) not in pair_wl:
                continue
            pr = self.pair_results[idx]
            for field_name, rname in [("gini_weights_i", r_i),
                                      ("gini_weights_j", r_j)]:
                v = pr.get(field_name)
                if v is None:
                    continue
                try:
                    g = float(np.asarray(v).ravel()[0])
                except Exception:
                    continue
                if np.isfinite(g):
                    gini_by_region.setdefault(rname, []).append(g)

        present = [r for r in region_order if r in gini_by_region]
        if not present:
            print(f"[{self.session_name}] no Gini values to plot")
            return None

        fig, ax = plt.subplots(
            figsize=(max(5.0, 0.9 * len(present)), 4.5),
            constrained_layout=True,
        )
        rng = np.random.default_rng(seed=42)
        for xi, rname in enumerate(present):
            vals = np.array(gini_by_region[rname])
            jitter = rng.uniform(-0.12, 0.12, size=vals.size)
            ax.scatter(xi + jitter, vals, s=28, alpha=0.65, color="C2",
                       zorder=3, linewidths=0)
            bp = ax.boxplot(
                vals, positions=[xi], widths=0.4,
                patch_artist=True, notch=False,
                boxprops=dict(facecolor="C2", alpha=0.35),
                medianprops=dict(color="black", lw=2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                flierprops=dict(marker=""),
                zorder=2,
            )

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Gini coefficient of |W| (dominant dim)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"{self.session_name}  —  Subspace weight sparsity (Gini)\n"
            "Higher Gini = fewer neurons drive the subspace",
            fontsize=10,
        )
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        if save:
            out = self.output_dir / f"{self.session_name}_gini_panel.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Session-wide figure 8 — Subspace angles (pivot-region pairs)
    # ------------------------------------------------------------------

    def create_session_subspace_angles(
        self,
        pivot_regions: Optional[List[str]] = None,
        region_order: Optional[List[str]] = None,
        selected_pairs: Optional[List[Tuple[str, str]]] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Pairwise maximum principal angle between pCCA subspaces that share
        a common 'pivot' region.

        For each pivot region, we collect all pairs (pivot, X) where X
        varies.  The subspace spanned by the significant columns of
        mean_A_matrix (the pivot's canonical weights) for each (pivot, X)
        pair is compared pairwise: angle = max principal angle between the
        two weight-matrix column spaces.

        Large angles (→ 90°) indicate that the same pivot neurons are
        recombined into nearly orthogonal patterns when communicating with
        different partners — the defining signature of subspace rotation
        described by Gonzalez et al. (2026) Fig. 1g.

        Parameters
        ----------
        pivot_regions : regions to treat as the shared pivot.  If None,
            all regions present in ≥ 2 pairs are treated as pivot.
        """
        if region_order is None:
            region_order = ANATOMICAL_ORDER
        if pivot_regions is None:
            from collections import Counter
            region_counts: Counter = Counter()
            for (r_i, r_j) in self.pair_index:
                region_counts[r_i] += 1
                region_counts[r_j] += 1
            pivot_regions = [r for r, cnt in region_counts.items() if cnt >= 2]

        pair_wl = (
            set(map(tuple, selected_pairs)) if selected_pairs is not None else None
        )

        # For each pivot, collect {partner: weight_matrix_for_pivot}
        figs_out = []
        for pivot in pivot_regions:
            partner_weights: Dict[str, np.ndarray] = {}
            for (r_i, r_j), idx in self.pair_index.items():
                if pair_wl is not None and (r_i, r_j) not in pair_wl:
                    continue
                if r_i != pivot and r_j != pivot:
                    continue
                pr  = self.pair_results[idx]
                sig_raw = pr.get("significant_components")
                if sig_raw is None:
                    continue
                sig_comps = [int(s) for s in np.asarray(sig_raw).ravel()]
                if not sig_comps:
                    continue
                # Extract the weight matrix corresponding to the pivot region
                if r_i == pivot:
                    W_raw = safe_array(pr.get("mean_A_matrix"))
                    partner = r_j
                else:
                    W_raw = safe_array(pr.get("mean_B_matrix"))
                    partner = r_i
                if W_raw is None:
                    continue
                if W_raw.ndim == 1:
                    W_raw = W_raw[:, np.newaxis]
                # Keep only significant columns (convert to 0-indexed)
                sig_idx = [s - 1 for s in sig_comps if 0 < s <= W_raw.shape[1]]
                if not sig_idx:
                    continue
                partner_weights[partner] = W_raw[:, sig_idx]

            if len(partner_weights) < 2:
                continue

            partners = sorted(partner_weights.keys(),
                              key=lambda r: region_order.index(r)
                              if r in region_order else 999)
            P = len(partners)
            angle_mat = np.full((P, P), np.nan)

            for pi, pa in enumerate(partners):
                for pj, pb in enumerate(partners):
                    if pa == pb:
                        angle_mat[pi, pj] = 0.0
                        continue
                    Wa = partner_weights[pa]
                    Wb = partner_weights[pb]
                    # Orthonormalise each subspace
                    Qa, _ = np.linalg.qr(Wa, mode="reduced")
                    Qb, _ = np.linalg.qr(Wb, mode="reduced")
                    try:
                        # ── first component only ─────────────────────────────
                        angles_rad = subspace_angles(Qa[:, 0:1], Qb[:, 0:1])
                        angle_mat[pi, pj] = float(np.degrees(angles_rad[0]))
                    except Exception:
                        pass

            fig, ax = plt.subplots(
                figsize=(max(4.0, 0.8 * P), max(3.5, 0.7 * P)),
                constrained_layout=True,
            )
            im = ax.imshow(
                angle_mat, aspect="equal", cmap="viridis",
                vmin=0, vmax=90, interpolation="nearest",
            )
            ax.set_xticks(range(P)); ax.set_xticklabels(partners, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(P)); ax.set_yticklabels(partners, fontsize=9)
            for pi in range(P):
                for pj in range(P):
                    v = angle_mat[pi, pj]
                    if np.isfinite(v):
                        tc = "white" if v < 50 else "black"
                        ax.text(pj, pi, f"{v:.0f}°", ha="center", va="center",
                                fontsize=8, color=tc)
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
            cb.set_label("Max principal angle (°)", fontsize=9)
            ax.set_title(
                f"{self.session_name}  —  Subspace rotation at pivot: {pivot}\n"
                "90° = orthogonal (maximum recombination)",
                fontsize=10,
            )
            figs_out.append(fig)

            if save:
                out = self.output_dir / (
                    f"{self.session_name}_subspace_angles_pivot_{pivot}.png"
                )
                fig.savefig(out, dpi=300, bbox_inches="tight")
                print(f"  saved: {out}")

        return figs_out[0] if figs_out else None

    # ------------------------------------------------------------------
    # Session-wide figure 9 — Bipartite pCCA weight graph (pivot region)
    # ------------------------------------------------------------------

    def create_session_subspace_angles2(
            self,
            pivot_region: str,
            left_partner: str,
            right_partner: str,
            n_dims: int = 2,
            n_neurons_show: int = 100,
            save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Bipartite connection diagram replicating the schematic in Fig. 1 of
        Gonzalez et al. (2026).

        Layout
        ------
        *Centre column*  : individual neurons of ``pivot_region`` (black dots),
            ordered top-to-bottom by descending max |weight| so the most
            informative neurons occupy the visible top.
        *Left nodes*     : blue circles, one per pCCA dimension for
            ``left_partner``.
        *Right nodes*    : rose circles, one per pCCA dimension for
            ``right_partner``.
        *Edges*          : line width and opacity both scale linearly with the
            absolute canonical weight |w_{n,d}|, normalised by the global
            maximum across all displayed edges.  Edges are rasterised for
            speed when n_neurons_show is large.

        The figure makes immediately apparent whether a neuron participates in
        *both* subspaces (thick edges on both sides) — the defining hallmark of
        the "weight alignment" phenomenon documented in Gonzalez et al. (2026)
        Fig. 1d–e — or specialises exclusively for one partner.

        Pair-index orientation
        ----------------------
        MATLAB may store a pair in either order, (A, B) or (B, A).  For each
        requested pair the method first attempts the exact key
        ``(pivot_region, partner)`` in ``pair_index``; if that key is absent it
        retries with the reversed key ``(partner, pivot_region)`` and swaps the
        weight-matrix field accordingly (``mean_A_matrix`` ↔ ``mean_B_matrix``).
        ``pair_index`` itself is never modified.

        Parameters
        ----------
        pivot_region : str
            Region whose canonical weight columns are displayed.  In the
            Oxford dataset this is typically ``"VPMPO"``, ``"MOp"``, etc.
        left_partner : str
            Partner rendered on the left (blue dimension nodes).
        right_partner : str
            Partner rendered on the right (rose dimension nodes).
        n_dims : int
            Number of canonical dimensions shown per partner side.  Columns of
            ``mean_A_matrix`` / ``mean_B_matrix`` are taken in order
            (dim 1 … dim n_dims).
        n_neurons_show : int
            Maximum number of neurons rendered.  The top-``n_neurons_show``
            neurons by max |weight| across all displayed edges are retained;
            the remainder are silently dropped.
        save : bool
            If True the figure is written to
            ``{session_name}_subspace_angles2_pivot_{pivot}_{L}_vs_{R}.png``.
        """

        # ── 1.  Targeted pair-lookup with orientation fallback ────────────
        #
        # For each (pivot, partner) request we try both storage orderings.
        # The w_field tuple encodes which weight matrix belongs to the pivot:
        #   (pivot, partner) stored as-is  → pivot weights are in mean_A_matrix
        #   (pivot, partner) stored reversed → pivot weights are in mean_B_matrix

        def _lookup_pair(
                partner: str,
        ) -> Optional[Tuple[np.ndarray, float]]:
            """
            Return (W_pivot, dominant_rho) for the pair (pivot_region, partner).

            Tries ``(pivot_region, partner)`` first; falls back to
            ``(partner, pivot_region)``.  Returns None if neither orientation
            is present in pair_index.
            """
            candidates = [
                ((pivot_region, partner), "mean_A_matrix"),
                ((partner, pivot_region), "mean_B_matrix"),
            ]
            for key, w_field in candidates:
                if key not in self.pair_index:
                    continue
                pr = self.pair_results[self.pair_index[key]]
                W_raw = safe_array(pr.get(w_field))
                if W_raw is None:
                    print(
                        f"  [{self.session_name}] pair {key}: "
                        f"'{w_field}' is missing or None — skipping."
                    )
                    return None
                if W_raw.ndim == 1:
                    W_raw = W_raw[:, np.newaxis]
                rho_raw = pr.get("dominant_rho")
                rho_val = (
                    float(np.asarray(rho_raw).ravel()[0])
                    if rho_raw is not None
                    else 0.0
                )
                return W_raw, rho_val

            # Neither orientation found
            print(
                f"  [{self.session_name}] pair ({pivot_region}, {partner}) not found "
                f"in pair_index in either orientation — check region names."
            )
            return None

        result_left = _lookup_pair(left_partner)
        result_right = _lookup_pair(right_partner)

        if result_left is None or result_right is None:
            return None

        W_left, rho_left = result_left
        W_right, rho_right = result_right

        # Consistency check: both weight matrices must have the same number of
        # rows (pivot neuron count) — a mismatch signals a session-level
        # neuron-selection inconsistency that should be diagnosed upstream.
        if W_left.shape[0] != W_right.shape[0]:
            print(
                f"  [{self.session_name}] pivot '{pivot_region}': neuron count "
                f"mismatch between pairs — left {W_left.shape[0]} vs "
                f"right {W_right.shape[0]}.  Truncating to the smaller."
            )
            n_common = min(W_left.shape[0], W_right.shape[0])
            W_left = W_left[:n_common, :]
            W_right = W_right[:n_common, :]

        # ── 2.  Clip weight matrices to the requested dimensionality ──────
        n_left_dims = min(n_dims, W_left.shape[1])
        n_right_dims = min(n_dims, W_right.shape[1])
        W_left = W_left[:, :n_left_dims]
        W_right = W_right[:, :n_right_dims]

        n_neurons_total = W_left.shape[0]

        # ── 3.  Neuron selection by descending max |weight| ───────────────
        #
        # Rank neurons by the largest absolute weight they carry across *all*
        # displayed dimensions and both partners, so the selected subset is
        # maximally informative about the communication subspaces.
        combined_abs = np.concatenate(
            [np.abs(W_left), np.abs(W_right)], axis=1
        )  # (n_neurons, n_L + n_R)
        max_per_neuron = combined_abs.max(axis=1)  # (n_neurons,)
        order = np.argsort(max_per_neuron)[::-1]
        sel = np.sort(order[: min(n_neurons_show, n_neurons_total)])
        n_sel = int(sel.size)

        W_left_s = W_left[sel, :]  # (n_sel, n_left_dims)
        W_right_s = W_right[sel, :]  # (n_sel, n_right_dims)

        # Global normalisation — shared across both sides so edge widths are
        # directly comparable between the left and right subspaces.
        w_all = np.concatenate(
            [np.abs(W_left_s).ravel(), np.abs(W_right_s).ravel()]
        )
        w_max = max(float(w_all.max()), 1e-9)

        # ── 4.  Figure canvas ─────────────────────────────────────────────
        fig_h = max(7.0, n_sel * 0.18)
        fig, ax = plt.subplots(figsize=(9.5, fig_h), constrained_layout=True)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.08, 1.10)
        ax.axis("off")

        # ── 5.  Coordinate layout ─────────────────────────────────────────
        y_neurons = np.linspace(0.93, 0.05, n_sel)
        x_neuron = 0.50
        x_left_node = 0.12
        x_right_node = 0.88

        def _dim_ycoords(n: int) -> np.ndarray:
            return np.array([0.5]) if n == 1 else np.linspace(0.74, 0.26, n)

        y_left_nodes = _dim_ycoords(n_left_dims)
        y_right_nodes = _dim_ycoords(n_right_dims)

        _BLUE = "#4A6FD4"
        _ROSE = "#C96070"
        LW_MAX, ALPHA_LO, ALPHA_HI = 8, 0.03, 0.88

        # ── 6.  Draw edges ────────────────────────────────────────────────
        def _draw_edges(
                weight_mat: np.ndarray,
                x_node: float,
                y_nodes: np.ndarray,
                colour: str,
        ) -> None:
            for di in range(weight_mat.shape[1]):
                for ni in range(n_sel):
                    t = abs(float(weight_mat[ni, di])) / w_max
                    lw = LW_MAX * t
                    alp = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * t
                    ax.plot(
                        [x_neuron, x_node],
                        [y_neurons[ni], y_nodes[di]],
                        color=colour, lw=lw, alpha=alp,
                        solid_capstyle="round",
                        rasterized=True, zorder=5,
                    )

        _draw_edges(W_left_s, x_left_node, y_left_nodes, _BLUE)
        _draw_edges(W_right_s, x_right_node, y_right_nodes, _ROSE)

        # ── 7.  Neuron dots ───────────────────────────────────────────────
        dot_sizes = 14 + 28 * (max_per_neuron[sel] / w_max)
        ax.scatter(
            np.full(n_sel, x_neuron), y_neurons,
            s=dot_sizes, color="black", zorder=10, linewidths=0,
        )
        ax.annotate(
            "",
            xy=(x_neuron, 0.975), xytext=(x_neuron, 1.03),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=3),
            annotation_clip=False,
        )
        ax.text(
            x_neuron, 1.05,
            f"{pivot_region} neurons\n",
            ha="center", va="bottom", fontsize=15, fontweight="bold",
            clip_on=False,
        ) #fr"($n = {n_sel}$ / $N = {n_neurons_total}$ shown)",

        # ── 8.  Dimension nodes + partner labels ──────────────────────────
        NODE_R = 0.036

        def _draw_dim_nodes(
                n_d: int,
                y_nodes: np.ndarray,
                x_node: float,
                colour: str,
                label_side: str,
        ) -> None:
            for di in range(n_d):
                ax.add_patch(
                    plt.Circle(
                        (x_node, y_nodes[di]), NODE_R,
                        color=colour, zorder=6, transform=ax.transData,
                    )
                )
                offset = -0.065 if label_side == "left" else 0.065
                ax.text(
                    x_node + offset, y_nodes[di], f"Dim {di + 1}",
                    ha="right" if label_side == "left" else "left",
                    va="center", fontsize=15, color=colour, fontweight="bold",
                )

        _draw_dim_nodes(n_left_dims, y_left_nodes, x_left_node, _BLUE, "left")
        _draw_dim_nodes(n_right_dims, y_right_nodes, x_right_node, _ROSE, "right")

        y_lbl = 0.07
        ax.text(
            x_left_node, float(y_left_nodes.min()) - y_lbl,
                         f"{left_partner}\n" + r"$\rho_1 = $" + f"{rho_left:.3f}",
            ha="center", va="top", fontsize=15, color=_BLUE,
        )
        ax.text(
            x_right_node, float(y_right_nodes.min()) - y_lbl,
                          f"{right_partner}\n" + r"$\rho_1 = $" + f"{rho_right:.3f}",
            ha="center", va="top", fontsize=15, color=_ROSE,
        )

        fig.suptitle(
            f"{self.session_name}\n"
            fr"$\leftarrow$ {left_partner}   {right_partner} $\rightarrow$",
            fontsize=15, fontweight="bold",
        )

        if save:
            out = self.output_dir / (
                f"{self.session_name}_subspace_angles2"
                f"_pivot_{pivot_region}"
                f"_{left_partner}_vs_{right_partner}.png"
            )
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")

        return fig


    # ------------------------------------------------------------------
    # Master dispatcher — per-pair figures
    # ------------------------------------------------------------------

    def create_pcca_pair_figures(
        self,
        region_i: str,
        region_j: str,
        component_idx: int = 0,
        n_neurons_show: int = 60,
        save: bool = True,
    ) -> None:
        """Build and save all four per-pair figures."""
        pair = self._build_session_pair(region_i, region_j)
        if pair is None or pair.n_sig <= component_idx:
            print(f"  skipping ({region_i}, {region_j}): "
                  f"n_sig={getattr(pair, 'n_sig', 0)}, need >{component_idx}")
            return

        pr_raw = self.pair_results[self.pair_index[(region_i, region_j)]]

        for label, fn in [
            ("rastermap",           lambda: self._figure_rastermap(pair, component_idx, n_neurons_show, save)),
            ("projections",         lambda: self._figure_projections(pr_raw, pair, save)),
            ("canonical spectrum",  lambda: self._figure_canonical_spectrum(pair, save)),
            ("weight scatter",      lambda: self._figure_weight_scatter(pair, save)),
        ]:
            print(f"    [{region_i} ↔ {region_j}]  {label}…")
            fig = fn()
            if fig is not None:
                plt.close(fig)

    # ------------------------------------------------------------------
    # Full runner
    # ------------------------------------------------------------------

    def run_all_pairs(
        self,
        component_idx: int = 0,
        n_neurons_show: int = 60,
        selected_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Per-pair figures for each pair, then all session-wide figures."""
        raw_pairs = (
            selected_pairs if selected_pairs is not None
            else list(self.pair_index.keys())
        )

        pairs_to_run = []
        for (r_i, r_j) in raw_pairs:
            if (r_i, r_j) in self.pair_index:
                pairs_to_run.append((r_i, r_j))
            elif (r_j, r_i) in self.pair_index:
                pairs_to_run.append((r_j, r_i))  # canonical order restored
            else:
                print(f"  [skip] ({r_i}, {r_j}) not found in either orientation")

        for (r_i, r_j) in pairs_to_run:
            self.create_pcca_pair_figures(
                r_i, r_j,
                component_idx=component_idx,
                n_neurons_show=n_neurons_show,
            )

        for label, fn in [
            ("rho heatmap",       lambda: self.create_session_rho_heatmap(selected_pairs=pairs_to_run)),
            ("MI bar",            lambda: self.create_session_mi_bar(selected_pairs=pairs_to_run)),
            ("Gini panel",        lambda: self.create_session_gini_panel(selected_pairs=pairs_to_run)),
            ("subspace angles",   lambda: self.create_session_subspace_angles(selected_pairs=pairs_to_run)), # ← add this line
        ]:
            print(f"  [session] {label}…")
            fig = fn()
            if fig is not None:
                plt.close(fig)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_str(v) -> Optional[str]:
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


# =============================================================================
# Driver
# =============================================================================

def main() -> None:
    # SELECTED_PAIRS: List[Tuple[str, str]] = [
    #     ("ORB", "MOs"),
    #     ("ORB", "OLF"),
    #     ("mPFC", "MOp"),
    #     ("mPFC", "MOs"),
    #     ("MOp", "STR"),
    #     ("MOs", "STR"),
    #     ("MOp", "MOs"),
    #     ("MOp", "VPMPO"),
    #     ("MOs", "VPMPO"),
    # ]

    # SELECTED_PAIRS: List[Tuple[str, str]] = [
    # #     ( "ORB","MOs"),
    # #     ("ORB", "MOp"),
    # #     ("ORB", "VPMPO"),
    # # ]
    #
    # #     ("MOs", "ORB"),
    #
    #     # ("MOs", "ORB"),
    #     # ("MOs", "VPMPO"),
    #
    #     # ("VPMPO", "ORB"),
    #     # ("MOs", "ORB"),
    #
    #     # ("VPMPO", "ORB"),
    #     # ("MOs", "VPMPO"),
    #
    #     # ("VPMPO", "MOp"),
    #     # ("MOs", "MOp"),
    # ]

    base_dir     = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    session_name = "yp021_220407" #"yp020_220331"
    session_file = (
        base_dir / "pcca_sessions_cued_hit_long_results"
        / f"{session_name}_analysis_results.mat"
    )
    out_dir = base_dir / "Paper_output" / "figures_pcca_session" / session_name

    viz = OxfordPCCASessionVisualizer(
        session_results_path=str(session_file),
        session_name=session_name,
        output_dir=str(out_dir),
    )

    if not viz.load_session():
        return

    viz.compute_global_rastermap()

    # viz.run_all_pairs(
    #     component_idx=0,
    #     n_neurons_show=60,
    #     selected_pairs=SELECTED_PAIRS,
    # )
    viz.run_all_pairs(
        component_idx=0,
        n_neurons_show=50
    )

    viz.create_session_subspace_angles2(
            pivot_region="MOp",
            left_partner="MOs",
            right_partner="VPMPO",
            n_dims=2,
            n_neurons_show=50,
        )

if __name__ == "__main__":
    main()
"""
single_session_tkcca_visualizer.py
==================================
Single-session visualisation for temporal-kernel CCA outputs.

PCA panels are absent.  For each region pair the visualiser produces four
independent figures (component 1 by default):

  Figure 1  ``*_1_lag_latents.png``
      2 × K grid.  Row 1: per-lag region-i latents
      $z_i^{(\\tau_k)}(t) = w_x(\\tau_k)^\\top X_i^z(t-\\tau_k)$.
      Row 2: stationary region-j latent $z_j(t) = w_y^\\top X_j^z(t)$, repeated
      for visual column pairing.  Each panel renders individual trial traces as
      thin translucent lines, with the trial mean and ±1 SD band overlaid.

  Figure 2  ``*_2_rastermap.png``
      Two rows (region i, region j).  Each row: Rastermap-ordered,
      trial-averaged PSTH (RdBu_r, real time axis, red dashed Bar-Off line)
      beside a horizontal CCA weight bar (barh):
          region i  ->  w_x(tau*, c=1)   dominant-lag spatial filter
          region j  ->  w_y(c=1)         stationary spatial filter
      Global Rastermap ordering shared across all pairs for comparability.

  Figure 3  ``*_3_dominant_lag.png``
      Two stacked panels, one per region.  Same trial+mean+/-SD style, showing
      the dominant-lag component-1 projection for region i alongside the
      stationary projection for region j.

  Figure 4  ``*_4_projections.png``
      2 x n_sig grid of the pre-computed projections stored in
      ``pair_results.projections.components`` (the full lag-summed latents
      produced by ``calculate_tkcca_projections`` in MATLAB).  Parsed
      following the convention in ``Useful_definition._extract_single_trial``:
          components[k]      -> list of one dict  (mat73 MATLAB cell)
          components[k][0]   -> dict with keys region_i_trials, region_j_trials,
                               component_number, R2

Additionally, a session-wide figure is produced by
``create_session_correlogram_heatmap``:

  Heatmap  ``*_correlogram_heatmap.png``
      Rows = region pairs (ANATOMICAL_ORDER).  Cols = lag tau_k (s).
      Colour = rho(tau_k, c=1).

Mathematical sign convention (from perform_session_tkcca.m header):
    tau > 0  =>  X_i(t - tau) leads X_j(t)   (region_i drives region_j)
    tau < 0  =>  region_j drives region_i
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import zscore

try:
    import mat73
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mat73 is required: pip install mat73") from exc

try:
    from rastermap import Rastermap
    RASTERMAP_AVAILABLE = True
except ImportError:
    RASTERMAP_AVAILABLE = False
    warnings.warn("rastermap not available; falling back to identity neuron ordering.")

# Re-use helpers established in the framework
from Useful_definition import ANATOMICAL_ORDER, safe_array, first_present


# =============================================================================
# 1. Lightweight pair container
# =============================================================================

@dataclass
class _SessionTkPair:
    """
    Parsed view of one ``pair_results`` entry for intra-session use.

    Shape annotations follow ``tkcca_analysis.TkCCAPair`` conventions;
    column index 0 corresponds to component 1 after the MATLAB-side
    significance-based slicing (wx_temporal[:, :, sig], etc.).
    """
    region_i: str
    region_j: str

    X_i: np.ndarray                    # (n_trials, n_i, T) -- selected neurons
    X_j: np.ndarray                    # (n_trials, n_j, T)

    wx_temporal: np.ndarray            # (n_i, K, n_sig)
    wy_stationary: np.ndarray          # (n_j, n_sig)

    lag_bins: np.ndarray               # (K,) integer tau_k
    lag_secs: np.ndarray               # (K,) tau_k * dt  in seconds

    correlogram: np.ndarray            # (K, n_sig)  rho(tau_k, c)
    mean_cv_rho: np.ndarray            # (n_components,)

    @property
    def n_sig(self) -> int:
        return self.correlogram.shape[1]

    def dominant_lag_idx(self, c: int = 0) -> int:
        """k* = arg max_k |rho(tau_k, c)|."""
        return int(np.argmax(np.abs(self.correlogram[:, c])))

    def dominant_lag_seconds(self, c: int = 0) -> float:
        return float(self.lag_secs[self.dominant_lag_idx(c)])

    def dominant_lag_bins(self, c: int = 0) -> int:
        return int(self.lag_bins[self.dominant_lag_idx(c)])


# =============================================================================
# 2. Session visualiser
# =============================================================================

class OxfordTkCCASessionVisualizer:
    """
    One ``*_analysis_results.mat`` file in; four per-pair figures + one
    session-wide correlogram heatmap out.

    Typical call sequence::

        viz = OxfordTkCCASessionVisualizer(
            session_results_path="…/yp010_220209_analysis_results.mat",
            session_name="yp010_220209",
            output_dir="…/figures_tkcca/yp010_220209",
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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.time_range_s = time_range_s

        # Filled by load_session()
        self.region_data: Optional[dict] = None
        self.cca_results: Optional[dict] = None
        self.pair_results: List[dict] = []
        self.pair_index: Dict[Tuple[str, str], int] = {}
        self.valid_regions: List[str] = []
        self.T: Optional[int] = None
        self.time_vec: Optional[np.ndarray] = None

        # Filled by compute_global_rastermap()
        # global_sort[region] -> local neuron indices in Rastermap-embedding order
        self.global_sort: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_session(self) -> bool:
        """
        Read one ``*_analysis_results.mat`` and index ``pair_results`` by
        (r_i, r_j), following the navigation path used in
        ``tkcca_analysis._extract_tkcca_fields``.
        """
        try:
            data = mat73.loadmat(str(self.session_path))
        except Exception as exc:
            print(f"[{self.session_name}] load failed: {exc}")
            return False

        self.region_data = data.get('region_data', {})
        self.cca_results = data.get('cca_results', {})

        if not isinstance(self.region_data, dict) or 'regions' not in self.region_data:
            print(f"[{self.session_name}] no region_data.regions field")
            return False

        pair_results = self.cca_results.get('pair_results', []) \
            if isinstance(self.cca_results, dict) else []
        if isinstance(pair_results, np.ndarray):
            pair_results = pair_results.tolist()
        if not isinstance(pair_results, (list, tuple)):
            pair_results = []
        self.pair_results = list(pair_results)

        for idx, pr in enumerate(self.pair_results):
            if not isinstance(pr, dict):
                continue
            r_i = self._as_str(pr.get('region_i'))
            r_j = self._as_str(pr.get('region_j'))
            if r_i and r_j:
                self.pair_index[(r_i, r_j)] = idx

        regions_dict = self.region_data['regions']
        self.valid_regions = [
            r for r, info in regions_dict.items()
            if isinstance(info, dict) and 'spike_data' in info
        ]

        if self.valid_regions:
            sd0 = safe_array(regions_dict[self.valid_regions[0]]['spike_data'])
            if sd0 is not None and sd0.ndim == 3:
                self.T = int(sd0.shape[2])
                self.time_vec = np.linspace(
                    self.time_range_s[0], self.time_range_s[1], self.T
                )

        print(f"[{self.session_name}] loaded: {len(self.valid_regions)} regions, "
              f"{len(self.pair_index)} pairs, T={self.T}")
        return True

    # ------------------------------------------------------------------
    # Global rastermap
    # ------------------------------------------------------------------

    def compute_global_rastermap(
        self,
        n_pcs: int = 200,
        locality: float = 0.0,
        grid_upsample: int = 5,
    ) -> None:
        """
        Pool every region's (n_trials, n_neurons, T) spike data into a single
        (N_total, T*n_trials) matrix, fit Rastermap once, and store per-region
        local ordering vectors.

        Using one global ordering (rather than per-region embeddings) ensures
        that the neuron row positions in Figure 2 are comparable across the
        two region panels, exactly as in the existing pipeline.
        """
        if self.region_data is None:
            raise RuntimeError("Call load_session() first.")
        regions_dict = self.region_data['regions']

        pooled, offsets = [], {}
        cursor = 0
        for rname in self.valid_regions:
            sd = safe_array(regions_dict[rname].get('spike_data'))
            if sd is None or sd.ndim != 3:
                continue
            # Mirror the MATLAB 1-indexed neuron selection used by tkCCA
            sel = safe_array(regions_dict[rname].get('selected_neurons'))
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

        pooled = np.vstack(pooled)
        pooled_z = zscore(pooled, axis=1, nan_policy='omit')
        pooled_z = np.nan_to_num(pooled_z, nan=0.0)

        if not RASTERMAP_AVAILABLE:
            for rname, (lo, hi) in offsets.items():
                self.global_sort[rname] = np.arange(hi - lo)
            return

        model = Rastermap(
            n_PCs=min(n_pcs, pooled.shape[0]),
            locality=locality,
            grid_upsample=grid_upsample,
        )
        model.fit(pooled_z)
        g = model.isort

        for rname, (lo, hi) in offsets.items():
            keep = (g >= lo) & (g < hi)
            self.global_sort[rname] = g[keep] - lo

        print(f"[{self.session_name}] global Rastermap fitted "
              f"({pooled.shape[0]} neurons across {len(offsets)} regions)")

    # ------------------------------------------------------------------
    # Per-pair extraction
    # ------------------------------------------------------------------

    def _build_session_pair(
        self, region_i: str, region_j: str
    ) -> Optional[_SessionTkPair]:
        """Parse one ``pair_results`` entry into ``_SessionTkPair``."""
        key = (region_i, region_j)
        if key not in self.pair_index:
            print(f"  pair ({region_i}, {region_j}) not in pair_results")
            return None
        pr = self.pair_results[self.pair_index[key]]

        wx   = safe_array(pr.get('wx_temporal'))
        wy   = safe_array(pr.get('wy_stationary'))
        corr = safe_array(pr.get('canonical_correlogram'))
        lb   = safe_array(pr.get('tkcca_lags_bins'))
        ls   = safe_array(pr.get('tkcca_lags_seconds'))

        if any(x is None for x in (wx, wy, corr, lb, ls)):
            print(f"  ({region_i}, {region_j}) is not a tkCCA pair_result "
                  f"(missing canonical_correlogram or filter fields)")
            return None

        # Coerce to 3-D / 2-D when n_sig == 1
        if wx.ndim == 2:
            wx = wx[:, :, np.newaxis]
        if wy.ndim == 1:
            wy = wy[:, np.newaxis]
        if corr.ndim == 1:
            corr = corr[:, np.newaxis]

        regions_dict = self.region_data['regions']
        X_i = safe_array(regions_dict[region_i].get('spike_data'))
        X_j = safe_array(regions_dict[region_j].get('spike_data'))
        if X_i is None or X_j is None:
            return None

        sel_i = safe_array(pr.get('selected_neurons_i'))
        sel_j = safe_array(pr.get('selected_neurons_j'))
        if sel_i is not None:
            X_i = X_i[:, sel_i.ravel().astype(int) - 1, :]
        if sel_j is not None:
            X_j = X_j[:, sel_j.ravel().astype(int) - 1, :]

        mean_cv_rho = np.array([])
        cvr = pr.get('cv_results', {})
        if isinstance(cvr, dict):
            r = safe_array(first_present(cvr, ['mean_cv_R2', 'mean_cv_rho']))
            if r is not None:
                mean_cv_rho = r.ravel()

        return _SessionTkPair(
            region_i=region_i, region_j=region_j,
            X_i=X_i, X_j=X_j,
            wx_temporal=wx, wy_stationary=wy,
            lag_bins=lb.ravel().astype(int),
            lag_secs=ls.ravel(),
            correlogram=corr,
            mean_cv_rho=mean_cv_rho,
        )

    # ------------------------------------------------------------------
    # Projection primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore_per_neuron(X: np.ndarray) -> np.ndarray:
        """
        Per-neuron z-score across all (trial, time) samples.

        Mirrors ``zscore(Xi_flat, 0, 1)`` in ``calculate_tkcca_projections``:
        each neuron is independently standardised to zero mean and unit variance
        across the pooled (trial * time) dimension, so that neurons with
        different baseline firing rates contribute equally to the projection.

        Parameters
        ----------
        X : (n_trials, n, T)

        Returns
        -------
        Xz : (n_trials, n, T)
        """
        n_trials, n, T = X.shape
        flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)   # (n, T*n_trials)
        flat = zscore(flat, axis=1, nan_policy='omit')
        flat = np.nan_to_num(flat, nan=0.0)
        return flat.reshape(n, T, n_trials).transpose(2, 0, 1)

    @staticmethod
    def _per_lag_latent(
        X_z: np.ndarray, w_k: np.ndarray, tau_bins: int
    ) -> np.ndarray:
        """
        Per-lag contribution to the region-i latent:

            z_i^(tau_k)(r, t) = w_x(tau_k)^T X_i^z(r, :, t - tau_bins)

        Out-of-range positions (where t - tau_bins is outside [0, T)) are set
        to NaN so that boundary truncation is displayed honestly as gaps rather
        than being silently zero-padded (as the MATLAB convolution does).

        Parameters
        ----------
        X_z      : (n_trials, n, T)
        w_k      : (n,)
        tau_bins : int   positive => region i leads by tau_bins samples

        Returns
        -------
        z : (n_trials, T)  -- NaN at boundary positions
        """
        n_trials, n, T = X_z.shape
        z = np.full((n_trials, T), np.nan)
        src = np.arange(T) - tau_bins
        valid = np.where((src >= 0) & (src < T))[0]
        if valid.size:
            z[:, valid] = np.einsum('rnt,n->rt', X_z[:, :, src[valid]], w_k)
        return z

    @staticmethod
    def _stationary_latent(X_z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """z_j(r, t) = w_y^T X_j^z(r, :, t)  (no lag)."""
        return np.einsum('rnt,n->rt', X_z, w)

    # ------------------------------------------------------------------
    # Z2 spectral sign alignment for per-lag latents
    # ------------------------------------------------------------------

    @staticmethod
    def _align_lag_signs(
        z_lag_trials: np.ndarray,
        epoch: Tuple[int, int],
    ) -> np.ndarray:
        """
        Determine a per-lag sign convention for the region-i latents via
        Z2 spectral synchronisation, adapted from the cross-session procedure
        in ``_align_signs_spectral``.

        Conceptual mapping
        ------------------
        Cross-session original          This function
        ────────────────────────────    ────────────────────────────────────────
        sessions  (index i)             lags  k = 1 … K
        session trajectory  U_i(t)      trial-averaged latent  z̄_i^(τ_k)(t)
        session-level sign  s_i         lag-level sign  s_k

        Algorithm (mirrors the five-step original)
        ------------------------------------------
        Step 1  Build the (K × K) pairwise Pearson correlation matrix C of the
                K trial-averaged trajectories z̄^(τ_k).  NaN time-points
                (boundary truncation from ``_per_lag_latent``) are zeroed before
                the correlation is computed so as not to propagate missing data.

        Step 2  Z2 synchronisation: extract the leading eigenvector of C via
                ``np.linalg.eigh`` (eigenvalues returned in ascending order, so
                the leading eigenvector is the *last* column).  Take the
                element-wise sign to obtain the raw lag-sign proposal s_k.
                Degenerate zero entries (numerically possible for near-zero
                eigenvector components) are clamped to +1.

        Step 3  Apply s_k to each lag-mean trajectory, producing the
                provisionally oriented ensemble ``aligned_means`` of shape
                (K, T).

        Step 4  Global epoch-mean orientation: compute the scalar mean of
                ``aligned_means`` over the specified epoch window.  If negative,
                flip all s_k simultaneously (and update ``aligned_means``).

        Step 5  Peak-polarity refinement: examine the group mean
                ``aligned_means.mean(axis=0)`` over the epoch window.  Find the
                time-point of maximum absolute deviation (the dominant peak).
                If that peak value is negative, flip all s_k (and
                ``aligned_means``) once more.

        Parameters
        ----------
        z_lag_trials : ndarray, shape (K, n_trials, T)
            Per-lag trial trajectories, NaN-padded at boundary positions.
        epoch : (t_start, t_end)
            Index range [t_start, t_end) into the T dimension that defines the
            task epoch used for orientation (Steps 4–5).  Typically
            corresponds to [t=0, t=T) or [Bar-Off index, T).

        Returns
        -------
        lag_signs : ndarray, shape (K,)
            Each element is +1.0 or -1.0.  Multiplying z_lag_trials[k] by
            lag_signs[k] yields the orientation-corrected latent for lag k.
        """
        K, n_trials, T = z_lag_trials.shape
        t_start, t_end = epoch

        # ------------------------------------------------------------------ #
        # Step 1: trial-mean per lag → (K, T); zero-fill NaN boundary bins   #
        # ------------------------------------------------------------------ #
        lag_means = np.nanmean(z_lag_trials, axis=1)          # (K, T)
        lag_means = np.nan_to_num(lag_means, nan=0.0)

        # ------------------------------------------------------------------ #
        # Step 2: Z2 synchronisation via leading eigenvector of C_{K×K}      #
        # ------------------------------------------------------------------ #
        C = np.corrcoef(lag_means)                            # (K, K)
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 1.0)

        _, lag_evecs = np.linalg.eigh(C)
        lag_signs = np.sign(lag_evecs[:, -1]).astype(float)   # (K,)
        lag_signs[lag_signs == 0.0] = 1.0

        # ------------------------------------------------------------------ #
        # Step 3: apply per-lag signs to lag means                            #
        # ------------------------------------------------------------------ #
        aligned_means = lag_signs[:, np.newaxis] * lag_means  # (K, T)

        # ------------------------------------------------------------------ #
        # Step 4: global epoch-mean → uniform orientation across all lags     #
        # ------------------------------------------------------------------ #
        epoch_mean = aligned_means[:, t_start:t_end].mean()
        if epoch_mean < 0:
            lag_signs    *= -1.0
            aligned_means *= -1.0

        # ------------------------------------------------------------------ #
        # Step 5: peak-polarity refinement on the group mean                  #
        # ------------------------------------------------------------------ #
        group_mean   = aligned_means.mean(axis=0)             # (T,)
        epoch_window = group_mean[t_start:t_end]
        if epoch_window.size > 0:
            peak_val = epoch_window[np.argmax(np.abs(epoch_window))]
            if peak_val < 0:
                lag_signs *= -1.0

        return lag_signs

    # ------------------------------------------------------------------
    # Shared trial-plotting primitive
    # ------------------------------------------------------------------

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
        Render individual trial traces as thin translucent lines, then overlay
        the trial mean as a solid opaque line and +-1 SD as a shaded band.

        NaN entries (boundary truncation from _per_lag_latent) are plotted
        only over their valid support, so no interpolation artefact appears.

        Parameters
        ----------
        trials : (n_trials, T)
        """
        for r in range(trials.shape[0]):
            tr = trials[r]
            v = np.isfinite(tr)
            if v.any():
                ax.plot(time_vec[v], tr[v],
                        color=color, linewidth=lw_trial, alpha=alpha_trial,
                        rasterized=True)

        mean = np.nanmean(trials, axis=0)
        std  = np.nanstd(trials,  axis=0)/trials.shape[0]
        vm = np.isfinite(mean)
        ax.plot(time_vec[vm], mean[vm],
                color=color, linewidth=2.0, label=label, zorder=3)
        ax.fill_between(time_vec[vm],
                        (mean - std)[vm], (mean + std)[vm],
                        color=color, alpha=0.25, zorder=2)

    # ------------------------------------------------------------------
    # Figure 1 -- per-lag latents
    # ------------------------------------------------------------------

    def _figure_lag_latents(
        self,
        pair: _SessionTkPair,
        c: int,
        save: bool,
    ) -> Optional[plt.Figure]:
        """
        K-column x 2-row grid of lag-resolved latents.

        Row 1: z_i^(tau_k) -- one panel per lag.
        Row 2: z_j          -- stationary projection, repeated across columns
               for direct visual pairing with each lag panel above.
        All panels share a common y-scale within their respective row.
        """
        K = pair.lag_bins.size
        X_i_z = self._zscore_per_neuron(pair.X_i)
        X_j_z = self._zscore_per_neuron(pair.X_j)

        wy_c       = pair.wy_stationary[:, c]
        z_j_trials = self._stationary_latent(X_j_z, wy_c)        # (n_trials, T)

        z_i_lag_trials = np.full((K, pair.X_i.shape[0], self.T), np.nan)
        for k in range(K):
            z_i_lag_trials[k] = self._per_lag_latent(
                X_i_z, pair.wx_temporal[:, k, c], int(pair.lag_bins[k])
            )

        # ------------------------------------------------------------------ #
        # Sign alignment via Z2 spectral synchronisation across lags.         #
        # ``_align_lag_signs`` treats the K trial-averaged trajectories as an  #
        # ensemble of "sessions" and returns a per-lag sign vector             #
        # lag_flip_signs ∈ {-1, +1}^K.  Multiplying in-place ensures that    #
        # both the individual trial lines and the mean ± SD band (computed     #
        # downstream by ``_plot_with_trials`` from the same array) are         #
        # consistently oriented.                                               #
        # The epoch convention mirrors the cross-session implementation:       #
        # index 0 corresponds to Bar-Off (t = 0 s); we orient relative to     #
        # the post-stimulus window [t=0, end-of-trial].                        #
        # ------------------------------------------------------------------ #
        t0_idx = int(np.searchsorted(self.time_vec, 0.0))
        lag_flip_signs = self._align_lag_signs(
            z_i_lag_trials, epoch=(t0_idx, self.T)
        )
        for k in range(K):
            z_i_lag_trials[k] *= lag_flip_signs[k]

        k_star = pair.dominant_lag_idx(c)

        # Robust y-limits (1st-99th percentile avoids artefact spike influence)
        flat_i = z_i_lag_trials.ravel()
        flat_i = flat_i[np.isfinite(flat_i)]
        y_lo_i = float(np.percentile(flat_i, 1))  if flat_i.size else -1.0
        y_hi_i = float(np.percentile(flat_i, 99)) if flat_i.size else  1.0
        flat_j = z_j_trials.ravel()[np.isfinite(z_j_trials.ravel())]
        y_lo_j = float(np.percentile(flat_j, 1))  if flat_j.size else -1.0
        y_hi_j = float(np.percentile(flat_j, 99)) if flat_j.size else  1.0

        fig, axes = plt.subplots(
            2, K,
            figsize=(max(18.0, 1.7 * K), 7.0),
            gridspec_kw={'hspace': 0.35, 'wspace': 0.20},
            squeeze=False,
        )

        for k in range(K):
            ax_i, ax_j = axes[0, k], axes[1, k]

            # Row 1: region i at lag tau_k
            self._plot_with_trials(ax_i, self.time_vec, z_i_lag_trials[k], color='C3')
            ax_i.axvline(0, color='k', linestyle='--', linewidth=0.7, alpha=0.4)
            ax_i.set_ylim(y_lo_i, y_hi_i)
            ax_i.set_xticks([])
            title_tag = r"  $(\tau^*)$" if k == k_star else ""
            ax_i.set_title(
                fr"$\tau={pair.lag_secs[k]:+.2f}$s{title_tag}",
                fontsize=8,
                color=('red' if k == k_star else 'black'),
            )
            if k == 0:
                ax_i.set_ylabel(
                    fr'{pair.region_i}  $z_i^{{(\tau)}}(t)$', fontsize=10
                )
            ax_i.tick_params(axis='y', labelsize=7)
            for s in ('top', 'right'):
                ax_i.spines[s].set_visible(False)

            # Row 2: region j (stationary, same trace repeated)
            self._plot_with_trials(ax_j, self.time_vec, z_j_trials, color='C0')
            ax_j.axvline(0, color='k', linestyle='--', linewidth=0.7, alpha=0.4)
            ax_j.set_ylim(y_lo_j, y_hi_j)
            ax_j.set_xlabel('t (s)', fontsize=8)
            ax_j.tick_params(axis='both', labelsize=7)
            if k == 0:
                ax_j.set_ylabel(
                    fr'{pair.region_j}  $z_j(t)$', fontsize=10
                )
            for s in ('top', 'right'):
                ax_j.spines[s].set_visible(False)

        fig.suptitle(
            f"{self.session_name}   |   "
            f"{pair.region_i} \u2192 {pair.region_j}   "
            f"(component 1,  K={K} lags)",
            fontsize=15, fontweight='normal',
        )

        if save:
            out = (self.output_dir /
                   f"{self.session_name}_{pair.region_i}_{pair.region_j}"
                   f"_1_lag_latents.png")
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Figure 2 -- Rastermap (Image-3 style)
    # ------------------------------------------------------------------

    def _figure_rastermap(
        self,
        pair: _SessionTkPair,
        c: int,
        n_show: int,
        save: bool,
    ) -> Optional[plt.Figure]:
        """
        Two-row figure (region i, region j).

        Each row: Rastermap-ordered, trial-averaged PSTH (RdBu_r, real time
        axis, red dashed Bar-Off line) beside a horizontal CCA weight barh.
        Matches the style of the existing
        ``neural_single_session_package_oxford_enhanced.py`` output.
        """
        k_star     = pair.dominant_lag_idx(c)
        tau_star_s = pair.dominant_lag_seconds(c)

        X_i_z = self._zscore_per_neuron(pair.X_i)
        X_j_z = self._zscore_per_neuron(pair.X_j)

        wx_star = pair.wx_temporal[:, k_star, c]
        wy_c    = pair.wy_stationary[:, c]

        fig, axes = plt.subplots(
            2, 2,
            figsize=(12, 11),
            gridspec_kw={
                'width_ratios': [4.5, 1.0],
                'hspace': 0.45,
                'wspace': 0.08,
            },
        )

        for row, (X_z, weight, rname, bar_lbl) in enumerate([
            (X_i_z, wx_star, pair.region_i,
             fr"$w_x(\tau^*\!=\!{tau_star_s:+.2f}$ s$)$"),
            (X_j_z, wy_c, pair.region_j, r"$w_y$"),
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

        fig.suptitle(
            f"{self.session_name}   |   "
            f"{pair.region_i} \u2192 {pair.region_j}   (component 1)",
            fontsize=13, fontweight='bold',
        )

        if save:
            out = (self.output_dir /
                   f"{self.session_name}_{pair.region_i}_{pair.region_j}"
                   f"_2_rastermap.png")
            fig.savefig(out, dpi=300, bbox_inches='tight')
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
        """
        One Rastermap row: trial-averaged PSTH (left) + horizontal CCA weight
        bar (right).

        Neuron sub-sampling preserves Rastermap order (every step-th neuron
        from the globally sorted list), so displayed rows are globally ordered.
        The y-axis runs 0 (top) to n_sel (bottom), matching Image-3.
        """
        n_neurons = X_z.shape[1]
        if sort_idx.size != n_neurons:
            sort_idx = np.arange(n_neurons)

        if n_neurons > n_show:
            step = max(1, n_neurons // n_show)
            sel = sort_idx[::step][:n_show]
        else:
            sel = sort_idx
        n_sel = len(sel)

        # Trial-averaged PSTH in Rastermap order
        psth = X_z.mean(axis=0)[sel]   # (n_sel, T)

        vmax = max(float(np.nanpercentile(np.abs(psth), 99)), 0.5)
        im = ax_raster.imshow(
            psth, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[time_vec[0], time_vec[-1], n_sel, 0],
            origin='upper',
        )
        ax_raster.axvline(
            0, color='red', linestyle='--', linewidth=1.5, alpha=0.9,
            label='Bar Off',
        )
        ax_raster.legend(
            loc='upper right', fontsize=8, framealpha=0.4, handlelength=1.2
        )
        ax_raster.set_xlabel('Time (s)', fontsize=11)
        ax_raster.set_ylabel('Neurons (Global sorted)', fontsize=11)
        ax_raster.set_title(
            f'{region_name} \u2013 Rastermap Ordered Activity', fontsize=12
        )
        cbar = fig.colorbar(im, ax=ax_raster, fraction=0.03, pad=0.03)
        cbar.set_label('Z-scored firing rate', fontsize=9)

        pos = ax_bar.get_position()
        offset = 0.1
        ax_bar.set_position([pos.x0 + offset, pos.y0, pos.width, pos.height])

        # Horizontal CCA weight bar (barh)
        ypos  = np.arange(n_sel) + 0.5
        w_sel = weight[sel].ravel()
        ax_bar.barh(ypos, w_sel, height=0.8, color='steelblue', alpha=0.85)
        ax_bar.axvline(0, color='k', linewidth=0.8, alpha=0.6)
        ax_bar.set_ylim(n_sel, 0)   # 0 at top -- matches imshow orientation
        ax_bar.set_xlabel('CCA Weight', fontsize=9)
        ax_bar.set_title('CCA', fontsize=10)
        ax_bar.tick_params(axis='both', labelsize=7)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['left'].set_visible(False)
        plt.setp(ax_bar.get_yticklabels(), visible=False)

    # ------------------------------------------------------------------
    # Figure 3 -- dominant-lag trajectory
    # ------------------------------------------------------------------

    def _figure_dominant_lag(
        self,
        pair: _SessionTkPair,
        c: int,
        save: bool,
    ) -> Optional[plt.Figure]:
        """
        Two stacked panels (region i top, region j bottom) showing the
        dominant-lag component-1 latent trajectories with individual trials
        and trial mean +/- 1 SD band.
        """
        k_star     = pair.dominant_lag_idx(c)
        tau_star_s = pair.dominant_lag_seconds(c)
        rho_star   = float(pair.correlogram[k_star, c])

        X_i_z = self._zscore_per_neuron(pair.X_i)
        X_j_z = self._zscore_per_neuron(pair.X_j)

        z_i_trials = self._per_lag_latent(
            X_i_z, pair.wx_temporal[:, k_star, c], int(pair.lag_bins[k_star])
        )
        z_j_trials = self._stationary_latent(X_j_z, pair.wy_stationary[:, c])

        fig, axes = plt.subplots(
            2, 1, figsize=(12, 6),
            gridspec_kw={'hspace': 0.38},
        )

        self._plot_with_trials(
            axes[0], self.time_vec, z_i_trials, color='C3',
            label=fr'{pair.region_i} @ $\tau^* = {tau_star_s:+.2f}$ s',
        )
        axes[0].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[0].set_ylabel('Latent projection (a.u.)', fontsize=10)
        axes[0].legend(fontsize=9, frameon=False, loc='best')
        for s in ('top', 'right'):
            axes[0].spines[s].set_visible(False)

        self._plot_with_trials(
            axes[1], self.time_vec, z_j_trials, color='C0',
            label=fr'{pair.region_j} (stationary)',
        )
        axes[1].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[1].set_ylabel('Latent projection (a.u.)', fontsize=10)
        axes[1].set_xlabel('Time (s)', fontsize=11)
        axes[1].legend(fontsize=9, frameon=False, loc='best')
        for s in ('top', 'right'):
            axes[1].spines[s].set_visible(False)

        fig.suptitle(
            f"{self.session_name}   |   "
            f"Dominant-lag component-1 trajectory   "
            fr"$\rho(\tau^*) = {rho_star:+.3f}$",
            fontsize=13, fontweight='bold',
        )

        if save:
            out = (self.output_dir /
                   f"{self.session_name}_{pair.region_i}_{pair.region_j}"
                   f"_3_dominant_lag.png")
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Figure 4 -- pre-computed projections
    # ------------------------------------------------------------------

    def _figure_projections(
        self,
        pr: dict,
        pair: _SessionTkPair,
        save: bool,
    ) -> Optional[plt.Figure]:
        """
        2 x n_sig grid of the lag-summed tkCCA projections stored in
        ``pair_results.projections``.

        These are the **full** canonical latents produced by
        ``calculate_tkcca_projections`` in MATLAB:

            z_i(r, t) = sum_k  w_x(tau_k)^T X_i^z(r, :, t - tau_k)
            z_j(r, t) = w_y^T X_j^z(r, :, t)

        They differ from Figure 1 (per-lag isolation) and Figure 3 (dominant
        lag only).

        Parsing follows ``Useful_definition._extract_single_trial``
        (confirmed from debugger):
            projections['components'][k]     -> list of one dict  (mat73 cell)
            projections['components'][k][0]  -> dict with keys:
                region_i_trials  (n_trials, T)
                region_j_trials  (n_trials, T)
                R2               scalar
        """
        proj = pr.get('projections')
        if not isinstance(proj, dict):
            print(f"  no projections field for "
                  f"({pair.region_i}, {pair.region_j})")
            return None

        components = proj.get('components')
        if not isinstance(components, (list, tuple)) or not components:
            return None

        # Parse each component block
        comp_data = []
        for comp in components:
            # mat73 wraps each MATLAB cell element in a one-element Python list
            if isinstance(comp, (list, tuple)):
                if not comp:
                    continue
                comp_dict = comp[0]
            elif isinstance(comp, dict):
                comp_dict = comp
            else:
                continue
            if not isinstance(comp_dict, dict):
                continue

            zi = safe_array(comp_dict.get('region_i_trials'))   # (n_trials, T)
            zj = safe_array(comp_dict.get('region_j_trials'))
            if zi is None or zj is None or zi.ndim != 2 or zj.ndim != 2:
                continue

            R2_raw = comp_dict.get('R2', 0.0)
            R2 = float(np.asarray(R2_raw).ravel()[0])
            comp_data.append({'zi': zi, 'zj': zj, 'R2': R2})

        if not comp_data:
            print(f"  no valid projection components for "
                  f"({pair.region_i}, {pair.region_j})")
            return None

        n_comps = len(comp_data)
        T_proj  = comp_data[0]['zi'].shape[1]

        # Time vector: prefer the stored time_axis (step 4 of _extract_single_trial)
        time_raw = safe_array(first_present(proj, ['time_axis']))
        if time_raw is not None and time_raw.size == T_proj:
            time_vec = time_raw.ravel()
        else:
            time_vec = np.linspace(
                self.time_range_s[0], self.time_range_s[1], T_proj
            )

        fig, axes = plt.subplots(
            2, n_comps,
            figsize=(max(10.0, 3.5 * n_comps), 7.0),
            gridspec_kw={'hspace': 0.38, 'wspace': 0.25},
            squeeze=False,
        )

        for ci, cd in enumerate(comp_data):
            ax_i, ax_j = axes[0, ci], axes[1, ci]

            self._plot_with_trials(ax_i, time_vec, cd['zi'], color='C3')
            ax_i.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax_i.set_title(
                fr"Comp {ci + 1}   $\rho = {cd['R2']:.3f}$", fontsize=9
            )
            ax_i.tick_params(labelsize=7)
            if ci == 0:
                ax_i.set_ylabel(
                    fr'{pair.region_i}  $z_i(t)$', fontsize=10
                )
            for s in ('top', 'right'):
                ax_i.spines[s].set_visible(False)

            self._plot_with_trials(ax_j, time_vec, cd['zj'], color='C0')
            ax_j.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax_j.set_xlabel('Time (s)', fontsize=9)
            ax_j.tick_params(labelsize=7)
            if ci == 0:
                ax_j.set_ylabel(
                    fr'{pair.region_j}  $z_j(t)$', fontsize=10
                )
            for s in ('top', 'right'):
                ax_j.spines[s].set_visible(False)

        fig.suptitle(
            f"{self.session_name}   |   "
            f"{pair.region_i} \u2192 {pair.region_j}   "
            "tkCCA global projections",
            fontsize=12, fontweight='normal',
        )

        if save:
            out = (self.output_dir /
                   f"{self.session_name}_{pair.region_i}_{pair.region_j}"
                   f"_4_projections.png")
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Master dispatcher -- four figures per pair
    # ------------------------------------------------------------------

    def create_tkcca_pair_figures(
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

        c      = component_idx
        pr_raw = self.pair_results[self.pair_index[(region_i, region_j)]]

        steps = [
            ("lag latents",  lambda: self._figure_lag_latents(pair, c, save)),
            ("rastermap",    lambda: self._figure_rastermap(pair, c, n_neurons_show, save)),
            ("dominant lag", lambda: self._figure_dominant_lag(pair, c, save)),
            ("projections",  lambda: self._figure_projections(pr_raw, pair, save)),
        ]
        for label, fn in steps:
            print(f"    [{region_i} -> {region_j}]  building {label}…")
            fig = fn()
            if fig is not None:
                plt.close(fig)

    # ------------------------------------------------------------------
    # Session correlogram heatmap
    # ------------------------------------------------------------------

    def create_session_correlogram_heatmap(
        self,
        component_idx: int = 0,
        region_order: Optional[List[str]] = None,
        selected_pairs: Optional[List[Tuple[str, str]]] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Heatmap of rho(tau_k, c=1) across region pairs in this session.
        Rows follow ANATOMICAL_ORDER for cross-session comparability.

        Parameters
        ----------
        selected_pairs : list of (region_i, region_j) tuples, optional
            When provided, only those pairs are included in the heatmap.
            The rows are still sorted according to ``region_order``
            (ANATOMICAL_ORDER by default) within that subset.  If None,
            every pair present in ``pair_index`` is shown.
        """
        if region_order is None:
            region_order = ANATOMICAL_ORDER

        # Build the set of admissible pairs for O(1) membership tests.
        pair_whitelist = (
            set(map(tuple, selected_pairs)) if selected_pairs is not None else None
        )

        rows = []
        for (r_i, r_j), idx in self.pair_index.items():
            # Apply whitelist filter when the caller has specified one.
            if pair_whitelist is not None and (r_i, r_j) not in pair_whitelist:
                continue
            pr  = self.pair_results[idx]
            cg  = safe_array(pr.get('canonical_correlogram'))
            ls  = safe_array(pr.get('tkcca_lags_seconds'))
            if cg is None or ls is None:
                continue
            if cg.ndim == 1:
                cg = cg[:, np.newaxis]
            if cg.shape[1] <= component_idx:
                continue
            rows.append((r_i, r_j, ls.ravel(), cg[:, component_idx]))

        if not rows:
            print(f"[{self.session_name}] no tkCCA correlograms to plot")
            return None

        def _key(item):
            ki = region_order.index(item[0]) if item[0] in region_order else 999
            kj = region_order.index(item[1]) if item[1] in region_order else 999
            return (ki, kj)
        rows.sort(key=_key)

        lag_secs = rows[0][2]
        K = lag_secs.size
        heat = np.full((len(rows), K), np.nan)
        labels = []
        for i, (r_i, r_j, _, rho_vec) in enumerate(rows):
            heat[i] = rho_vec[:K]
            labels.append(fr'{r_i} $\to$ {r_j}')

        fig, ax = plt.subplots(
            figsize=(max(8.0, 0.45 * K + 2.0),
                     max(4.0, 0.32 * len(rows) + 2.0)),
            constrained_layout=True,
        )
        vmax = max(float(np.nanmax(np.abs(heat))) if np.isfinite(heat).any() else 1.0,
                   0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(
            heat, aspect='auto', cmap='RdBu_r', norm=norm,
            extent=[lag_secs[0], lag_secs[-1], len(rows) - 0.5, -0.5],
        )
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(r'Lag $\tau$ (s)', fontsize=10)
        ax.set_title(
            f"{self.session_name}   \u2014   "
            r"$\rho(\tau,\,c\!=\!1)$  across region pairs",
            fontsize=11,
        )
        ax.axvline(0, color='k', linestyle='--', linewidth=0.6, alpha=0.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(r'$\rho(\tau,\,c\!=\!1)$', fontsize=9)

        if save:
            out = self.output_dir / f"{self.session_name}_correlogram_heatmap.png"
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------

    def run_all_pairs(
            self,
            component_idx: int = 0,
            n_neurons_show: int = 60,
            selected_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Produce all four figures for specific pairs (or all if None), then the heatmap."""

        # Determine which pairs to iterate over
        pairs_to_process = selected_pairs if selected_pairs is not None else list(self.pair_index.keys())

        for (r_i, r_j) in pairs_to_process:
            # Safety check: skip if the pair wasn't actually found in the .mat file
            if (r_i, r_j) not in self.pair_index:
                print(f"  [Skipping] ({r_i}, {r_j}) not found in pair_results index.")
                continue

            self.create_tkcca_pair_figures(
                r_i, r_j,
                component_idx=component_idx,
                n_neurons_show=n_neurons_show,
            )

        # The session heatmap respects the same pair whitelist that was used for
        # the per-pair figures, so the heatmap rows are directly comparable.
        fig = self.create_session_correlogram_heatmap(
            component_idx=component_idx,
            selected_pairs=selected_pairs,
        )
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
    SELECTED_PAIRS: List[Tuple[str, str]] = [
        ('ORB', 'STR'),
        ('mPFC', 'STR'),
        ('MOp', 'STR'),
        ('MOs', 'STR'),
        ('MOs', 'mPFC'),
        ('MOp', 'MOs'),
        ('MOp', 'VPMPO'),
        ('MOs', 'VPMPO'),
    ]

    base_dir = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    session_name = "yp020_220331"
    session_file = (
            base_dir / "tkcca_sessions_cued_hit_long_results"
            / f"{session_name}_analysis_results.mat"
    )
    out_dir = base_dir / "Paper_output" / "figures_tkcca_session" / session_name

    viz = OxfordTkCCASessionVisualizer(
        session_results_path=str(session_file),
        session_name=session_name,
        output_dir=str(out_dir),
    )

    if not viz.load_session():
        return

    viz.compute_global_rastermap()

    # PASS THE SELECTED_PAIRS HERE
    viz.run_all_pairs(
        component_idx=0,
        n_neurons_show=60,
        selected_pairs=SELECTED_PAIRS
    )
if __name__ == "__main__":
    main()
"""
pcca_single_session.py
======================
"""


from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
# 0b.  Core CCA / pCCA primitives — copied verbatim (cosmetic trimming only)
#      from pcca_sequential_ablation.py, §1 "Core mathematics" and
#      §2 "Data loading".
#
#      Per this project's established convention ("primitive copying over
#      importing"), these are duplicated here rather than imported from the
#      ablation script, so that every pCCA fit in this module follows a
#      byte-identical data path (same ridge whitening, same residualisation,
#      same per-region neuron-selection masking) and this file remains
#      independently runnable and independently auditable.
# =============================================================================

LAMBDA_CCA:   float = 1e-4   # ridge coefficient added to Cxx / Cyy in whitening
LAMBDA_HAT:   float = 1e-4   # ridge on Z'Z in the nuisance hat matrix
N_COMPONENTS: int   = 5      # canonical dimensions retained per pCCA fit

# Preprocessing regime used throughout the single-region-ablation pivot
# analyses below (§0c onward).  Held at the "raw" regime (no PSTH
# subtraction, no trial shuffling) by default, matching
# pcca_sequential_ablation.py's own SUBTRACT_PSTH / SHUFFLE_TRIALS
# defaults.  Flip these to explore the residual (noise-correlation-only)
# or shuffled (finite-sample-bias) regimes instead; see that script's §0
# for the full three-regime rationale.
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
    flatten (n_trials, n, T) → (T * n_trials, n).  Verbatim copy of
    pcca_sequential_ablation.py's ``_zscore_flat``."""
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
    """Verbatim copy of pcca_sequential_ablation.py's ``load_region_spikes``:
    every region's raw (n_trials, n_neurons, T) spike tensor, with each
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
# 0c.  Pivot-ablation constants and small statistics helpers
# =============================================================================

# Pivot regions for the single-region-ablation analyses
# (create_session_subspace_angles / create_session_gini_panel).  Each pivot
# is analysed independently: every other non-excluded region present in the
# session is both a candidate CCA partner and a candidate single-region
# nuisance regressor for that pivot.
PIVOT_REGIONS: List[str] = ["ORB", "MOp", "MOs", "OLF", "VALVM", "VPMPO"]

# User-configurable region blacklist (case-insensitive match).  Any region
# named here is treated exactly like a fiber-tract catch-all: it is barred
# from every pivot / partner / nuisance role in the single-region-ablation
# analyses below.  Edit this list to change which anatomical regions are
# considered eligible for the ablation sweep.
EXCLUDED_REGIONS: List[str] = ["STR", "STRv", "LP", "HY"]


def _is_fiber_region(region_name: str) -> bool:
    """True for catch-all white-matter / fiber-tract region labels (e.g.
    'fiber tracts')."""
    return "fiber" in region_name.lower()


def _is_excluded_region(region_name: str) -> bool:
    """
    True if `region_name` should be excluded from every pivot / partner /
    nuisance role in the ablation analyses below — i.e. it is either a
    fiber-tract catch-all region, or it appears (case-insensitively) in
    the user-configurable ``EXCLUDED_REGIONS`` list above.
    """
    if _is_fiber_region(region_name):
        return True
    excluded_lower = {r.lower() for r in EXCLUDED_REGIONS}
    return region_name.lower() in excluded_lower


def _gini_coefficient(w: np.ndarray) -> float:
    r"""
    Gini coefficient of the absolute canonical weight vector :math:`|w|`,
    via the standard sorted-cumulative formula

    .. math::
        G = \frac{\sum_{i=1}^{n} (2i - n - 1)\, x_{(i)}}
                 {n \sum_{i=1}^{n} x_{(i)}}

    for :math:`x` sorted ascending.  :math:`G \to 0`: weight spread
    uniformly across all neurons; :math:`G \to 1`: weight concentrated in a
    single neuron.  Returns NaN for a degenerate (empty or all-zero) vector.
    """
    x = np.sort(np.abs(np.asarray(w, dtype=float).ravel()))
    n = x.size
    total = float(x.sum())
    if n == 0 or total <= 1e-12:
        return float("nan")
    index = np.arange(1, n + 1, dtype=float)
    return float(np.sum((2.0 * index - n - 1.0) * x) / (n * total))


def _principal_angle_deg(
        a: Optional[np.ndarray], b: Optional[np.ndarray],
) -> float:
    """
    Principal angle (degrees) between two 1-D canonical weight vectors, via
    QR orthonormalisation followed by ``scipy.linalg.subspace_angles`` —
    the numerically safe pattern established elsewhere in this project (QR
    preserves the column space; slicing ``Qa[:, 0:1]`` rather than indexing
    ``Qa[:, 0]`` keeps the array 2-D, which ``subspace_angles`` requires).
    Sign-invariant by construction, since a CCA weight vector's global sign
    is arbitrary.
    """
    if a is None or b is None:
        return float("nan")
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return float("nan")
    Qa, _ = np.linalg.qr(a.reshape(-1, 1), mode="reduced")
    Qb, _ = np.linalg.qr(b.reshape(-1, 1), mode="reduced")
    ang = subspace_angles(Qa[:, 0:1], Qb[:, 0:1])
    return float(np.degrees(ang[0]))


# =============================================================================
# 0d.  A0 poster scaling — shared by every session-wide heatmap / bar
#      figure below (create_session_subspace_angles_single,
#      create_session_gini_panel, create_session_gini_panel_full_ablation,
#      create_session_mi_bar).  Figures sized for on-screen/paper viewing
#      become illegible once printed at A0 (33.1 x 46.8 in); POSTER_SCALE
#      multiplies canvas size, tick/annotation/label fontsize, and cell-
#      grid linewidth relative to the screen-resolution defaults used
#      elsewhere in this module.  _FS_FLOOR guards against per-cell
#      annotation text (e.g. "0.73" in a large R x R heatmap) shrinking
#      below legibility as R grows, since figsize scales with R but a
#      fixed base fontsize does not.
#
#      This is a single tunable knob, not a fixed physical mapping: the
#      "correct" POSTER_SCALE depends on the final placement width of
#      each panel inside your A0 layout (whatever page-layout tool you're
#      assembling the poster in), which this script has no visibility
#      into.  Render once, compare against the panel's placeholder size
#      in your layout, and adjust POSTER_SCALE up/down accordingly —
#      raising dpi below is a separate, orthogonal knob for print
#      sharpness once the physical size is right.
# =============================================================================

POSTER_SCALE: float = 2.6    # figsize + fontsize multiplier for A0 export
_FS_FLOOR:    float = 13.0   # minimum legible pt size regardless of R


def _fs(base_pt: float) -> float:
    """Scale a screen-resolution font size (pt) for A0 poster print,
    clipped at ``_FS_FLOOR`` so small annotation text never drops below
    legibility as matrix dimension R grows."""
    return max(base_pt * POSTER_SCALE, _FS_FLOOR)


def _grid_cells(ax: plt.Axes, R: int, lw: float = 0.5) -> None:
    """Draw thin white gridlines between heatmap cells via minor ticks —
    at A0 scale, undivided imshow cells blur together even when the
    color mapping is correct; an explicit cell boundary restores the
    grid structure a screen viewer gets 'for free' from anti-aliasing at
    smaller physical size."""
    ax.set_xticks(np.arange(-0.5, R, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, R, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=lw * POSTER_SCALE)
    ax.tick_params(which="minor", bottom=False, left=False)


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

        # ── NEW — raw spike tensors for the single-region-ablation pivot
        # analyses (create_session_subspace_angles / create_session_
        # gini_panel / create_session_mi_bar).  Populated in load_session()
        # via the borrowed load_region_spikes() primitive; kept independent
        # of self.region_data / self.pair_results, which remain the
        # MATLAB-precomputed source for the other, unchanged per-pair
        # figures (projections / canonical spectrum / weight scatter /
        # rho heatmap / bipartite pCCA graph).
        self.region_spikes: Dict[str, np.ndarray] = {}
        self.rs_n_trials: Optional[int] = None
        self.rs_T: Optional[int] = None
        self._ablation_weight_cache: Dict[
            Tuple[str, str, str], Optional[np.ndarray]
        ] = {}
        # Sibling cache for the *full* k-dimensional single-region-ablation
        # fit (weights + canonical correlations), as opposed to
        # _ablation_weight_cache above, which only ever keeps the dominant
        # column Wx[:, 0].  Needed by create_session_subspace_angles2 when
        # run in use_ablation=True mode, where up to n_dims columns per
        # side are required rather than just one.
        self._ablation_fit_cache: Dict[
            Tuple[str, str, str, int], Optional[Tuple[np.ndarray, np.ndarray]]
        ] = {}

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

        # ── NEW — raw spike tensors for the single-region-ablation pivot
        # analyses.  This re-reads the .mat file through the borrowed
        # load_region_spikes() primitive rather than reusing region_data
        # above, by design (see §0b): it keeps the ablation pipeline's data
        # path byte-identical to pcca_sequential_ablation.py, independent
        # of how this class parses region_data for the legacy
        # MATLAB-precomputed figures.
        try:
            self.region_spikes, self.rs_n_trials, self.rs_T = load_region_spikes(
                str(self.session_path)
            )
        except Exception as exc:
            warnings.warn(
                f"[{self.session_name}] load_region_spikes failed ({exc}); "
                f"pivot ablation analyses (subspace angles / Gini heatmap / "
                f"MI bar) will be unavailable."
            )
            self.region_spikes, self.rs_n_trials, self.rs_T = {}, None, None

        if self.rs_T is not None and self.T is not None and self.rs_T != self.T:
            warnings.warn(
                f"[{self.session_name}] T mismatch between region_data "
                f"({self.T}) and load_region_spikes ({self.rs_T}); "
                f"time_vec uses {self.T}, ablation fits use {self.rs_T}."
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
    # NEW — single-region-ablation pivot-analysis backbone
    # ------------------------------------------------------------------
    #
    # Shared by create_session_subspace_angles, create_session_gini_panel,
    # and (in its "full joint ablation" variant) create_session_mi_bar.
    # For a fixed pivot region P, every other candidate region can serve
    # as either the CCA partner or the single nuisance regressor; pCCA is
    # refit for every ordered (partner, ablate) pair with partner ≠ ablate,
    # using the byte-identical primitives copied in §0b above — i.e. this
    # bypasses self.pair_results (the MATLAB-precomputed field used by the
    # other, unchanged figures in this class) entirely.

    def _candidate_regions(
            self, exclude: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Regions eligible to serve as an ablation partner or nuisance
        regressor: everything present in self.region_spikes, ordered by
        ANATOMICAL_ORDER (any session region absent from ANATOMICAL_ORDER
        is appended afterwards, alphabetically, as a defensive fallback),
        excluding fiber-tract catch-all regions, regions named in
        EXCLUDED_REGIONS, and anything in `exclude`.
        """
        exclude_set = set(exclude) if exclude else set()
        present = set(self.region_spikes.keys())
        ordered = [r for r in ANATOMICAL_ORDER if r in present]
        extra   = sorted(present - set(ordered))
        return [
            r for r in ordered + extra
            if r not in exclude_set and not _is_excluded_region(r)
        ]

    def _compute_single_ablation_weight(
        self,
        pivot: str,
        partner: str,
        ablate: str,
    ) -> Optional[np.ndarray]:
        """
        Fit pCCA(pivot, partner | Z = ablate) on this session's raw spike
        tensors and return the pivot's first-canonical-dimension weight
        vector :math:`w_P \\in \\mathbb{R}^{n_{\\mathrm{pivot}}}`.  The
        pivot is always placed on the "X" side of ``pcca(X, Y, Z)`` — the
        returned vector is always ``Wx[:, 0]``, regardless of anatomical
        ordering, so angle / Gini comparisons across cells are always
        comparing the same object.  Results are memoised in
        ``self._ablation_weight_cache``.
        """
        key = (pivot, partner, ablate)
        if key in self._ablation_weight_cache:
            return self._ablation_weight_cache[key]

        if not all(r in self.region_spikes for r in (pivot, partner, ablate)):
            self._ablation_weight_cache[key] = None
            return None

        X_piv = _zscore_flat(self.region_spikes[pivot],
                              subtract_psth=SUBTRACT_PSTH,
                              shuffle_trials=SHUFFLE_TRIALS)
        X_par = _zscore_flat(self.region_spikes[partner],
                              subtract_psth=SUBTRACT_PSTH,
                              shuffle_trials=SHUFFLE_TRIALS)
        Z_abl = _zscore_flat(self.region_spikes[ablate],
                              subtract_psth=SUBTRACT_PSTH,
                              shuffle_trials=SHUFFLE_TRIALS)

        try:
            Wx, _, _, _, _ = pcca(X_piv, X_par, Z_abl)
            w = Wx[:, 0].copy()
        except Exception as exc:
            warnings.warn(
                f"[{self.session_name}] pCCA({pivot}, {partner} | "
                f"Z={ablate}) failed: {exc}"
            )
            w = None

        self._ablation_weight_cache[key] = w
        return w

    def _compute_single_ablation_fit(
        self,
        pivot: str,
        partner: str,
        ablate: str,
        n_components: int = N_COMPONENTS,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        r"""
        Fit :math:`\mathrm{pCCA}(P, \text{partner} \mid Z=\text{ablate})`
        on this session's raw spike tensors and return the *full*
        pivot-side canonical weight matrix
        :math:`W_P \in \mathbb{R}^{n_P \times k}` together with its
        canonical correlation vector :math:`\rho \in \mathbb{R}^k`
        (:math:`k = \min(\text{n\_components}, \dots)`, per ``ridge_cca``).

        This generalises ``_compute_single_ablation_weight`` — which keeps
        only the dominant column :math:`W_P[:,0]` for the pivot subspace-
        angle / Gini panels — to the multi-dimensional case needed by
        ``create_session_subspace_angles2(use_ablation=True)``, where up to
        ``n_dims`` canonical directions per side are displayed rather than
        just the first.  Uses the same byte-identical primitives (§0b) and
        regime flags (``SUBTRACT_PSTH`` / ``SHUFFLE_TRIALS``) as every
        other ablation fit in this module.  Memoised in
        ``self._ablation_fit_cache``.
        """
        key = (pivot, partner, ablate, n_components)
        if key in self._ablation_fit_cache:
            return self._ablation_fit_cache[key]

        if not all(r in self.region_spikes for r in (pivot, partner, ablate)):
            self._ablation_fit_cache[key] = None
            return None

        X_piv = _zscore_flat(self.region_spikes[pivot],
                              subtract_psth=SUBTRACT_PSTH,
                              shuffle_trials=SHUFFLE_TRIALS)
        X_par = _zscore_flat(self.region_spikes[partner],
                              subtract_psth=SUBTRACT_PSTH,
                              shuffle_trials=SHUFFLE_TRIALS)
        Z_abl = _zscore_flat(self.region_spikes[ablate],
                              subtract_psth=SUBTRACT_PSTH,
                              shuffle_trials=SHUFFLE_TRIALS)

        try:
            Wx, _, rho, _, _ = pcca(
                X_piv, X_par, Z_abl, n_components=n_components,
            )
        except Exception as exc:
            warnings.warn(
                f"[{self.session_name}] full-matrix pCCA({pivot}, "
                f"{partner} | Z={ablate}) failed: {exc}"
            )
            self._ablation_fit_cache[key] = None
            return None

        result = (Wx.copy(), rho.copy())
        self._ablation_fit_cache[key] = result
        return result

    def compute_pivot_ablation_matrix(
        self, pivot_region: str,
    ) -> Tuple[List[str], Dict[Tuple[str, str], Optional[np.ndarray]]]:
        """
        For a fixed pivot P, compute the full single-region-ablation weight
        matrix

            W[(row, col)]  =  w_P   from   pCCA(P, row | Z = col),  row ≠ col

        over every candidate region (ANATOMICAL_ORDER regions present in
        this session, excluding P itself, fiber-tract regions, and
        EXCLUDED_REGIONS).  Diagonal entries (row == col) are not defined
        — a region cannot simultaneously be the CCA partner and the sole
        nuisance regressor partialled out — and are simply absent from the
        returned dict.

        Returns
        -------
        regions : ordered list of candidate region names (shared row/column
                  labelling for both create_session_subspace_angles and
                  create_session_gini_panel, so the two figures are drawn
                  from a single consistent index).
        matrix  : {(row_region, col_region): w_P or None}
        """
        if pivot_region not in self.region_spikes:
            warnings.warn(
                f"[{self.session_name}] pivot '{pivot_region}' not present "
                f"in this session's region_spikes; skipping."
            )
            return [], {}

        regions = self._candidate_regions(exclude=[pivot_region])
        matrix: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
        n_fits = len(regions) * max(len(regions) - 1, 0)
        print(
            f"  [{self.session_name}]  pivot={pivot_region}:  fitting pCCA "
            f"over {len(regions)} candidate regions  ({n_fits} single-"
            f"region-ablation steps)…"
        )
        for row_region in regions:
            for col_region in regions:
                if row_region == col_region:
                    continue
                matrix[(row_region, col_region)] = (
                    self._compute_single_ablation_weight(
                        pivot_region, row_region, col_region,
                    )
                )
        return regions, matrix

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
        alpha_trial: float = 0.0,
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
                        color=color, alpha=0.5, zorder=2)
        total_min = np.nanmin(trials)
        total_max = np.nanmax(trials)

        ax.set_ylim(bottom=total_min * 0.5, top=total_max * 0.5)

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

            self._plot_with_trials(ax_i, time_vec, cd["zi"], color="k")
            ax_i.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax_i.set_title(fr"Dim {ci + 1}   $\rho = {cd['R2']:.3f}$", fontsize=15)
            ax_i.tick_params(labelsize=15)
            if ci == 0:
                ax_i.set_ylabel(fr"{pair.region_i}  $z_i(t)$", fontsize=15)
            for sp in ("top", "right"):
                ax_i.spines[sp].set_visible(False)

            self._plot_with_trials(ax_j, time_vec, cd["zj"], color="C0")
            ax_j.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax_j.set_xlabel("Time (s)", fontsize=15)
            ax_j.tick_params(labelsize=15)
            if ci == 0:
                ax_j.set_ylabel(fr"{pair.region_j}  $z_j(t)$", fontsize=15)
            for sp in ("top", "right"):
                ax_j.spines[sp].set_visible(False)

        fig.suptitle(
            f"{pair.pair_label}   "
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
                f"3_canonical_spectrum_{self.session_name}_{pair.region_i}_{pair.region_j}.png"
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
            f" {pair.pair_label}   "
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
            fr"pCCA dominant $\rho$ (dim 1)",
            fontsize=11,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cb.set_label(r"Dominant $\rho$", fontsize=9)

        if save:
            out = self.output_dir / f"rho_heatmap_{self.session_name}_.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Session-wide figure 6 — MI bar chart
    # ------------------------------------------------------------------

    def create_session_mi_bar(
        self,
        fixed_region: str = "ORB",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        r"""
        Mutual-information bar chart, retaining the *original*
        remove-all-other-regions ("full joint ablation") definition:

            Z = every candidate region except fixed_region and partner
            MI(fixed_region, partner) = -\sum_k log(1 - \rho_k^2)

        with :math:`\rho_k` the canonical correlations returned by
        ``pcca()`` (script-1's "full pCCA reference" construction — Z is
        the concatenation of every remaining candidate region's flat
        matrix), as opposed to the single-region ablation used by
        create_session_subspace_angles / create_session_gini_panel.

        ``fixed_region`` (ORB by default) is always r_i and is always
        written first in the pair label — "ORB↔MOs", never "MOs↔ORB" —
        i.e. it is always the leftmost region name in each tick label.
        Bars remain sorted by descending MI; only the naming convention is
        pinned, not the bar order.
        """
        if fixed_region not in self.region_spikes:
            print(f"[{self.session_name}] fixed region '{fixed_region}' "
                  f"not present in region_spikes; cannot build MI bar.")
            return None

        all_candidates = self._candidate_regions()
        partners = [r for r in all_candidates if r != fixed_region]
        if not partners:
            print(f"[{self.session_name}] no partner regions for MI bar.")
            return None

        X_fix = _zscore_flat(self.region_spikes[fixed_region],
                              subtract_psth=SUBTRACT_PSTH,
                              shuffle_trials=SHUFFLE_TRIALS)

        records: List[Tuple[str, float]] = []
        for partner in partners:
            nuisance_regions = [
                r for r in all_candidates if r not in (fixed_region, partner)
            ]
            X_par = _zscore_flat(self.region_spikes[partner],
                                  subtract_psth=SUBTRACT_PSTH,
                                  shuffle_trials=SHUFFLE_TRIALS)
            Z_full = (
                np.concatenate(
                    [_zscore_flat(self.region_spikes[r],
                                   subtract_psth=SUBTRACT_PSTH,
                                   shuffle_trials=SHUFFLE_TRIALS)
                     for r in nuisance_regions],
                    axis=1,
                )
                if nuisance_regions else None
            )

            try:
                _, _, rho, _, _ = pcca(X_fix, X_par, Z_full)
            except Exception as exc:
                warnings.warn(
                    f"[{self.session_name}] full-ablation pCCA("
                    f"{fixed_region}, {partner} | all others) failed: {exc}"
                )
                continue

            rho_sq = np.clip(rho.astype(float), 0.0, 1.0 - 1e-6) ** 2
            mi_val = float(-np.sum(np.log1p(-rho_sq)))
            if not np.isfinite(mi_val):
                continue
            records.append((f"{fixed_region}↔{partner}", mi_val))

        if not records:
            print(f"[{self.session_name}] no MI values to plot")
            return None

        records.sort(key=lambda x: x[1], reverse=True)
        labels, vals = zip(*records)

        fig, ax = plt.subplots(
            figsize=(7 * POSTER_SCALE, max(3.0, 0.4 * len(records)) * POSTER_SCALE),
            constrained_layout=True,
        )
        ypos = np.arange(len(records))
        ax.barh(ypos, vals, height=0.6, color="steelblue", alpha=0.85,
                edgecolor="black", linewidth=0.5 * POSTER_SCALE)
        ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=_fs(16))
        ax.tick_params(axis="x", labelsize=_fs(16))
        ax.set_xlabel(r"Mutual information  $-\sum_k\log(1-\rho_k^2)$", fontsize=_fs(18))
        ax.set_title(
            f"pCCA mutual information",
            fontsize=_fs(20),
        )
        ax.invert_yaxis()
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.8 * POSTER_SCALE / 2.6)

        if save:
            out = self.output_dir / (
                f"mi_bar_anchored_{fixed_region}_{self.session_name}.png"
            )
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    # ------------------------------------------------------------------
    # Session-wide figure 7 — Gini coefficient panel
    # ------------------------------------------------------------------
    def create_session_gini_panel_full_ablation(
            self,
            save: bool = True,
    ) -> Optional[plt.Figure]:
        r"""
        Full-joint-ablation Gini heatmap, companion to
        ``create_session_gini_panel``.

        Unlike that method — which fixes a pivot P and ablates one
        nuisance region at a time (single-region ablation) — this panel
        is NOT pivot-restricted.  For every unordered pair of candidate
        regions :math:`(i, j)`, a single pCCA fit is taken with the
        nuisance set Z equal to *every other* candidate region — your
        original "full joint ablation" construction, identical in spirit
        to the Z used in ``create_session_mi_bar``:

        .. math::
            (W_i,\, W_j,\, \boldsymbol\rho) \;=\; \mathrm{pCCA}\Big(
                X_i,\, X_j \;\Big|\; Z = \{\text{all candidates}\}
                \setminus \{i, j\}\Big)

        Cell (row i, col j), :math:`i > j` only (lower triangle), reports

        .. math::
            \mathrm{Gini}(|w_i|)

        the Gini coefficient of region *i*'s (the row region's) own
        canonical weight vector from this fit — region i is always
        placed on the "X" side of ``pcca(X, Y, Z)``.  Only the lower
        triangle is drawn: for a fixed unordered pair {i, j} the
        underlying fit is symmetric in Z (Z does not depend on which
        region is X vs Y), so, unlike the single-region-ablation matrix
        above, (i, j) and (j, i) are not independent replicates but two
        readouts (Wx-side vs Wy-side Gini) of the *same* fit.  Reporting
        only one triangle keeps each cell an unambiguous single number
        rather than implying a false second degree of freedom.  Region
        j's own Gini for a given pair is the Wy-side coefficient of that
        same fit, and is deliberately not plotted here.

        Diagonal is undefined (a region cannot be ablated against
        itself).
        """
        candidates = self._candidate_regions()
        R = len(candidates)
        if R < 2:
            print(f"[{self.session_name}] fewer than 2 candidate regions; "
                  f"nothing to plot.")
            return None

        gini_mat = np.full((R, R), np.nan)
        n_fits = R * (R - 1) // 2
        print(
            f"  [{self.session_name}]  full-joint-ablation Gini panel: "
            f"fitting pCCA over {R} candidate regions  ({n_fits} unordered "
            f"pairs)…"
        )

        # Pre-compute each region's flattened, z-scored matrix once —
        # every pair's Z is a different subset of these, but recomputing
        # _zscore_flat per (i, j) would be O(R^3) redundant work.
        X_flat: Dict[str, np.ndarray] = {
            r: _zscore_flat(self.region_spikes[r],
                            subtract_psth=SUBTRACT_PSTH,
                            shuffle_trials=SHUFFLE_TRIALS)
            for r in candidates
        }

        for i in range(R):
            region_i = candidates[i]
            for j in range(0, i):
                region_j = candidates[j]
                nuisance_regions = [
                    r for r in candidates if r not in (region_i, region_j)
                ]
                Z_full = (
                    np.concatenate(
                        [X_flat[r] for r in nuisance_regions], axis=1,
                    )
                    if nuisance_regions else None
                )
                try:
                    Wx, Wy, _, _, _ = pcca(
                        X_flat[region_i], X_flat[region_j], Z_full,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[{self.session_name}] full-ablation pCCA("
                        f"{region_i}, {region_j} | all others) failed: "
                        f"{exc}"
                    )
                    continue
                gini_mat[i, j] = _gini_coefficient(Wx[:, 0])
                gini_mat[j, i] = _gini_coefficient(Wy[:, 0])
        fig, ax = plt.subplots(
            figsize=(max(6.0, 0.9 * R) * POSTER_SCALE,
                     max(5.0, 0.85 * R) * POSTER_SCALE),
            constrained_layout=True,
        )
        #mask = np.triu(np.ones((R, R), dtype=bool))  # hide diagonal + upper
        #display = np.where(mask, np.nan, gini_mat)
        im = ax.imshow(
            gini_mat, aspect="equal", cmap="magma",
            vmin=0.0, vmax=1.0, interpolation="nearest",
        )
        _grid_cells(ax, R)
        for i in range(R):
            for j in range(R):
                v = gini_mat[i, j]
                if np.isfinite(v):
                    tc = "white" if v < 0.55 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=_fs(7.5), color=tc)
        ax.set_xticks(range(R));
        ax.set_xticklabels(candidates, rotation=45, ha="right", fontsize=_fs(16))
        ax.set_yticks(range(R));
        ax.set_yticklabels(candidates, fontsize=_fs(16))
        ax.set_xlabel("column j", fontsize=_fs(18))
        ax.set_ylabel(r"row i   $(\mathrm{Gini}(|w_i|)$ shown)", fontsize=_fs(18))
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cb.ax.tick_params(labelsize=_fs(8))
        cb.set_label(r"Gini$(|w_i|)$", fontsize=_fs(9))
        ax.set_title(
            r"Gini$(|w_i|)$ from pCCA(region $i$, region $j$ $\vert$ "
            r"Z = all others)",
            fontsize=_fs(20),
        )

        if save:
            out = self.output_dir / (
                f"gini_heatmap_full_ablation_{self.session_name}.png"
            )
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

    def create_session_gini_panel(
        self,
        pivot_region: str,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        r"""
        Single-region-ablation Gini heatmap for one fixed pivot P.

        Cell (row i, col j) shows :math:`\mathrm{Gini}(|w_P|)`, where
        :math:`w_P` is P's canonical weight vector from
        pCCA(P, region_i | Z = region_j) — pivot P paired with region_i,
        with region_j ablated as the sole nuisance regressor.

        Unlike the subspace-angle matrix below, (i, j) and (j, i) are
        genuinely different pCCA fits — "P + i, ablate j" vs.
        "P + j, ablate i" — and their Gini coefficients need not agree, so
        the FULL square matrix is drawn (diagonal left undefined: ablating
        a region from its own pairing is not a valid configuration).

        Higher Gini ⇒ P's coupling to region_i, with region_j partialled
        out, is carried by a sparser subset of P's neurons.
        """
        regions, W = self.compute_pivot_ablation_matrix(pivot_region)
        R = len(regions)
        if R < 2:
            print(f"[{self.session_name}] pivot '{pivot_region}': fewer "
                  f"than 2 candidate partner regions; nothing to plot.")
            return None

        gini_mat = np.full((R, R), np.nan)
        for i, row_region in enumerate(regions):
            for j, col_region in enumerate(regions):
                if i == j:
                    continue
                w = W.get((row_region, col_region))
                if w is None:
                    continue
                gini_mat[i, j] = _gini_coefficient(w)

        fig, ax = plt.subplots(
            figsize=(max(6.0, 0.9 * R) * POSTER_SCALE,
                     max(5.0, 0.85 * R) * POSTER_SCALE),
            constrained_layout=True,
        )
        im = ax.imshow(
            gini_mat, aspect="equal", cmap="magma",
            vmin=0.0, vmax=1.0, interpolation="nearest",
        )
        _grid_cells(ax, R)
        for i in range(R):
            for j in range(R):
                v = gini_mat[i, j]
                if np.isfinite(v):
                    tc = "white" if v < 0.55 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=_fs(7.5), color=tc)
        ax.set_xticks(range(R)); ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=_fs(16))
        ax.set_yticks(range(R)); ax.set_yticklabels(regions, fontsize=_fs(16))
        ax.set_xlabel("column j  (region ablated,  Z)", fontsize=_fs(18))
        ax.set_ylabel("row i  (CCA partner region)", fontsize=_fs(18))
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cb.ax.tick_params(labelsize=_fs(8))
        cb.set_label(r"Gini$(|w_P|)$", fontsize=_fs(9))
        ax.set_title(
            f"{self.session_name}  —  single-region-ablation weight sparsity\n"
            f"pivot P = {pivot_region}   "
            r"Gini$(|w_P|)$ from pCCA(P, row $\vert$ Z = col)",
            fontsize=_fs(20),
        )

        if save:
            out = self.output_dir / (
                f"gini_heatmap_ablation_pivot_{self.session_name}_"
                f"{pivot_region}.png"
            )
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
            pivot_regions = [
                r for r, cnt in region_counts.items()
                if cnt >= 2 and not _is_excluded_region(r)
            ]

        pair_wl = (
            set(map(tuple, selected_pairs)) if selected_pairs is not None else None
        )

        # For each pivot, collect {partner: weight_matrix_for_pivot}
        figs_out = []
        for pivot in pivot_regions:
            if _is_excluded_region(pivot):
                continue
            partner_weights: Dict[str, np.ndarray] = {}
            for (r_i, r_j), idx in self.pair_index.items():
                if pair_wl is not None and (r_i, r_j) not in pair_wl:
                    continue
                if r_i != pivot and r_j != pivot:
                    continue
                if _is_excluded_region(r_i) or _is_excluded_region(r_j):
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
                f"Subspace rotation at pivot: {pivot}",
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

    def create_session_subspace_angles_single(
        self,
        pivot_region: str,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        r"""
        Single-region-ablation subspace-rotation matrix for one fixed
        pivot P.

        Once the pivot is fixed, cell (row i, col j), :math:`i < j` only
        (upper triangle), shows the principal angle between two P-side
        canonical weight vectors:

            A  =  w_P   from   pCCA(P, region_i | Z = region_j)
            B  =  w_P   from   pCCA(P, region_j | Z = region_i)

        i.e. "P paired with region_i, region_j ablated" vs. "P paired with
        region_j, region_i ablated".  Swapping (i, j) swaps A and B without
        changing the angle between them, so this matrix is symmetric and
        only the upper triangle is drawn (the lower triangle and diagonal
        are masked).

        :math:`\theta \to 0°`: the two single-region-ablation solutions
        recruit the same direction in P regardless of which of
        :math:`\{i, j\}` is the CCA partner and which is the nuisance
        regressor — P's coupling geometry is insensitive to this
        particular swap.
        :math:`\theta \to 90°`: swapping which region is "partner" vs.
        "ablated" rotates P's communication axis into an orthogonal
        subspace — the single-region-ablation analogue of the subspace
        rotation described in Gonzalez et al. (2026) Fig. 1g.
        """
        regions, W = self.compute_pivot_ablation_matrix(pivot_region)
        R = len(regions)
        if R < 2:
            print(f"[{self.session_name}] pivot '{pivot_region}': fewer "
                  f"than 2 candidate partner regions; nothing to plot.")
            return None

        angle_mat = np.full((R, R), np.nan)
        for i in range(R):
            for j in range(0, i):
                a = W.get((regions[i], regions[j]))  # P + row_i, ablate col_j
                b = W.get((regions[j], regions[i]))  # P + row_j, ablate col_i
                angle_mat[i, j] = _principal_angle_deg(a, b)

        fig, ax = plt.subplots(
            figsize=(max(6.0, 0.9 * R) * POSTER_SCALE,
                     max(5.0, 0.85 * R) * POSTER_SCALE),
            constrained_layout=True,
        )
        im = ax.imshow(
            angle_mat, aspect="equal", cmap="viridis",
            vmin=0, vmax=90, interpolation="nearest",
        )
        _grid_cells(ax, R)
        for i in range(R):
            for j in range(0, i):
                v = angle_mat[i, j]
                if np.isfinite(v):
                    tc = "white" if v < 50 else "black"
                    ax.text(j, i, f"{v:.0f}°", ha="center", va="center",
                            fontsize=_fs(12), color=tc)
        ax.set_xticks(range(R)); ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=_fs(15))
        ax.set_yticks(range(R)); ax.set_yticklabels(regions, fontsize=_fs(15))
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
        cb.ax.tick_params(labelsize=_fs(15))
        cb.set_label("Principal angle (°)", fontsize=_fs(15))
        ax.set_title(
            f"pivot R = {pivot_region}"
            r"$\theta$[pCCA(P,row$\vert$Z=col), pCCA(P,col$\vert$Z=row)]",
            fontsize=_fs(12)
        )

        if save:
            out = self.output_dir / (
                f"subspace_angles_ablation_pivot_{self.session_name}_"
                f"{pivot_region}.png"
            )
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
        return fig

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
            use_ablation: bool = False,
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
        use_ablation : bool
            If False (default), pivot-side weights are read from the
            MATLAB-precomputed ``pair_results`` entries, exactly as before
            — whatever nuisance set those two pairs happened to be fit
            with.  If True, both sides are instead refit from raw spikes
            via single-region ablation, with the two partner regions used
            as *each other's* nuisance regressor:

                W_left  = W_x  from  pCCA(pivot, left_partner  | Z = right_partner)
                W_right = W_x  from  pCCA(pivot, right_partner | Z = left_partner)

            i.e. the left side shows pivot's coupling to left_partner with
            right_partner's contribution specifically partialled out, and
            vice versa — a genuinely paired, symmetric ablation contrast,
            computed with byte-identical primitives on both sides (see
            §0b/0c and ``_compute_single_ablation_fit``), rather than
            whatever heterogeneous nuisance conditioning the precomputed
            MATLAB pairs used.
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

        def _lookup_pair_ablation(
                partner: str, ablate: str,
        ) -> Optional[Tuple[np.ndarray, float]]:
            """
            Single-region-ablation counterpart to ``_lookup_pair``: refits
            pCCA(pivot_region, partner | Z = ablate) from raw spikes
            (n_components = n_dims) and returns (W_pivot, dominant_rho),
            matching _lookup_pair's return signature so every downstream
            step (dimension clipping, neuron ranking, edge drawing) is
            untouched regardless of which mode produced the weights.
            """
            fit = self._compute_single_ablation_fit(
                pivot_region, partner, ablate, n_components=n_dims,
            )
            if fit is None:
                print(
                    f"  [{self.session_name}] single-region-ablation "
                    f"pCCA({pivot_region}, {partner} | Z={ablate}) "
                    f"unavailable — check region_spikes for these three "
                    f"regions."
                )
                return None
            Wx, rho = fit
            rho_val = float(rho[0]) if rho.size > 0 else 0.0
            return Wx, rho_val

        if use_ablation:
            result_left  = _lookup_pair_ablation(left_partner, right_partner)
            result_right = _lookup_pair_ablation(right_partner, left_partner)
        else:
            result_left  = _lookup_pair(left_partner)
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
        # ax.annotate(
        #     "",
        #     xy=(x_neuron, 0.975), xytext=(x_neuron, 1.03),
        #     arrowprops=dict(arrowstyle="-|>", color="black", lw=3),
        #     annotation_clip=False,
        # )

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
                    va="center", fontsize=25, color=colour, fontweight="bold",
                )

        _draw_dim_nodes(n_left_dims, y_left_nodes, x_left_node, _BLUE, "left")
        _draw_dim_nodes(n_right_dims, y_right_nodes, x_right_node, _ROSE, "right")

        y_lbl = 0.07
        ax.text(
            x_left_node, float(y_left_nodes.min()) - y_lbl,
                         f"{left_partner}\n" + r"$\rho_1 = $" + f"{rho_left:.3f}",
            ha="center", va="top", fontsize=25, color=_BLUE,
        )
        ax.text(
            x_right_node, float(y_right_nodes.min()) - y_lbl,
                          f"{right_partner}\n" + r"$\rho_1 = $" + f"{rho_right:.3f}",
            ha="center", va="top", fontsize=25, color=_ROSE,
        )

        title_note = (
            fr"single-region ablation: $Z={{{right_partner}}}$ (left) / "
            fr"$Z={{{left_partner}}}$ (right)"
            if use_ablation else "weights from precomputed pair_results"
        )

        if right_partner == 'VALVM' :
            right_partner = 'motor Thal'
        if left_partner == 'VPMPO':
            left_partner = 'sens Thal'

        if pivot_region == 'VALVM':
            pivot_region = 'motor Thal'
        elif pivot_region == 'VPMPO':
            pivot_region = 'sens Thal'


        fig.suptitle(
            fr"$\leftarrow$ {left_partner}   {right_partner} $\rightarrow$",
            fontsize=30,
            fontweight="bold",y=0.95
        )
        ax.text(
            x_neuron, 0.87,
            f"{pivot_region} neurons\n",
            ha="center", va="bottom", fontsize=30, fontweight="bold",
            clip_on=False,
        ) #fr"($n = {n_sel}$ / $N = {n_neurons_total}$ shown)",


        if save:
            tag = "_ablation" if use_ablation else ""
            out = self.output_dir / (
                f"subspace_angles2_{self.session_name}_"
                f"_pivot_{pivot_region}"
                f"_{left_partner}_vs_{right_partner}{tag}.png"
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
            #("projections",         lambda: self._figure_projections(pr_raw, pair, save)),
            # ("canonical spectrum",  lambda: self._figure_canonical_spectrum(pair, save)),
            # ("weight scatter",      lambda: self._figure_weight_scatter(pair, save)),
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
            ("rho heatmap", lambda: self.create_session_rho_heatmap(selected_pairs=pairs_to_run)),
            ("MI bar (ORB-anchored, full joint ablation)", lambda: self.create_session_mi_bar()),
            ("subspace angles", lambda: self.create_session_subspace_angles(selected_pairs=pairs_to_run)),
            # ← add this line
        ]:
            print(f"  [session] {label}…")
            fig = fn()
            if fig is not None:
                plt.close(fig)

        print(
            "  [session] pivot-based single-region-ablation figures "
            "(subspace-angle matrix / Gini heatmap) are pivot-specific and "
            "are not part of this wholesale dispatch — call "
            "create_session_subspace_angles(pivot_region=...) / "
            "create_session_gini_panel(pivot_region=...) explicitly for "
            "each of PIVOT_REGIONS, or see main() below."
        )

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
    # NOTE — selected_pairs is an OPTIONAL filter for run_all_pairs() /
    # create_session_rho_heatmap(); leaving it unset (as below) runs every
    # pair present in self.pair_index.  Uncomment to restrict the per-pair
    # figures + rho heatmap to a specific subset:
    #
    # SELECTED_PAIRS: List[Tuple[str, str]] = [
    #     ("MOs", "VPMPO"), ("MOs", "VALVM"), ("VPMPO", "VALVM"),
    # ]

    base_dir     = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
    session_name = "yp021_220404" #"yp020_220331"
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

    viz.run_all_pairs(
        component_idx=0,
        n_neurons_show=50
    )

    # Example: bipartite diagram with single-region-ablation weights
    # instead of the precomputed pair_results.  With use_ablation=True,
    # the MOs-side weights on the left branch come from
    # pCCA(MOs, VALVM | Z=VPMPO), and on the right branch from
    # pCCA(MOs, VPMPO | Z=VALVM) — a genuinely paired ablation contrast.
    #
    viz.create_session_subspace_angles2(
            pivot_region="MOs",
            left_partner="VPMPO",
            right_partner="VALVM",
            n_dims=2,
            n_neurons_show=50,
            use_ablation=True,
        )

    # viz.create_session_subspace_angles2(
    #         pivot_region="MOs",
    #         left_partner="VALVM",
    #         right_partner="MOp",
    #         n_dims=2,
    #         n_neurons_show=50,
    #         use_ablation=True,
    #     )

    viz.create_session_subspace_angles2(
            pivot_region="MOp",
            left_partner="VPMPO",
            right_partner="VALVM",
            n_dims=2,
            n_neurons_show=50,
            use_ablation=True,
        )


    viz.create_session_subspace_angles2(
            pivot_region="VALVM",
            left_partner="MOs",
            right_partner="MOp",
            n_dims=2,
            n_neurons_show=50,
            use_ablation=True,
        )

    viz.create_session_subspace_angles2(
            pivot_region="VPMPO",
            left_partner="MOs",
            right_partner="MOp",
            n_dims=2,
            n_neurons_show=50,
            use_ablation=True,
        )

    viz.create_session_subspace_angles2(
            pivot_region="VALVM",
            left_partner="MOs",
            right_partner="ORB",
            n_dims=2,
            n_neurons_show=50,
            use_ablation=True,
        )

    viz.create_session_subspace_angles2(
            pivot_region="VPMPO",
            left_partner="MOs",
            right_partner="ORB",
            n_dims=2,
            n_neurons_show=50,
            use_ablation=True,
        )

    # viz.create_session_subspace_angles2(
    #         pivot_region="ORB",
    #         left_partner="VALVM",
    #         right_partner="VPMPO",
    #         n_dims=2,
    #         n_neurons_show=50,
    #         use_ablation=True,
    #     )


    # viz.create_session_subspace_angles2(
    #         pivot_region="VPMPO",
    #         left_partner="MOs",
    #         right_partner="VALVM",
    #         n_dims=2,
    #         n_neurons_show=50,
    #         use_ablation=True,
    #     )

    # viz.create_session_subspace_angles2(
    #         pivot_region="VPMPO",
    #         left_partner="ORB",
    #         right_partner="MOp",
    #         n_dims=2,
    #         n_neurons_show=50,
    #         use_ablation=True,
    #     )
    #
    # viz.create_session_subspace_angles2(
    #         pivot_region="VPMPO",
    #         left_partner="ORB",
    #         right_partner="MOs",
    #         n_dims=2,
    #         n_neurons_show=50,
    #         use_ablation=True,
    #     )

    # ── NEW — single-region-ablation pivot analyses ───────────────────────
    # Recomputed from raw spikes via load_region_spikes() + pcca(), bypassing
    # self.pair_results entirely (see §0b–0c and compute_pivot_ablation_matrix).
    # PIVOT_REGIONS = ["ORB", "MOp", "MOs", "OLF", "VALVM", "VPMPO"]; each is
    # paired against every other non-excluded region present in the session
    # (fiber tracts and EXCLUDED_REGIONS are both barred — see §0c).
    # for pivot in PIVOT_REGIONS:
    #     fig_angles = viz.create_session_subspace_angles_single(pivot_region=pivot)
    #     if fig_angles is not None:
    #         plt.close(fig_angles)
    #
    #     fig_gini = viz.create_session_gini_panel(pivot_region=pivot)
    #     if fig_gini is not None:
    #         plt.close(fig_gini)
    #
    #     fig_gini_full = viz.create_session_gini_panel_full_ablation()
    #     if fig_gini_full is not None:
    #         plt.close(fig_gini_full)
    #
    # fig_mi = viz.create_session_mi_bar(fixed_region="VPMPO")
    # if fig_mi is not None:
    #     plt.close(fig_mi)
    # fig_mi = viz.create_session_mi_bar(fixed_region="MOs")
    # if fig_mi is not None:
    #     plt.close(fig_mi)
    #
    # fig_mi = viz.create_session_mi_bar(fixed_region="ORB")
    # if fig_mi is not None:
    #     plt.close(fig_mi)

if __name__ == "__main__":
    main()


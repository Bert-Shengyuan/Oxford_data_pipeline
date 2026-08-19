import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import mat73
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "mat73 is required to read MATLAB v7.3 files: pip install mat73"
    ) from exc

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ANATOMICAL_ORDER: List[str] = [
    "mPFC", "ORB", "MOp", "MOs", "OLF",
    "STR", "STRv",
    "MD", "LP", "VALVM", "VPMPO", "ILM",
    "HY",
]


def align_projection_signs(
        projections: dict,
        n_components: int,
        reference_type: str,
        time_bins: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """Align canonical-variate signs across trial types via Z2 spectral sync.

    Intended for the multi-condition (cross-session) pipeline where the same
    pCCA pair is computed on several trial types (cued hit, spont hit, …) and
    all must share a common positive orientation before plotting.

    Algorithm — for each component k
    ----------------------------------
    Step 1  Build (C, C) pairwise Pearson-correlation matrices from the
            trial-averaged time-courses stacked into  U  of shape (C, T).

    Step 2  Z2 synchronisation: leading eigenvector of C_u gives sign vector
            s ∈ {-1, +1}^C whose ith entry says "flip condition i or not".

    Step 3  Apply per-condition signs to u_stack and v_stack.

    Step 4  Global orientation anchor: epoch mean of the reference condition
            over [0, 1.5 s] must be ≥ 0.

    Step 5  Fine correction: signed peak of the reference epoch must be ≥ 0.

    Step 6  Record net flip decisions; write aligned means back and propagate
            to per-trial arrays.  (STD / SEM are sign-invariant, unchanged.)

    Parameters
    ----------
    projections : dict
        {trial_type: {'u_mean':   np.ndarray  (T, n_comp),
                      'v_mean':   np.ndarray  (T, n_comp),
                      'u_trials': np.ndarray  (n_trials, T, n_comp),
                      'v_trials': np.ndarray  (n_trials, T, n_comp)}}
    n_components : int
        Number of CCA / pCCA components to align.
    reference_type : str
        Trial-type key used as the positive anchor (Steps 4–5).
    time_bins : np.ndarray, optional
        Time axis (seconds).  Used to locate the [0, 1.5 s] epoch by
        searchsorted.  Falls back to the hard-coded 226-bin default if None.

    Returns
    -------
    flip_decisions : dict
        {trial_type -> {comp_idx -> {'u_flip': bool, 'v_flip': bool}}}
        Net flip relative to the original input signs.
    """
    trial_types = list(projections.keys())

    # ── Resolve epoch bin indices ─────────────────────────────────────────────
    if time_bins is not None:
        t_start = int(np.searchsorted(time_bins, 0.0))
        t_end = int(np.searchsorted(time_bins, 1.5))
    else:
        t_start, t_end = 75, 151  # fallback for the 226-bin Oxford axis

    flip_decisions: Dict[str, Dict] = {
        tt: {k: {'u_flip': False, 'v_flip': False} for k in range(n_components)}
        for tt in trial_types
    }

    # u_stack / v_stack shape: (n_conditions, T, n_comp)
    u_stack = np.stack(
        [projections[tt]['u_mean'] for tt in trial_types], axis=0
    )
    v_stack = np.stack(
        [projections[tt]['v_mean'] for tt in trial_types], axis=0
    )
    ref_idx = trial_types.index(reference_type)

    for comp_idx in range(n_components):

        # ── Steps 1–2  Build correlation matrix; leading eigenvector ──────────
        U = u_stack[:, :, comp_idx]  # (n_conditions, T)
        V = v_stack[:, :, comp_idx]
        C_u = np.atleast_2d(np.corrcoef(U))  # (n_conditions, n_conditions)
        C_v = np.atleast_2d(np.corrcoef(V))

        _, evecs_u = np.linalg.eigh(C_u)
        s_u = np.sign(evecs_u[:, -1])  # (n_conditions,)

        _, evecs_v = np.linalg.eigh(C_v)
        s_v = np.sign(evecs_v[:, -1])

        # Guard: eigh may return 0.0 for perfectly degenerate entries
        s_u[s_u == 0] = 1
        s_v[s_v == 0] = 1

        # ── Step 3  Apply per-condition signs ────────────────────────────────
        u_stack[:, :, comp_idx] = s_u[:, np.newaxis] * U
        v_stack[:, :, comp_idx] = s_v[:, np.newaxis] * V

        # ── Step 4  Global anchor — epoch mean of reference must be ≥ 0 ──────
        u_ref_mean = u_stack[ref_idx, t_start:t_end, comp_idx].mean()
        v_ref_mean = v_stack[ref_idx, t_start:t_end, comp_idx].mean()

        if u_ref_mean < 0:
            s_u *= -1
            u_stack[:, :, comp_idx] *= -1

        if v_ref_mean < 0:
            s_v *= -1
            v_stack[:, :, comp_idx] *= -1

        # ── Step 5  Fine correction — signed peak must be ≥ 0 ───────────────
        u_ref_epoch = u_stack[ref_idx, t_start:t_end, comp_idx]
        v_ref_epoch = v_stack[ref_idx, t_start:t_end, comp_idx]

        if u_ref_epoch[np.argmax(np.abs(u_ref_epoch))] < 0:
            s_u *= -1
            u_stack[:, :, comp_idx] *= -1

        if v_ref_epoch[np.argmax(np.abs(v_ref_epoch))] < 0:
            s_v *= -1
            v_stack[:, :, comp_idx] *= -1

        # ── Step 6  Record net flip decisions ─────────────────────────────────
        for tt_idx, tt in enumerate(trial_types):
            flip_decisions[tt][comp_idx] = {
                'u_flip': bool(s_u[tt_idx] < 0),
                'v_flip': bool(s_v[tt_idx] < 0),
            }

    # ── Write aligned means back; propagate flips to per-trial arrays ─────────
    # SEM / STD are sign-invariant — they are NOT modified.
    for tt_idx, tt in enumerate(trial_types):
        proj = projections[tt]
        for comp_idx in range(n_components):
            fd = flip_decisions[tt][comp_idx]
            if fd['u_flip']:
                proj['u_trials'][:, :, comp_idx] *= -1
            if fd['v_flip']:
                proj['v_trials'][:, :, comp_idx] *= -1
        proj['u_mean'] = u_stack[tt_idx]
        proj['v_mean'] = v_stack[tt_idx]

    print(f"    [align_projection_signs]  "
          f"{n_components} component(s), {len(trial_types)} condition(s).")
    return flip_decisions


def apply_latent_sign_correction(
        z_u: np.ndarray,
        z_v: np.ndarray,
        time_vec: np.ndarray,
        t_epoch_start: float = 0.0,
        t_epoch_end: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """Single-condition sign correction for a canonical-variate pair.

    This is the C = 1 special case of ``align_projection_signs``.  With a
    single condition, Steps 1–3 (Z2 synchronisation) are degenerate — the
    1 × 1 Pearson matrix has leading eigenvector [1], so no inter-condition
    alignment is needed.  The substantive constraints reduce to:

        Step 4  Trial-mean epoch in [t_epoch_start, t_epoch_end] must be ≥ 0.
        Step 5  Signed peak within that epoch must be ≥ 0.

    Each variate (u, v) is corrected **independently**.

    Note on weight-vector propagation
    ----------------------------------
    Because  z = X_resid @ w,  a sign flip on z is equivalent to negating w.
    The caller must therefore propagate any flip to the corresponding canonical
    weight vector before displaying the weight-bar panel::

        z_i_p, z_j_p, flip_ip, flip_jp = apply_latent_sign_correction(
            z_i_p, z_j_p, time_vec)
        w_pcca_i = Wx_pcca[:, 0] * (-1 if flip_ip else 1)
        w_pcca_j = Wy_pcca[:, 0] * (-1 if flip_jp else 1)

    Parameters
    ----------
    z_u, z_v       : (n_trials, T) — latent projections for regions I and J.
    time_vec       : (T,) — time axis in seconds.
    t_epoch_start  : left edge of the anchor epoch (default 0.0 s, onset).
    t_epoch_end    : right edge of the anchor epoch (default 1.5 s).

    Returns
    -------
    z_u_out, z_v_out : sign-corrected arrays, shape (n_trials, T).
    flip_u, flip_v   : True if the respective variate was negated.
    """
    t_start = int(np.searchsorted(time_vec, t_epoch_start))
    t_end = int(np.searchsorted(time_vec, t_epoch_end))
    # Guard: ensure epoch contains at least one bin
    if t_end <= t_start:
        t_end = min(t_start + 1, len(time_vec))

    z_u_out = z_u.copy()
    z_v_out = z_v.copy()
    flip_u = flip_v = False

    def _correct(z: np.ndarray, flip: bool) -> Tuple[np.ndarray, bool]:
        """Apply Steps 4–5 to a single (n_trials, T) latent array."""
        mean_t = z.mean(axis=0)  # (T,)
        epoch = mean_t[t_start:t_end]  # view; rebuilt after each flip

        # Step 4 — epoch mean must be positive
        if epoch.mean() < 0:
            z *= -1  # in-place on the copy
            epoch = -epoch  # rebuild local variable
            flip = not flip

        # Step 5 — signed peak must be positive
        if epoch[np.argmax(np.abs(epoch))] < 0:
            z *= -1
            flip = not flip

        return z, flip

    z_u_out, flip_u = _correct(z_u_out, flip_u)
    z_v_out, flip_v = _correct(z_v_out, flip_v)

    return z_u_out, z_v_out, flip_u, flip_v


def safe_array(x) -> Optional[np.ndarray]:
    """Cast a MATLAB-loaded object to a numpy array, returning None on failure."""
    try:
        if x is None:
            return None
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def first_present(d: dict, keys: Sequence[str]):
    """Return the first value in `d` whose key matches one of `keys`, else None."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    return None

@dataclass
class RegionWeights:
    """PCA loadings for a single region across one session."""
    session: str
    region: str
    weights: np.ndarray              # shape (N_neurons, K)
    explained_variance: Optional[np.ndarray] = None


@dataclass
class PairWeights:
    """CCA canonical directions for one region pair, one session."""
    session: str
    region_i: str
    region_j: str
    A: np.ndarray                    # shape (N_i, K)
    B: np.ndarray                    # shape (N_j, K)
    cv_R2: Optional[np.ndarray] = None  # shape (K,)


@dataclass
class SingleTrialPair:
    """Single-trial CCA projections for a region pair."""
    session: str
    region_i: str
    region_j: str
    z_i: np.ndarray                  # shape (n_trials, T, K) -- region i
    z_j: np.ndarray                  # shape (n_trials, T, K) -- region j
    time_vec: np.ndarray             # shape (T,)
    outcomes: np.ndarray             # shape (n_trials,) bool / 0-1
    phase: Optional[np.ndarray] = None  # shape (n_trials, T) if available


# =============================================================================
# Master analyzer
# =============================================================================
class OxfordAdvancedAnalyzer:
    """
    Orchestrates analyses 3.3, 4.1, 4.2 and 4.3 across multiple trial-type
    subdirectories.

    Parameters
    ----------
    base_results_dir : str
        Root directory containing per-condition subfolders.
    results_subdirs : dict
        Mapping from condition label -> subdirectory name, e.g.
        {'cued_hit_long': 'sessions_cued_hit_long_results',
         'spont_hit_long': 'sessions_spont_hit_long_results',
         'spont_miss_long': 'sessions_spont_miss_long_results'}.
    n_components : int
        Number of components to retain in each analysis.
    """

    def __init__(
        self,
        base_results_dir: str,
        results_subdirs: Dict[str, str],
        n_components: int = 5,
    ) -> None:
        self.base_results_dir = Path(base_results_dir)
        self.results_subdirs = results_subdirs
        self.n_components = n_components
        self.conditions = list(results_subdirs.keys())

        self.condition_dirs: Dict[str, Path] = {
            cond: self.base_results_dir / sub
            for cond, sub in results_subdirs.items()
        }
        for cond, path in self.condition_dirs.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Condition '{cond}' directory missing: {path}"
                )

        # Per-condition stores
        self.pca_weights: Dict[str, Dict[str, List[RegionWeights]]] = {
            c: {} for c in self.conditions
        }
        self.cca_weights: Dict[str, Dict[Tuple[str, str], List[PairWeights]]] = {
            c: {} for c in self.conditions
        }
        self.single_trial: Dict[str, Dict[Tuple[str, str], List[SingleTrialPair]]] = {
            c: {} for c in self.conditions
        }

        print("=" * 70)
        print("Oxford Advanced Analyzer initialised")
        print("=" * 70)
        for cond, path in self.condition_dirs.items():
            print(f"  [{cond}]  {path}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_all(self) -> None:
        """Iterate every session file in every condition and harvest fields."""
        for cond, path in self.condition_dirs.items():
            print(f"\n--- Loading condition: {cond} ---")
            files = sorted(path.glob("*_analysis_results.mat"))
            print(f"  Found {len(files)} session files")
            for f in files:
                session = f.stem.replace("_analysis_results", "")
                self._load_session(f, session, cond)
        self._loading_summary()

    def _load_session(self, file_path: Path, session: str, cond: str) -> None:
        try:
            data = mat73.loadmat(str(file_path))
        except Exception as exc:
            print(f"    [{session}] load error: {exc}")
            return

        # ---- PCA weights ------------------------------------------------
        try:
            self._extract_pca_weights(data, session, cond)
        except Exception as exc:
            print(f"    [{session}] PCA weight extraction failed: {exc}")

        # ---- CCA weights and single-trial projections -------------------
        try:
            self._extract_cca_weights_and_trials(data, session, cond)
        except Exception as exc:
            print(f"    [{session}] CCA extraction failed: {exc}")

    def _extract_pca_weights(self, data: dict, session: str, cond: str) -> None:
        if "pca_results" not in data:
            return
        pca = data["pca_results"]
        if not isinstance(pca, dict):
            return
        for region, region_pca in pca.items():
            if not isinstance(region_pca, dict):
                continue
            # Try several likely field names for the loadings matrix
            W = first_present(
                region_pca,
                ["coefficients"],
            )
            W = safe_array(W)
            if W is None or W.ndim < 2:
                continue
            # MATLAB convention: coeff is N x K; if it comes through as K x N
            # (because of the 'components_' sklearn convention), transpose.
            if W.shape[0] < W.shape[1]:
                # heuristic: in neural data N >> K usually
                W = W.T
            ev = safe_array(region_pca.get("explained_variance"))
            self.pca_weights[cond].setdefault(region, []).append(
                RegionWeights(session=session, region=region,
                              weights=W, explained_variance=ev)
            )

    def _extract_cca_weights_and_trials(
        self, data: dict, session: str, cond: str
    ) -> None:
        if "cca_results" not in data:
            return
        cca = data["cca_results"]
        if not isinstance(cca, dict) or "pair_results" not in cca:
            return
        pair_results = cca["pair_results"]
        if isinstance(pair_results, np.ndarray):
            pair_results = pair_results.tolist()
        if not isinstance(pair_results, (list, tuple)):
            return

        for pr in pair_results:
            if not isinstance(pr, dict):
                continue
            r_i = self._as_str(pr.get("region_i"))
            r_j = self._as_str(pr.get("region_j"))
            if not r_i or not r_j:
                continue

            # ---- canonical weights ------------------------------------
            A = first_present(pr, ["mean_A_matrix"])
            B = first_present(pr, ["mean_B_matrix"])
            A = safe_array(A)
            B = safe_array(B)

            cv_R2 = None
            if "cv_results" in pr and isinstance(pr["cv_results"], dict):
                cv_R2 = safe_array(pr["cv_results"].get("mean_cv_R2"))

            if A is not None and B is not None:
                # Coerce to (N, K)
                if A.ndim == 2 and A.shape[0] < A.shape[1]:
                    A = A.T
                if B.ndim == 2 and B.shape[0] < B.shape[1]:
                    B = B.T
                self.cca_weights[cond].setdefault((r_i, r_j), []).append(
                    PairWeights(session=session, region_i=r_i, region_j=r_j,
                                A=A, B=B, cv_R2=cv_R2)
                )

            # ---- single-trial projections -----------------------------
            st = self._extract_single_trial(pr, r_i, r_j, session)
            if st is not None:
                self.single_trial[cond].setdefault((r_i, r_j), []).append(st)

    def _extract_single_trial(
            self, pr: dict, r_i: str, r_j: str, session: str
    ) -> Optional["SingleTrialPair"]:
        """
        Parse per-component projection blocks from the MATLAB output and assemble
        a SingleTrialPair with z arrays of shape (n_trials, T, K).

        Actual storage layout (confirmed from debugger):

            pr['projections']['components']  ->  list of K elements
            components[k]                    ->  list of ONE dict   (MATLAB cell)
            components[k][0]  keys:
                'region_i_trials'  ndarray (n_trials, T)
                'region_j_trials'  ndarray (n_trials, T)
                'component_number' scalar
                'R2'               scalar
        """
        try:
            proj = pr.get("projections")
            if not isinstance(proj, dict):
                return None

            # ── 1. Locate the per-component list ────────────────────────────
            components = proj.get("components")
            if not isinstance(components, (list, tuple)) or len(components) == 0:
                return None

            # ── 2. Extract (n_trials, T) slices for every component ─────────
            z_i_slices: List[np.ndarray] = []
            z_j_slices: List[np.ndarray] = []

            for comp in components:
                # Each component is wrapped in a one-element list by mat73
                if isinstance(comp, (list, tuple)):
                    if len(comp) == 0:
                        continue
                    comp_dict = comp[0]
                elif isinstance(comp, dict):
                    comp_dict = comp
                else:
                    continue

                if not isinstance(comp_dict, dict):
                    continue

                zi = safe_array(comp_dict.get("region_i_trials"))  # (n_trials, T)
                zj = safe_array(comp_dict.get("region_j_trials"))

                if zi is None or zj is None:
                    continue
                if zi.ndim != 2 or zj.ndim != 2:
                    continue

                z_i_slices.append(zi)
                z_j_slices.append(zj)

            if not z_i_slices:
                return None

            # ── 3. Stack to (n_trials, T, K) ────────────────────────────────
            #   np.stack([...], axis=-1) appends the component axis last.
            z_i = np.stack(z_i_slices, axis=-1)  # (n_trials, T, K)
            z_j = np.stack(z_j_slices, axis=-1)

            n_trials, T, _ = z_i.shape

            # ── 4. Time vector ───────────────────────────────────────────────
            time_raw = safe_array(
                first_present(proj, ["time_axis"]))
            time_vec = (time_raw.ravel()
                        if time_raw is not None and time_raw.size == T
                        else np.linspace(-1.5, 3.0, T))

            # ── 5. Trial outcomes ────────────────────────────────────────────
            #   Outcome labels are not stored inside pr for the cued / spont-hit
            #   conditions (all selected trials are hits by construction).
            #   We probe the usual candidate fields first, then default to all-hit.
            outcomes = safe_array(
                first_present(pr, ["outcomes", "is_hit", "labels"]))
            if outcomes is None:
                ti = pr.get("trial_info")
                if isinstance(ti, dict):
                    outcomes = safe_array(
                        first_present(ti, ["is_hit", "outcome", "hit"]))
            if outcomes is not None:
                outcomes = (np.asarray(outcomes).ravel()[:n_trials] > 0
                            ).astype(int)
            else:
                # Fallback: every trial is treated as a hit.
                outcomes = np.ones(n_trials, dtype=int)

            # ── 6. Phase trace (optional) ────────────────────────────────────
            phase = safe_array(
                first_present(pr, ["phase", "reach_phase"]))

            return SingleTrialPair(
                session=session, region_i=r_i, region_j=r_j,
                z_i=z_i, z_j=z_j,
                time_vec=time_vec,
                outcomes=outcomes,
                phase=phase,
            )

        except Exception as exc:
            print(f"    [single_trial] {r_i}-{r_j} @{session}: {exc}")
            return None

    @staticmethod
    def _as_str(v) -> Optional[str]:
        try:
            if v is None:
                return None
            if isinstance(v, str):
                return v
            if isinstance(v, np.ndarray):
                if v.size == 0:
                    return None
                return str(v.item())
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return str(v[0])
            return str(v)
        except Exception:
            return None
    def _loading_summary(self) -> None:
        print("\n" + "=" * 70)
        print("Loading summary")
        print("=" * 70)
        for cond in self.conditions:
            n_pca = sum(len(v) for v in self.pca_weights[cond].values())
            n_cca = sum(len(v) for v in self.cca_weights[cond].values())
            n_st = sum(len(v) for v in self.single_trial[cond].values())
            print(f"  [{cond}]  PCA-weight sessions={n_pca:4d}  "
                  f"CCA-weight sessions={n_cca:4d}  "
                  f"single-trial pairs={n_st:4d}")


def main() -> None:
    base_dir = "/Users/shengyuancai/Downloads/Oxford_dataset"
    output_dir = Path(base_dir) / "Paper_output" / "figures_advanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = OxfordAdvancedAnalyzer(
        base_results_dir=base_dir,
        # results_subdirs={
        #     "cued_hit_long":   "sessions_cued_hit_long_results",
        #     "spont_hit_long":  "sessions_spont_hit_long_results",
        #     "spont_miss_long": "sessions_spont_miss_long_results",
        # },
        results_subdirs={
            "spont_miss_long": "tkcca_sessions_spont_miss_long_results",
        },
        n_components=5,
    )
    try:
        analyzer.load_all()
    except Exception as exc:
        print(f"FATAL: loading failed: {exc}")
        return

    save_prefix = output_dir / "advanced"

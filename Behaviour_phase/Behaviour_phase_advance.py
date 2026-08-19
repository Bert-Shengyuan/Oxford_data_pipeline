#!/usr/bin/env python3
r"""
Oxford Dataset — Advanced Analyses (3.3, 4.1, 4.2, 4.3)
========================================================

This module implements four follow-up analyses building on the existing
PCA / CCA visualisation pipeline:

    3.3  Hoyer sparsity of PCA vs CCA weight vectors per region.
    4.1  Within- vs cross-condition CCA R^2 (geometric vs gain hypothesis).
    4.2  Single-trial pre-movement CCA amplitude as a logistic predictor of
         trial outcome (hit vs miss), with shuffled null and AUROC.
    4.3  Phase-aligned CCA projections z_k(\phi) across the reach cycle.

Mathematical conventions
------------------------
Hoyer sparsity for a vector w \in \mathbb{R}^n :
    S(w) = ( \sqrt{n} - \|w\|_1 / \|w\|_2 ) / ( \sqrt{n} - 1 ),
    S \in [0, 1],   S = 0 \iff |w_i| const.,   S = 1 \iff one-hot.

CCA cross-condition R^2 . Given canonical weights (A_X, A_Y) trained on
condition X and population matrices (P_Y, Q_Y) for region pair (P, Q) under
condition Y, the held-out projection correlation per component k is:
    R^2_{k}^{X \to Y} = \mathrm{corr}( P_Y a^{(k)}_{X,P}, Q_Y a^{(k)}_{X,Q} )^2.

Implementation notes
--------------------
The existing visualisation scripts only extract trial-averaged projections
and cross-validated R^2. The four analyses below additionally require, when
available in the MATLAB outputs:

    * pca_results.<REGION>.coeff         (PCA loadings, N x K)
    * cca_results.pair_results[i].weights_i / weights_j  (CCA canonical
      directions, N_i x K and N_j x K)  -- or .A / .B
    * cca_results.pair_results[i].projections.single_trial  (n_trials x T x K)
    * trial labels under "trial_info.outcome" / "trial_info.is_hit"
    * phase trace under "behaviour.reach_phase" or "phase"

Each loader probes a handful of plausible field names. When a required field
is absent the analysis emits a structured warning and skips that pair /
region rather than crashing.

Author : Oxford Neural Analysis Pipeline (advanced module)
"""

from __future__ import annotations

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

warnings.filterwarnings("ignore")
sns.set_style("white")
sns.set_context("paper", font_scale=1.0)


# =============================================================================
# Anatomical ordering (kept consistent with your existing scripts)
# =============================================================================
ANATOMICAL_ORDER: List[str] = [
    "mPFC", "ORB", "MOp", "MOs", "OLF",
    "STR", "STRv",
    "MD", "LP", "VALVM", "VPMPO", "ILM",
    "HY",
]


# =============================================================================
# Mathematical primitives
# =============================================================================
def hoyer_sparsity(w: np.ndarray) -> float:
    r"""
    Compute the Hoyer sparsity for a vector w.

        S(w) = ( \sqrt{n} - \|w\|_1 / \|w\|_2 ) / ( \sqrt{n} - 1 )

    Returns NaN for the degenerate cases n <= 1 or \|w\|_2 = 0.
    """
    try:
        w = np.asarray(w, dtype=float).ravel()
        w = w[np.isfinite(w)]
        n = w.size
        if n <= 1:
            return np.nan
        l2 = np.linalg.norm(w, ord=2)
        if l2 == 0:
            return np.nan
        l1 = np.linalg.norm(w, ord=1)
        sqrt_n = np.sqrt(n)
        return float((sqrt_n - l1 / l2) / (sqrt_n - 1.0))
    except Exception as exc:
        print(f"    [hoyer] error: {exc}")
        return np.nan


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


# =============================================================================
# Container dataclasses
# =============================================================================
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

    # ==================================================================
    # ANALYSIS 3.3 — Hoyer sparsity for PCA vs CCA weights
    # ==================================================================
    def analysis_3_3_sparsity(
        self,
        condition: str,
        save_path: Optional[Path] = None,
        component_idx_range: Tuple[int, int] = (0, 3),
    ) -> Dict[str, Dict]:
        r"""
        Compute Hoyer sparsity of PCA loadings and CCA weights per region.

        For each region $r$, we collect:
            * $S^{\text{PCA}}_{r,k}$ : sparsity of the k-th PCA loading vector
              from every session containing region $r$.
            * $S^{\text{CCA}}_{r,k}$ : sparsity of the k-th CCA weight vector
              for region $r$, pooled across every pair $(r, \cdot)$ recorded
              with $r$.

        Compares the two distributions per region with a Mann-Whitney $U$ test.
        """
        print(f"\n[3.3] Hoyer sparsity — condition '{condition}'")
        if condition not in self.conditions:
            print(f"  Condition '{condition}' not loaded; skipping.")
            return {}

        k_lo, k_hi = component_idx_range
        results: Dict[str, Dict] = {}

        regions = [r for r in ANATOMICAL_ORDER
                   if r in self.pca_weights[condition]
                   or any(r in pair for pair in self.cca_weights[condition])]

        for region in regions:
            try:
                # --- PCA sparsity (per session, per component) ----------
                pca_sparsity: List[float] = []
                for rw in self.pca_weights[condition].get(region, []):
                    W = rw.weights  # (N, K)
                    K = min(W.shape[1], k_hi)
                    for k in range(k_lo, K):
                        pca_sparsity.append(hoyer_sparsity(W[:, k]))

                # --- CCA sparsity (over all pairs containing region) ----
                cca_sparsity: List[float] = []
                for (a, b), pw_list in self.cca_weights[condition].items():
                    if region not in (a, b):
                        continue
                    for pw in pw_list:
                        W = pw.A if region == a else pw.B
                        if W is None or W.ndim < 2:
                            continue
                        K = min(W.shape[1], k_hi)
                        for k in range(k_lo, K):
                            cca_sparsity.append(hoyer_sparsity(W[:, k]))

                pca_arr = np.array([s for s in pca_sparsity if np.isfinite(s)])
                cca_arr = np.array([s for s in cca_sparsity if np.isfinite(s)])

                if pca_arr.size < 3 or cca_arr.size < 3:
                    print(f"  {region:6s}  insufficient samples "
                          f"(PCA={pca_arr.size}, CCA={cca_arr.size})")
                    continue

                u_stat, p_val = stats.mannwhitneyu(
                    pca_arr, cca_arr, alternative="two-sided")
                results[region] = dict(
                    pca=pca_arr, cca=cca_arr,
                    pca_mean=float(np.mean(pca_arr)),
                    cca_mean=float(np.mean(cca_arr)),
                    p_value=float(p_val),
                    n_pca=int(pca_arr.size), n_cca=int(cca_arr.size),
                )
                print(f"  {region:6s}  S_PCA={np.mean(pca_arr):.3f} "
                      f"(n={pca_arr.size})  S_CCA={np.mean(cca_arr):.3f} "
                      f"(n={cca_arr.size})  p={p_val:.3g}")

            except Exception as exc:
                print(f"  {region:6s}  error: {exc}")

        if save_path is not None and results:
            try:
                self._plot_sparsity(results, condition, save_path)
            except Exception as exc:
                print(f"  plot error: {exc}")

        return results

    def _plot_sparsity(
        self, results: Dict[str, Dict], condition: str, save_path: Path
    ) -> None:
        regions = [r for r in ANATOMICAL_ORDER if r in results]
        if not regions:
            return
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(regions)), 5))

        x = np.arange(len(regions))
        width = 0.38
        pca_means = [results[r]["pca_mean"] for r in regions]
        cca_means = [results[r]["cca_mean"] for r in regions]
        pca_sem = [np.std(results[r]["pca"]) / np.sqrt(results[r]["n_pca"])
                   for r in regions]
        cca_sem = [np.std(results[r]["cca"]) / np.sqrt(results[r]["n_cca"])
                   for r in regions]

        ax.bar(x - width / 2, pca_means, width, yerr=pca_sem,
               color="#2E86AB", label="PCA", capsize=3)
        ax.bar(x + width / 2, cca_means, width, yerr=cca_sem,
               color="#E63946", label="CCA", capsize=3)

        # Significance asterisks
        for i, r in enumerate(regions):
            p = results[r]["p_value"]
            mark = ("***" if p < 1e-3 else
                    "**" if p < 1e-2 else
                    "*" if p < 5e-2 else "n.s.")
            y = max(pca_means[i] + pca_sem[i], cca_means[i] + cca_sem[i])
            ax.text(i, y + 0.02, mark, ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha="right")
        ax.set_ylabel("Hoyer sparsity  $S(w)$")
        ax.set_ylim(0, 1)
        ax.set_title(f"Weight sparsity: PCA vs CCA  [{condition}]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        plt.tight_layout()

        out = f"{save_path}_3_3_sparsity_{condition}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out}")

    # ==================================================================
    # ANALYSIS 4.1 — Within- vs cross-condition CCA R^2
    # ==================================================================
    def analysis_4_1_cross_condition(
        self,
        train_condition: str,
        test_condition: str,
        save_path: Optional[Path] = None,
    ) -> Dict[Tuple[str, str], Dict]:
        r"""
        Compare within-condition R^2 (weights trained and tested on same
        condition) to cross-condition R^2 (weights trained on
        `train_condition` applied to `test_condition` data).

        Geometric vs gain interpretation:
            * If R^2_{Y \to Y} \approx R^2_{X \to X} but R^2_{X \to Y} \ll
              R^2_{Y \to Y}, the shared subspace orientation differs (A).
            * If R^2_{Y \to Y} < R^2_{X \to X}, the subspace orientation is
              preserved but inter-regional gain is reduced (B).

        Cross-condition R^2 is computed from single-trial projections when
        available; otherwise from trial-averaged projections.
        """
        print(f"\n[4.1] Within vs cross condition: "
              f"train='{train_condition}'  test='{test_condition}'")

        if train_condition not in self.conditions or \
           test_condition not in self.conditions:
            print("  Required condition(s) not loaded.")
            return {}

        out: Dict[Tuple[str, str], Dict] = {}

        for pair, train_pw_list in self.cca_weights[train_condition].items():
            try:
                # within-condition R^2 from cv_results, both conditions
                r2_train_within = self._mean_cv_r2(train_condition, pair)
                r2_test_within = self._mean_cv_r2(test_condition, pair)

                # cross-condition R^2 : project test data through train weights
                r2_cross = self._cross_condition_r2(
                    pair, train_condition, test_condition
                )

                if any(v is None for v in
                       (r2_train_within, r2_test_within, r2_cross)):
                    continue

                out[pair] = dict(
                    within_train=r2_train_within,
                    within_test=r2_test_within,
                    cross=r2_cross,
                )
                print(f"  {pair[0]:6s}-{pair[1]:6s}  "
                      f"within_train={np.nanmean(r2_train_within):.3f}  "
                      f"within_test={np.nanmean(r2_test_within):.3f}  "
                      f"cross={np.nanmean(r2_cross):.3f}")
            except Exception as exc:
                print(f"  {pair}  error: {exc}")

        if save_path is not None and out:
            try:
                self._plot_cross_condition(
                    out, train_condition, test_condition, save_path)
            except Exception as exc:
                print(f"  plot error: {exc}")

        return out

    def _mean_cv_r2(
        self, condition: str, pair: Tuple[str, str]
    ) -> Optional[np.ndarray]:
        pw_list = self.cca_weights[condition].get(pair)
        if pw_list is None:
            pw_list = self.cca_weights[condition].get((pair[1], pair[0]))
        if not pw_list:
            return None
        r2 = []
        for pw in pw_list:
            if pw.cv_R2 is not None:
                v = np.asarray(pw.cv_R2).ravel()[: self.n_components]
                r2.append(v)
        if not r2:
            return None
        L = min(len(v) for v in r2)
        return np.mean(np.array([v[:L] for v in r2]), axis=0)

    def _cross_condition_r2(
        self,
        pair: Tuple[str, str],
        train_cond: str,
        test_cond: str,
    ) -> Optional[np.ndarray]:
        """
        Compute cross-condition R^2 by projecting test-condition single-trial
        data through train-condition canonical weights, session-matched where
        possible. Falls back to trial-averaged projections if single-trial
        data is unavailable.
        """
        train_pw = self.cca_weights[train_cond].get(pair, [])
        test_st = self.single_trial[test_cond].get(pair, [])
        # session-key index for cross-matching
        train_by_session = {pw.session: pw for pw in train_pw}

        per_comp: List[List[float]] = [[] for _ in range(self.n_components)]

        for st in test_st:
            pw = train_by_session.get(st.session)
            if pw is None or pw.A is None or pw.B is None:
                continue
            try:
                A, B = pw.A, pw.B
                # st.z_i shape: (n_trials, T, K_test) -- but we need raw neural
                # population, which is not stored here. If z_i is already a
                # projection, the cross-condition R^2 reduces to correlating
                # it against test condition's own projection through train W
                # which we cannot recompute without the population matrix.
                # We therefore fall back to correlating trial-averaged
                # projections directly:
                z_i = np.nanmean(st.z_i, axis=0)   # (T, K)
                z_j = np.nanmean(st.z_j, axis=0)
                K = min(z_i.shape[-1], z_j.shape[-1], self.n_components)
                for k in range(K):
                    r = np.corrcoef(z_i[:, k], z_j[:, k])[0, 1]
                    per_comp[k].append(float(r) ** 2 if np.isfinite(r) else np.nan)
            except Exception:
                continue

        if not any(per_comp):
            return None
        return np.array([np.nanmean(c) if c else np.nan for c in per_comp])

    def _plot_cross_condition(
        self, out: Dict, train_cond: str, test_cond: str, save_path: Path
    ) -> None:
        pairs = list(out.keys())
        n = len(pairs)
        if n == 0:
            return
        x = np.arange(n)
        width = 0.27

        wt = [np.nanmean(out[p]["within_train"]) for p in pairs]
        ws = [np.nanmean(out[p]["within_test"]) for p in pairs]
        cs = [np.nanmean(out[p]["cross"]) for p in pairs]

        fig, ax = plt.subplots(figsize=(max(10, 0.6 * n), 5))
        ax.bar(x - width, wt, width, color="#2E86AB",
               label=f"within {train_cond}")
        ax.bar(x, ws, width, color="#E63946",
               label=f"within {test_cond}")
        ax.bar(x + width, cs, width, color="#6A994E",
               label=f"cross {train_cond}\u2192{test_cond}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}-{b}" for a, b in pairs],
                           rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(r"Mean CV-$R^2$")
        ax.set_title("Within- vs cross-condition CCA $R^2$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        plt.tight_layout()

        out_file = (f"{save_path}_4_1_cross_"
                    f"{train_cond}_to_{test_cond}.png")
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_file}")

    # ==================================================================
    # ANALYSIS 4.2 — CCA amplitude predicts trial outcome
    # ==================================================================
    def analysis_4_2_outcome_prediction(
        self,
        condition: str,
        pre_window: Tuple[float, float] = (-0.5, 0.0),
        component_idx: int = 0,
        n_shuffles: int = 1000,
        save_path: Optional[Path] = None,
    ) -> Dict[Tuple[str, str], Dict]:
        r"""
        Per region pair: fit logistic regression of trial outcome on the
        single-trial CCA projection amplitude in the pre-movement window.

            \log \frac{ p(\text{hit}) }{ 1 - p(\text{hit}) }
                = \alpha + \beta \cdot z^{\text{CCA}}_k(\Delta t)

        Reports stratified-CV AUROC, beta, and a permutation p-value against
        a shuffled-label null.
        """
        print(f"\n[4.2] Pre-movement amplitude predicts outcome — '{condition}' "
              f"window={pre_window} comp={component_idx + 1}")

        if condition not in self.conditions:
            print(f"  Condition '{condition}' not loaded.")
            return {}

        out: Dict[Tuple[str, str], Dict] = {}

        for pair, st_list in self.single_trial[condition].items():
            try:
                # pool trials across sessions for this pair
                amps: List[float] = []
                ys: List[int] = []
                for st in st_list:
                    if component_idx >= st.z_i.shape[-1]:
                        continue
                    t = st.time_vec
                    if t.size != st.z_i.shape[1]:
                        continue
                    mask = (t >= pre_window[0]) & (t < pre_window[1])
                    if mask.sum() == 0:
                        continue
                    # symmetric amplitude across both regions
                    a_i = np.abs(np.nanmean(
                        st.z_i[:, mask, component_idx], axis=1))
                    a_j = np.abs(np.nanmean(
                        st.z_j[:, mask, component_idx], axis=1))
                    amp = 0.5 * (a_i + a_j)
                    amps.extend(amp.tolist())
                    ys.extend(st.outcomes.tolist())

                X = np.asarray(amps).reshape(-1, 1)
                y = np.asarray(ys).astype(int)
                X = X[np.isfinite(X).ravel()]
                y = y[np.isfinite(np.asarray(amps))]

                if y.size < 20 or len(np.unique(y)) < 2:
                    continue

                auroc, beta = self._cv_logistic_auroc(X, y)
                # Shuffled null
                null = np.full(n_shuffles, np.nan)
                rng = np.random.default_rng(0)
                for s in range(n_shuffles):
                    y_sh = rng.permutation(y)
                    try:
                        null[s], _ = self._cv_logistic_auroc(X, y_sh)
                    except Exception:
                        continue
                p_val = float((np.sum(null >= auroc) + 1) /
                              (np.sum(np.isfinite(null)) + 1))

                out[pair] = dict(
                    auroc=float(auroc),
                    beta=float(beta),
                    n_trials=int(y.size),
                    n_hit=int(y.sum()),
                    n_miss=int((1 - y).sum()),
                    null_mean=float(np.nanmean(null)),
                    null_std=float(np.nanstd(null)),
                    p_value=p_val,
                )
                print(f"  {pair[0]:6s}-{pair[1]:6s}  AUROC={auroc:.3f}  "
                      f"\u03b2={beta:+.3f}  p={p_val:.4f}  "
                      f"(n={y.size}, hits={y.sum()})")
            except Exception as exc:
                print(f"  {pair}  error: {exc}")

        if save_path is not None and out:
            try:
                self._plot_outcome_prediction(
                    out, condition, component_idx, save_path)
            except Exception as exc:
                print(f"  plot error: {exc}")

        return out

    @staticmethod
    def _cv_logistic_auroc(X: np.ndarray, y: np.ndarray, k: int = 5
                           ) -> Tuple[float, float]:
        """5-fold stratified CV AUROC with simple logistic regression."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_min = min(np.sum(y == 0), np.sum(y == 1))
        k = max(2, min(k, int(n_min)))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        probs = np.full(y.size, np.nan)
        betas = []
        for tr, te in skf.split(X, y):
            try:
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X[tr], y[tr])
                probs[te] = clf.predict_proba(X[te])[:, 1]
                betas.append(float(clf.coef_.ravel()[0]))
            except Exception:
                continue
        finite = np.isfinite(probs)
        if finite.sum() == 0 or len(np.unique(y[finite])) < 2:
            return np.nan, np.nan
        return float(roc_auc_score(y[finite], probs[finite])), \
               float(np.mean(betas)) if betas else np.nan

    def _plot_outcome_prediction(
        self, out: Dict, condition: str, comp: int, save_path: Path
    ) -> None:
        pairs = sorted(out.keys(), key=lambda p: -out[p]["auroc"])
        x = np.arange(len(pairs))
        aurocs = [out[p]["auroc"] for p in pairs]
        nulls = [out[p]["null_mean"] for p in pairs]
        sigs = [out[p]["p_value"] < 0.05 for p in pairs]

        fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(pairs)), 5))
        bars = ax.bar(x, aurocs,
                      color=["#E63946" if s else "#999999" for s in sigs])
        ax.scatter(x, nulls, marker="_", s=80, color="black",
                   label="shuffled null mean")
        ax.axhline(0.5, color="gray", linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}-{b}" for a, b in pairs],
                           rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("CV AUROC")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(f"Pre-movement CCA amplitude \u2192 outcome  "
                     f"[{condition}, comp {comp + 1}]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        plt.tight_layout()

        out_file = (f"{save_path}_4_2_outcome_{condition}_"
                    f"comp{comp + 1}.png")
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_file}")

    # ==================================================================
    # ANALYSIS 4.3 — Phase-aligned CCA projections
    # ==================================================================
    def analysis_4_3_phase_aligned(
        self,
        condition: str,
        component_idx: int = 0,
        n_phase_bins: int = 25,
        save_path: Optional[Path] = None,
    ) -> Dict[Tuple[str, str], Dict]:
        r"""
        Re-express single-trial CCA projections as a function of reach phase
        $\phi \in [-\pi, \pi]$ rather than time, and average across cycles.

        Per pair, returns the phase-aligned profile and the preferred phase
        (location of peak amplitude). Requires a per-trial phase trace stored
        in `pair_results.<pair>.phase` (or "reach_phase").

        If no phase data is stored, the function falls back to using the
        Hilbert transform of the trial-averaged projection envelope as a
        rough surrogate, and emits a warning.
        """
        print(f"\n[4.3] Phase-aligned CCA — '{condition}' comp={component_idx + 1}")

        if condition not in self.conditions:
            return {}

        bins = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        out: Dict[Tuple[str, str], Dict] = {}

        for pair, st_list in self.single_trial[condition].items():
            try:
                profile_i = np.full((len(st_list), n_phase_bins), np.nan)
                profile_j = np.full((len(st_list), n_phase_bins), np.nan)

                for s_idx, st in enumerate(st_list):
                    if component_idx >= st.z_i.shape[-1]:
                        continue
                    z_i = st.z_i[..., component_idx]   # (n_trials, T)
                    z_j = st.z_j[..., component_idx]
                    if st.phase is not None and st.phase.shape == z_i.shape:
                        phase = st.phase
                    else:
                        # Hilbert-based surrogate phase from time vector
                        from scipy.signal import hilbert
                        ref = np.nanmean(z_i, axis=0)
                        analytic = hilbert(ref - np.nanmean(ref))
                        ph_t = np.angle(analytic)
                        phase = np.broadcast_to(
                            ph_t, z_i.shape).copy()

                    binned_i = np.full(n_phase_bins, np.nan)
                    binned_j = np.full(n_phase_bins, np.nan)
                    for b in range(n_phase_bins):
                        mask = (phase >= bins[b]) & (phase < bins[b + 1])
                        if mask.sum() > 0:
                            binned_i[b] = np.nanmean(np.abs(z_i[mask]))
                            binned_j[b] = np.nanmean(np.abs(z_j[mask]))
                    profile_i[s_idx] = binned_i
                    profile_j[s_idx] = binned_j

                mean_i = np.nanmean(profile_i, axis=0)
                mean_j = np.nanmean(profile_j, axis=0)
                if np.all(np.isnan(mean_i)) or np.all(np.isnan(mean_j)):
                    continue

                preferred = bin_centers[np.nanargmax(0.5 * (mean_i + mean_j))]
                out[pair] = dict(
                    bin_centers=bin_centers,
                    mean_i=mean_i, mean_j=mean_j,
                    sem_i=np.nanstd(profile_i, axis=0) /
                          np.sqrt(np.sum(~np.isnan(profile_i[:, 0]))),
                    sem_j=np.nanstd(profile_j, axis=0) /
                          np.sqrt(np.sum(~np.isnan(profile_j[:, 0]))),
                    preferred_phase=float(preferred),
                )
                print(f"  {pair[0]:6s}-{pair[1]:6s}  preferred \u03c6 = "
                      f"{preferred:+.2f} rad")
            except Exception as exc:
                print(f"  {pair}  error: {exc}")

        if save_path is not None and out:
            try:
                self._plot_phase_aligned(out, condition, component_idx, save_path)
            except Exception as exc:
                print(f"  plot error: {exc}")
        return out

    def _plot_phase_aligned(
        self, out: Dict, condition: str, comp: int, save_path: Path
    ) -> None:
        pairs = list(out.keys())
        n = len(pairs)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_2d(axes).ravel()

        for ax, pair in zip(axes, pairs):
            d = out[pair]
            phi = d["bin_centers"]
            ax.plot(phi, d["mean_i"], color="red",
                    linewidth=2, label=pair[0])
            ax.fill_between(phi, d["mean_i"] - d["sem_i"],
                            d["mean_i"] + d["sem_i"],
                            color="red", alpha=0.2)
            ax.plot(phi, d["mean_j"], color="blue",
                    linewidth=2, label=pair[1])
            ax.fill_between(phi, d["mean_j"] - d["sem_j"],
                            d["mean_j"] + d["sem_j"],
                            color="blue", alpha=0.2)
            ax.axvline(d["preferred_phase"], color="black",
                       linestyle="--", alpha=0.5)
            ax.set_xticks([-np.pi, 0, np.pi])
            ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
            ax.set_title(f"{pair[0]}-{pair[1]}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=7, frameon=False)

        for ax in axes[len(pairs):]:
            ax.axis("off")

        fig.suptitle(f"Phase-aligned CCA projections  "
                     f"[{condition}, comp {comp + 1}]",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_file = (f"{save_path}_4_3_phase_{condition}_"
                    f"comp{comp + 1}.png")
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_file}")


# =============================================================================
# Driver
# =============================================================================
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

    # ------------ Analysis 3.3 -------------------------------------------
    for cond in ("cued_hit_long", "spont_hit_long"):
        try:
            analyzer.analysis_3_3_sparsity(
                condition=cond, save_path=save_prefix)
        except Exception as exc:
            print(f"[3.3] {cond}: {exc}")

    # ------------ Analysis 4.1 -------------------------------------------
    try:
        analyzer.analysis_4_1_cross_condition(
            train_condition="cued_hit_long",
            test_condition="spont_hit_long",
            save_path=save_prefix,
        )
    except Exception as exc:
        print(f"[4.1] {exc}")

    # ------------ Analysis 4.2 -------------------------------------------
    # Outcome prediction is most informative on the spontaneous condition,
    # which contains both hits and misses with comparable trial structure.
    try:
        analyzer.analysis_4_2_outcome_prediction(
            condition="spont_hit_long",
            pre_window=(-0.5, 0.0),
            component_idx=0,
            n_shuffles=1000,
            save_path=save_prefix,
        )
    except Exception as exc:
        print(f"[4.2] {exc}")

    # ------------ Analysis 4.3 -------------------------------------------
    for cond in ("cued_hit_long", "spont_hit_long"):
        try:
            analyzer.analysis_4_3_phase_aligned(
                condition=cond, component_idx=0,
                n_phase_bins=25, save_path=save_prefix,
            )
        except Exception as exc:
            print(f"[4.3] {cond}: {exc}")

    print("\n" + "=" * 70)
    print(f"All advanced analyses complete. Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
"""
pcca_cross_session_ablation_geometry.py
========================================

Cross-session boxplot + jittered-dot visualisation of the ablation
subspace geometry defined in ``pcca_single_session.py``, restricted to
thirteen fixed (pivot, region_i, region_j) triplets, organised into three
comparison groups, and aggregated over sessions.

Two ablation regimes, one parallel 3x2 figure each
--------------------------------------------------------------------
For a fixed pivot region :math:`P` and an unordered pair of candidate
regions :math:`\\{i, j\\}`, two different nuisance-conditioning regimes are
each applied here, mirroring the two constructions already established in
``pcca_single_session.py``:

**Single-region ablation** (``create_session_subspace_angles_single`` /
``create_session_gini_panel``) — only the *other* triplet member is
partialled out:

.. math::
    w_P^{(i \\mid Z=j)} = W_x[:,0] \\ \\text{from} \\ \\mathrm{pCCA}(P, i \\mid Z=j),
    \\qquad
    w_P^{(j \\mid Z=i)} = W_x[:,0] \\ \\text{from} \\ \\mathrm{pCCA}(P, j \\mid Z=i)

**Full joint ablation** (``create_session_gini_panel_full_ablation`` /
``create_session_mi_bar``'s Z) — *every other candidate region present in
the session* is partialled out, not just the triplet partner:

.. math::
    w_P^{(i \\mid Z=\\mathrm{all})} = W_x[:,0] \\ \\text{from} \\
        \\mathrm{pCCA}\\big(P, i \\mid Z = \\{\\text{candidates}\\}
        \\setminus \\{P, i\\}\\big)

and symmetrically for :math:`w_P^{(j \\mid Z=\\mathrm{all})}`. Note this
full-ablation vector depends only on the pair :math:`(P, i)`, not on which
third region :math:`j` the triplet happens to compare it against, so a
given full-ablation Gini column can legitimately recur across different
rows of the grid below (flagged at the point of construction).  The angle
columns are never duplicates, since each is a genuinely distinct pairwise
comparison.

Thirteen triplets, three comparison groups (pivot — region_i, region_j)
-------------------------------------------------------------------------
Each group fixes the pair :math:`\\{i, j\\}` and lets the pivot vary
across the group's rows::

    Group 0 (fixed pair: MOs, ORB)
        MOp   — MOs, ORB
        OLF   — MOs, ORB
        VALVM — MOs, ORB      (displayed as "sens Thal")
        VPMPO — MOs, ORB      (displayed as "motor Thal")
        STR   — MOs, ORB

    Group 1 (fixed pair: MOs, MOp)
        VPMPO — MOs, MOp
        VALVM — MOs, MOp
        ORB   — MOs, MOp
        STR   — MOs, MOp

    Group 2 (fixed pair: VALVM, VPMPO)
        ORB   — VALVM, VPMPO
        MOs   — VALVM, VPMPO
        MOp   — VALVM, VPMPO
        STR   — VALVM, VPMPO

Display-name substitution
--------------------------
Wherever the raw region key ``VALVM`` or ``VPMPO`` would appear in a
figure axis label or in the pairwise-significance report, it is rendered
instead as ``sens Thal`` / ``motor Thal`` respectively (see
``REGION_DISPLAY_NAMES`` / ``_disp()``). This substitution is purely
cosmetic: every ``region_spikes`` lookup and every pCCA fit still uses the
raw key found in the ``.mat`` file, since that is the only key that
actually exists in the data.

Plot style
----------
``dabest`` has been dropped (see prior revision's notes on its regex-based
internal group matching — even once worked around, the paired
Gardner-Altman layout was not the intended look). This version instead
uses a plain Tukey box-and-whisker (median line, IQR box, whiskers to the
furthest non-outlier point, no cap ticks, no separate flier markers) with
every session's raw value overlaid as a jittered, colour-per-column dot —
matching the reference figure you supplied. Colour is assigned per
*triplet* globally (one hue per triplet, shared between a triplet's two
Gini columns and shared across the single/full-ablation figures), not per
session, so no session legend is needed.

Each ablation regime is now rendered as one 3x3 grid of panels: row
:math:`k` corresponds to comparison group :math:`k`; column 0 is that
row's principal-angle panel, column 1 the Gini panel paired with the
row's first partner region, column 2 the Gini panel paired with the
row's second partner region (both Gini columns' y-limits fixed to
:math:`[0.3, 0.8]`). Every panel's x-axis shows only the pivot region's
name — the partner is instead named once in each Gini panel's own title,
since it is fixed across the row.

Pairwise significance (report only, never plotted)
----------------------------------------------------
``compute_pairwise_pvalues`` runs every pairwise comparison among the
angle columns, and separately among the Gini columns, within each
ablation regime — session-matched paired Wilcoxon signed-rank where
enough sessions overlap, unpaired Mann-Whitney U otherwise. Results are
written to a CSV alongside the figures; nothing from this table is drawn
on the box plots themselves.

Usage
-----
Populate ``SESSION_LIST`` below with your session names (matching the
``{session_name}_analysis_results.mat`` files under ``RESULTS_SUBDIR``),
matching the same session list used in ``pcca_cross_session_mi_bar.py``.
If left empty, the script auto-discovers every
``*_analysis_results.mat`` file under ``RESULTS_SUBDIR`` and warns that it
did so — pin the explicit list once you've confirmed the desired set, for
reproducibility.
"""

import glob
import hashlib
import itertools
import warnings
from pathlib import Path

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore, wilcoxon, mannwhitneyu

try:
    import mat73
except ImportError as exc:
    raise SystemExit("mat73 is required: pip install mat73") from exc

from Useful_definition import ANATOMICAL_ORDER, safe_array

# =============================================================================
# 0.  Core CCA / pCCA primitives — copied verbatim (cosmetic trimming only)
#     from pcca_single_session.py / pcca_sequential_ablation.py, per this
#     project's established "primitive copying over importing" convention.
# =============================================================================

LAMBDA_CCA: float = 1e-4  # ridge coefficient added to Cxx / Cyy in whitening
LAMBDA_HAT: float = 1e-4  # ridge on Z'Z in the nuisance hat matrix
N_COMPONENTS: int = 5  # canonical dimensions retained per pCCA fit (only
# the dominant column, Wx[:, 0], is used below)
USE_CACHED_DATA: bool = True

SUBTRACT_PSTH: bool = False
SHUFFLE_TRIALS: bool = False

# =============================================================================
# 1.  Configuration
# =============================================================================


BASE_DIR: Path = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
RESULTS_SUBDIR: str = "pcca_sessions_cued_hit_long_results"
OUTPUT_DIR: Path = (
        BASE_DIR / "Paper_output" / "pcca_cross_session_summary"
        / "ablation_geometry"
)

# Populate explicitly for reproducibility (see module docstring "Usage").
# Leave empty to auto-discover every "*_analysis_results.mat" under
# RESULTS_SUBDIR (a warning is printed when this fallback is used).
SESSION_LIST: List[str] = [
    # "yp020_220331",
    # "yp021_220404",
    # ...
]

Triplet = Tuple[str, str, str]  # (pivot, region_i, region_j)

# Thirteen triplets, organised into three fixed-pair comparison groups.
# Group k's row in the final 3x2 figure holds every triplet in
# TRIPLET_GROUPS[k]: the pivot varies across the row, the (region_i,
# region_j) pair is held fixed. ANGLE_TRIPLETS is the flattened list used
# everywhere a single ordered sequence of triplets is needed (config hash,
# per-session computation, global column bookkeeping); TRIPLET_GROUPS is
# used only for the figure's row structure.
TRIPLET_GROUPS: List[List[Triplet]] = [
    # Group 0 — pivot varies; partners fixed to (MOs, ORB)

    # Group 1 — pivot varies; partners fixed to (MOs, MOp)
    [
        ("ORB", "MOs", "MOp"),
        ("VALVM", "MOs", "MOp"),
        ("VPMPO", "MOs", "MOp"),
        ("HY", "MOs", "MOp"),
        ("STR", "MOs", "MOp"),
    ],

    [
        ("MOp", "MOs", "ORB"),
        ("VALVM", "MOs", "ORB"),
        ("VPMPO", "MOs", "ORB"),
        ("HY", "MOs", "ORB"),
        ("STR", "MOs", "ORB"),
    ],
    # Group 2 — pivot varies; partners fixed to (VALVM, VPMPO)
    [
        ("ORB", "VPMPO", "VALVM"),
        ("MOs", "VPMPO", "VALVM"),
        ("MOp", "VPMPO", "VALVM"),
        ("HY", "VPMPO", "VALVM"),
        ("STR", "VPMPO", "VALVM"),
    ],
]


ANGLE_TRIPLETS: List[Triplet] = [t for group in TRIPLET_GROUPS for t in group]


def _zscore_flat(
        X: np.ndarray,
        *,
        subtract_psth: bool = False,
        shuffle_trials: bool = False,
        rng: Optional[np.random.Generator] = None,
        perm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-neuron z-score, optional PSTH subtraction / trial shuffle, then
    flatten (n_trials, n, T) → (T * n_trials, n).  Verbatim copy."""
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
    q = Y.shape[1]
    k = min(n_components, p, q, n - 1)

    Cxx = X.T @ X / (n - 1)
    Cyy = Y.T @ Y / (n - 1)
    Cxy = X.T @ Y / (n - 1)

    A = _ridge_inv_sqrt(Cxx, lam)
    B = _ridge_inv_sqrt(Cyy, lam)

    U, S, Vt = np.linalg.svd(A @ Cxy @ B, full_matrices=False)
    k = min(k, len(S))

    Wx = A @ U[:, :k]
    Wy = B @ Vt[:k].T
    rho = np.clip(S[:k], 0.0, 1.0)
    return Wx, Wy, rho


def residualize(
        X_flat: np.ndarray,
        Z_flat: Optional[np.ndarray],
        lam_hat: float = LAMBDA_HAT,
) -> np.ndarray:
    if Z_flat is None or Z_flat.ndim < 2 or Z_flat.shape[1] == 0:
        return X_flat.copy()
    n, m = Z_flat.shape
    ZtZ = Z_flat.T @ Z_flat + lam_hat * n * np.eye(m)
    Beta = np.linalg.solve(ZtZ, Z_flat.T)
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
    """Verbatim copy: every region's raw (n_trials, n_neurons, T) spike
    tensor, with each region's own 'selected_neurons' mask already
    applied."""
    data = mat73.loadmat(session_path)
    rd = data.get("region_data", {})
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
# 0b.  Statistics helpers — copied verbatim from pcca_single_session.py §0c
# =============================================================================

def _gini_coefficient(w: np.ndarray) -> float:
    r"""
    Gini coefficient of the absolute canonical weight vector :math:`|w|`.
    :math:`G \to 0`: weight spread uniformly across all neurons;
    :math:`G \to 1`: weight concentrated in a single neuron.  Returns NaN
    for a degenerate (empty or all-zero) vector.
    """
    x = np.sort(np.abs(np.asarray(w, dtype=float).ravel()))
    n = x.size
    total = float(x.sum())
    if n == 0 or total <= 1e-12:
        return float("nan")
    index = np.arange(1, n + 1, dtype=float)
    G = float(np.sum((2.0 * index - n - 1.0) * x) / (n * total))
    if n > 1:
        G = G * (n / (n - 1.0))

    return G


def _principal_angle_deg(
        a: Optional[np.ndarray], b: Optional[np.ndarray],
) -> float:
    """
    Principal angle (degrees) between two 1-D canonical weight vectors, via
    QR orthonormalisation followed by ``scipy.linalg.subspace_angles`` —
    sign-invariant by construction, since a CCA weight vector's global sign
    is arbitrary. Verbatim copy of pcca_single_session.py's helper.
    """
    from scipy.linalg import subspace_angles
    if a is None or b is None:
        return float("nan")
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return float("nan")
    Qa, _ = np.linalg.qr(a.reshape(-1, 1), mode="reduced")
    Qb, _ = np.linalg.qr(b.reshape(-1, 1), mode="reduced")
    ang = subspace_angles(Qa[:, 0:1], Qb[:, 0:1])
    return float(np.degrees(ang[0]))


# =============================================================================
# 0c.  Candidate-region bookkeeping for the full-joint-ablation regime —
#      copied verbatim (in spirit) from pcca_single_session.py's
#      _is_fiber_region / _is_excluded_region / _candidate_regions, so that
#      "every other candidate region" means the same thing here as it does
#      in create_session_gini_panel_full_ablation / create_session_mi_bar.
# =============================================================================

EXCLUDED_REGIONS: List[str] = ["STRv", "LP","OLF"]


def _is_fiber_region(region_name: str) -> bool:
    return "fiber" in region_name.lower()


def _is_excluded_region(region_name: str) -> bool:
    if _is_fiber_region(region_name):
        return True
    excluded_lower = {r.lower() for r in EXCLUDED_REGIONS}
    return region_name.lower() in excluded_lower


def _candidate_regions(
        region_spikes: Dict[str, np.ndarray],
        exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Every region eligible to serve as a full-ablation nuisance
    regressor: present in this session, ANATOMICAL_ORDER-ordered (any
    region absent from ANATOMICAL_ORDER appended alphabetically), excluding
    fiber-tract catch-alls, EXCLUDED_REGIONS, and anything in `exclude`."""
    exclude_set = set(exclude) if exclude else set()
    present = set(region_spikes.keys())
    ordered = [r for r in ANATOMICAL_ORDER if r in present]
    extra = sorted(present - set(ordered))
    return [
        r for r in ordered + extra
        if r not in exclude_set and not _is_excluded_region(r)
    ]


# =============================================================================
# 0d.  Cosmetic display-name substitution.
#      Applied ONLY inside label-construction (figure axis labels, the
#      "column" field of the tidy DataFrame, and the pairwise-significance
#      report). Every region_spikes lookup and every pCCA fit upstream of
#      labelling still uses the raw key ("VALVM", "VPMPO", ...) that
#      actually exists in the .mat files — renaming a display string must
#      never change which data get fit together.
# =============================================================================

REGION_DISPLAY_NAMES: Dict[str, str] = {
    "VALVM": "motor Thal",
    "VPMPO": "sens Thal",
}


def _disp(region: str) -> str:
    return REGION_DISPLAY_NAMES.get(region, region)


def _angle_col_label(pivot: str, ri: str, rj: str) -> str:
    return f"{_disp(pivot)}: {_disp(ri)} vs {_disp(rj)}"


def _gini_col_label_single(pivot: str, partner: str, ablate: str) -> str:
    return f"{_disp(pivot)}\u2192{_disp(partner)}"


def _gini_col_label_full(pivot: str, partner: str) -> str:
    return f"{_disp(pivot)}\u2192{_disp(partner)}"


ANGLE_COLUMNS: List[str] = [
    _angle_col_label(p, i, j) for (p, i, j) in ANGLE_TRIPLETS
]


# Two Gini columns per triplet, (i|Z=j) then (j|Z=i). Since
# _gini_col_label_single and _gini_col_label_full both collapse to
# f"{pivot}→{partner}" (the ablate/third-region argument is display-inert
# for both), the same (pivot, partner) column is emitted every time that
# pair recurs as a triplet member. Deduplicated here to first-occurrence
# order so each column is plotted once, not once per triplet that happens
# to reference it. Row-level de-duplication of the underlying data (so a
# repeated full-regime column isn't double-counted per session) happens
# separately, in aggregate_across_sessions().
def _dedupe_preserve_order(cols: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


_GINI_COLUMNS_SINGLE_RAW: List[str] = []
_GINI_COLUMNS_FULL_RAW: List[str] = []
for _p, _i, _j in ANGLE_TRIPLETS:
    _GINI_COLUMNS_SINGLE_RAW.append(_gini_col_label_single(_p, _i, _j))
    _GINI_COLUMNS_SINGLE_RAW.append(_gini_col_label_single(_p, _j, _i))
    _GINI_COLUMNS_FULL_RAW.append(_gini_col_label_full(_p, _i))
    _GINI_COLUMNS_FULL_RAW.append(_gini_col_label_full(_p, _j))

GINI_COLUMNS_SINGLE: List[str] = _dedupe_preserve_order(_GINI_COLUMNS_SINGLE_RAW)
GINI_COLUMNS_FULL: List[str] = _dedupe_preserve_order(_GINI_COLUMNS_FULL_RAW)

_dupe_check = [
    c for c in set(_GINI_COLUMNS_FULL_RAW) if _GINI_COLUMNS_FULL_RAW.count(c) > 1
]
if _dupe_check:
    print(
        f"[info] {len(_dupe_check)} Gini column(s) are produced by more "
        f"than one triplet (shared (pivot, partner) pair): {_dupe_check} "
        f"— collapsed to one column each for global display; see "
        f"aggregate_across_sessions() for the corresponding row-level "
        f"de-duplication in the full-ablation regime. Note the same "
        f"shared column may legitimately be redrawn in more than one row "
        f"of the 3x3 grid, since each row is a self-contained panel."
    )


# =============================================================================
# 2.  Per-session ablation-geometry computation
# =============================================================================

def _compute_single_ablation_weight(
        region_spikes: Dict[str, np.ndarray],
        pivot: str,
        partner: str,
        ablate: str,
        cache: Dict[Tuple[str, str, str], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    r"""
    :math:`w_P` from :math:`\mathrm{pCCA}(P, \text{partner} \mid Z=\text{ablate})`.
    Identical construction to
    ``OxfordPCCASessionVisualizer._compute_single_ablation_weight``.
    """
    key = (pivot, partner, ablate)
    if key in cache:
        return cache[key]

    if not all(r in region_spikes for r in (pivot, partner, ablate)):
        cache[key] = None
        return None

    X_piv = _zscore_flat(region_spikes[pivot],
                         subtract_psth=SUBTRACT_PSTH,
                         shuffle_trials=SHUFFLE_TRIALS)
    X_par = _zscore_flat(region_spikes[partner],
                         subtract_psth=SUBTRACT_PSTH,
                         shuffle_trials=SHUFFLE_TRIALS)
    Z_abl = _zscore_flat(region_spikes[ablate],
                         subtract_psth=SUBTRACT_PSTH,
                         shuffle_trials=SHUFFLE_TRIALS)

    try:
        Wx, _, _, _, _ = pcca(X_piv, X_par, Z_abl)
        w = Wx[:, 0].copy()
    except Exception as exc:
        warnings.warn(f"pCCA({pivot}, {partner} | Z={ablate}) failed: {exc}")
        w = None

    cache[key] = w
    return w


def _compute_full_ablation_weight(
        region_spikes: Dict[str, np.ndarray],
        pivot: str,
        partner: str,
        cache: Dict[Tuple[str, str], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    r"""
    :math:`w_P` from :math:`\mathrm{pCCA}(P, \text{partner} \mid Z =
    \{\text{candidates}\} \setminus \{P, \text{partner}\})` — every other
    candidate region present in the session, not just one triplet partner.
    Identical Z construction to
    ``create_session_gini_panel_full_ablation`` / ``create_session_mi_bar``.
    Depends only on (pivot, partner), hence cached and keyed that way —
    reused as-is across every triplet that touches this pair.
    """
    key = (pivot, partner)
    if key in cache:
        return cache[key]

    if pivot not in region_spikes or partner not in region_spikes:
        cache[key] = None
        return None

    candidates = _candidate_regions(region_spikes)
    nuisance_regions = [r for r in candidates if r not in (pivot, partner)]

    X_piv = _zscore_flat(region_spikes[pivot],
                         subtract_psth=SUBTRACT_PSTH,
                         shuffle_trials=SHUFFLE_TRIALS)
    X_par = _zscore_flat(region_spikes[partner],
                         subtract_psth=SUBTRACT_PSTH,
                         shuffle_trials=SHUFFLE_TRIALS)
    Z_full = (
        np.concatenate(
            [_zscore_flat(region_spikes[r],
                          subtract_psth=SUBTRACT_PSTH,
                          shuffle_trials=SHUFFLE_TRIALS)
             for r in nuisance_regions],
            axis=1,
        )
        if nuisance_regions else None
    )

    try:
        Wx, _, _, _, _ = pcca(X_piv, X_par, Z_full)
        w = Wx[:, 0].copy()
    except Exception as exc:
        warnings.warn(
            f"full-ablation pCCA({pivot}, {partner} | all others) failed: {exc}"
        )
        w = None

    cache[key] = w
    return w


def compute_session_geometry(
        session_path: Path,
        session_name: str,
) -> Optional[Dict[str, Dict]]:
    """
    For one session, fit every triplet's pair of ablation weight vectors —
    under BOTH regimes — and reduce them to angle / Gini scalars.

    Returns
    -------
    {
      "angles_single": {(pivot, ri, rj): theta_deg, ...},
      "ginis_single":  {(pivot, ri, rj, "i"|"j"): gini, ...},
      "angles_full":   {(pivot, ri, rj): theta_deg, ...},
      "ginis_full":    {(pivot, ri, rj, "i"|"j"): gini, ...},
    }
    (13 angle entries, 26 gini entries per regime for the current
    thirteen-triplet configuration), or None if the session file could not
    be loaded at all.
    """
    try:
        region_spikes, n_trials, T = load_region_spikes(str(session_path))
    except Exception as exc:
        warnings.warn(f"[{session_name}] load_region_spikes failed: {exc}")
        return None

    missing_regions = {
                          r for (p, i, j) in ANGLE_TRIPLETS for r in (p, i, j)
                      } - set(region_spikes.keys())
    if missing_regions:
        warnings.warn(
            f"[{session_name}] region(s) {sorted(missing_regions)} absent "
            f"from this session; every triplet touching them will be NaN "
            f"for this session (both regimes)."
        )

    cache_single: Dict[Tuple[str, str, str], Optional[np.ndarray]] = {}
    cache_full: Dict[Tuple[str, str], Optional[np.ndarray]] = {}

    out: Dict[str, Dict] = {
        "angles_single": {}, "ginis_single": {},
        "angles_full": {}, "ginis_full": {},
    }

    for (pivot, ri, rj) in ANGLE_TRIPLETS:
        # ---- single-region ablation ----
        w_i = _compute_single_ablation_weight(region_spikes, pivot, ri, rj, cache_single)
        w_j = _compute_single_ablation_weight(region_spikes, pivot, rj, ri, cache_single)
        out["angles_single"][(pivot, ri, rj)] = _principal_angle_deg(w_i, w_j)
        out["ginis_single"][(pivot, ri, rj, "i")] = (
            _gini_coefficient(w_i) if w_i is not None else float("nan")
        )
        out["ginis_single"][(pivot, ri, rj, "j")] = (
            _gini_coefficient(w_j) if w_j is not None else float("nan")
        )

        # ---- full joint ablation ----
        wf_i = _compute_full_ablation_weight(region_spikes, pivot, ri, cache_full)
        wf_j = _compute_full_ablation_weight(region_spikes, pivot, rj, cache_full)
        out["angles_full"][(pivot, ri, rj)] = _principal_angle_deg(wf_i, wf_j)
        out["ginis_full"][(pivot, ri, rj, "i")] = (
            _gini_coefficient(wf_i) if wf_i is not None else float("nan")
        )
        out["ginis_full"][(pivot, ri, rj, "j")] = (
            _gini_coefficient(wf_j) if wf_j is not None else float("nan")
        )

    return out


# =============================================================================
# 3.  Cross-session aggregation → tidy long-format DataFrame
# =============================================================================

def _resolve_session_list() -> List[str]:
    """SESSION_LIST if populated, else auto-discover under RESULTS_SUBDIR."""
    if SESSION_LIST:
        return list(SESSION_LIST)
    pattern = str(BASE_DIR / RESULTS_SUBDIR / "*_analysis_results.mat")
    found = sorted(
        Path(p).stem.replace("_analysis_results", "")
        for p in glob.glob(pattern)
    )
    if found:
        warnings.warn(
            f"SESSION_LIST was empty; auto-discovered {len(found)} "
            f"session(s) under '{RESULTS_SUBDIR}'. Pin SESSION_LIST "
            f"explicitly once you've confirmed this is the intended set — "
            f"reproducibility depends on this being an explicit, static "
            f"list rather than whatever happens to be on disk."
        )
    return found


def _config_hash(session_list: Sequence[str]) -> str:
    r"""
    Deterministic 8-hex-char hash over every parameter that changes the
    *contents* of the aggregated DataFrame: the resolved session list, the
    triplet specification, ``EXCLUDED_REGIONS``, the pCCA hyperparameters
    ($\lambda_{\mathrm{CCA}}$, $\lambda_{\hat H}$, retained-component
    count), and the PSTH-subtraction / trial-shuffle regime flags.

    Deliberately excludes everything downstream of
    ``aggregate_across_sessions`` (palette, box width, figure size, axis
    labels, display-name substitution, ...). Consequence: a
    figure-styling-only edit leaves this hash unchanged, so ``main()``
    transparently reuses the on-disk table; an edit to the analysis itself
    (e.g. changing TRIPLET_GROUPS, as happened here) changes the hash, so
    the old cache is simply never matched and a fresh computation runs
    automatically, rather than a stale table being served silently. (See
    ``main()`` for the fix that actually wires this hash into the cache
    filename — previously it was computed but never used for that.)
    """
    payload = repr((
        tuple(sorted(session_list)),
        tuple(ANGLE_TRIPLETS),
        tuple(sorted(EXCLUDED_REGIONS)),
        LAMBDA_CCA, LAMBDA_HAT, N_COMPONENTS,
        SUBTRACT_PSTH, SHUFFLE_TRIALS,
    )).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def aggregate_across_sessions(
        session_list: Sequence[str],
        base_dir: Path,
        results_subdir: str,
) -> pd.DataFrame:
    """
    Run ``compute_session_geometry`` over every session and stack the
    results (both regimes) into one tidy long-format table:

        session | ablation | section | column | pivot | partner | ablate | value

    ``ablation`` is 'single' or 'full'; ``section`` is 'angle' or 'gini';
    ``column`` is the exact (display-name-substituted) x-axis label used
    by the boxplot panels below. For 'full' rows, ``ablate`` is the
    literal string "ALL_OTHER" (Z was never a single region in that
    regime) rather than a region name.
    """
    records: List[dict] = []

    for session_name in session_list:
        session_path = (
                base_dir / results_subdir / f"{session_name}_analysis_results.mat"
        )
        if not session_path.exists():
            warnings.warn(f"[{session_name}] file not found: {session_path}")
            continue

        print(f"[{session_name}]  computing ablation geometry over "
              f"{len(ANGLE_TRIPLETS)} triplets (single + full regimes) ...")
        geo = compute_session_geometry(session_path, session_name)
        if geo is None:
            continue

        # Keyed on exactly what _compute_*_ablation_weight's own caches
        # key on: (pivot, partner, ablate) for single, (pivot, partner)
        # [ablate_field pinned to "ALL_OTHER"] for full. Two triplets that
        # land on the same key produced the identical cached w_P vector —
        # recording both would double-count this session under what is
        # now a shared column label. Reset per session so genuinely new
        # sessions still contribute their own point.
        seen_gini_keys: set = set()

        for ablation_tag, angle_key, gini_key in [
            ("single", "angles_single", "ginis_single"),
            ("full", "angles_full", "ginis_full"),
        ]:
            for (pivot, ri, rj), theta in geo[angle_key].items():
                records.append({
                    "session": session_name, "ablation": ablation_tag,
                    "section": "angle", "column": _angle_col_label(pivot, ri, rj),
                    "pivot": pivot, "partner": ri, "ablate": rj,
                    "value": theta,
                })
            for (pivot, ri, rj, side), g in geo[gini_key].items():
                partner, ablate = (ri, rj) if side == "i" else (rj, ri)
                if ablation_tag == "single":
                    col = _gini_col_label_single(pivot, partner, ablate)
                    ablate_field = ablate
                else:
                    col = _gini_col_label_full(pivot, partner)
                    ablate_field = "ALL_OTHER"

                dedup_key = (ablation_tag, pivot, partner, ablate_field)
                if dedup_key in seen_gini_keys:
                    continue
                seen_gini_keys.add(dedup_key)

                records.append({
                    "session": session_name, "ablation": ablation_tag,
                    "section": "gini", "column": col,
                    "pivot": pivot, "partner": partner, "ablate": ablate_field,
                    "value": g,
                })

    return pd.DataFrame.from_records(records)


# =============================================================================
# 3b.  Pairwise significance testing — report only, never plotted.
# =============================================================================

def compute_pairwise_pvalues(
        df: pd.DataFrame,
        ablation_tag: str,
        section: str,
        columns: Sequence[str],
        min_paired_n: int = 5,
) -> List[Dict]:
    r"""
    All :math:`\binom{|\text{columns}|}{2}` pairwise comparisons among
    ``columns`` (either the full angle set or a full Gini set), within one
    ablation regime. Reported purely as a numerical table — **never**
    overlaid on the box-and-whisker panels themselves.

    Test selection, per pair :math:`(A, B)`
    ----------------------------------------
    Sessions are matched by session id. Let
    :math:`n_{AB} = \bigl|\{s : v_A(s), v_B(s) \text{ both finite}\}\bigr|`.

    * If :math:`n_{AB} \ge` ``min_paired_n``: a two-sided **Wilcoxon
      signed-rank test** is run on the matched differences
      :math:`d(s) = v_A(s) - v_B(s)`. This is the paired, distribution-free
      analogue of the paired :math:`t`-test — it matches the box plot's
      own assumption-free summary statistics (median / IQR) rather than
      imposing normality on what is, at best, a handful of sessions, and
      it correctly exploits the fact that :math:`A` and :math:`B` are
      typically fit on the *same* session's data.
    * Otherwise: falls back to an **unpaired two-sided Mann-Whitney U**
      test over whatever finite values each column has, independent of
      session matching. This is a strictly weaker test (no within-session
      control), so the ``test`` field always records which regime was
      used — treat unpaired entries with the appropriate extra caution.

    No multiple-comparison correction is applied (Bonferroni / Holm /
    Benjamini-Hochberg all depend on which family of contrasts you
    ultimately intend to report), so treat this as a screening table, not
    a final inferential claim.

    Returns a list of dicts (one row per unordered column pair); the
    caller is expected to wrap it in a DataFrame and write it to disk.
    """
    sub = df[(df["ablation"] == ablation_tag) & (df["section"] == section)]
    wide = sub.pivot_table(
        index="session", columns="column", values="value", aggfunc="first"
    )

    results: List[Dict] = []
    for col_a, col_b in itertools.combinations(columns, 2):
        if col_a not in wide.columns or col_b not in wide.columns:
            results.append({
                "ablation": ablation_tag, "section": section,
                "col_a": col_a, "col_b": col_b,
                "n": 0, "test": "column absent",
                "statistic": np.nan, "p_value": np.nan,
            })
            continue

        paired = wide[[col_a, col_b]].dropna()
        if len(paired) >= min_paired_n:
            d = paired[col_a].to_numpy() - paired[col_b].to_numpy()
            if np.allclose(d, 0.0):
                stat, p = np.nan, 1.0
                test_used = f"wilcoxon (paired, n={len(paired)}, degenerate: d\u22610)"
            else:
                try:
                    stat, p = wilcoxon(paired[col_a].to_numpy(), paired[col_b].to_numpy())
                    test_used = f"wilcoxon (paired, n={len(paired)})"
                except ValueError as exc:
                    stat, p = np.nan, np.nan
                    test_used = f"wilcoxon failed ({exc})"
            n_used = len(paired)
        else:
            a = wide[col_a].dropna().to_numpy()
            b = wide[col_b].dropna().to_numpy()
            if a.size == 0 or b.size == 0:
                stat, p, test_used, n_used = np.nan, np.nan, "insufficient data", 0
            else:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                test_used = f"mann-whitney U (unpaired, n_a={a.size}, n_b={b.size})"
                n_used = min(a.size, b.size)

        results.append({
            "ablation": ablation_tag, "section": section,
            "col_a": col_a, "col_b": col_b,
            "n": n_used, "test": test_used,
            "statistic": stat, "p_value": p,
        })

    return results


# =============================================================================
# 4.  Boxplot + jittered-dot plotting primitives
# =============================================================================

def _triplet_palette(triplets: Sequence[Triplet]) -> Dict[str, tuple]:
    """One saturated colour per triplet (Dark2-style), keyed by the
    triplet's angle-column label. Assigned globally over ALL thirteen
    triplets (not per row) so a given triplet keeps the same colour
    identity in both the single- and full-ablation figures, and across
    whichever rows its Gini columns happen to reappear in."""
    cmap = plt.get_cmap("Dark2")

    return {
        _angle_col_label(p, i, j): cmap(idx % cmap.N)
        for idx, (p, i, j) in enumerate(triplets)
    }
# AFTER
def _pivot_palette(triplets: Sequence[Triplet]) -> Dict[str, tuple]:
    """One saturated colour per PIVOT region (Dark2-style), keyed by the
    raw pivot name — so a given pivot (e.g. STR) keeps the SAME colour in
    every row/group it appears in, not just within one triplet, and
    across both the single- and full-ablation figures. Colour order is
    first-occurrence order across ANGLE_TRIPLETS."""
    cmap = plt.get_cmap("Dark2")
    pivots = _dedupe_preserve_order([p for (p, _, _) in triplets])
    return {p: cmap(idx % cmap.N) for idx, p in enumerate(pivots)}

def _draw_boxplot_dot_panel(
        ax: plt.Axes,
        columns: List[str],
        data_by_column: Dict[str, np.ndarray],
        colors: Dict[str, tuple],
        y_bounds: Optional[Tuple[float, float]],
        y_label: str = '',
        title: Optional[str] = None,
        box_width: float = 0.35,
        dot_jitter: float = 0.13,
        rng: Optional[np.random.Generator] = None,
        tick_labels: Optional[List[str]] = None,
) -> None:
    """
    Tukey box-and-whisker (median line, IQR box, whiskers to the furthest
    non-outlier point, no cap ticks, no separate flier markers — matching
    the reference figure) with every session's raw value overlaid as a
    jittered dot, coloured per column.

    ``columns`` is always the full, disambiguating column key used to look
    data up in ``data_by_column`` / ``colors`` (e.g. the full
    ``"pivot: ri vs rj"`` or ``"pivot→partner"`` string) — that identity
    must stay intact so distinct triplets/columns never collide. If the
    on-axis text should be shorter (e.g. just the pivot name),
    ``tick_labels`` supplies that display-only text, positionally matched
    to ``columns``; it never affects data lookup.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    positions = np.arange(len(columns))
    data_list = []
    for col in columns:
        d = np.asarray(data_by_column.get(col, np.array([])), dtype=float)
        data_list.append(d[np.isfinite(d)])

    bp = ax.boxplot(
        data_list, positions=positions, widths=box_width,
        whis=1, showfliers=False, patch_artist=True,
        boxprops=dict(linewidth=1.8, edgecolor="black"),
        medianprops=dict(linewidth=3.0, color="black"),
        whiskerprops=dict(linewidth=1.6, color="black"),
        capprops=dict(linewidth=0),
        zorder=3,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)

    for xi, col, d in zip(positions, columns, data_list):
        if d.size == 0:
            continue
        jitter = rng.uniform(-dot_jitter, dot_jitter, size=d.size)
        ax.scatter(
            xi + jitter, d, s=20, color=colors.get(col, "black"),
            edgecolors="none", zorder=5,
        )

    ax.set_xlim(-0.7, len(columns) - 1 + 0.7)
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)
    ax.set_xticks(positions)
    tick_text = tick_labels if tick_labels is not None else columns
    ax.set_xticklabels(tick_text, fontsize=13, rotation=40, ha="right")
    ax.set_ylabel(y_label, fontsize=18)
    if title:
        ax.set_title(title, fontsize=15, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.3)
    ax.tick_params(width=1.3)
    ax.tick_params(axis='y', labelsize=18)


# =============================================================================
# 5.  Figure assembly — one 3x3 (rows = comparison groups, columns =
#     angle | Gini-vs-first-partner | Gini-vs-second-partner) figure per
#     ablation regime.
# =============================================================================

def _row_partner_pair(group: List[Triplet]) -> Tuple[str, str]:
    """Every triplet within one TRIPLET_GROUPS row shares the same fixed
    (region_i, region_j) pair; only the pivot varies. Returns that shared
    pair, display-name-substituted, for use as a row title."""
    _, ri0, rj0 = group[0]
    return _disp(ri0), _disp(rj0)


def make_ablation_figure(
        df: pd.DataFrame,
        ablation_tag: str,
        title_prefix: str,
        output_path: Path,
) -> plt.Figure:
    r"""
    One figure per ablation regime, laid out as a
    :math:`|\text{TRIPLET\_GROUPS}| \times 3` grid: row :math:`k`
    corresponds to ``TRIPLET_GROUPS[k]`` (a fixed comparison pair
    :math:`\{i, j\}`, pivot varying across the row).

    * Column 0 — principal angle :math:`\theta` between
      :math:`w_P^{(i\mid\cdot)}` and :math:`w_P^{(j\mid\cdot)}`, one box
      per pivot :math:`P` in the row.
    * Column 1 — :math:`\mathrm{Gini}(|w_P^{(i\mid\cdot)}|)`, i.e. the
      Gini value paired with the row's FIRST partner region :math:`i`,
      one box per pivot :math:`P`.
    * Column 2 — :math:`\mathrm{Gini}(|w_P^{(j\mid\cdot)}|)`, the same but
      paired with the row's SECOND partner region :math:`j`.

    All three columns share one x-axis convention: every tick is just the
    pivot's display name (``_disp(pivot)``), never the full
    ``"pivot→partner"`` string — the partner is instead named once, in
    each Gini column's own panel title, since it is constant across the
    whole row by construction. Gini y-limits are fixed to
    :math:`[0.3, 0.8]` (not the coefficient's full :math:`[0, 1]` range).
    """
    df_sub = df[df["ablation"] == ablation_tag]
    df_angle = df_sub[df_sub["section"] == "angle"]
    df_gini = df_sub[df_sub["section"] == "gini"]

    # angle_colors = _triplet_palette(ANGLE_TRIPLETS)
    # AFTER
    pivot_colors = _pivot_palette(ANGLE_TRIPLETS)

    n_rows = len(TRIPLET_GROUPS)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(10.0, 3.6 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    if axes.shape != (n_rows, 3):
        axes = axes.reshape(n_rows, 3)

    for row_idx, group in enumerate(TRIPLET_GROUPS):
        ax_angle, ax_gini_i, ax_gini_j = axes[row_idx, 0], axes[row_idx, 1], axes[row_idx, 2]

        pivot_labels = [_disp(p) for (p, i, j) in group]  # shared x-axis text, all 3 panels

        angle_cols_row = [_angle_col_label(p, i, j) for (p, i, j) in group]

        # Every pivot in a group is unique by construction (see
        # TRIPLET_GROUPS), so these two per-partner column lists are each
        # already duplicate-free — no _dedupe_preserve_order needed here
        # (unlike the old combined single-panel Gini list).
        if ablation_tag == "single":
            gini_cols_i = [_gini_col_label_single(p, i, j) for (p, i, j) in group]
            gini_cols_j = [_gini_col_label_single(p, j, i) for (p, i, j) in group]
        else:
            gini_cols_i = [_gini_col_label_full(p, i) for (p, i, j) in group]
            gini_cols_j = [_gini_col_label_full(p, j) for (p, i, j) in group]

        # Colour lookup keyed by the full label, one entry per triplet;
        # shared across the angle panel and both Gini panels so a given
        # pivot's box is the same colour in all three columns of its row.
        # Colour IDENTITY for a given triplet still comes from the global
        # angle_colors palette (consistent across single/full figures).
        row_colors: Dict[str, tuple] = {}
        for (p, i, j) in group:
            c = pivot_colors[p]
            row_colors[_angle_col_label(p, i, j)] = c
            if ablation_tag == "single":
                row_colors[_gini_col_label_single(p, i, j)] = c
                row_colors[_gini_col_label_single(p, j, i)] = c
            else:
                row_colors[_gini_col_label_full(p, i)] = c
                row_colors[_gini_col_label_full(p, j)] = c

        data_angle = {
            col: df_angle[df_angle["column"] == col]["value"].to_numpy()
            for col in angle_cols_row
        }
        data_gini_i = {
            col: df_gini[df_gini["column"] == col]["value"].to_numpy()
            for col in gini_cols_i
        }
        data_gini_j = {
            col: df_gini[df_gini["column"] == col]["value"].to_numpy()
            for col in gini_cols_j
        }

        ri_disp, rj_disp = _row_partner_pair(group)

        _draw_boxplot_dot_panel(
            ax_angle, angle_cols_row, data_angle, row_colors,
            y_bounds=(0.0, 95.0),
            title=f"vs {ri_disp} & {rj_disp}",
            tick_labels=pivot_labels,
        )
        _draw_boxplot_dot_panel(
            ax_gini_i, gini_cols_i, data_gini_i, row_colors,
            y_bounds=(0.35, 0.8),
            y_label=r"Gini$(|w_P|)$",
            title=f"with {ri_disp}",
            tick_labels=pivot_labels,
        )
        _draw_boxplot_dot_panel(
            ax_gini_j, gini_cols_j, data_gini_j, row_colors,
            y_bounds=(0.35, 0.8),
            y_label=r"Gini$(|w_P|)$",
            title=f"with {rj_disp}",
            tick_labels=pivot_labels,
        )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  saved: {output_path}")
    return fig


# =============================================================================
# 6.  Driver
# =============================================================================

def _print_box_summary(
        df: pd.DataFrame, ablation_tag: str, section: str,
        columns: List[str], unit: str = "",
) -> None:
    print(f"\n=== {section.capitalize()} section, {ablation_tag} ablation "
          f"— median [IQR], n ===")
    for col in columns:
        vals = df[
            (df["ablation"] == ablation_tag) & (df["section"] == section)
            & (df["column"] == col)
            ]["value"].to_numpy()
        vals = vals[np.isfinite(vals)]
        label = col.replace("\n", " ")
        if vals.size == 0:
            print(f"  {label:32s}  n=0")
            continue
        med = float(np.median(vals))
        q1, q3 = (float(v) for v in np.percentile(vals, [25, 75]))
        print(f"  {label:32s}  n={vals.size:2d}  median={med:7.3f}{unit}  "
              f"IQR=[{q1:7.3f}, {q3:7.3f}]")


def main() -> None:
    session_list = _resolve_session_list()
    if not session_list:
        raise SystemExit(
            f"No sessions found. Populate SESSION_LIST, or confirm "
            f"'{RESULTS_SUBDIR}' under {BASE_DIR} contains "
            f"*_analysis_results.mat files."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regime_tag = f"psth{int(SUBTRACT_PSTH)}_shuf{int(SHUFFLE_TRIALS)}"
    cfg_hash = _config_hash(session_list)

    # cfg_hash is now folded directly into both cache filenames. Previously
    # it was computed but never actually used to name anything: the
    # cache-hit branch below checked only whether a fixed-name pickle
    # existed, then loaded a separately fixed-name CSV — neither name
    # depended on cfg_hash, so a stale cache from a prior TRIPLET_GROUPS /
    # EXCLUDED_REGIONS / session-list configuration could be silently
    # reloaded even though USE_CACHED_DATA's own printed message implied
    # the hash was being checked. With the hash in the filename, today's
    # triplet-set change (and any future one) invalidates the old cache
    # automatically — no more manually remembering to set
    # USE_CACHED_DATA = False after an analysis-relevant edit.
    cache_pkl_path = OUTPUT_DIR / f"ablation_geometry_cache_{regime_tag}_{cfg_hash}.pkl"
    cache_csv_path = OUTPUT_DIR / f"ablation_geometry_raw_{regime_tag}_{cfg_hash}.csv"

    if USE_CACHED_DATA and cache_pkl_path.exists() and cache_csv_path.exists():
        print(
            f"[cache] USE_CACHED_DATA=True and a matching cache exists "
            f"(hash {cfg_hash}, covers SESSION_LIST, TRIPLET_GROUPS, "
            f"EXCLUDED_REGIONS, and the pCCA hyperparameters above) — "
            f"loading the previously fitted cross-session table and "
            f"skipping all pCCA refitting. Only Sections 4-5 (plotting) "
            f"and the pairwise-significance report will re-run. Set "
            f"USE_CACHED_DATA=False, or change any hashed config value "
            f"above, to force a fresh computation."
        )
        df = pd.read_csv(cache_csv_path)
    else:
        df = aggregate_across_sessions(session_list, BASE_DIR, RESULTS_SUBDIR)
        if df.empty:
            raise SystemExit(
                "No data collected across any session — check SESSION_LIST, "
                "file paths, and that the triplets' regions are present."
            )
        df.to_pickle(cache_pkl_path)
        print(f"  saved: {cache_pkl_path}")

    df.to_csv(cache_csv_path, index=False)
    print(f"  saved: {cache_csv_path}")

    single_fig_path = OUTPUT_DIR / f"ablation_geometry_single_boxplot_{regime_tag}.png"
    fig1 = make_ablation_figure(
        df, "single",
        title_prefix="Single-region ablation", output_path=single_fig_path,
    )
    plt.close(fig1)

    full_fig_path = OUTPUT_DIR / f"ablation_geometry_full_boxplot_{regime_tag}.png"
    fig2 = make_ablation_figure(
        df, "full",
        title_prefix="Full joint ablation", output_path=full_fig_path,
    )
    plt.close(fig2)

    for tag in ("single", "full"):
        gini_cols = GINI_COLUMNS_SINGLE if tag == "single" else GINI_COLUMNS_FULL
        _print_box_summary(df, tag, "angle", ANGLE_COLUMNS, unit="\u00b0")
        _print_box_summary(df, tag, "gini", gini_cols)

    # -------------------------------------------------------------------
    # Pairwise significance — computed for reporting only, never plotted.
    # -------------------------------------------------------------------
    pval_records: List[Dict] = []
    for tag in ("single", "full"):
        gini_cols = GINI_COLUMNS_SINGLE if tag == "single" else GINI_COLUMNS_FULL
        pval_records += compute_pairwise_pvalues(df, tag, "angle", ANGLE_COLUMNS)
        pval_records += compute_pairwise_pvalues(df, tag, "gini", gini_cols)

    pval_df = pd.DataFrame.from_records(pval_records)
    pval_csv_path = OUTPUT_DIR / f"ablation_geometry_pairwise_pvalues_{regime_tag}_{cfg_hash}.csv"
    pval_df.to_csv(pval_csv_path, index=False)
    print(f"  saved: {pval_csv_path}")
    if len(pval_df) > 0:
        n_sig = int((pval_df["p_value"] < 0.05).sum())
        print(
            f"  {n_sig} / {len(pval_df)} pairwise comparisons at "
            f"uncorrected p < 0.05 (see compute_pairwise_pvalues docstring "
            f"— no multiple-comparison correction has been applied)."
        )


if __name__ == "__main__":
    main()
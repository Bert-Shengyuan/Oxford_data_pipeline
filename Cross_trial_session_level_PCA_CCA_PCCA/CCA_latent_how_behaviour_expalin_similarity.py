#!/usr/bin/env python3
r"""
Cross-Session Behavioural Variance of CCA.py
=========================================================================================

Quantifies, across all sessions and region pairs, *how much of a CCA
communication latent is behaviourally driven* (position / speed), and *how
self-consistent a region's CCA weight vector is* across its different pairings.
The output is one vertical bar figure with four categories and overlaid
per-sample points (styled after the attached reference — terracotta bars, cream
field, dark jittered dots) plus two tidy CSV tables.

Design
------
This is a **pure CCA** script — there is no pCCA / nuisance-partialling control
logic here.  The canonical weights and per-trial latents are **already computed**
by the MATLAB pipeline and stored in the ``*_analysis_results.mat`` files; by
default we simply **load the previous results** through
``Useful_definition.OxfordAdvancedAnalyzer`` (imported directly), so nothing is
re-fit:

  * ``cca_weights``  →  ``PairWeights.A``  (N_i x K),  ``PairWeights.B`` (N_j x K)
       — the leading column A[:,d] / B[:,d] is region i's / j's CCA weight (dim d).
  * ``single_trial`` →  ``SingleTrialPair.z_i`` (n_trials x T x K), ``.z_j``
       — z_i[:,:,d] / z_j[:,:,d] is region i's / j's per-trial CCA latent.

A ``RECOMPUTE_FROM_SPIKES`` switch (default off) refits plain ridge-CCA from the
raw spike tensors if you ever choose to re-run rather than reuse.

Behavioural loading (per-trial position (x, y, z) and speed on t in [-1, 2] s)
follows ``pCCA_sensitive_realsingle_Session_8panel.py``.

Mathematics
-----------
For a pair (i, j), weights w_i = A[:, d], w_j = B[:, d] (d = LATENT_DIM) and
per-trial latents L_i, L_j in R^{n_tr x T}.  Every latent is truncated to
t in [-1, 2] s, flattened over (trial, time), and regressed on a behavioural
design Z in {Z_pos (3), Z_speed (1), Z_pos+speed (4)} via the ridge coefficient
of determination

    R^2(L, Z) = 1 - ||l_c - z_hat||^2 / ||l_c||^2,
    z_hat = Z_c (Z_c^T Z_c + lam n I)^{-1} Z_c^T l_c,

with mean-centring supplying the intercept; R^2 in [0, 1], sign-invariant.

  * Columns 1-3: within a session, average R^2(L_i,Z) and R^2(L_j,Z) into the
    pair value; average that across sessions -> one dot per region pair.
  * Column 4:  within a session, for region A with partners {B, C, D, ...},
        s_A = mean over unordered partner pairs of |cos(w_{A|p}, w_{A|q})|,
    (needs >= 2 partners); average across sessions -> one dot per region.

CSV outputs
-----------
* ``csv1_pairwise_latent_variance_explained.csv``  (columns 1-3)
  One row per (predictor, pair, session), carrying the two per-region values
  R2_latent_i / R2_latent_j (session named) plus their mean.

* ``csv2_weightvector_similarity_by_comparison.csv``  (column 4)
  One row per session in which a comparison type (region; partner_1, partner_2)
  occurs.

Author:  Oxford Neural Analysis Pipeline
Date:    2026
"""

from __future__ import annotations

import csv
import math
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageOps import scale
from mpmath.math2 import sqrt2
from scipy.stats import zscore

# =============================================================================
# 0.  Configuration
# =============================================================================

# ---- Paths (edit for your machine) ------------------------------------------
BASE_DIR = Path("/Users/shengyuancai/Downloads/Oxford_dataset")
SESSION_SUBDIR = "sessions_cued_hit_long_results"           # neural .mat folder
CONDITION_LABEL = "cued_hit_long"                                # key inside the analyzer
BEHAVIOR_DIR = BASE_DIR / "Paper_output" / "tapproach_sessions"  # *_pos / *_speed / *_task_label .npy
OUTPUT_DIR = BASE_DIR / "Paper_output" / "cross_session_behavior_latent"

# All sessions by default.  Set to an explicit list to restrict.
SESSIONS_TO_RUN: Optional[List[str]] = None

# ---- Source of CCA weights / latents ----------------------------------------
# Default: reuse the previously computed CCA results on disk (no re-fitting).
RECOMPUTE_FROM_SPIKES: bool = False   # True -> refit plain ridge-CCA from spikes

# ---- Analysis hyperparameters -----------------------------------------------
LATENT_DIM: int = 0                   # CCA dimension used everywhere (0 = leading)
LAMBDA_R2: float = 1e-4               # ridge on the behavioural R^2 design (~ OLS)
MIN_SESSIONS: int = 1                 # a pair / region must appear in >= this many sessions

# ridge-CCA settings -- only used when RECOMPUTE_FROM_SPIKES is True
LAMBDA_CCA: float = 1e-4
N_COMPONENTS: int = 5
SUBTRACT_PSTH: bool = False

# ---- Behavioural window -----------------------------------------------------
TIME_RANGE_S: Tuple[float, float] = (-1.5, 3.0)            # neural acquisition window
BEHAVIOR_TIME_RANGE_S: Tuple[float, float] = (-1.0, 2.0)   # behavioural tracking window
BEHAVIOR_FS: float = 50.0
BEHAVIOR_T_OFFSET: float = -1.0
BEHAVIOR_TRIAL_LABEL: str = "cued hit long"                # must match CONDITION_LABEL folder

# ---- Region handling --------------------------------------------------------
# Regions excluded from BOTH the analysis and the displayed figure/CSVs.
EXCLUDED_REGIONS: List[str] = ["mPFC,""STRv", "LP", "OLF","MD", "LP", "ILM"]
# Optional display remapping (analysis always keys on canonical names).
DISPLAY_NAME: Dict[str, str] = {"VALVM": "sens Thal", "VPMPO": "motor Thal"}

# ---- Figure aesthetics (reference palette) ----------------------------------
POSTER_SCALE: float = 1.0             # bump to ~2.0 for A0 poster export
_FS_FLOOR: float = 6.0

BAR_FACE_BEHAVIOR = "#DE6E4B"         # terracotta / burnt orange -- behavioural R^2 bars
BAR_FACE_SIMILARITY = "#4B7DDE"       # steel blue -- similarity bars (distinct from behaviour)
FIG_BG = "#ffffff"                    # cream field
DOT_FACE = "#848385"                  # dark charcoal dots
DOT_EDGE = "#848385"
ERR_COLOR = "#141414"
BAR_ALPHA = 0.95
DOT_ALPHA = 0.80
ERROR_KIND: str = "std"               # "sem" or "std"

A0_WIDTH_IN: float = 33.1             # A0 short edge, portrait width in inches
A0_HEIGHT_IN: float = 46.8            # A0 long edge, portrait height in inches
BAR_GAP: float = 0.18

BEHAVIOR_LABELS = ["Position\n(x, y, z)", "Speed", "Position\n+ speed"]
SIMILARITY_LABELS = ["Weight\nvector", "Latent"]

# =============================================================================
# 0b.  Import the project loader (previous CCA results) + shared symbols
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from Useful_definition import (
        OxfordAdvancedAnalyzer,
        ANATOMICAL_ORDER,
        safe_array,
    )
    _USEFUL_OK = True
except (Exception, SystemExit) as _exc:   # Useful_definition raises SystemExit if mat73 missing
    _USEFUL_OK = False
    warnings.warn(
        f"Could not import Useful_definition ({_exc}); the stored-result loader "
        f"is unavailable. Set RECOMPUTE_FROM_SPIKES=True or fix the import."
    )
    ANATOMICAL_ORDER = [
        "mPFC", "ORB", "MOp", "MOs", "OLF",
        "STR", "STRv",
        "MD", "LP", "VALVM", "VPMPO", "ILM",
        "HY",
    ]

    def safe_array(x):
        try:
            if x is None:
                return None
            arr = np.asarray(x)
            return None if arr.size == 0 else arr
        except Exception:
            return None

try:
    import mat73
    _MAT73_OK = True
except Exception:
    _MAT73_OK = False


def _fs(base: float) -> float:
    """Poster-scaled font size with a hard floor."""
    return max(_FS_FLOOR, base * POSTER_SCALE)


# =============================================================================
# 1.  Core mathematics  (plain CCA only -- copied verbatim, no pCCA)
# =============================================================================
def _zscore_flat(X: np.ndarray, *, subtract_psth: bool = False) -> np.ndarray:
    """(n_trials, n, T) -> (T*n_trials, n) z-scored per neuron.  Observation axis
    is ordered (time, trial): row = t * n_trials + trial.  (Recompute path only.)"""
    n_trials, n, T = X.shape
    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
    flat = zscore(flat, axis=1, nan_policy="omit")
    np.nan_to_num(flat, nan=0.0, copy=False)
    X = flat.reshape(n, T, n_trials).transpose(2, 0, 1)
    if subtract_psth:
        X = X - X.mean(axis=0, keepdims=True)
    flat = X.transpose(1, 2, 0).reshape(n, T * n_trials)
    return flat.T


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
    """Ridge-regularised CCA.  Returns (Wx, Wy, rho).  (Recompute path only.)"""
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


def latent_projections(X_flat: np.ndarray, w: np.ndarray, n_trials: int, T: int) -> np.ndarray:
    """Project a flat matrix onto a weight and reshape to (n_trials, T)."""
    proj = X_flat @ w
    return proj.reshape(T, n_trials).T


def _cos_sim_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute cosine similarity |a.b| / (||a|| ||b||), 0 if either is degenerate."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.abs(np.dot(a, b)) / (na * nb))


def _flatten_latent(L: np.ndarray) -> np.ndarray:
    """Flatten a (n_trials, T) latent trajectory to a single vector so the same
    |cos| logic used for weight vectors can be applied to latents. Two latents
    being compared must first be truncated to a common (trial, time) support by
    the caller, exactly as _behavior_r2 does before regression."""
    return L.reshape(-1)


# =============================================================================
# 2.  Data loading
# =============================================================================
def load_region_spikes(session_path: str) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Load per-region (n_trials, n, T) spike tensors (recompute path only)."""
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
    print(f"  [load_region_spikes]  {len(region_spikes)} regions  "
          f"| n_trials={n_trials_out}  T={T_out}")
    return region_spikes, int(n_trials_out), int(T_out)


def crop_time_window(
        region_spikes: Dict[str, np.ndarray],
        time_vec_full: np.ndarray,
        window: Tuple[float, float],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Crop the trailing time axis of every (n_trials, n, T) tensor to `window`."""
    lo, hi = window
    mask = (time_vec_full >= lo - 1e-6) & (time_vec_full <= hi + 1e-6)
    if mask.sum() < 2:
        raise ValueError(f"Window {window} barely overlaps time_vec.")
    cropped = {r: X[:, :, mask] for r, X in region_spikes.items()}
    return cropped, time_vec_full[mask]


def load_behavior_regressors(
        session_name: str,
        behavior_dir: Path = BEHAVIOR_DIR,
        trial_label: str = BEHAVIOR_TRIAL_LABEL,
        fs: float = BEHAVIOR_FS,
        t_offset: float = BEHAVIOR_T_OFFSET,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-trial position (x, y, z) and speed for one session, filtered to
    trials matching `trial_label`.  Returns (pos (n,3,T), speed (n,1,T), t (T,))."""
    pos_path = behavior_dir / f"{session_name}_pos.npy"
    speed_path = behavior_dir / f"{session_name}_speed.npy"
    label_path = behavior_dir / f"{session_name}_task_label.npy"
    for p in (pos_path, speed_path, label_path):
        if not p.exists():
            raise FileNotFoundError(f"Behaviour file not found: {p}")
    pos = np.load(pos_path)
    speed = np.load(speed_path)
    labels = np.load(label_path, allow_pickle=True)
    if speed.ndim == 2:
        speed = speed[:, None, :]
    sel = (labels == trial_label)
    if not np.any(sel):
        available = sorted(set(labels.tolist()))
        raise ValueError(f"No behaviour trials labelled '{trial_label}' "
                         f"(available: {available}).")
    pos_sel = pos[sel].astype(np.float32)
    speed_sel = speed[sel].astype(np.float32)
    T_behav = pos_sel.shape[-1]
    t_behav = np.arange(T_behav, dtype=np.float64) / fs + t_offset
    return pos_sel, speed_sel, t_behav


# =============================================================================
# 3.  Region / pair bookkeeping
# =============================================================================
def get_anatomical_index(region: str) -> int:
    try:
        return ANATOMICAL_ORDER.index(region)
    except ValueError:
        return len(ANATOMICAL_ORDER)


def sort_pair_by_anatomy(region_i: str, region_j: str) -> Tuple[str, str]:
    """Canonicalise a pair so the anatomically earlier region comes first."""
    return (region_i, region_j) if get_anatomical_index(region_i) <= get_anatomical_index(region_j) \
        else (region_j, region_i)


def region_ok(r: str) -> bool:
    """A region survives if it is in the anatomical vocabulary and not excluded.
    EXCLUDED_REGIONS therefore drops the region from every pair, from column 4,
    and from the figure/CSVs alike."""
    return (r in ANATOMICAL_ORDER) and (r not in EXCLUDED_REGIONS)


# =============================================================================
# 4.  Behavioural variance explained (ridge R^2)
# =============================================================================
def variance_explained(latent_2d: np.ndarray, design_3d: np.ndarray,
                       lam: float = LAMBDA_R2) -> float:
    r"""Ridge R^2 for a latent (n_trials, T) explained by a behavioural design
    (n_trials, C, T).  Flattened over (trial, time), finite-masked, mean-centred
    (intercept), design columns z-scored for conditioning.  R^2 in [0, 1],
    clipped; invariant to a global sign flip of the latent."""
    n_tr, T = latent_2d.shape
    ell = latent_2d.reshape(-1).astype(np.float64)
    Z = np.transpose(design_3d, (0, 2, 1)).reshape(n_tr * T, -1).astype(np.float64)
    finite = np.isfinite(ell) & np.all(np.isfinite(Z), axis=1)
    if finite.sum() < (Z.shape[1] + 2):
        return 0.0
    ell = ell[finite]
    Z = Z[finite]
    zsd = Z.std(axis=0)
    zsd[zsd < 1e-12] = 1.0
    Z = (Z - Z.mean(axis=0, keepdims=True)) / zsd
    ell_c = ell - ell.mean()
    ss_tot = float(ell_c @ ell_c)
    if ss_tot < 1e-12:
        return 0.0
    n, m = Z.shape
    ZtZ = Z.T @ Z + lam * n * np.eye(m)
    beta = np.linalg.solve(ZtZ, Z.T @ ell_c)
    resid = ell_c - Z @ beta
    return float(np.clip(1.0 - float(resid @ resid) / ss_tot, 0.0, 1.0))


def _behavior_r2(L: np.ndarray, pos: np.ndarray, speed: np.ndarray) -> Dict[str, float]:
    """R^2 of a single latent against position / speed / both, after aligning the
    latent and behaviour to their common (trial, time) support."""
    n = min(L.shape[0], pos.shape[0], speed.shape[0])
    T = min(L.shape[1], pos.shape[2], speed.shape[2])
    L, P, S = L[:n, :T], pos[:n, :, :T], speed[:n, :, :T]
    return {
        "position": variance_explained(L, P),
        "speed": variance_explained(L, S),
        "position+speed": variance_explained(L, np.concatenate([P, S], axis=1)),
    }


# =============================================================================
# 5.  Per-session bundle + analysis
# =============================================================================
class SessionBundle:
    """One session's CCA products aligned for analysis.

    pair_weights : {canonical_pair: {region: leading weight vector w}}
    pair_latents : {canonical_pair: {region: latent (n_trials, T) in [-1, 2] s}}
    pos / speed  : (n, 3, T) / (n, 1, T) behavioural tensors, or None.
    """
    def __init__(self, session: str) -> None:
        self.session = session
        self.pair_weights: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        self.pair_latents: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        self.pos: Optional[np.ndarray] = None
        self.speed: Optional[np.ndarray] = None


class SessionResult:
    """One session's contribution to the five stored quantities (three
    behavioural predictors, weight self-similarity, latent self-similarity)."""
    def __init__(self, session_name: str) -> None:
        self.session = session_name
        self.pair_behavior: Dict[str, Dict[Tuple[str, str], Dict]] = {
            "position": {}, "speed": {}, "position+speed": {},
        }
        self.region_weight_selfsim: Dict[str, float] = {}
        self.weight_comparisons: List[Tuple[str, str, str, float]] = []
        self.region_latent_selfsim: Dict[str, float] = {}
        self.latent_comparisons: List[Tuple[str, str, str, float]] = []


def _crop_latent(L: np.ndarray, time_vec: Optional[np.ndarray],
                 window: Tuple[float, float] = BEHAVIOR_TIME_RANGE_S) -> np.ndarray:
    """Truncate a (n_trials, T) latent to the behavioural window along time."""
    lo, hi = window
    if time_vec is None or time_vec.size != L.shape[1]:
        time_vec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], L.shape[1])
    mask = (time_vec >= lo - 1e-6) & (time_vec <= hi + 1e-6)
    return L[:, mask] if mask.sum() >= 2 else L


def bundle_from_precomputed(
        session: str,
        pair_weights_list: Sequence,
        single_trial_list: Sequence,
        pos: Optional[np.ndarray],
        speed: Optional[np.ndarray],
        latent_dim: int = LATENT_DIM,
) -> SessionBundle:
    """Assemble a bundle from previously computed CCA results (PairWeights /
    SingleTrialPair objects from Useful_definition), applying EXCLUDED_REGIONS."""
    b = SessionBundle(session)
    b.pos, b.speed = pos, speed

    for pw in pair_weights_list:
        ri, rj = pw.region_i, pw.region_j
        if not (region_ok(ri) and region_ok(rj)):
            continue
        A, B = safe_array(pw.A), safe_array(pw.B)
        if A is None or B is None or A.ndim != 2 or B.ndim != 2:
            continue
        d = min(latent_dim, A.shape[1] - 1, B.shape[1] - 1)
        named = {ri: A[:, d], rj: B[:, d]}
        cp = sort_pair_by_anatomy(ri, rj)
        b.pair_weights[cp] = {cp[0]: named[cp[0]], cp[1]: named[cp[1]]}

    for st in single_trial_list:
        ri, rj = st.region_i, st.region_j
        if not (region_ok(ri) and region_ok(rj)):
            continue
        zi, zj = safe_array(st.z_i), safe_array(st.z_j)
        if zi is None or zj is None or zi.ndim != 3 or zj.ndim != 3:
            continue
        d = min(latent_dim, zi.shape[2] - 1, zj.shape[2] - 1)
        tvec = safe_array(st.time_vec)
        Li = _crop_latent(zi[:, :, d], tvec)
        Lj = _crop_latent(zj[:, :, d], tvec)
        named = {ri: Li, rj: Lj}
        cp = sort_pair_by_anatomy(ri, rj)
        b.pair_latents[cp] = {cp[0]: named[cp[0]], cp[1]: named[cp[1]]}
    return b


def bundle_from_spikes(
        session: str,
        region_spikes: Dict[str, np.ndarray],
        time_vec_full: np.ndarray,
        pos: Optional[np.ndarray],
        speed: Optional[np.ndarray],
        latent_dim: int = LATENT_DIM,
) -> SessionBundle:
    """Assemble a bundle by refitting plain ridge-CCA from spikes (recompute
    path).  All tensors are cropped to the behavioural window up front."""
    b = SessionBundle(session)
    region_spikes, _ = crop_time_window(region_spikes, time_vec_full, BEHAVIOR_TIME_RANGE_S)
    regions = [r for r in ANATOMICAL_ORDER if r in region_spikes and region_ok(r)]
    if len(regions) < 2:
        return b
    n_trials, T = region_spikes[regions[0]].shape[0], region_spikes[regions[0]].shape[2]
    b.pos, b.speed = pos, speed
    flat = {r: _zscore_flat(region_spikes[r], subtract_psth=SUBTRACT_PSTH) for r in regions}
    for a, c in combinations(regions, 2):
        ri, rj = sort_pair_by_anatomy(a, c)
        Wx, Wy, _ = ridge_cca(flat[ri], flat[rj], LAMBDA_CCA, N_COMPONENTS)
        d = min(latent_dim, Wx.shape[1] - 1, Wy.shape[1] - 1)
        w_i, w_j = Wx[:, d], Wy[:, d]
        b.pair_weights[(ri, rj)] = {ri: w_i, rj: w_j}
        b.pair_latents[(ri, rj)] = {
            ri: latent_projections(flat[ri], w_i, n_trials, T),
            rj: latent_projections(flat[rj], w_j, n_trials, T),
        }
    return b


def analyze_bundle(bundle: SessionBundle) -> SessionResult:
    """Compute the four-column contributions for one session bundle."""
    res = SessionResult(bundle.session)

    # -- Weight self-similarity: region -> {partner: weight} -> within-session
    weight_store: Dict[str, Dict[str, np.ndarray]] = {}
    for (r0, r1), wd in bundle.pair_weights.items():
        weight_store.setdefault(r0, {})[r1] = wd[r0]
        weight_store.setdefault(r1, {})[r0] = wd[r1]
    for region, partner_weights in weight_store.items():
        partners = sorted(partner_weights.keys(), key=get_anatomical_index)
        if len(partners) < 2:
            continue
        sims: List[float] = []
        for p, q in combinations(partners, 2):
            s = _cos_sim_abs(partner_weights[p], partner_weights[q])
            sims.append(s)
            p1, p2 = (p, q) if get_anatomical_index(p) <= get_anatomical_index(q) else (q, p)
            res.weight_comparisons.append((region, p1, p2, s))
        res.region_weight_selfsim[region] = float(np.mean(sims))

    # -- Latent self-similarity: identical logic, applied to the per-trial
    #    latent trajectories instead of the static weight vectors. A region's
    #    latent varies by partner (it is re-extracted per pair), so this asks
    #    the analogous question at the level of the projected activity itself.
    latent_store: Dict[str, Dict[str, np.ndarray]] = {}
    for (r0, r1), ld in bundle.pair_latents.items():
        latent_store.setdefault(r0, {})[r1] = ld[r0]
        latent_store.setdefault(r1, {})[r0] = ld[r1]
    for region, partner_latents in latent_store.items():
        partners = sorted(partner_latents.keys(), key=get_anatomical_index)
        if len(partners) < 2:
            continue
        sims = []
        for p, q in combinations(partners, 2):
            Lp, Lq = partner_latents[p], partner_latents[q]
            n = min(Lp.shape[0], Lq.shape[0])
            T = min(Lp.shape[1], Lq.shape[1])
            s = _cos_sim_abs(_flatten_latent(Lp[:n, :T]), _flatten_latent(Lq[:n, :T]))
            sims.append(s)
            p1, p2 = (p, q) if get_anatomical_index(p) <= get_anatomical_index(q) else (q, p)
            res.latent_comparisons.append((region, p1, p2, s))
        res.region_latent_selfsim[region] = float(np.mean(sims))

    # -- Columns 1-3: behavioural R^2 of each pair's two latents

    # -- Columns 1-3: behavioural R^2 of each pair's two latents
    if bundle.pos is not None and bundle.speed is not None:
        pos = np.asarray(bundle.pos, dtype=np.float64)
        speed = np.asarray(bundle.speed, dtype=np.float64)
        for (r0, r1), ld in bundle.pair_latents.items():
            r2_0 = _behavior_r2(ld[r0], pos, speed)
            r2_1 = _behavior_r2(ld[r1], pos, speed)
            for pred in ("position", "speed", "position+speed"):
                res.pair_behavior[pred][(r0, r1)] = {
                    "i": r2_0[pred], "j": r2_1[pred],
                    "mean": 0.5 * (r2_0[pred] + r2_1[pred]),
                    "region_i": r0, "region_j": r1,
                }
    return res


# =============================================================================
# 6.  Cross-session aggregation
# =============================================================================
class Aggregate:
    """Merges heterogeneous per-session results into cross-session summaries and
    the tidy row lists backing the three CSV files."""
    def __init__(self) -> None:
        self.pair_means: Dict[str, Dict[Tuple[str, str], List[float]]] = {
            "position": {}, "speed": {}, "position+speed": {},
        }
        self.region_weight_sim: Dict[str, List[float]] = {}
        self.region_latent_sim: Dict[str, List[float]] = {}
        self.csv1_rows: List[Dict] = []
        self.csv2_rows: List[Dict] = []
        self.csv3_rows: List[Dict] = []

    def add(self, res: SessionResult) -> None:
        for pred, pairs in res.pair_behavior.items():
            for pair, vals in pairs.items():
                self.pair_means[pred].setdefault(pair, []).append(vals["mean"])
                self.csv1_rows.append({
                    "predictor": pred,
                    "region_pair": f"{pair[0]}-{pair[1]}",
                    "region_i": vals["region_i"], "region_j": vals["region_j"],
                    "session": res.session,
                    "R2_latent_i": round(vals["i"], 6),
                    "R2_latent_j": round(vals["j"], 6),
                    "R2_pair_mean": round(vals["mean"], 6),
                })
        for region, sim in res.region_weight_selfsim.items():
            self.region_weight_sim.setdefault(region, []).append(sim)
        for region, p1, p2, sim in res.weight_comparisons:
            self.csv2_rows.append({
                "region": region, "partner_1": p1, "partner_2": p2,
                "comparison": f"{region}|{p1} vs {region}|{p2}",
                "session": res.session,
                "abs_cosine_similarity": round(sim, 6),
            })
        for region, sim in res.region_latent_selfsim.items():
            self.region_latent_sim.setdefault(region, []).append(sim)
        for region, p1, p2, sim in res.latent_comparisons:
            self.csv3_rows.append({
                "region": region, "partner_1": p1, "partner_2": p2,
                "comparison": f"{region}|{p1} vs {region}|{p2}",
                "session": res.session,
                "abs_cosine_similarity": round(sim, 6),
            })

    def column_points(self, predictor: str) -> Tuple[List[str], np.ndarray]:
        labels, values = [], []
        for pair, vals in self.pair_means[predictor].items():
            if len(vals) < MIN_SESSIONS:
                continue
            labels.append(f"{pair[0]}-{pair[1]}")
            values.append(float(np.mean(vals)))
        return labels, np.asarray(values, dtype=float)

    def region_points(self, kind: str = "weight") -> Tuple[List[str], np.ndarray]:
        """kind: 'weight' -> column-4-style weight self-similarity,
                 'latent' -> latent self-similarity (new)."""
        store = self.region_weight_sim if kind == "weight" else self.region_latent_sim
        labels, values = [], []
        for region, vals in store.items():
            if len(vals) < MIN_SESSIONS:
                continue
            labels.append(region)
            values.append(float(np.mean(vals)))
        return labels, np.asarray(values, dtype=float)

# =============================================================================
# 7.  Figure
# =============================================================================
def _err(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    if ERROR_KIND == "std":
        return float(np.std(values, ddof=1))
    return float(np.std(values, ddof=1) / math.sqrt(values.size))

def _bar_figure(
        point_clouds: List[np.ndarray],
        labels: List[str],
        units: List[str],
        bar_color: str,
        ylabel: str,
        output_path: Path,
        scale: float,
        title: str =None,
) -> plt.Figure:
    """Shared renderer for a vertical bar figure (mean +/- error, jittered
    points) — used by both the behaviour figure and the similarity figure so
    the two only differ in their data, labels, and bar colour."""
    means = [float(np.mean(v)) if v.size else 0.0 for v in point_clouds]
    errs = [_err(v) for v in point_clouds]
    n = len(point_clouds)

    fig_w = min(A0_WIDTH_IN, 1.4 * n * POSTER_SCALE + 1)
    fig_h = min(A0_HEIGHT_IN, 5.2 * POSTER_SCALE)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)

    slot = 1.0 - BAR_GAP           # bars occupy this fraction of each unit slot
    x = np.arange(n) * slot        # slots pulled together -> less inter-bar gap
    bar_w = slot * 0.60

    ax.bar(x, means, width=bar_w, color=bar_color, alpha=BAR_ALPHA, edgecolor="none", zorder=1)
    ax.errorbar(x, means, yerr=errs, fmt="none", ecolor=ERR_COLOR,
                elinewidth=2.0 * POSTER_SCALE, capsize=6 * POSTER_SCALE,
                capthick=2.0 * POSTER_SCALE, zorder=3)

    rng = np.random.default_rng(0)
    jitter_w = bar_w * 0.72
    for xi, v in zip(x, point_clouds):
        if v.size == 0:
            continue
        jitter = (rng.random(v.size) - 0.5) * jitter_w
        ax.scatter(xi + jitter, v, s=42 * POSTER_SCALE, facecolor=DOT_FACE,
                   edgecolor=DOT_EDGE, linewidth=0.6 * POSTER_SCALE,
                   alpha=DOT_ALPHA, zorder=4)

    tick_labels = [f"{lab}"
                   for lab, v, u in zip(labels, point_clouds, units)]
    ax.set_xticks(x)
    ax.set_ylabel(ylabel, fontsize=_fs(18))
    ax.set_ylim(0.0, 1)
    ax.set_xlim(x[0] - slot * 0.75, x[-1] + slot * 0.75)
    ax.set_xticklabels(tick_labels, fontsize=_fs(18))
    ax.set_title(title, fontsize=_fs(18), y=1.03)


    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#8A8178")
    ax.tick_params(colors="#000000", labelsize=_fs(18))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    print(f"  [figure] saved -> {output_path}")
    return fig


def make_behavior_barplot(agg: Aggregate, output_path: Path) -> plt.Figure:
    """Three-column figure: behavioural variance explained (position, speed,
    position+speed), one dot per region pair."""
    point_clouds = [
        agg.column_points("position")[1],
        agg.column_points("speed")[1],
        agg.column_points("position+speed")[1],
    ]
    src = "refit" if RECOMPUTE_FROM_SPIKES else "stored"
    return _bar_figure(
        point_clouds, BEHAVIOR_LABELS, ["pairs", "pairs", "pairs"],
        BAR_FACE_BEHAVIOR,
        "Behavioural variance explained  ($R^2$)",
        output_path,
        scale=1,
        # f"Cross-session behavioural drive of CCA communication latents "
        # f"(dim {LATENT_DIM}, {src})",
    )


def make_similarity_barplot(agg: Aggregate, output_path: Path) -> plt.Figure:
    """Two-column figure: weight-vector self-similarity and latent
    self-similarity, one dot per region in each column."""
    point_clouds = [
        agg.region_points("weight")[1],
        agg.region_points("latent")[1],
    ]
    src = "refit" if RECOMPUTE_FROM_SPIKES else "stored"
    return _bar_figure(
        point_clouds, SIMILARITY_LABELS, ["regions", "regions"],
        BAR_FACE_SIMILARITY,
        r"Cross-session similarity ($|\cos\theta|$)",
        output_path,
        scale=0.7,#title = f"Similarity of CCA weights and latents ",
    )


# =============================================================================
# 8.  CSV writers
# =============================================================================
def write_csv1(agg: Aggregate, path: Path) -> None:
    """Columns 1-3: session-level values per region pair (two values + mean)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["predictor", "region_pair", "region_i", "region_j", "session",
              "R2_latent_i", "R2_latent_j", "R2_pair_mean"]
    order = {"position": 0, "speed": 1, "position+speed": 2}
    rows = sorted(agg.csv1_rows,
                  key=lambda r: (order.get(r["predictor"], 9), r["region_pair"], r["session"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  [csv] {len(rows)} rows -> {path}")


def write_csv2(agg: Aggregate, path: Path) -> None:
    """Weight self-similarity: within-session weight self-similarity by
    comparison type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["region", "partner_1", "partner_2", "comparison", "session",
              "abs_cosine_similarity"]
    rows = sorted(agg.csv2_rows,
                  key=lambda r: (get_anatomical_index(r["region"]),
                                 get_anatomical_index(r["partner_1"]),
                                 get_anatomical_index(r["partner_2"]), r["session"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  [csv] {len(rows)} rows -> {path}")


def write_csv3(agg: Aggregate, path: Path) -> None:
    """Latent self-similarity: within-session latent self-similarity by
    comparison type (same organisation as write_csv2, applied to latents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["region", "partner_1", "partner_2", "comparison", "session",
              "abs_cosine_similarity"]
    rows = sorted(agg.csv3_rows,
                  key=lambda r: (get_anatomical_index(r["region"]),
                                 get_anatomical_index(r["partner_1"]),
                                 get_anatomical_index(r["partner_2"]), r["session"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  [csv] {len(rows)} rows -> {path}")


def load_aggregate_from_csvs(csv1_path: Path, csv2_path: Path, csv3_path: Path) -> Aggregate:
    """Reconstruct an Aggregate purely from the three CSV files on disk, so a
    cached run can go straight to plotting without touching .mat / .npy data."""
    agg = Aggregate()

    with open(csv1_path, newline="") as f:
        for row in csv.DictReader(f):
            pred = row["predictor"]
            pair = tuple(row["region_pair"].split("-", 1))
            agg.pair_means[pred].setdefault(pair, []).append(float(row["R2_pair_mean"]))
            agg.csv1_rows.append(row)

    with open(csv2_path, newline="") as f:
        for row in csv.DictReader(f):
            agg.csv2_rows.append(row)
    # region_weight_sim must hold, per region, the SAME per-session values that
    # produced region_weight_selfsim originally -- i.e. one value per session,
    # already averaged over that session's partner-pair comparisons.
    _fill_region_sim_from_rows(agg.csv2_rows, agg.region_weight_sim)

    with open(csv3_path, newline="") as f:
        for row in csv.DictReader(f):
            agg.csv3_rows.append(row)
    _fill_region_sim_from_rows(agg.csv3_rows, agg.region_latent_sim)

    return agg


def _fill_region_sim_from_rows(rows: List[Dict], dest: Dict[str, List[float]]) -> None:
    """Average a comparison-type CSV's rows back to one value per (region,
    session), then collect those session-level values per region -- this
    reverses exactly what Aggregate.add()/analyze_bundle() computed originally,
    so cached plots match a fresh run."""
    per_region_session: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        key = (row["region"], row["session"])
        per_region_session.setdefault(key, []).append(float(row["abs_cosine_similarity"]))
    for (region, _session), vals in per_region_session.items():
        dest.setdefault(region, []).append(float(np.mean(vals)))


# =============================================================================
# 9.  Drivers
# =============================================================================


# =============================================================================
# 9.  Drivers
# =============================================================================

# =============================================================================
# 9.  Drivers
# =============================================================================
def _load_behavior_safe(session: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        pos, speed, _ = load_behavior_regressors(session)
        return pos, speed
    except (FileNotFoundError, ValueError) as exc:
        warnings.warn(f"[{session}] behaviour unavailable -- columns 1-3 skipped ({exc}).")
        return None, None


def run_precomputed(sessions_filter: Optional[Sequence[str]], agg: Aggregate) -> None:
    """Load previously computed CCA weights + latents for ALL sessions (default)
    via Useful_definition, then aggregate."""
    if not _USEFUL_OK:
        raise RuntimeError("Useful_definition import failed; cannot use stored results.")
    analyzer = OxfordAdvancedAnalyzer(
        base_results_dir=str(BASE_DIR),
        results_subdirs={CONDITION_LABEL: SESSION_SUBDIR},
        n_components=max(N_COMPONENTS, LATENT_DIM + 1),
    )
    analyzer.load_all()
    cw = analyzer.cca_weights.get(CONDITION_LABEL, {})
    st = analyzer.single_trial.get(CONDITION_LABEL, {})

    # reorganise pair-keyed lists into per-session lists (the merge step)
    sess_w: Dict[str, List] = {}
    for lst in cw.values():
        for pw in lst:
            sess_w.setdefault(pw.session, []).append(pw)
    sess_l: Dict[str, List] = {}
    for lst in st.values():
        for stp in lst:
            sess_l.setdefault(stp.session, []).append(stp)

    sessions = sorted(set(sess_w) | set(sess_l))
    if sessions_filter:
        sessions = [s for s in sessions if s in set(sessions_filter)]
    print(f"  [precomputed] {len(sessions)} sessions with stored CCA results")

    for k, s in enumerate(sessions, 1):
        print(f"\n[{k}/{len(sessions)}] {s}")
        pos, speed = _load_behavior_safe(s)
        bundle = bundle_from_precomputed(s, sess_w.get(s, []), sess_l.get(s, []), pos, speed)
        agg.add(analyze_bundle(bundle))


def run_recompute(sessions_filter: Optional[Sequence[str]], agg: Aggregate) -> None:
    """Refit plain ridge-CCA from spikes for the chosen sessions (default: all
    session files found on disk)."""
    if not _MAT73_OK:
        raise RuntimeError("mat73 required for the recompute path.")
    sess_dir = BASE_DIR / SESSION_SUBDIR
    if sessions_filter:
        sessions = list(sessions_filter)
    else:
        sessions = sorted(p.stem.replace("_analysis_results", "")
                          for p in sess_dir.glob("*_analysis_results.mat"))
    print(f"  [recompute] {len(sessions)} sessions")
    for k, s in enumerate(sessions, 1):
        print(f"\n[{k}/{len(sessions)}] {s}")
        sf = sess_dir / f"{s}_analysis_results.mat"
        if not sf.exists():
            print(f"  [skip] not found: {sf}")
            continue
        region_spikes, n_trials, T = load_region_spikes(str(sf))
        tvec = np.linspace(TIME_RANGE_S[0], TIME_RANGE_S[1], T)
        pos, speed = _load_behavior_safe(s)
        bundle = bundle_from_spikes(s, region_spikes, tvec, pos, speed)
        agg.add(analyze_bundle(bundle))


# Set False to force regeneration even when the three CSVs already exist.
USE_CACHED_CSVS_IF_PRESENT: bool = True


def run_all(sessions_filter: Optional[Sequence[str]] = SESSIONS_TO_RUN,
            output_dir: Path = OUTPUT_DIR) -> Aggregate:
    """Full pipeline -> two figures + three CSVs.  Default: all sessions,
    CCA results reused from disk.  If the three CSVs already exist on disk
    (and USE_CACHED_CSVS_IF_PRESENT is True), analysis is skipped entirely and
    the run goes straight from the cached CSVs to plotting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv1_path = output_dir / "csv1_pairwise_latent_variance_explained.csv"
    csv2_path = output_dir / "csv2_weightvector_similarity_by_comparison.csv"
    csv3_path = output_dir / "csv3_latent_similarity_by_comparison.csv"

    print("=" * 70)
    print("Cross-session behavioural drive of CCA communication latents")
    print(f"  source={'refit' if RECOMPUTE_FROM_SPIKES else 'stored'}  "
          f"latent_dim={LATENT_DIM}  min_sessions={MIN_SESSIONS}")
    print(f"  excluded regions: {EXCLUDED_REGIONS}")
    print("=" * 70)

    cached = USE_CACHED_CSVS_IF_PRESENT and csv1_path.exists() and csv2_path.exists() and csv3_path.exists()
    if cached:
        print(f"\n  [cache] found existing CSVs in {output_dir} -- skipping "
              f"regeneration, loading straight into the figures.")
        agg = load_aggregate_from_csvs(csv1_path, csv2_path, csv3_path)
    else:
        agg = Aggregate()
        if RECOMPUTE_FROM_SPIKES:
            run_recompute(sessions_filter, agg)
        else:
            run_precomputed(sessions_filter, agg)
        write_csv1(agg, csv1_path)
        write_csv2(agg, csv2_path)
        write_csv3(agg, csv3_path)

    make_behavior_barplot(agg, output_dir / "behavior_variance_explained_bar.png")
    make_similarity_barplot(agg, output_dir / "weight_latent_similarity_bar.png")

    print("\n" + "-" * 70)
    for pred in ("position", "speed", "position+speed"):
        _, vals = agg.column_points(pred)
        if vals.size:
            print(f"  {pred:16s}: {vals.size} pairs, mean R^2={vals.mean():.3f} +/- {_err(vals):.3f}")
    for kind in ("weight", "latent"):
        _, vals = agg.region_points(kind)
        if vals.size:
            print(f"  {kind + '-selfsim':16s}: {vals.size} regions, "
                  f"mean |cos|={vals.mean():.3f} +/- {_err(vals):.3f}")
    print("-" * 70)
    return agg


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
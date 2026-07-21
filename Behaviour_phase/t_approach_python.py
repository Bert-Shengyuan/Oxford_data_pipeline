"""
tapproach_extract_and_segment.py
=================================================================
Convert the MATLAB `tapproach` struct (position-trajectory dataset,
50 Hz, t = sample/Fs - 1) into per-session .npy tensors, and assign
each trial a movement-type label (b2) via hysteresis bout-detection
on the 3-D speed profile.

Outputs, one triplet per session, written to `out_dir`:

    {session}_pos.npy            float32, shape (n_trials, 3, T)
                                  axis-1 order: [x, y, z]
    {session}_task_label.npy     object array, shape (n_trials,)   -- b1
    {session}_movement_label.npy object array, shape (n_trials,)   -- b2
    {session}_qc.csv             per-trial QC: n_bouts, bout times/peaks

Author: pipeline extension for Shengyuan's Oxford tapproach analysis.
=================================================================
"""

from __future__ import annotations

import argparse
import csv
import scipy
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

try:
    import mat73
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "mat73 is required to read MATLAB v7.3 files: pip install mat73"
    ) from exc

try:
    from scipy.io import loadmat as _scipy_loadmat
except ImportError:  # pragma: no cover
    _scipy_loadmat = None


# =============================================================================
# 0.  Small utilities (mirrors Useful_definition.safe_array convention)
# =============================================================================

def safe_array(x) -> Optional[np.ndarray]:
    try:
        if x is None:
            return None
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _as_str_array(x) -> np.ndarray:
    """Coerce a MATLAB cellstr / object array into a 1-D numpy object array of str."""
    arr = np.asarray(x, dtype=object).ravel()
    out = np.empty(arr.shape[0], dtype=object)
    for i, v in enumerate(arr):
        if isinstance(v, np.ndarray):
            v = v.item() if v.size == 1 else str(v)
        out[i] = str(v)
    return out


# =============================================================================
# 1.  Loading tapproach
# =============================================================================

def load_tapproach(mat_path: str) -> dict:
    """
    Load the `tapproach` struct, robust to v7.3 (HDF5, via mat73) and
    older v7/v6 formats (via scipy.io.loadmat with simplify_cells=True).
    """
    mat_path = str(mat_path)
    # if _scipy_loadmat is not None:
    #     try:
    #         raw = _scipy_loadmat(mat_path, simplify_cells=True)
    #         return raw.get("tapproach", raw)
    #     except Exception as exc5:
    #         warnings.warn(f"scipy.io.loadmat failed ({exc5}); trying mat73 (v7.3/HDF5).")
    try:
        data = mat73.loadmat(mat_path)
        return data.get("tap_struct", data)
    except Exception as exc73:
        raise RuntimeError(
            f"Could not load '{mat_path}' with either scipy.io.loadmat or mat73: {exc73}"
        ) from exc73




def extract_position_tensor(
    tap: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    pos      : (N_trials, 3, T) float array, axis-1 = [x, y, z]
    labels   : (N_trials,) object array of task-type strings (b1 source)
    sessions : (N_trials,) object array of session-name strings
    """
    pos_x = safe_array(tap.get("pos_x"))
    pos_y = safe_array(tap.get("pos_y"))
    pos_z = safe_array(tap.get("pos_z"))
    if pos_x is None or pos_y is None or pos_z is None:
        raise KeyError("tapproach struct is missing pos_x / pos_y / pos_z.")

    if not (pos_x.shape == pos_y.shape == pos_z.shape):
        raise ValueError(
            f"pos_x/y/z shape mismatch: {pos_x.shape}, {pos_y.shape}, {pos_z.shape}"
        )

    pos = np.stack([pos_x, pos_y, pos_z], axis=1).astype(np.float64)  # (N,3,T)

    labels = _as_str_array(tap.get("label"))
    sessions = _as_str_array(tap.get("session_name"))

    n_trials = pos.shape[0]
    if labels.shape[0] != n_trials or sessions.shape[0] != n_trials:
        raise ValueError(
            f"Label/session length mismatch vs. n_trials={n_trials}: "
            f"labels={labels.shape[0]}, sessions={sessions.shape[0]}"
        )

    return pos, labels, sessions


# =============================================================================
# 2.  Speed profile
# =============================================================================

def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    """Linear-interpolate interior NaNs; forward/back-fill edge NaNs."""
    x = x.copy()
    n = x.shape[0]
    idx = np.arange(n)
    bad = np.isnan(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def compute_speed(
    pos_trial: np.ndarray,
    fs: float = 50.0,
    savgol_window: int = 7,
    savgol_poly: int = 3,
) -> np.ndarray:
    """
    pos_trial : (3, T) position trace for one trial, possibly with NaNs.
    Returns   : (T,) Euclidean speed profile ||d p / dt|| after
                Savitzky-Golay smoothing (denoises tracking jitter
                without the phase lag of a causal low-pass filter).
    """
    T = pos_trial.shape[1]
    pos_filled = np.stack([_interp_nan_1d(pos_trial[d]) for d in range(3)], axis=0)

    w = min(savgol_window, T if T % 2 == 1 else T - 1)
    if w >= 5:
        poly = min(savgol_poly, w - 1)
        pos_smooth = savgol_filter(pos_filled, window_length=w, polyorder=poly, axis=1)
    else:
        pos_smooth = pos_filled

    vel = np.gradient(pos_smooth, 1.0 / fs, axis=1)  # (3, T)
    speed = np.linalg.norm(vel, axis=0)
    return speed


# =============================================================================
# 3.  Hysteresis bout detection  (the core "how many reaches" algorithm)
# =============================================================================

def _contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs = []
    in_run = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_run:
            start, in_run = i, True
        elif not m and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def _merge_close_runs(
    runs: List[Tuple[int, int]], min_gap_samples: int
) -> List[Tuple[int, int]]:
    if not runs:
        return runs
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe - 1 <= min_gap_samples:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def detect_reach_bouts(
    speed: np.ndarray,
    t: np.ndarray,
    baseline_mask: np.ndarray,
    window_mask: Optional[np.ndarray] = None,
    k_on: float = 5.0,
    k_off: float = 2.5,
    min_bout_dur_s: float = 0.06,
    min_gap_merge_s: float = 0.08,
    min_baseline_sd: float = 1e-9,
) -> Tuple[List[Tuple[int, int, float]], float, float]:
    """
    Schmitt-trigger bout detector on a speed trace.

    Per-trial adaptive thresholds:
        theta_on  = mu_base + k_on  * sigma_base
        theta_off = mu_base + k_off * sigma_base     (k_off < k_on)

    A bout is seeded wherever speed >= theta_on, then grown outward in
    both directions while speed remains >= theta_off (hysteresis), which
    prevents a single reach from fragmenting into several bouts when its
    speed profile ripples near threshold. Bouts separated by gaps shorter
    than `min_gap_merge_s` are merged; bouts shorter than `min_bout_dur_s`
    are discarded as noise.

    Returns
    -------
    bouts     : list of (start_idx, end_idx, peak_speed)
    theta_on, theta_off : the thresholds used (for QC / plotting)
    """
    mu = float(np.nanmean(speed[baseline_mask]))
    sd = float(np.nanstd(speed[baseline_mask]))
    if not np.isfinite(sd) or sd < min_baseline_sd:
        sd = max(float(np.nanstd(speed)), min_baseline_sd)

    theta_on = mu + k_on * sd
    theta_off = mu + k_off * sd

    if window_mask is None:
        window_mask = np.ones_like(speed, dtype=bool)

    dt = float(np.median(np.diff(t)))
    min_dur_samples = max(1, int(round(min_bout_dur_s / dt)))
    min_gap_samples = max(0, int(round(min_gap_merge_s / dt)))

    T = speed.shape[0]
    above_on = (speed >= theta_on) & window_mask
    visited = np.zeros(T, dtype=bool)
    in_bout = np.zeros(T, dtype=bool)

    for seed in np.flatnonzero(above_on):
        if visited[seed]:
            continue
        lo, hi = seed, seed
        while lo > 0 and speed[lo - 1] >= theta_off and window_mask[lo - 1]:
            lo -= 1
        while hi < T - 1 and speed[hi + 1] >= theta_off and window_mask[hi + 1]:
            hi += 1
        in_bout[lo : hi + 1] = True
        visited[lo : hi + 1] = True

    raw_runs = _contiguous_runs(in_bout)
    merged_runs = _merge_close_runs(raw_runs, min_gap_samples)

    bouts: List[Tuple[int, int, float]] = []
    for s, e in merged_runs:
        dur = e - s + 1
        peak = float(np.max(speed[s : e + 1]))
        if dur >= min_dur_samples and peak >= theta_on:
            bouts.append((s, e, peak))

    return bouts, theta_on, theta_off


def classify_movement_type(n_bouts: int) -> str:
    if n_bouts == 0:
        return "no_reach"
    elif n_bouts == 1:
        return "single"
    elif n_bouts == 2:
        return "double"
    else:
        return "triple_plus"


# =============================================================================
# 4.  Per-trial wrapper
# =============================================================================

def label_trial_movement(
    pos_trial: np.ndarray,
    t: np.ndarray,
    fs: float,
    baseline_tmax: float = -0.05,
    window_tmin: float = 0.0,
    **bout_kwargs,
) -> Tuple[str, List[Tuple[int, int, float]], float, float]:
    """Compute speed, detect bouts, return (b2_label, bouts, theta_on, theta_off)."""
    speed = compute_speed(pos_trial, fs=fs)
    baseline_mask = t < baseline_tmax
    window_mask = t >= window_tmin
    bouts, theta_on, theta_off = detect_reach_bouts(
        speed, t, baseline_mask, window_mask=window_mask, **bout_kwargs
    )
    return classify_movement_type(len(bouts)), bouts, theta_on, theta_off


# =============================================================================
# 5.  Session-level pipeline
# =============================================================================

def process_sessions(
    mat_path: str,
    out_dir: str,
    fs: float = 50.0,
    t_offset: float = -1.0,
    baseline_tmax: float = -0.05,
    window_tmin: float = 0.0,
    k_on: float = 5.0,
    k_off: float = 2.5,
    min_bout_dur_s: float = 0.06,
    min_gap_merge_s: float = 0.08,
    make_overview_plots: bool = False,  # <-- add this line
) -> None:


    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tap = load_tapproach(mat_path)
    pos, labels, sessions = extract_position_tensor(tap)
    n_trials, _, T = pos.shape
    t = np.arange(T) / fs + t_offset
    global_idx = np.arange(n_trials)  # 原始表中的全局行号，session 切分前建立

    reach_idx_raw = safe_array(tap.get("idx_since_cue"))
    if reach_idx_raw is None:
        reach_idx = np.full(n_trials, np.nan)
    else:
        reach_idx = np.asarray(reach_idx_raw).reshape(-1)
        if reach_idx.shape[0] != n_trials:
            raise ValueError(
                f"idx_since_cue length mismatch vs. n_trials={n_trials}: "
                f"got {reach_idx.shape[0]}"
            )

    if T != 226:
        warnings.warn(f"Expected T=226 samples; got T={T}. Proceeding with actual T.")

    unique_sessions = sorted(set(sessions.tolist()))
    print(f"Loaded {n_trials} trials across {len(unique_sessions)} sessions "
          f"(T={T} samples, Fs={fs} Hz).")

    for sess in unique_sessions:
        sel = sessions == sess
        pos_s = pos[sel]                      # (n_s, 3, T)
        b1_s = labels[sel].copy()              # task trial type
        global_idx_s = global_idx[sel]  # 对应每个 trial 在原始表中的行号
        b3_s = reach_idx[sel]  # idx_since_cue, taken as-is from the original table
        n_s = pos_s.shape[0]

        b2_s = np.empty(n_s, dtype=object)
        qc_rows = []

        for i in range(n_s):
            b2, bouts, th_on, th_off = label_trial_movement(
                pos_s[i], t, fs,
                baseline_tmax=baseline_tmax, window_tmin=window_tmin,
                k_on=k_on, k_off=k_off,
                min_bout_dur_s=min_bout_dur_s, min_gap_merge_s=min_gap_merge_s,
            )
            b2_s[i] = b2
            bout_times = ";".join(
                f"[{t[s]:.3f},{t[e]:.3f}]@{pk:.4g}" for s, e, pk in bouts
            )
            qc_rows.append([i, b1_s[i], b2, len(bouts), th_on, th_off, bout_times])


        np.save(out_dir / f"{sess}_pos.npy", pos_s.astype(np.float32))
        np.save(out_dir / f"{sess}_task_label.npy", b1_s)
        np.save(out_dir / f"{sess}_movement_label.npy", b2_s)

        # --- new: compute + save speed ---
        speed_s = np.stack(
            [compute_speed(pos_s[i], fs=fs) for i in range(n_s)], axis=0
        )  # (n_s, T)
        np.save(out_dir / f"{sess}_speed.npy", speed_s.astype(np.float32))

        if make_overview_plots:
            plot_session_grid(
                pos_s, speed_s[:, None, :], b1_s, b2_s, t,
                out_path=out_dir / f"{sess}_overview.png",
                channel_names=("x", "y", "z", "speed"),
                ylim_p=(-0.015, 0.015),
                ylim_speed=(0.0, float(np.nanpercentile(speed_s, 100)) + 0.02),
                global_idx_s=global_idx_s,
                b3_s=b3_s,
            )

        with open(out_dir / f"{sess}_qc.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial_idx", "task_label", "movement_label",
                        "n_bouts", "theta_on", "theta_off", "bouts(start_s,end_s@peak)"])
            w.writerows(qc_rows)

        counts = {lbl: int(np.sum(b2_s == lbl))
                  for lbl in ("no_reach", "single", "double", "triple_plus")}
        print(f"  [{sess}] n={n_s:4d}  "
              f"no_reach={counts['no_reach']:3d}  single={counts['single']:3d}  "
              f"double={counts['double']:3d}  triple_plus={counts['triple_plus']:3d}")

    print(f"\nDone. Per-session .npy + QC .csv files written to: {out_dir}")


# =============================================================================
# 6.  QC visualization helper (run interactively to calibrate k_on/k_off)
# =============================================================================

def plot_qc_trial(pos_trial: np.ndarray, t: np.ndarray, fs: float = 50.0,
                   baseline_tmax: float = -0.05, window_tmin: float = 0.0,
                   k_on: float = 5.0, k_off: float = 2.5,
                   min_bout_dur_s: float = 0.06, min_gap_merge_s: float = 0.08,
                   ax=None):
    """Plot speed(t), thresholds, and detected bouts for one trial. For calibration."""
    import matplotlib.pyplot as plt

    speed = compute_speed(pos_trial, fs=fs)
    label, bouts, th_on, th_off = label_trial_movement(
        pos_trial, t, fs, baseline_tmax=baseline_tmax, window_tmin=window_tmin,
        k_on=k_on, k_off=k_off, min_bout_dur_s=min_bout_dur_s,
        min_gap_merge_s=min_gap_merge_s,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, speed, color="k", lw=1.2, label="speed")
    ax.axhline(th_on, color="r", ls="--", lw=0.8, label=r"$\theta_{on}$")
    ax.axhline(th_off, color="orange", ls="--", lw=0.8, label=r"$\theta_{off}$")
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    for s, e, pk in bouts:
        ax.axvspan(t[s], t[e], color="tab:blue", alpha=0.25)
    ax.set_title(f"n_bouts={len(bouts)} -> {label}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("speed (mm/s)")
    ax.legend(fontsize=7, loc="upper right")
    return ax


def plot_session_grid(
        data_s: np.ndarray,
        speed_s:np.array,# <-- renamed from pos_s
        b1_s: np.ndarray,
        b2_s: np.ndarray,
        t: np.ndarray,
        out_path: str,
        channel_names: Sequence[str] = ("x", "y", "z","speed"),   # <-- new
        row_height_in: float = 1,
        fig_width_in: float = 10.0,
        dpi: int = 200,
        ylim_p: Optional[Tuple[float, float]] = (-0.015, 0.015),
        ylim_speed: Optional[Tuple[float, float]] = (-0.015, 0.015),
        linewidth: float = 0.4,
        fontsize: float = 10.0,
        global_idx_s: Optional[np.ndarray] = None,  # 每个 trial 在原始表中的全局行号
        b3_s: Optional[np.ndarray] = None,  # idx_since_cue, from the original table
):
    """
    Session-wide QC overview: one compact row per trial, one column per
    channel in `data_s` (e.g. x/y/z position, or a single speed trace),
    each row labeled on the left with `b1 | b2`.
    Only shows up to 100 trials where b1_s == 'cued hit long'.
    """
    n_channels = data_s.shape[1] +1               # <-- was hardcoded 3


    valid_idx = np.where(b1_s == "cued hit long")[0]
    show_idx = valid_idx[:100]
    show_trials = len(show_idx)
    if show_trials == 0:
        print("Warning: No trials found with b1_s == 'cued hit long'. Plot not saved.")
        return

    col_names = list(channel_names)                # <-- was col_names = ["x","y","z"]

    fig, axes = plt.subplots(
        show_trials, n_channels,
        figsize=(fig_width_in, row_height_in * show_trials),
        dpi=dpi,
        sharex=True,
        squeeze=False,          # <-- always 2-D, regardless of trials/channels count
    )
    if show_trials == 1:
        axes = axes[None, :]

    for row_idx, i in enumerate(show_idx):
        for c in range(n_channels):                # <-- was range(3)
            ax = axes[row_idx, c]
            # --------------------------------------
            if c <=2:
                data = data_s[i, c, :]
                if ylim_p is not None:
                    ax.set_ylim(ylim_p)
            elif c>2:
                data = speed_s[i, 0, :]
                if ylim_speed is not None:
                    ax.set_ylim(ylim_speed)

            ax.plot(t, data, color="k", lw=linewidth)   # <-- was pos_s
            ax.axvline(0.0, color="0", lw=0.5, ls=":")
            ax.axvline(1.5, color="0", lw=0.5, ls=":")
            ax.axvline(2, color="0", lw=0.5, ls=":")
            ax.set_xticks([])
            ax.set_yticks([])

            # --- ✨ 核心修改：去除上方和右边的边框 ---
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.3)
            ax.spines["bottom"].set_linewidth(0.3)
            if row_idx == 0:
                ax.set_title(col_names[c], fontsize=fontsize + 2)

        global_i = global_idx_s[i] if global_idx_s is not None else i
        reach_i = b3_s[i] if b3_s is not None else "NA"
        # axes[row_idx, 0].set_ylabel(
        #     f"#{global_i + 1} | {b1_s[i]} | {b2_s[i]} | reach {reach_i}",
        #     fontsize=fontsize, rotation=0, ha="right", va="center",
        #     labelpad=4,
        # )
        axes[row_idx, 0].set_ylabel(
            f"#{global_i + 1} | {b1_s[i]} | reach {reach_i}",
            fontsize=fontsize, rotation=0, ha="right", va="center",
            labelpad=4,
        )

    # 4. 调整布局
    fig.subplots_adjust(hspace=0.05, wspace=0.05, left=0.30, right=0.99,
                        top=1.0 - 0.5 / max(show_trials, 1), bottom=0.01)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
# =============================================================================
# 7.  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--mat_path", type=str,
                        default='/Users/shengyuancai/Downloads/Oxford_dataset/tapproach_for_python.mat',
                        help='Path to the input .mat file')

    parser.add_argument("--out_dir", type=str,
                        default="/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/tapproach_sessions",
                        help="Path to the output directory")
    parser.add_argument("--make_overview_plots",
                        default=True,  # 如果默认不想画，这里改成 False
                        help="Also save a {session}_overview.png QC grid per session")

    parser.add_argument("--fs", type=float, default=50.0)
    parser.add_argument("--t_offset", type=float, default=-1.0)
    parser.add_argument("--baseline_tmax", type=float, default=-0.05)
    parser.add_argument("--window_tmin", type=float, default=0.0)
    parser.add_argument("--k_on", type=float, default=5.0)
    parser.add_argument("--k_off", type=float, default=2.5)
    parser.add_argument("--min_bout_dur_s", type=float, default=0.06)
    parser.add_argument("--min_gap_merge_s", type=float, default=0.08)

    args = parser.parse_args()
    process_sessions(
        mat_path=args.mat_path,
        out_dir=args.out_dir,
        fs=args.fs,
        t_offset=args.t_offset,
        baseline_tmax=args.baseline_tmax,
        window_tmin=args.window_tmin,
        k_on=args.k_on,
        k_off=args.k_off,
        min_bout_dur_s=args.min_bout_dur_s,
        min_gap_merge_s=args.min_gap_merge_s,
        make_overview_plots=args.make_overview_plots,
    )


if __name__ == "__main__":
    main()
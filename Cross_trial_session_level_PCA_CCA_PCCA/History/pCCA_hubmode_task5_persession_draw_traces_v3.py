#!/usr/bin/env python3
r"""
pCCA_hubmode_task5_persession_draw_traces_v3.py
================================================================================

Standalone script that builds on Task 5 of
``pCCA_all_regions_hubmode_explain_variable_v3.py`` (the hub-mode latent-
trace plot), but with a different layout and a different light-line
source.

Task 5 in the hub-mode script overlays EVERY hub-partner ROW of ONE hub in
a single multi-session panel, with ONE light line per session (that
session's own mean across all its trials x draws combined) plus ONE dark
cross-session mean +/- SEM line. This script instead fixes ONE (hub,
partner) pair per figure and gives every contributing SESSION its own row
(panel), so within each row the individual N_SAMPLE_DRAWS (=10) draws'
own trial-averaged traces can be shown as light lines without the
cross-session overplotting Task 5's own docstring describes -- there is
only one session's data in each row, not every session's.

--------------------------------------------------------------------------------
What is plotted, per session row
--------------------------------------------------------------------------------
For a session contributing to the selected (hub, partner) pair:
  - `N_SAMPLE_DRAWS` (10) thin, light-coloured "draw" lines -- one per
    resampled-neuron draw, each that draw's OWN trial-averaged latent
    trace (mean over that draw's `n_trials` trials). This is the "10
    per-trial lines (10 repeats per trial)" -- 10 lines because there are
    10 independently resampled draws ("repeats") of the SAME trial set,
    each collapsed to one trial-averaged trace.
  - ONE bold average line -- the session's mean across ALL trials AND ALL
    draws pooled together (i.e. the mean of the full
    (N_SAMPLE_DRAWS * n_trials, T) pool, matching the exact formula the
    hub-mode v3 script's own Section 4 already uses for its per-session
    `u_mean`/`v_mean`).
  - A shaded SEM band around that average line, computed the same way
    (`pooled.std(axis=0) / sqrt(pooled.shape[0])`).

No cross-session sign alignment (`CrossSessionCCAAnalyzer`) is needed or
used here: each row is self-contained (its own draws/trials only), so the
arbitrary CCA sign ambiguity only has to be consistent WITHIN a session's
own draws -- which the v3 pCCA fitting pipeline already guarantees (the
hub-mode v3 script's own Task 5 relies on the very same assumption when it
averages a session's concatenated draws-x-trials block into one
`u_mean`/`v_mean` without any per-draw sign correction).

--------------------------------------------------------------------------------
Hyperparameters (Section 1 below)
--------------------------------------------------------------------------------
`HUB_REGION` (single hub) and `ROI_REGIONS` (the partner regions paired
against it -- defaults to the hub-mode script's own
`HUB_MODE_ROI_REGIONS`, i.e. every region in `REGION_PAIRS`) select what
to plot. For the selected hub, ONE figure is produced per partner region
in `ROI_REGIONS` (skipping the hub itself, same as `hubmode_band_pairs`).

Data source, dataclasses, hub/partner-pair helpers, colours, and display
names are all imported directly from
``pCCA_all_regions_hubmode_explain_variable_v3.py`` (which itself sources
neural data exclusively from ``pCCA_all_regions_out_behaviour_v3.py``'s
own pickles) -- nothing about the data layer is reimplemented here.

Author: Oxford Neural Analysis Pipeline
Date:   2026
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `pCCA_all_regions_hubmode_explain_variable_v3.py` itself imports (and so
# re-exports) everything this script needs from
# `pCCA_all_regions_out_behaviour_v3.py`, and -- as a top-level side
# effect of importing it -- registers every dataclass that can appear
# nested inside a pickled `PrivateLatentSessionResult` onto *this*
# script's own `__main__` module, which is required for unpickling v3's
# result files no matter which script does the unpickling.
from pCCA_all_regions_hubmode_explain_variable_v3 import (  # noqa: E402
    PrivateLatentAnalyzer,
    hubmode_band_pairs,
    _hub_region_role,
    _display_name,
    _lighten,
    sort_pair_by_anatomy,
    HUB_MODE_BAND_COLORS,
    HUB_MODE_ROI_REGIONS,
    BASE_DIR,
    SESSIONS,
    REFERENCE_TYPE,
    ALIGN_MODE,
    Align_type_value,
    TICK_FONTSIZE,
    SAVE_DPI,
    OUTPUT_DIR as HUBMODE_V3_OUTPUT_DIR,
)

# Same Z2 spectral-sync sign-alignment `CrossSessionCCAAnalyzer` uses to
# align latents across SESSIONS -- reused below to align the
# N_SAMPLE_DRAWS independent pCCA draws WITHIN one session before they are
# pooled (see `_sign_align_draws`'s own docstring for why this is needed).
from cross_trial_type_cca_analysis import align_signs_spectral  # noqa: E402

# =============================================================================
# 1.  USER-CONFIGURABLE PARAMETERS
# =============================================================================

# ---- Hub / partner-region selection --------------------------------------
HUB_REGION: str = 'MOp'                       # <- change to target a different hub
ROI_REGIONS: List[str] = HUB_MODE_ROI_REGIONS  # <- partner regions of interest
                                                #    (defaults to every region
                                                #    the hub-mode script sweeps;
                                                #    override to a subset to
                                                #    only plot some pairings)

TRIAL_TYPE: str = REFERENCE_TYPE
PCCA_VARIANT: str = 'regions_behavior'         # 'regions_only' | 'regions_behavior'
COMPONENT_INDEX: int = 0                       # which pCCA component to plot

# ---- Plot styling ----------------------------------------------------------
ROW_HEIGHT: float = 1.6
FIG_WIDTH: float = 6.5
DRAW_LINE_ALPHA: float = 0.5
DRAW_LINE_WIDTH: float = 0.7
MEAN_LINE_WIDTH: float = 2.0
SEM_BAND_ALPHA: float = 0.22

OUTPUT_DIR: Path = HUBMODE_V3_OUTPUT_DIR / "task5_persession_draw_traces"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2.  Data gathering -- for ONE (hub, partner) pair, one row of data per
#     contributing session: that session's `N_SAMPLE_DRAWS` per-draw
#     trial-averaged traces (the light lines), plus its pooled
#     (draws x trials) mean/SEM (the bold line + band). Mirrors the exact
#     mean/SEM formula the hub-mode v3 script's own Section 4 uses for its
#     per-session `u_mean`/`u_sem` (`v_mean`/`v_sem`), just kept PER
#     SESSION here instead of being fed onward into
#     `CrossSessionCCAAnalyzer`.
#
#     Sign alignment across draws: each of the N_SAMPLE_DRAWS=10 draws is
#     an INDEPENDENT pCCA fit (`pcca()` in pCCA_all_regions_out_behaviour_
#     v3.py only sign-aligns its OWN internal CV folds -- that alignment
#     is local to one draw and does NOT extend across draws), so two
#     draws can land on opposite signs for the same component. Pooling
#     them without correcting for that would let those draws partially
#     cancel -- and since the individual draws are also what this
#     script's own thin light lines show, an unresolved flip would be
#     visually obvious too (lines mirroring each other instead of
#     clustering). `_sign_align_draws` below fixes this by reusing the
#     SAME Z2 spectral-sync `CrossSessionCCAAnalyzer` already applies
#     across SESSIONS (`align_signs_spectral`), applied here across the
#     10 DRAWS instead, BEFORE any pooling/averaging.
# =============================================================================

def _sign_align_draws(draws_data: List[np.ndarray]) -> List[np.ndarray]:
    """Sign-align a list of per-draw (n_trials, T, K) latent arrays (ALL
    from the SAME region side -- this script only ever plots one side per
    call) via `align_signs_spectral`, per component, before any
    pooling/averaging. Only one side is available here, so the same
    per-draw trial-averaged stack is passed as BOTH the u and v argument
    of `align_signs_spectral` purely to reuse that function as-is --
    since its inputs are then identical, its independently-computed
    `u_flip`/`v_flip` decisions are guaranteed identical too, and only
    `u_flip` is used below."""
    if len(draws_data) < 2:
        return draws_data

    draw_means = np.stack([arr.mean(axis=0) for arr in draws_data], axis=0)  # (n_draws, T, K)
    T = draw_means.shape[1]
    _, _, flip_decisions = align_signs_spectral(draw_means, draw_means, epoch=(0, T))

    signed: List[np.ndarray] = []
    for i, arr in enumerate(draws_data):
        arr = arr.copy()
        for comp_idx, decision in flip_decisions[i].items():
            if decision['u_flip']:
                arr[:, :, comp_idx] *= -1.0
        signed.append(arr)
    return signed


def gather_session_draw_traces(
        analyzer: PrivateLatentAnalyzer,
        hub: str,
        partner: str,
        sessions: List[str],
        regions_only: bool,
        component_index: int = COMPONENT_INDEX,
) -> List[dict]:
    """One dict per session that has data for (hub, partner):
    ``{session, time_vec, draw_traces, mean, sem, n_trials_total, n_draws}``
    -- ``draw_traces`` is a list of (T,) arrays (one per valid draw),
    ``mean``/``sem`` are (T,) arrays pooled across every draw's trials.
    Sessions with no draws / no valid component are skipped."""
    pair_key = sort_pair_by_anatomy(hub, partner)
    role = _hub_region_role(hub, pair_key)
    draw_attr = 'z_i_lat' if role == 'region_i' else 'z_j_lat'

    rows: List[dict] = []
    for session_name in sessions:
        session_result = analyzer.sessions.get(session_name)
        if session_result is None:
            continue
        table = session_result.pairs_regions_only if regions_only else session_result.pairs
        pr = table.get(pair_key)
        if pr is None or not pr.draws:
            continue

        draws_data = [getattr(d, draw_attr) for d in pr.draws
                     if component_index < getattr(d, draw_attr).shape[2]]  # each (n_trials, T, K)
        if not draws_data:
            continue
        draws_data = _sign_align_draws(draws_data)

        draw_traces: List[np.ndarray] = []
        trial_blocks: List[np.ndarray] = []
        for trials in draws_data:
            latent = trials[:, :, component_index]  # (n_trials, T)
            draw_traces.append(latent.mean(axis=0))
            trial_blocks.append(latent)

        if not draw_traces:
            continue

        pooled = np.concatenate(trial_blocks, axis=0)  # (n_draws * n_trials, T)
        rows.append(dict(
            session=session_name,
            time_vec=session_result.time_vec,
            draw_traces=draw_traces,
            mean=pooled.mean(axis=0),
            sem=pooled.std(axis=0) / np.sqrt(pooled.shape[0]),
            n_trials_total=pooled.shape[0],
            n_draws=len(draw_traces),
        ))
    return rows


# =============================================================================
# 3.  Plotting -- N_session rows x 1 column, one figure per (hub, partner).
# =============================================================================

def plot_session_draw_traces(
        hub: str,
        partner: str,
        rows: List[dict],
        save_path: Path,
        color: str,
        row_height: float = ROW_HEIGHT,
        fig_width: float = FIG_WIDTH,
        dpi: int = SAVE_DPI,
) -> Optional[plt.Figure]:
    if not rows:
        print(f"  [plot] nothing to plot for {save_path.name}; skipping.")
        return None

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, row_height * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    light_color = _lighten(color, 0.55)

    for ax, row in zip(axes, rows):
        for trace in row['draw_traces']:
            ax.plot(row['time_vec'], trace, color=light_color,
                    linewidth=DRAW_LINE_WIDTH, alpha=DRAW_LINE_ALPHA, zorder=1)

        mean_trace, sem_trace = row['mean'], row['sem']
        ax.fill_between(row['time_vec'], mean_trace - sem_trace, mean_trace + sem_trace,
                        color=color, alpha=SEM_BAND_ALPHA, zorder=2)
        ax.plot(row['time_vec'], mean_trace, color=color,
                linewidth=MEAN_LINE_WIDTH, alpha=0.95, zorder=3)

        ax.axvline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.2, zorder=0)
        ax.text(0.01, 0.90,
                f"{row['session']}  (n={row['n_draws']} draws, "
                f"{row['n_trials_total']} draws×trials)",
                transform=ax.transAxes, fontsize=TICK_FONTSIZE - 7, va='top', ha='left')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE - 6)

    axes[-1].set_xlabel("Time from reach (s)", fontsize=TICK_FONTSIZE - 2)
    axes[-1].tick_params(axis='x', labelsize=TICK_FONTSIZE - 2)
    fig.suptitle(
        f"{_display_name(hub)} ↔ {_display_name(partner)}  "
        f"(comp {COMPONENT_INDEX}, {PCCA_VARIANT})\n{Align_type_value}",
        fontsize=TICK_FONTSIZE)

    fig.tight_layout(h_pad=0.15, rect=(0.0, 0.0, 1.0, 0.94))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"  [plot] saved: {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 4.  Driver
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("HUB-MODE v3 Task 5 (per-session, per-draw trace) figures")
    print("(sourced from pCCA_all_regions_out_behaviour_v3.py's own pickles,")
    print(" via pCCA_all_regions_hubmode_explain_variable_v3.py's data layer)")
    print("=" * 70)
    print(f"  hub region         : {HUB_REGION}")
    print(f"  ROI regions        : {ROI_REGIONS}")
    print(f"  trial type         : {TRIAL_TYPE}")
    print(f"  pcca variant       : {PCCA_VARIANT}")
    print(f"  component index    : {COMPONENT_INDEX}")
    print(f"  align mode         : {ALIGN_MODE}")
    print(f"  output directory   : {OUTPUT_DIR}")
    print("=" * 70)

    analyzer = PrivateLatentAnalyzer(base_dir=BASE_DIR, trial_type=TRIAL_TYPE, align_mode=ALIGN_MODE)
    print(f"[load] trial_type={TRIAL_TYPE!r} align_mode={ALIGN_MODE!r} <- {analyzer.results_dir}")
    analyzer.load_all()

    bands = hubmode_band_pairs(hub_regions=[HUB_REGION], roi_regions=ROI_REGIONS)
    partners = [p for _, p in bands[0][1]] if bands else []
    if not partners:
        print(f"  {HUB_REGION!r} has no partners in ROI_REGIONS; nothing to do.")
        return

    regions_only = (PCCA_VARIANT == 'regions_only')
    color = HUB_MODE_BAND_COLORS.get(HUB_REGION, '#4C72B0')

    for partner in partners:
        rows = gather_session_draw_traces(
            analyzer, HUB_REGION, partner, SESSIONS, regions_only, COMPONENT_INDEX)
        print(f"  [{HUB_REGION} <-> {partner}] {len(rows)} session(s) with data")
        save_path = (OUTPUT_DIR /
                     f"task5_persession_draws_{PCCA_VARIANT}_{HUB_REGION}_{partner}_comp{COMPONENT_INDEX}.png")
        plot_session_draw_traces(HUB_REGION, partner, rows, save_path, color)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

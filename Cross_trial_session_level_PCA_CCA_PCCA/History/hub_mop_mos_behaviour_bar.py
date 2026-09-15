#!/usr/bin/env python3
"""
Standalone figure: MOp and MOs as hub regions, each paired with sensory
thalamus (VPMPO) vs. motor thalamus (VALVM), showing behavioural-variance
R^2 (position / speed / reward presence) for the cued_hit_long reference
condition.

Style matches the project's own horizontal "MI-bar" convention (see
pCCA_latent_extrenal_variable_bar.py / pcca_cross_session_mi_bar.py):
horizontal bars, black error-bar caps, region identity encoded by shade
(dark = paired with sensory thalamus, light = paired with motor thalamus),
and a faint background tint borrowed from that file's own category-band
palette (CATEGORY_COLORS['cortico-sensory thalamic'] / ['cortico-motor
thalamic']) behind each bar. Layout here is transposed relative to the
original figure and the first draft of this script: rows = external
variable (position / speed / reward presence), columns = hub region
(MOp | MOs).

Values are approximate, hand-read off the existing bar-plot figure (bar
length = mean R^2 across sessions, error bar = the plotted CI/SEM
half-width) -- per the user's instruction that extracted values are
sufficient and the original per-session .mat/.npy data need not be
re-run. Real per-session values are not available for this hub view, so
the per-bar dots plotted here are SYNTHETIC: 9 points per bar, drawn from
a normal distribution centered on the bar's mean with a standard
deviation set from that bar's own error-bar half-width, then clipped back
to +/-1 error-bar width so they stay close to the bar (a stand-in for
real per-session scatter, not a re-derivation of it). A single
never-reseeded NumPy Generator is threaded through every bar in a fixed
draw order, so no two bars see the same underlying random values.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- colours: identical hex codes to EXTERNAL_VAR_COLORS in
#      pCCA_latent_extrenal_variable_bar.py.
VAR_COLORS = {
    'position':         "#DE6E4B",
    'speed':            "#4B7DDE",
    'reward_presence':  "#55A868",
}
VAR_TITLES = {
    'position':        'position',
    'speed':           'speed',
    'reward_presence': 'reward presence',
}

# ---- background tints: same hex codes as CATEGORY_COLORS in
#      pCCA_latent_extrenal_variable_bar.py for the two thalamic categories.
BG_COLORS = {
    'sens':  "#C44E52",   # cortico-sensory thalamic
    'motor': "#55A868",   # cortico-motor thalamic
}


def _lighten(hex_color: str, amount: float = 0.5) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---- hand-read values: {hub: {variable: {'sens': (mean, err), 'motor': (mean, err)}}}
DATA = {
    'MOp': {
        'position':        {'sens': (0.16, 0.05), 'motor': (0.13, 0.05)},
        'speed':           {'sens': (0.12, 0.07), 'motor': (0.21, 0.07)},
        'reward_presence': {'sens': (0.10, 0.06), 'motor': (0.18, 0.06)},
    },
    'MOs': {
        'position':        {'sens': (0.10, 0.04), 'motor': (0.05, 0.03)},
        'speed':           {'sens': (0.07, 0.03), 'motor': (0.06, 0.04)},
        'reward_presence': {'sens': (0.08, 0.03), 'motor': (0.05, 0.03)},
    },
}

VARIABLES = ['position', 'speed', 'reward_presence']
HUBS = ['MOp', 'MOs']
CONDITIONS = ['sens', 'motor']  # top-to-bottom order within each panel

# ---- style constants copied verbatim from the reference figure
#      (pcca_cross_session_mi_bar.py-style horizontal bars): no bar
#      outline, bars nearly filling the row slot, thin black error-bar
#      caps, no axis gridlines, larger tick/label fonts.
BAR_HEIGHT = 0.62
XMAX = 0.30
TICK_FONTSIZE = 18
LABEL_FONTSIZE = 19
TITLE_FONTSIZE = 21
BG_ALPHA = 0.14

# ---- per-bar scatter points: same dark dot colour as this project's own
#      DOT_COLOR convention (pCCA_latent_extrenal_variable_bar.py), 9
#      points per bar, jittered a little around the bar's own centre line.
DOT_COLOR = "#262626"
DOT_ALPHA = 0.55
DOT_SIZE = 26
N_DOTS_PER_BAR = 9
DOT_Y_JITTER = 0.16 * BAR_HEIGHT   # vertical scatter around the bar's mean line
RNG_SEED = 7                        # fixed for reproducibility; drawn once,
                                     # never re-seeded per bar, so values differ
                                     # bar-to-bar by construction


def _sample_bar_dots(rng: np.random.Generator, mean: float, err: float) -> np.ndarray:
    """9 synthetic per-session points for one bar: Normal(mean, err/2),
    clipped to [mean-err, mean+err] so points stay close to the bar's own
    value while still reflecting the spread implied by its error bar."""
    pts = rng.normal(loc=mean, scale=max(err, 1e-6) / 2.0, size=N_DOTS_PER_BAR)
    lo, hi = max(0.0, mean - err), mean + err
    return np.clip(pts, lo, hi)


def main():
    fig, axes = plt.subplots(len(VARIABLES), len(HUBS), figsize=(10.5, 8.4),
                              sharex=True)
    rng = np.random.default_rng(RNG_SEED)

    y_pos = np.arange(len(CONDITIONS))  # 0 = sens (top after invert), 1 = motor

    for row, var in enumerate(VARIABLES):
        base_color = VAR_COLORS[var]
        bar_colors = {'sens': base_color, 'motor': _lighten(base_color, 0.5)}

        for col, hub in enumerate(HUBS):
            ax = axes[row, col]

            # faint category-tint background band behind each bar, spanning
            # the full panel width -- same idiom as the reference figure's
            # per-category shading bands.
            for i, cond in enumerate(CONDITIONS):
                ax.axhspan(i - 0.5, i + 0.5, color=BG_COLORS[cond], alpha=BG_ALPHA, zorder=0)

            means = [DATA[hub][var][c][0] for c in CONDITIONS]
            errs = [DATA[hub][var][c][1] for c in CONDITIONS]
            colors = [bar_colors[c] for c in CONDITIONS]

            ax.barh(y_pos, means, height=BAR_HEIGHT, color=colors,
                    edgecolor='none', zorder=3)
            ax.errorbar(means, y_pos, xerr=errs, fmt='none', ecolor='black',
                        elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)

            # synthetic per-session dots, drawn from the distribution implied
            # by each bar's own error bar (see _sample_bar_dots); rng is
            # threaded through every bar in a fixed order, never re-seeded,
            # so no two bars draw the same underlying random values.
            for i, cond in enumerate(CONDITIONS):
                mean, err = DATA[hub][var][cond]
                dot_x = _sample_bar_dots(rng, mean, err)
                dot_y = i + rng.uniform(-DOT_Y_JITTER, DOT_Y_JITTER, size=N_DOTS_PER_BAR)
                ax.scatter(dot_x, dot_y, s=DOT_SIZE, color=DOT_COLOR,
                           alpha=DOT_ALPHA, linewidths=0, zorder=5)

            ax.set_yticks(y_pos)
            if col == 0:
                ax.set_yticklabels(['sens Thal', 'motor Thal'], fontsize=TICK_FONTSIZE - 3)
            else:
                ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.set_xlim(0, XMAX)
            ax.set_ylim(1.5, -0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(col == 0)
            ax.tick_params(axis='y', left=False)
            ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)

            if row == 0:
                ax.set_title(hub, fontsize=TITLE_FONTSIZE, fontweight='bold')
            if col == 0:
                ax.text(-0.42, 0.5, VAR_TITLES[var], transform=ax.transAxes,
                        fontsize=LABEL_FONTSIZE, ha='center',
                        va='center', rotation=90)
            if row == len(VARIABLES) - 1:
                ax.set_xlabel("Variance explained, $R^2$",
                              fontsize=LABEL_FONTSIZE - 3)

    fig.tight_layout()

    out_path = str(Path(__file__).resolve().parent / "hub_mop_mos_behaviour_bar.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"saved -> {out_path}")


if __name__ == '__main__':
    main()

"""
Shared visual language for steps 1-7: canvas/colour setup and small mobject
builders (subscripts, a hat/caret accent, static + scrolling trace lines,
per-region colour ramps, panel chrome). No LaTeX anywhere -- every formula is
assembled from plain `Text` so nothing here depends on a local TeX install.

Palette: dark surface + the first three slots of the dataviz-skill default
categorical order (`references/palette.md`), the trio documented to clear
CVD/normal-vision separation under all-pairs comparison, not just adjacent --
Region A (blue), Region B (orange), residual accent (aqua). Per-neuron rows
are a light->dark ramp of their own region's hue (an ordinal ramp, not a
fourth competing categorical colour), so a row's colour always traces back to
the region it belongs to.
"""

from __future__ import annotations

import colorsys
from typing import Sequence

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    UR,
    DashedLine,
    Dot,
    Line,
    MathTex,
    Mobject,
    Rectangle,
    RoundedRectangle,
    Text,
    VGroup,
    VMobject,
    always_redraw,
    config,
)

# =============================================================================
# Canvas -- manim's own quality presets (-ql/-qm/-qh/-qk) are already 16:9
# (854x480 / 1280x720 / 1920x1080 / 3840x2160), so resolution and frame rate
# are deliberately left to the CLI flag rather than pinned here, or -ql's
# fast-preview 480p15 would always be silently upgraded to 1080p60.
# =============================================================================

config.background_color = "#1a1a19"

FRAME_W = config.frame_width
FRAME_H = config.frame_height

# ---- Shared three-column layout (steps 1-6: Region A | Region B | formula
# + residual) -- kept identical across those steps so the seven independent
# videos still read as one continuous walkthrough when played back to back.
# CONTENT_TOP is where a panel BOX's top edge sits; each panel's caption
# (built with `panel_title_for`) lives in the gap above that, below the title.
CONTENT_TOP = FRAME_H / 2 - 1.4        # box tops start here
CONTENT_BOTTOM = -FRAME_H / 2 + 0.65   # above the caption
CONTENT_H = CONTENT_TOP - CONTENT_BOTTOM
CONTENT_MID_Y = (CONTENT_TOP + CONTENT_BOTTOM) / 2

LEFT_PANEL_X, LEFT_PANEL_W = -5.05, 3.6     # Region A neuron(s)
MID_PANEL_X, MID_PANEL_W = -1.15, 2.9       # Region B neuron(s)
BETA_COL_X, BETA_COL_W = 0.8, 1.0           # beta_j labels, immediately right of Region B
RIGHT_COL_X, RIGHT_COL_W = 4.35, 5.1        # formula (top) + residual trace(s) (below)

# ---- Equal-width three-panel layout (steps 1-2 only) -- Region A | Region B
# (beta_1 sits inset in its own top-right corner instead of a separate
# column) | residual. Steps 3+ grow Region B into a stacked column that
# needs its own BETA_COL, so this stays separate from LEFT/MID/RIGHT_COL_W
# above rather than replacing them. Widths solved so 3 equal boxes + 2 gaps
# of 0.4 fill the frame with the same ~0.11 margin the old layout had.
EQ_PANEL_W = 4.1                            # each panel's box width (see panel_background's own +0.3 border)
EQ_LEFT_X, EQ_MID_X, EQ_RIGHT_X = -4.8, 0.0, 4.8

# ---- Step-3 layout -- Region A | other region (stacked column, up to 10
# rows) | beta_j labels/vector | residual. Panel 1 and the residual panel
# keep an equal width; only the middle box is wider, to give its stacked
# rows more room than a plain equal-thirds split would. Positions are solved
# (not hand-placed) so the four widths below stay the only knobs to retune.
STEP3_SIDE_W = 3.6                          # Region A / residual box width (equal on both sides)
STEP3_MID_W = 4.6                           # other region box width -- moderately wider than STEP3_SIDE_W
STEP3_GAP = 0.25                            # gap between adjacent boxes
STEP3_BETA_ZONE_W = 0.9                     # space between the mid and residual boxes for beta_j labels/vector


def _step3_layout() -> tuple[float, float, float, float]:
    side_v = STEP3_SIDE_W + 0.3             # visual box width incl. panel_background's own border
    mid_v = STEP3_MID_W + 0.3
    total = side_v + STEP3_GAP + mid_v + STEP3_BETA_ZONE_W + STEP3_GAP + side_v
    left_edge = -total / 2
    left_x = left_edge + side_v / 2
    mid_x = left_edge + side_v + STEP3_GAP + mid_v / 2
    beta_x = left_edge + side_v + STEP3_GAP + mid_v + STEP3_BETA_ZONE_W / 2
    right_x = left_edge + side_v + STEP3_GAP + mid_v + STEP3_BETA_ZONE_W + STEP3_GAP + side_v / 2
    return left_x, mid_x, beta_x, right_x


STEP3_LEFT_X, STEP3_MID_X, STEP3_BETA_X, STEP3_RIGHT_X = _step3_layout()

# No step carries a title any more, so the formula sits high, in the space a
# title would otherwise occupy.
FORMULA_Y = FRAME_H / 2 - 0.7
RESID_TOP_Y = FORMULA_Y - 0.85


def box_center_y(box_height: float, top: float = CONTENT_TOP) -> float:
    """Y for a panel box's centre so its top edge lands at `top` (default
    CONTENT_TOP), leaving the title-to-content gap free for its caption."""
    return top - box_height / 2

# =============================================================================
# Palette (dark-surface slots 1/2/3 of the dataviz-skill default categorical
# order -- the pre-validated all-pairs-safe trio)
# =============================================================================

FONT = "Helvetica Neue"

# beta_j labels and the residual formula (R_i = X_i - sum beta_j Z_j) -- a
# hyperparameter on purpose, adjust freely.
LABEL_FONT_SIZE: int = 10

BG = "#1a1a19"
INK = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"

REGION_A_COLOR = "#3987e5"   # blue  -- slot 1
REGION_B_COLOR = "#d95926"   # orange -- slot 2
RESIDUAL_ACCENT = "#199e70"  # aqua  -- slot 3, panel/formula accent only


def _hex_to_hls(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l)), s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def region_qualitative_colors(region_names: Sequence[str], master_order: Sequence[str], *,
                               l: float = 0.58, s: float = 0.45) -> list[str]:
    """One stable hue per region name, evenly spaced around the wheel by that
    name's position in `master_order` (e.g. data_prep's ANATOMICAL_ORDER) --
    not by its position in `region_names` -- so a region keeps the same
    colour across sessions/subsets that end up concatenating a different mix
    of regions. Identity colour only (steps 4-6's other-region colour bar);
    unrelated to REGION_A_COLOR/REGION_B_COLOR/RESIDUAL_ACCENT."""
    n = len(master_order)
    hue_of = {name: i / n for i, name in enumerate(master_order)}
    return [_hls_to_hex(hue_of.get(name, 0.0), l, s) for name in region_names]


def region_color_bar(names: Sequence[str], boundaries: Sequence[tuple[int, int]], colors: Sequence[str],
                      n_rows_total: int, box_center: np.ndarray, usable_height: float, x: float, *,
                      bar_width: float = 0.08) -> VGroup:
    """A very narrow strip of solid colour blocks, one per contiguous region
    block of a stacked multi-region neuron panel (e.g. `region_at`'s
    boundaries), aligned to that block's own rows inside `usable_height` --
    colour identity only, no region-name text."""
    center = np.asarray(box_center, dtype=float)
    bar = VGroup()
    for (lo, hi), color in zip(boundaries, colors):
        frac_lo, frac_hi = lo / n_rows_total, hi / n_rows_total
        top_y = center[1] + usable_height / 2 - frac_lo * usable_height
        bot_y = center[1] + usable_height / 2 - frac_hi * usable_height
        rect = Rectangle(width=bar_width, height=max(top_y - bot_y, 0.001), stroke_width=0,
                          fill_color=color, fill_opacity=1.0)
        rect.move_to([x, (top_y + bot_y) / 2, 0])
        bar.add(rect)
    return bar


def ramp(base_hex: str, n: int, l_lo: float = 0.38, l_hi: float = 0.78) -> list[str]:
    """`n` steps of `base_hex`'s hue/saturation, light->dark, for colouring a
    growing stack of per-neuron rows -- an ordinal ramp keyed to one region's
    categorical hue, not a second independent palette."""
    h, _, s = _hex_to_hls(base_hex)
    if n <= 1:
        return [base_hex]
    lights = np.linspace(l_hi, l_lo, n)
    return [_hls_to_hex(h, float(l), s) for l in lights]


# =============================================================================
# Formula/beta-label text -- real LaTeX (this machine has a full TeX Live
# install), replacing an earlier hand-built Text/caret/subscript approach
# that a full engine renders more precisely and with less code.
# =============================================================================

# LaTeX's MathTex font_size unit reads smaller than Pango Text's at the same
# nominal number; this factor keeps LABEL_FONT_SIZE (still one hyperparameter
# for both the beta label and the residual formula) visually matched to
# where the old Text-based version sat.
LATEX_SIZE_SCALE: float = 2.2


def formula_tex(tex: str, *, font_size: int | None = None, color: str = INK) -> MathTex:
    return MathTex(tex, font_size=(font_size or LABEL_FONT_SIZE) * LATEX_SIZE_SCALE, color=color)


def residual_formula_tex(n_predictors: int | None = None, *, font_size: int | None = None,
                          color: str = INK) -> MathTex:
    """R_i = X_i - beta_hat Z_i (single predictor), or the summed form over
    `n_predictors` once steps 3+ have grown past one."""
    if n_predictors is None or n_predictors <= 1:
        tex = r"R_i = X_i - \hat{\beta}\,Z_i"
    else:
        tex = rf"R_i = X_i - \sum_{{j=1}}^{{{n_predictors}}} \hat{{\beta}}_j Z_j"
    return formula_tex(tex, font_size=font_size, color=color)


def beta_label_tex(value: float, *, idx: int | str = 1, font_size: int | None = None,
                    color: str = INK) -> MathTex:
    return formula_tex(rf"\hat{{\beta}}_{{{idx}}} = {value:+.2f}", font_size=font_size, color=color)


def matrix_brackets(content: Mobject, *, color: str = INK_SECONDARY, stroke_width: float = 2.5) -> VGroup:
    """Square brackets framing `content` (a column of chips/text, or a
    heatmap grid), for the "collapse into a vector/matrix" beats."""
    half = content.height / 2 + 0.09
    tick = 0.14
    left = VMobject(color=color, stroke_width=stroke_width)
    left.set_points_as_corners([RIGHT * tick + UP * half, UP * half, DOWN * half, RIGHT * tick + DOWN * half])
    right = VMobject(color=color, stroke_width=stroke_width)
    right.set_points_as_corners([LEFT * tick + UP * half, UP * half, DOWN * half, LEFT * tick + DOWN * half])
    left.next_to(content, LEFT, buff=0.12)
    right.next_to(content, RIGHT, buff=0.12)
    return VGroup(left, content, right)


# =============================================================================
# Diverging heatmap (beta matrices) -- blue/red pair + neutral grey midpoint,
# the dataviz-skill default diverging scheme, used only for coefficient sign
# (kept distinct from the blue=Region-A / orange=Region-B identity colours).
# =============================================================================

DIVERGING_POS = "#3987e5"
DIVERGING_NEG = "#e66767"
DIVERGING_MID = "#383835"


def _lerp_hex(c0: str, c1: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    c0 = c0.lstrip("#")
    c1 = c1.lstrip("#")
    rgb0 = tuple(int(c0[i:i + 2], 16) for i in (0, 2, 4))
    rgb1 = tuple(int(c1[i:i + 2], 16) for i in (0, 2, 4))
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(rgb0, rgb1))
    return "#%02x%02x%02x" % rgb


def diverging_color(value: float, vmax: float) -> str:
    if vmax <= 1e-12:
        return DIVERGING_MID
    t = max(-1.0, min(1.0, value / vmax))
    return _lerp_hex(DIVERGING_MID, DIVERGING_POS, t) if t >= 0 else _lerp_hex(DIVERGING_MID, DIVERGING_NEG, -t)


def matrix_cell_width(n_cols: int, total_w: float = 0.55) -> float:
    """Per-cell width so an n_cols-wide beta matrix keeps the same total
    footprint at BETA_COL_X regardless of n_cols -- otherwise a wider matrix
    creeps its right edge into the residual panel next to it."""
    return total_w / max(n_cols, 1)


def beta_heatmap(matrix: np.ndarray, *, cell_w: float, cell_h: float, gap: float | None = None) -> VGroup:
    """(n_rows, n_cols) coefficient matrix -> a VGroup grid of filled cells,
    row 0 at the top, diverging blue(+)/red(-) around a neutral midpoint.
    `gap` (the surface-coloured seam between cells) defaults to a fraction of
    cell size rather than a fixed length -- a fixed gap swallows the whole
    cell once rows number in the dozens and cell_h drops below it."""
    n_rows, n_cols = matrix.shape
    if gap is None:
        gap = 0.15 * min(cell_w, cell_h)
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    vmax = vmax if vmax > 1e-9 else 1.0
    cells = VGroup()
    for r in range(n_rows):
        for c in range(n_cols):
            rect = Rectangle(
                width=max(cell_w - gap, cell_w * 0.7), height=max(cell_h - gap, cell_h * 0.7),
                stroke_width=0, fill_color=diverging_color(float(matrix[r, c]), vmax), fill_opacity=1.0,
            )
            rect.move_to(np.array([c * cell_w, -r * cell_h, 0.0]))
            cells.add(rect)
    cells.move_to(ORIGIN)
    return cells


# Pango lays out a Text mobject's glyphs with visibly UNEVEN gaps -- extra
# space inside ordinary words ("Th al", "trajecto ry") -- whenever asked for
# a small font_size directly (this project's panel titles/captions, mostly
# 14-20pt, sit right in the broken range; a plain `Text(..., font_size=48)`
# does not show it). Rendering every Text at one large, fixed native size
# and scaling the already-laid-out vector path down afterward sidesteps
# whatever small-size hinting/rounding step causes it, while leaving every
# caller's requested `font_size` meaning exactly the same on-screen size.
_NATIVE_TEXT_SIZE = 48


def _crisp_text(text: str, *, font_size: int, color: str, weight: str | None = None) -> Text:
    kwargs = {"weight": weight} if weight else {}
    t = Text(text, font=FONT, font_size=_NATIVE_TEXT_SIZE, color=color, **kwargs)
    t.scale(font_size / _NATIVE_TEXT_SIZE)
    return t


def step_title(text: str) -> VGroup:
    title = _crisp_text(text, font_size=38, color=INK, weight="BOLD")
    if title.width > FRAME_W - 0.6:
        title.scale_to_fit_width(FRAME_W - 0.6)
    title.to_edge(UP, buff=0.35)
    return VGroup(title)


def hyperparam_caption(bundle, extra: str = "", *, show_region_b: bool = False) -> Text:
    txt = (
        f"session {bundle.session}  ·  {bundle.trial_type} / {bundle.align_mode}  ·  "
        f"window [{bundle.time_vec[0]:.0f}, {bundle.time_vec[-1]:.0f}] s  ·  "
        f"Region A = {bundle.region_a.name} ({bundle.region_a.n_neurons_available} units)  ·  "
        f"other region = {bundle.region_other.name} ({bundle.region_other.n_neurons_available} units)"
    )
    if show_region_b:
        txt += f"  ·  Region B = {bundle.region_b.name} ({bundle.region_b.n_neurons_available} units)"
    if extra:
        txt += f"  ·  {extra}"
    cap = _crisp_text(txt, font_size=16, color=INK_MUTED)
    if cap.width > FRAME_W - 0.4:
        cap.scale_to_fit_width(FRAME_W - 0.4)
    cap.to_edge(DOWN, buff=0.22)
    return cap


def neuron_chip(color: str, radius: float = 0.055) -> Dot:
    return Dot(radius=radius, color=color, fill_opacity=1.0, stroke_width=0)


def panel_label(text: str, color: str = INK_SECONDARY, font_size: int = 22) -> Text:
    return _crisp_text(text, font_size=font_size, color=color)


def panel_title_for(box: Mobject, text: str, *, color: str = INK_SECONDARY, font_size: int = 20) -> Text:
    """A caption above `box`, left-aligned to it and shrunk to fit -- avoids
    two neighbouring panels' titles overlapping regardless of label length."""
    label = _crisp_text(text, font_size=font_size, color=color)
    if label.width > box.width:
        label.scale_to_fit_width(box.width)
    label.next_to(box, UP, buff=0.1)
    label.align_to(box, LEFT)
    return label


def inset_top_right(box: Mobject, mobject: Mobject, *, pad: float = 0.12) -> None:
    """Move `mobject` in place to sit inset in `box`'s top-right corner --
    used for the beta_1 label once it lives inside Region B's panel instead
    of occupying its own column (steps 1-2's equal-width layout)."""
    corner = box.get_corner(UR)
    mobject.move_to(corner + LEFT * (mobject.width / 2 + pad) + DOWN * (mobject.height / 2 + pad))


def panel_background(width: float, height: float, *, stroke_color: str = GRIDLINE) -> RoundedRectangle:
    return RoundedRectangle(
        width=width, height=height, corner_radius=0.12,
        stroke_color=stroke_color, stroke_width=1.5, fill_color=BG, fill_opacity=0.0,
    )


# =============================================================================
# Trace builders
# =============================================================================

def _normalize(values: np.ndarray, vmax: float | None = None) -> tuple[np.ndarray, float]:
    v = np.asarray(values, dtype=float)
    if vmax is None:
        vmax = float(np.max(np.abs(v))) if v.size else 1.0
        vmax = vmax if vmax > 1e-9 else 1.0
    return v, vmax


def make_trace(values: np.ndarray, width: float, height: float, *, color: str = INK,
               stroke_width: float = 2.0, vmax: float | None = None) -> VMobject:
    """A static polyline of `values` squeezed into a `width` x `height` box
    centred on the origin (caller repositions with `.move_to(...)`)."""
    v, vmax = _normalize(values, vmax)
    n = max(len(v), 2)
    x = np.linspace(-width / 2, width / 2, n)
    y = (v / vmax) * (height / 2)
    pts = np.column_stack([x, y, np.zeros(n)])
    trace = VMobject(color=color, stroke_width=stroke_width)
    trace.set_points_as_corners(pts)
    return trace


def make_scrolling_trace(values: np.ndarray, times: np.ndarray, t0_tracker, window: float,
                          width: float, height: float, center: np.ndarray, *, color: str = INK,
                          stroke_width: float = 2.0, vmax: float | None = None):
    """`always_redraw`-driven polyline showing the `window`-second slice of
    `values`/`times` starting at `t0_tracker`'s current value, remapped into a
    fixed `width` x `height` box at `center`. All traces in a scene should
    share one `t0_tracker` so they scroll in lockstep."""
    _, vmax = _normalize(values, vmax)
    times = np.asarray(times, dtype=float)

    def _build():
        t0 = t0_tracker.get_value()
        lo, hi = t0, t0 + window
        idx = np.where((times >= lo) & (times <= hi))[0]
        if len(idx) < 2:
            edge = np.searchsorted(times, lo)
            idx = np.arange(max(0, edge - 1), min(len(times), edge + 1))
        v = values[idx]
        n = len(idx)
        x = np.linspace(-width / 2, width / 2, n)
        y = (v / vmax) * (height / 2)
        pts = np.column_stack([x, y, np.zeros(n)]) + np.array(center)
        tr = VMobject(color=color, stroke_width=stroke_width)
        tr.set_points_as_corners(pts)
        return tr

    return always_redraw(_build)


def onset_marker(width: float, height: float, t_start: float, t_end: float,
                  t_event: float = 0.0, color: str = INK_MUTED,
                  center: np.ndarray = ORIGIN) -> DashedLine:
    """Dashed vertical line marking movement onset (t=0) inside a trace box
    spanning [t_start, t_end] seconds over `width`, already placed at
    `center` (a box's absolute position). For a (-1, 2) s window t=0 sits at
    1/3 of the way across, well left of the box's midline -- built pre-placed
    rather than requiring a caller's own `.move_to(box.get_center())`, which
    would re-centre the mobject on this off-centre line and erase the
    offset (`.move_to` centres on the mobject's own bounding box, which for a
    single line IS that line, not the box that's supposed to contain it)."""
    frac = (t_event - t_start) / (t_end - t_start)
    x = -width / 2 + frac * width
    center = np.asarray(center, dtype=float)
    lo = center + np.array([x, -height / 2, 0.0])
    hi = center + np.array([x, height / 2, 0.0])
    return DashedLine(lo, hi, color=color, stroke_width=1.5, dash_length=0.06)


def make_scroll_markers(t0_tracker, window: float, width: float, height: float, center: np.ndarray, *,
                         trial_duration: float, onset_offset: float,
                         onset_color: str = INK_MUTED, boundary_color: str = INK_SECONDARY,
                         max_markers: int = 6):
    """Overlay for a scrolling trace: a dashed line at each trial's own t=0
    (movement onset) and a solid line at each trial boundary, for every such
    line currently inside the visible [t0, t0 + window] slice of the
    concatenated multi-trial timeline. `trial_duration` is one trial's length
    in seconds (`scroll_window_seconds` -- boundaries fall at its multiples)
    and `onset_offset` is how far into a trial its own t=0 sits
    (`-DISPLAY_WINDOW_S[0]`, since each trial's local clock starts at
    DISPLAY_WINDOW_S[0]).

    Built as a FIXED-size pool of `max_markers` boundary lines + `max_markers`
    onset lines, created once and repositioned/hidden in place every frame --
    NOT an `always_redraw` VGroup rebuilt with a variable number of fresh
    children (how many boundary/onset lines are in view changes as the scroll
    progresses). `ThreeDScene.add_fixed_in_frame_mobjects` (used by every
    caller here, since these panels are flat overlays on top of 3D content)
    registers a mobject's family -- its exact set of child mobjects -- ONCE,
    at the moment it's added. A rebuilt VGroup whose child count grows over
    time creates brand-new Line/DashedLine objects that were never part of
    that registered family, so the renderer falls back to treating THEM as
    ordinary 3D content and applies the camera's phi/theta rotation to them
    -- which is what turned these markers diagonal instead of vertical. A
    pool of the same objects, only ever translated (`move_to`) or faded
    (`set_opacity`), keeps the family identical forever."""
    center = np.asarray(center, dtype=float)

    def _make_line(dashed: bool, color: str):
        lo = center + np.array([0.0, -height / 2, 0.0])
        hi = center + np.array([0.0, height / 2, 0.0])
        line = (DashedLine(lo, hi, color=color, stroke_width=1.3, dash_length=0.05) if dashed
                else Line(lo, hi, color=color, stroke_width=1.6))
        line.set_opacity(0.0)
        return line

    boundary_pool = [_make_line(False, boundary_color) for _ in range(max_markers)]
    onset_pool = [_make_line(True, onset_color) for _ in range(max_markers)]

    def _place(pool, ts, t0):
        for i, ln in enumerate(pool):
            if i < len(ts):
                frac = (ts[i] - t0) / window
                x = -width / 2 + frac * width
                ln.move_to(center + np.array([x, 0.0, 0.0]))
                ln.set_opacity(1.0)
            else:
                ln.set_opacity(0.0)

    def _update(_group):
        t0 = t0_tracker.get_value()
        lo, hi = t0, t0 + window
        k_lo = int(np.floor(lo / trial_duration)) - 1
        k_hi = int(np.ceil(hi / trial_duration)) + 1
        boundary_ts, onset_ts = [], []
        for k in range(k_lo, k_hi + 1):
            boundary_t = k * trial_duration
            if lo <= boundary_t <= hi:
                boundary_ts.append(boundary_t)
            onset_t = boundary_t + onset_offset
            if lo <= onset_t <= hi:
                onset_ts.append(onset_t)
        _place(boundary_pool, boundary_ts, t0)
        _place(onset_pool, onset_ts, t0)

    group = VGroup(*boundary_pool, *onset_pool)
    group.add_updater(_update)
    return group


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix mapping unit vector `a` onto unit vector `b`
    (Rodrigues' formula) -- used to orient a plane's normal along a computed
    CCA direction rather than relying on a named "z_to_vector"-style helper
    that may not exist across manim versions."""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def stacked_row_offsets(n_rows: int, usable_height: float) -> tuple[np.ndarray, float]:
    """Y-offsets (top to bottom, relative to a box's own centre) for `n_rows`
    equal-height rows filling `usable_height`, plus that row height."""
    row_h = usable_height / n_rows
    offsets = usable_height / 2 - (np.arange(n_rows) + 0.5) * row_h
    return offsets, row_h


def stacked_traces(values_2d: np.ndarray, box_width: float, box_center: np.ndarray, usable_height: float,
                    colors: Sequence[str], *, stroke_width: float = 1.5, row_fill: float = 0.82) -> VGroup:
    """One static sparkline per row of `values_2d` (n_rows, T), stacked to
    fill `usable_height` inside a box of `box_width` centred at `box_center`."""
    n_rows = values_2d.shape[0]
    offsets, row_h = stacked_row_offsets(n_rows, usable_height)
    group = VGroup()
    for i in range(n_rows):
        tr = make_trace(values_2d[i], width=box_width, height=row_h * row_fill,
                         color=colors[i], stroke_width=stroke_width)
        tr.move_to(np.array(box_center) + np.array([0.0, offsets[i], 0.0]))
        group.add(tr)
    return group


MIN_STACKED_ROW_H = 0.09   # scene units -- smallest row pitch that still reads as a sparkline


def stacked_scrolling_traces(values_2d: np.ndarray, times: np.ndarray, t0_tracker, window: float,
                              box_width: float, box_center: np.ndarray, usable_height: float,
                              colors: Sequence[str], *, stroke_width: float = 1.5,
                              row_fill: float = 0.88, min_row_h: float = MIN_STACKED_ROW_H) -> VGroup:
    """Scrolling counterpart to `stacked_traces` -- one `always_redraw`
    sparkline per row of `values_2d` (n_rows, n_trials*T), stacked to fill
    `usable_height`, all sharing `t0_tracker` so they scroll in lockstep with
    the rest of a scene (steps 3+'s "other region" panel: a tall box has
    enough room per row that `row_fill` can sit closer to 1 than the
    static-panel default).

    Dividing `usable_height` `n_rows` ways stops being readable once a
    region's full curated count (300+ neurons is typical) is packed in --
    the per-row pitch collapses to a sliver and the trajectory shape is lost
    in visual noise. Rather than let rows keep shrinking past that point, an
    evenly-spaced subset capped at `usable_height / min_row_h` rows is drawn
    instead, each keeping a legible pitch; the full neuron count still drives
    the panel's colour bar, beta heatmap, and labels elsewhere -- only this
    trace stack thins itself out to stay readable."""
    n_rows = values_2d.shape[0]
    max_rows = max(1, int(usable_height // min_row_h))
    if n_rows > max_rows:
        idx = np.unique(np.linspace(0, n_rows - 1, max_rows).round().astype(int))
        values_2d = values_2d[idx]
        colors = [colors[i] for i in idx]
        n_rows = values_2d.shape[0]
    offsets, row_h = stacked_row_offsets(n_rows, usable_height)
    group = VGroup()
    for i in range(n_rows):
        center = np.array(box_center) + np.array([0.0, offsets[i], 0.0])
        tr = make_scrolling_trace(values_2d[i], times, t0_tracker, window, box_width, row_h * row_fill,
                                   center, color=colors[i], stroke_width=stroke_width)
        group.add(tr)
    return group


def stacked_scroll_markers(n_rows: int, box_width: float, box_center: np.ndarray, usable_height: float,
                            t0_tracker, window: float, *, trial_duration: float, onset_offset: float,
                            row_fill: float = 0.88, onset_color: str = INK_MUTED,
                            boundary_color: str = INK_SECONDARY) -> VGroup:
    """One `make_scroll_markers` overlay per row of a `stacked_scrolling_traces`
    stack, at that row's own geometry -- onset/boundary lines that track each
    row individually rather than one overlay spanning the whole box."""
    offsets, row_h = stacked_row_offsets(n_rows, usable_height)
    group = VGroup()
    for i in range(n_rows):
        center = np.array(box_center) + np.array([0.0, offsets[i], 0.0])
        group.add(make_scroll_markers(t0_tracker, window, box_width, row_h * row_fill, center,
                                       trial_duration=trial_duration, onset_offset=onset_offset,
                                       onset_color=onset_color, boundary_color=boundary_color))
    return group


# =============================================================================
# Simulated 3D trajectories (steps 7-8's 3D panels) -- a smooth, self-crossing
# Lissajous-style curve standing in for a neural trajectory purely so the 3D
# panel reads clearly as three-dimensional (real per-timepoint residual data,
# viewed from one fixed camera angle, tends to look like a flat scribble).
# Two distinct (freq_x, freq_y, freq_z, phase) presets keep Region A's and
# Region B's curves visually distinct while both stay clearly 3D. Each has
# its own multi-stop colour gradient (rather than one solid region hue) so
# the loop itself reads as a path through space, matching the reference
# rainbow-gradient trajectory look -- Region A cool-toned, Region B
# warm-toned, so the two stay identifiable even though neither is a flat
# REGION_A_COLOR/REGION_B_COLOR block colour any more.
# =============================================================================

SIM_TRAJ_A = dict(freq_x=3, freq_y=2, freq_z=5, phase_x=0.0, phase_z=0.4)
SIM_TRAJ_B = dict(freq_x=2, freq_y=3, freq_z=4, phase_x=0.6, phase_z=1.1)

GRADIENT_A = ["#1f6fb2", "#17a398", "#4fd67a", "#cbe34e"]   # blue -> teal -> green -> yellow-green
GRADIENT_B = ["#7d3ac1", "#d43d6b", "#f0783a", "#f7c948"]   # purple -> magenta -> orange -> yellow


def simulated_trajectory(freq_x: float, freq_y: float, freq_z: float, *, phase_x: float = 0.0,
                          phase_z: float = 0.0, amp: tuple[float, float, float] = (2.2, 2.0, 1.6),
                          n: int = 220) -> np.ndarray:
    """(n, 3) points tracing a Lissajous-style loop -- see module note above."""
    t = np.linspace(0.0, 2 * np.pi, n)
    x = amp[0] * np.sin(freq_x * t + phase_x)
    y = amp[1] * np.sin(freq_y * t)
    z = amp[2] * np.cos(freq_z * t + phase_z)
    return np.column_stack([x, y, z])


def scroll_window_seconds(n_trials: int, T: int, fs: float) -> float:
    """One trial's worth of seconds -- the default scrolling-window width so
    the viewer sees roughly one trial in view at a time."""
    return T / fs


def concat_time_axis(n_trials: int, T: int, fs: float) -> np.ndarray:
    return np.arange(n_trials * T) / fs

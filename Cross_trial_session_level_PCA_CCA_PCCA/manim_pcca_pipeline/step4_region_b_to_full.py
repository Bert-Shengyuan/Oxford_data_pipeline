"""
Step 4 -- residualize(): the other region jumps from the 10 predictors step 3
grew (one example region) to every curated neuron in every OTHER remaining
region of the session -- not just the second predictor region's own full
count, but Region A's/Region B's session concatenated cortical-to-subcortical
across every anatomical region besides Region A/Region B itself (see
data_prep's ANATOMICAL_ORDER / `load_other_regions_multi`). A direct jump,
not an incremental grow. Region A is the target throughout; Region B is held
back for step 7.

Opens exactly where step 3 left off -- same 10 other-region neurons (row 0
still the neuron step 2 picked), same equal-width Region A/residual panels,
same wide, top-aligned, window-filling other-region panel -- reconstructed
as a starting frame rather than replayed. The jump beat then swaps that
single-region stack for the full multi-region one and fades in a very
narrow colour bar to its left, one solid colour per contiguous source
region (colour identity only, no region-name text). Every other-region row
(both the inherited 10 and the jumped-to full set) is a live scrolling
trace sharing the scene's one `t0`, so the closing slide scrolls the whole
panel, not just Region A and the residual; the full set's onset/boundary
lines are drawn as one overlay spanning all of its rows rather than one per
row (mathematically identical once rows are packed this densely -- every
row shares the same trial_duration/onset_offset/t0 -- at a fraction of the
always_redraw cost).

    manim -pqh step4_region_b_to_full.py Step4RegionBToFull
"""

import numpy as np
from manim import FadeIn, FadeOut, ReplacementTransform, Scene, ValueTracker, VGroup, linear

import data_prep as dp
import theme as th
from step2_ten_trial_scroll import _pick_beta_extreme_neurons


class Step4RegionBToFull(Scene):
    def construct(self):
        bundle = dp.load_pipeline_data()
        T, fs = bundle.T, bundle.fs
        n_trials = dp.N_TRIALS_DEMO
        trials_demo = bundle.trials_demo
        n_b_prev = bundle.region_other.step_counts[1]

        neuron_a = bundle.region_a.neurons(1)
        # Same pick step 2/3 make -- row 0 of the inherited 10-stack is that
        # exact neuron, not just neuron_order[0].
        (idx_high, _beta_high), _ = _pick_beta_extreme_neurons(bundle, neuron_a[0])
        order = bundle.region_other.neuron_order
        rest = [int(i) for i in order if i != idx_high]
        neurons_b_prev = np.array([idx_high] + rest[:n_b_prev - 1])
        colors_b_prev = th.ramp(th.REGION_B_COLOR, n_b_prev)

        window = th.scroll_window_seconds(n_trials, T, fs)
        times = th.concat_time_axis(n_trials, T, fs)
        max_t0 = max(float(times[-1]) - window, 0.01)
        t0 = ValueTracker(0.0)
        panel_y = th.box_center_y(1.7)
        trial_duration = window
        onset_offset = -dp.DISPLAY_WINDOW_S[0]

        def markers_for(width, center, height=1.2):
            return th.make_scroll_markers(t0, window, width, height, center,
                                           trial_duration=trial_duration, onset_offset=onset_offset)

        # ---- already at step 3's end state -- no entrance animation, this is
        #      the starting point, not new content ------------------------------
        color_a = th.ramp(th.REGION_A_COLOR, 1)[0]
        box_a = th.panel_background(th.STEP3_SIDE_W + 0.3, 1.7)
        box_a.move_to([th.STEP3_LEFT_X, panel_y, 0])
        label_a = th.panel_title_for(box_a, f"Region A · {dp.display_name(bundle.region_a.name)} · neuron 1 · {n_trials} trials")
        vals_a = bundle.region_a.zscored[trials_demo][:, neuron_a[0], :].reshape(-1)
        trace_a = th.make_scrolling_trace(vals_a, times, t0, window, th.STEP3_SIDE_W, 1.2,
                                           box_a.get_center(), color=color_a)
        marks_a = markers_for(th.STEP3_SIDE_W, box_a.get_center())
        self.add(box_a, label_a, marks_a, trace_a)

        # ---- other-region panel: wide, top-aligned with panels 1/3, tall
        #      enough to fill the window -- same geometry step 3 ended on ----
        mid_h = th.CONTENT_H
        box_b = th.panel_background(th.STEP3_MID_W + 0.3, mid_h)
        box_b.move_to([th.STEP3_MID_X, th.box_center_y(mid_h), 0])
        label_b = th.panel_title_for(box_b, f"Other region · {dp.display_name(bundle.region_other.name)} · {n_trials} trials")

        usable_h = mid_h * 0.88
        vals_b_prev = bundle.region_other.zscored[trials_demo][:, neurons_b_prev, :].transpose(1, 0, 2)
        vals_b_prev = vals_b_prev.reshape(n_b_prev, -1)
        rows_prev = th.stacked_scrolling_traces(vals_b_prev, times, t0, window, th.STEP3_MID_W,
                                                 box_b.get_center(), usable_h, colors_b_prev)
        marks_b_prev = th.stacked_scroll_markers(n_b_prev, th.STEP3_MID_W, box_b.get_center(), usable_h,
                                                  t0, window, trial_duration=trial_duration,
                                                  onset_offset=onset_offset)

        dots_prev = VGroup(*[th.neuron_chip(c, radius=0.065) for c in colors_b_prev])
        for i, chip in enumerate(dots_prev):
            chip.move_to([0.0, (n_b_prev / 2 - i - 0.5) * 0.2, 0.0])
        dots_prev.move_to([th.STEP3_BETA_X, box_b.get_center()[1], 0])
        brackets_prev = th.matrix_brackets(dots_prev)

        formula_prev = th.residual_formula_tex(n_b_prev)
        formula_prev.move_to([th.STEP3_RIGHT_X, th.FORMULA_Y, 0])

        resid_prev, _ = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neuron_a,
                                                       bundle.region_other, neurons_b_prev)
        resid_prev_vals = dp.flat_to_trial_major(resid_prev, n_trials, T)[:, :, 0].reshape(-1)
        box_r = th.panel_background(th.STEP3_SIDE_W + 0.3, 1.7)
        box_r.move_to([th.STEP3_RIGHT_X, panel_y, 0])
        label_r = th.panel_title_for(box_r, f"residual · Region A neuron 1 · vs {n_b_prev} predictors")
        trace_r_prev = th.make_scrolling_trace(resid_prev_vals, times, t0, window, th.STEP3_SIDE_W, 1.2,
                                                box_r.get_center(), color=color_a)
        marks_r = markers_for(th.STEP3_SIDE_W, box_r.get_center())

        self.add(box_b, label_b, rows_prev, marks_b_prev, dots_prev, brackets_prev, formula_prev,
                 box_r, label_r, marks_r, trace_r_prev)
        self.wait(0.3)

        # ---- 4a/4b: the other region jumps directly to every neuron in every
        #      remaining anatomical region (not just OTHER_REGION's own full
        #      count) -- concatenated cortical -> subcortical, each source
        #      region its own colour in a very narrow bar to the panel's left.
        #      Beta collapses into a compact heatmap strip (individual values
        #      no longer legible at this count -- colour there reads
        #      coefficient sign, not identity) -------------------------------
        multi = bundle.region_other_multi
        n_b = multi.n_neurons
        block_colors = th.region_qualitative_colors(multi.names, dp.ANATOMICAL_ORDER)
        colors_b = np.repeat(block_colors, [hi - lo for lo, hi in multi.boundaries])

        vals_b = multi.zscored[trials_demo].transpose(1, 0, 2).reshape(n_b, -1)
        rows_full = th.stacked_scrolling_traces(vals_b, times, t0, window, th.STEP3_MID_W,
                                                 box_b.get_center(), usable_h, colors_b, stroke_width=0.5)
        # One overlay spanning every row rather than 300+ individual ones --
        # every row shares the same t0/trial_duration/onset_offset, so the
        # lines land at the exact same x in every row; stacked this densely
        # the two are visually identical, at a fraction of the update cost.
        marks_b_full = markers_for(th.STEP3_MID_W, box_b.get_center(), height=usable_h)

        color_bar_x = box_b.get_left()[0] - 0.12
        color_bar = th.region_color_bar(multi.names, multi.boundaries, block_colors, n_b,
                                         box_b.get_center(), usable_h, color_bar_x)

        new_label_b = th.panel_title_for(box_b, f"Other regions ({len(multi.names)}) · {n_trials} trials")

        resid, coeff = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neuron_a,
                                                      multi, np.arange(n_b))

        strip = th.beta_heatmap(coeff, cell_w=th.matrix_cell_width(1), cell_h=usable_h / n_b)
        strip.move_to([th.STEP3_BETA_X, box_b.get_center()[1], 0])
        brackets_full = th.matrix_brackets(strip)

        formula_full = th.residual_formula_tex(n_b)
        formula_full.move_to([th.STEP3_RIGHT_X, th.FORMULA_Y, 0])

        self.play(
            FadeOut(rows_prev), FadeIn(rows_full),
            FadeOut(marks_b_prev), FadeIn(marks_b_full),
            FadeIn(color_bar),
            ReplacementTransform(label_b, new_label_b),
            ReplacementTransform(brackets_prev, brackets_full),
            ReplacementTransform(formula_prev, formula_full),
            run_time=1.4,
        )
        label_b = new_label_b
        self.wait(0.3)

        # ---- 4c: updated residual for the one Region A neuron -- aligned with
        #      panels 1/2, not enlarged ------------------------------------------
        resid_vals = dp.flat_to_trial_major(resid, n_trials, T)[:, :, 0].reshape(-1)
        label_r2 = th.panel_title_for(box_r, f"residual · Region A neuron 1 · vs {n_b} predictors")
        trace_r = th.make_scrolling_trace(resid_vals, times, t0, window, th.STEP3_SIDE_W, 1.2,
                                           box_r.get_center(), color=color_a)
        self.play(ReplacementTransform(label_r, label_r2), FadeOut(trace_r_prev), FadeIn(trace_r), run_time=0.5)
        self.wait(0.3)

        # ---- final slide: Region A, the residual, AND every other-region row
        #      (plus its onset/boundary overlay) scroll together ------------
        self.play(t0.animate.set_value(max_t0), run_time=5, rate_func=linear)
        self.wait(1.0)

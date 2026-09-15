"""
Step 3 -- residualize(): the other region grows from one predictor to ten,
one neuron at a time, each with its own readable beta_j = xx label -- then
all ten collapse into a colour-coded 10x1 vector. Region A is the target
throughout; Region B is held back for step 7.

Opens exactly where step 2 left off -- same neuron, same beta_1 (inset in
the other-region panel's own top-right corner), same equal-width
three-panel layout -- reconstructed as a starting frame rather than replayed,
so the two videos read as one continuous walkthrough. From there, panels 1
and 3 (Region A / residual) settle to a shared, equal width while only the
middle "other region" panel widens and grows tall, its top edge staying
aligned with panels 1/3 while its bottom is free to reach down toward the
bottom of the frame -- room for its stacked rows to grow without much
shrinking. Every other-region row is a live scrolling trace with its own
onset/boundary markers from the moment it appears, so the closing slide
scrolls the whole panel, not just Region A and the residual.

    manim -pqh step3_region_b_to_ten.py Step3RegionBToTen
"""

import numpy as np
from manim import Create, FadeIn, FadeOut, ReplacementTransform, Scene, ValueTracker, VGroup, linear

import data_prep as dp
import theme as th
from step2_ten_trial_scroll import _pick_beta_extreme_neurons

ROW_BETA_FONT_SIZE = 5   # small per-row beta_j = xx labels during growth (3a)
ROW_FILL = 0.88          # fraction of a row's slot the trace fills -- the tall mid panel leaves
                          # rows enough room that a smaller reduction than steps 1-2's is enough


class Step3RegionBToTen(Scene):
    def construct(self):
        bundle = dp.load_pipeline_data()
        T, fs = bundle.T, bundle.fs
        n_trials = dp.N_TRIALS_DEMO
        trials_demo = bundle.trials_demo
        n_b = bundle.region_other.step_counts[1]

        neuron_a = bundle.region_a.neurons(1)
        # Same pick step 2 makes for its second segment -- row 0 of the
        # growing stack is that exact neuron, not just neuron_order[0].
        (idx_high, _beta_high), _ = _pick_beta_extreme_neurons(bundle, neuron_a[0])
        order = bundle.region_other.neuron_order
        rest = [int(i) for i in order if i != idx_high][:n_b - 1]
        neurons_b = np.array([idx_high] + rest)

        color_a = th.ramp(th.REGION_A_COLOR, 1)[0]
        color_o_solid = th.ramp(th.REGION_B_COLOR, 1)[0]   # matches step 2's single-neuron colour
        colors_b = th.ramp(th.REGION_B_COLOR, n_b)

        # caption = th.hyperparam_caption(bundle, extra=f"trials {dp.TRIAL_START_IDX}-{dp.TRIAL_START_IDX + n_trials - 1}")
        # self.add(caption)

        window = th.scroll_window_seconds(n_trials, T, fs)
        times = th.concat_time_axis(n_trials, T, fs)
        max_t0 = max(float(times[-1]) - window, 0.01)
        t0 = ValueTracker(0.0)
        panel_y = th.box_center_y(1.7)
        trial_duration = window
        onset_offset = -dp.DISPLAY_WINDOW_S[0]

        def markers_for(width, height, center):
            return th.make_scroll_markers(t0, window, width, height, center,
                                           trial_duration=trial_duration, onset_offset=onset_offset)

        # ---- reconstruct step 2's frame exactly as it stood just before its
        #      final scroll -- the starting point, not new content, so it
        #      goes straight on screen with no entrance animation -----------
        label_a_text = f"Region A · {dp.display_name(bundle.region_a.name)} · neuron 1 · {n_trials} trials"
        box_a = th.panel_background(th.EQ_PANEL_W + 0.3, 1.7)
        box_a.move_to([th.EQ_LEFT_X, panel_y, 0])
        label_a = th.panel_title_for(box_a, label_a_text)
        vals_a = bundle.region_a.zscored[trials_demo][:, neuron_a[0], :].reshape(-1)
        trace_a = th.make_scrolling_trace(vals_a, times, t0, window, th.EQ_PANEL_W, 1.2,
                                           box_a.get_center(), color=color_a)
        marks_a = markers_for(th.EQ_PANEL_W, 1.2, box_a.get_center())

        box_o = th.panel_background(th.EQ_PANEL_W + 0.3, 1.7)
        box_o.move_to([th.EQ_MID_X, panel_y, 0])
        label_o = th.panel_title_for(
            box_o, f"Other region · {dp.display_name(bundle.region_other.name)} · neuron {idx_high + 1} · {n_trials} trials")
        vals_o0 = bundle.region_other.zscored[trials_demo][:, idx_high, :].reshape(-1)
        trace_o = th.make_scrolling_trace(vals_o0, times, t0, window, th.EQ_PANEL_W, 1.2,
                                           box_o.get_center(), color=color_o_solid)
        marks_o = markers_for(th.EQ_PANEL_W, 1.2, box_o.get_center())

        resid1, coeff1 = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neuron_a,
                                                        bundle.region_other, neurons_b[:1])
        beta_label = th.beta_label_tex(float(coeff1[0, 0]), idx=1)
        th.inset_top_right(box_o, beta_label)

        formula = th.residual_formula_tex(None)
        formula.move_to([th.EQ_RIGHT_X, th.FORMULA_Y, 0])

        resid1_vals = dp.flat_to_trial_major(resid1, n_trials, T)[:, :, 0].reshape(-1)
        label_r_text = f"residual · Region A neuron 1 · {n_trials} trials"
        box_r = th.panel_background(th.EQ_PANEL_W + 0.3, 1.7)
        box_r.move_to([th.EQ_RIGHT_X, panel_y, 0])
        label_r = th.panel_title_for(box_r, label_r_text)
        trace_r = th.make_scrolling_trace(resid1_vals, times, t0, window, th.EQ_PANEL_W, 1.2,
                                           box_r.get_center(), color=color_a)
        marks_r = markers_for(th.EQ_PANEL_W, 1.2, box_r.get_center())

        self.add(box_a, label_a, marks_a, trace_a,
                  box_o, label_o, marks_o, trace_o, beta_label,
                  formula,
                  box_r, label_r, marks_r, trace_r)
        self.wait(0.6)

        # ---- transition into step 3's own layout: panels 1/3 settle to an
        #      equal, shared width; the other-region panel widens and grows
        #      tall (top edge only stays aligned with panels 1/3 -- see
        #      th.box_center_y, which always pins a box's top at CONTENT_TOP
        #      regardless of its height); neuron 1 shrinks straight into row
        #      0 of the eventual 10-row stack, already a scrolling trace ----
        new_box_a = th.panel_background(th.STEP3_SIDE_W + 0.3, 1.7)
        new_box_a.move_to([th.STEP3_LEFT_X, panel_y, 0])
        new_label_a = th.panel_title_for(new_box_a, label_a_text)
        new_trace_a = th.make_scrolling_trace(vals_a, times, t0, window, th.STEP3_SIDE_W, 1.2,
                                               new_box_a.get_center(), color=color_a)
        new_marks_a = markers_for(th.STEP3_SIDE_W, 1.2, new_box_a.get_center())

        new_box_r = th.panel_background(th.STEP3_SIDE_W + 0.3, 1.7)
        new_box_r.move_to([th.STEP3_RIGHT_X, panel_y, 0])
        new_label_r = th.panel_title_for(new_box_r, label_r_text)
        new_trace_r = th.make_scrolling_trace(resid1_vals, times, t0, window, th.STEP3_SIDE_W, 1.2,
                                               new_box_r.get_center(), color=color_a)
        new_marks_r = markers_for(th.STEP3_SIDE_W, 1.2, new_box_r.get_center())

        mid_h = th.CONTENT_H
        box_b = th.panel_background(th.STEP3_MID_W + 0.3, mid_h)
        box_b.move_to([th.STEP3_MID_X, th.box_center_y(mid_h), 0])
        label_b = th.panel_title_for(box_b, f"Other region · {dp.display_name(bundle.region_other.name)} 10 neurons · {n_trials} trials")

        usable_h = mid_h * 0.88
        offsets, row_h = th.stacked_row_offsets(n_b, usable_h)
        row_trace_h = row_h * ROW_FILL

        row0_center = np.array(box_b.get_center()) + np.array([0.0, offsets[0], 0.0])
        row0_trace = th.make_scrolling_trace(vals_o0, times, t0, window, th.STEP3_MID_W, row_trace_h,
                                              row0_center, color=colors_b[0])
        row0_marks = markers_for(th.STEP3_MID_W, row_trace_h, row0_center)
        beta_row_1 = th.beta_label_tex(float(coeff1[0, 0]), idx=1, font_size=ROW_BETA_FONT_SIZE+5, color=colors_b[0])
        beta_row_1.move_to([th.STEP3_BETA_X, box_b.get_center()[1] + offsets[0], 0])

        self.play(
            ReplacementTransform(box_a, new_box_a), ReplacementTransform(label_a, new_label_a),
            FadeOut(trace_a), FadeIn(new_trace_a), FadeOut(marks_a), FadeIn(new_marks_a),

            ReplacementTransform(box_o, box_b), ReplacementTransform(label_o, label_b),
            FadeOut(trace_o), FadeIn(row0_trace), FadeOut(marks_o), FadeIn(row0_marks),
            FadeOut(beta_label), FadeIn(beta_row_1),

            formula.animate.move_to([th.STEP3_RIGHT_X, th.FORMULA_Y, 0]),

            ReplacementTransform(box_r, new_box_r), ReplacementTransform(label_r, new_label_r),
            FadeOut(trace_r), FadeIn(new_trace_r), FadeOut(marks_r), FadeIn(new_marks_r),
            run_time=1.4,
        )
        box_a, trace_a, marks_a = new_box_a, new_trace_a, new_marks_a
        box_r, label_r, trace_r, marks_r = new_box_r, new_label_r, new_trace_r, new_marks_r
        self.wait(0.2)

        # ---- 3a: add neurons 2..n_b one at a time, each its own frame, each
        #      already scrolling with its own onset/boundary markers, each
        #      with a readable beta_j = xx label (refit on however many
        #      predictors are visible at that moment) beside its row --------
        row_traces = VGroup(row0_trace)
        row_marks = VGroup(row0_marks)
        beta_rows = VGroup(beta_row_1)

        for i in range(1, n_b):
            row_center = np.array(box_b.get_center()) + np.array([0.0, offsets[i], 0.0])
            vals_i = bundle.region_other.zscored[trials_demo][:, neurons_b[i], :].reshape(-1)
            row_i_trace = th.make_scrolling_trace(vals_i, times, t0, window, th.STEP3_MID_W, row_trace_h,
                                                   row_center, color=colors_b[i])
            row_i_marks = markers_for(th.STEP3_MID_W, row_trace_h, row_center)

            _, coeff_i = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neuron_a,
                                                        bundle.region_other, neurons_b[:i + 1])
            beta_row_i = th.beta_label_tex(float(coeff_i[i, 0]), idx=i + 1,
                                            font_size=ROW_BETA_FONT_SIZE+5, color=colors_b[i])
            beta_row_i.move_to([th.STEP3_BETA_X, box_b.get_center()[1] + offsets[i], 0])

            row_traces.add(row_i_trace)
            row_marks.add(row_i_marks)
            beta_rows.add(beta_row_i)

            self.play(FadeIn(row_i_trace), FadeIn(row_i_marks), FadeIn(beta_row_i), run_time=0.45)
        self.wait(0.3)

        # ---- 3b: collapse the n_b beta labels into one n_b x 1 vector, each
        #      entry its own colour ------------------------------------------
        col = VGroup(*[th.neuron_chip(colors_b[i], radius=0.065) for i in range(n_b)])
        for i, chip in enumerate(col):
            chip.move_to([0.0, (n_b / 2 - i - 0.5) * 0.2, 0.0])
        col.move_to([th.STEP3_BETA_X, box_b.get_center()[1], 0])
        self.play(ReplacementTransform(beta_rows, col), run_time=1.0)
        brackets = th.matrix_brackets(col)
        self.play(Create(brackets[0]), Create(brackets[2]), run_time=0.6)
        self.wait(0.3)

        # ---- 3c: formula grows a sum over the n_b Region B predictors -----------
        formula2 = th.residual_formula_tex(n_b)
        formula2.move_to([th.STEP3_RIGHT_X, th.FORMULA_Y, 0])
        self.play(ReplacementTransform(formula, formula2), run_time=1.0)
        self.wait(0.3)

        # ---- 3d: scrolling residual of Region A neuron 1 vs all n_b predictors,
        #      aligned with panels 1/2, not enlarged --------------------------
        resid, _ = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neuron_a,
                                                 bundle.region_other, neurons_b)
        resid_vals = dp.flat_to_trial_major(resid, n_trials, T)[:, :, 0].reshape(-1)

        label_r2 = th.panel_title_for(box_r, f"residual · Region A neuron 1 · vs {n_b} predictors")
        trace_r2 = th.make_scrolling_trace(resid_vals, times, t0, window, th.STEP3_SIDE_W, 1.2,
                                            box_r.get_center(), color=color_a)
        self.play(ReplacementTransform(label_r, label_r2), FadeOut(trace_r), FadeIn(trace_r2), run_time=0.5)
        self.wait(0.3)

        # ---- final slide: Region A, the residual, AND every other-region row
        #      (plus each row's own onset/boundary markers) scroll together --
        self.play(t0.animate.set_value(max_t0), run_time=5, rate_func=linear)
        self.wait(1.0)

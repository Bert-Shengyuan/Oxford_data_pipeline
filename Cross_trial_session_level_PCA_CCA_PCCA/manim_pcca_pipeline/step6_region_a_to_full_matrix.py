"""
Step 6 -- residualize(): Region A jumps straight from three example neurons
to every curated neuron available for it (still residualized against every
other remaining anatomical region) -- a direct jump, not incremental. The
residual panel is then retired and the beta matrix expands into that space.
Region B is held back for step 7.

Opens exactly where step 5 left off -- same full multi-region other-region
panel (wide, top-aligned with panels 1/3, tall enough to fill the window,
its narrow per-region colour bar to its left), same equal Region A/residual
panel width, Panel 1 already using the "top-aligned, window-filling"
geometry step 5 introduced -- reconstructed as a starting frame rather than
replayed. Region A's jump re-lays its rows out across the same tall panel
(20 thinner slots instead of 3) rather than resizing the panel itself, each
row still its own scrolling trace with its own onset/boundary markers. Once
the residual panel is retired, the beta matrix expands into the space it
vacated -- sized to clear the other-region panel's own (now much wider)
right edge instead of a fixed offset, so the two no longer overlap. Unlike
steps 1-5, the closing beat has no scroll of its own in the original design,
but Region A and the other region both keep scrolling with their own
onset/boundary markers right up to it -- so a closing slide is added here
too rather than leaving them inert once the matrix reveal lands.

    manim -pqh step6_region_a_to_full_matrix.py Step6RegionAToFullMatrix
"""

import numpy as np
from manim import FadeIn, FadeOut, ReplacementTransform, Scene, ValueTracker, VGroup, linear

import data_prep as dp
import theme as th


class Step6RegionAToFullMatrix(Scene):
    def construct(self):
        bundle = dp.load_pipeline_data()
        T, fs = bundle.T, bundle.fs
        n_trials = dp.N_TRIALS_DEMO
        trials_demo = bundle.trials_demo
        n_a_prev = bundle.region_a.step_counts[1]
        n_a = bundle.region_a.step_counts[2]

        neurons_a_prev = bundle.region_a.neurons(n_a_prev)
        neurons_a = bundle.region_a.neurons(n_a)
        multi = bundle.region_other_multi
        n_b = multi.n_neurons
        block_colors = th.region_qualitative_colors(multi.names, dp.ANATOMICAL_ORDER)
        colors_b = np.repeat(block_colors, [hi - lo for lo, hi in multi.boundaries])

        colors_a_prev = th.ramp(th.REGION_A_COLOR, n_a_prev)
        colors_a = th.ramp(th.REGION_A_COLOR, n_a)

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

        # ---- already at step 5's end state -- no entrance animation. Panel 1's
        #      3 neurons are still individually scrolling in their own
        #      pre-allocated slots (as step 5 left them); the other-region
        #      panel is unchanged, still wide/top-aligned/window-filling with
        #      its own scrolling rows + colour bar ---------------------------
        panel_a_h = th.CONTENT_H
        usable_h1 = panel_a_h * 0.88
        box_a = th.panel_background(th.STEP3_SIDE_W + 0.3, panel_a_h)
        box_a.move_to([th.STEP3_LEFT_X, th.box_center_y(panel_a_h), 0])
        label_a = th.panel_title_for(box_a, f"Region A · {dp.display_name(bundle.region_a.name)} · {n_trials} trials")
        offsets_prev, row_h_prev = th.stacked_row_offsets(n_a_prev, usable_h1)
        vals_a_prev_all = bundle.region_a.zscored[trials_demo][:, neurons_a_prev, :].transpose(1, 0, 2)
        rows_a_prev = VGroup(*[
            th.make_scrolling_trace(vals_a_prev_all[k].reshape(-1), times, t0, window, th.STEP3_SIDE_W,
                                     row_h_prev * 0.88,
                                     np.array(box_a.get_center()) + np.array([0.0, offsets_prev[k], 0.0]),
                                     color=colors_a_prev[k])
            for k in range(n_a_prev)
        ])
        marks_a_prev = th.stacked_scroll_markers(n_a_prev, th.STEP3_SIDE_W, box_a.get_center(), usable_h1,
                                                  t0, window, trial_duration=trial_duration,
                                                  onset_offset=onset_offset, row_fill=0.88)

        mid_h = th.CONTENT_H
        box_b = th.panel_background(th.STEP3_MID_W + 0.3, mid_h)
        box_b.move_to([th.STEP3_MID_X, th.box_center_y(mid_h), 0])
        label_b = th.panel_title_for(box_b, f"Other regions ({len(multi.names)}) · {n_trials} trials")

        usable_h_mid = mid_h * 0.88
        vals_b = multi.zscored[trials_demo].transpose(1, 0, 2).reshape(n_b, -1)
        rows_b = th.stacked_scrolling_traces(vals_b, times, t0, window, th.STEP3_MID_W,
                                              box_b.get_center(), usable_h_mid, colors_b, stroke_width=0.5)
        marks_b = markers_for(th.STEP3_MID_W, box_b.get_center(), height=usable_h_mid)
        color_bar = th.region_color_bar(multi.names, multi.boundaries, block_colors, n_b,
                                         box_b.get_center(), usable_h_mid, box_b.get_left()[0] - 0.12)

        resid_prev, coeff_prev = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neurons_a_prev,
                                                                multi, np.arange(n_b))
        strip_prev = th.beta_heatmap(coeff_prev, cell_w=th.matrix_cell_width(n_a_prev), cell_h=usable_h_mid / n_b)
        strip_prev.move_to([th.STEP3_BETA_X, box_b.get_center()[1], 0])
        brackets_prev = th.matrix_brackets(strip_prev)

        formula = th.residual_formula_tex(n_b)
        formula.move_to([th.STEP3_RIGHT_X, th.FORMULA_Y, 0])

        # step 5 ends with its residual box exactly Panel 1's own height, its
        # rows sharing Panel 1's pre-allocated slots (`offsets_prev`/
        # `row_h_prev`) one-for-one -- reconstruct that exactly rather than
        # an independently-sized box.
        box_r_h_prev = panel_a_h
        box_r = th.panel_background(th.STEP3_SIDE_W + 0.3, box_r_h_prev)
        box_r.move_to([th.STEP3_RIGHT_X, th.box_center_y(box_r_h_prev, top=th.RESID_TOP_Y), 0])
        label_r = th.panel_title_for(box_r, f"residual · Region A ({n_a_prev} neurons) · {n_trials} trials")
        resid_traces = VGroup(*[
            th.make_scrolling_trace(
                dp.flat_to_trial_major(resid_prev, n_trials, T)[:, :, k].reshape(-1), times, t0, window,
                th.STEP3_SIDE_W, row_h_prev * 0.88,
                np.array(box_r.get_center()) + np.array([0.0, offsets_prev[k], 0.0]), color=colors_a_prev[k])
            for k in range(n_a_prev)
        ])
        marks_r = th.stacked_scroll_markers(n_a_prev, th.STEP3_SIDE_W, box_r.get_center(), usable_h1,
                                             t0, window, trial_duration=trial_duration,
                                             onset_offset=onset_offset, row_fill=0.88)

        self.add(box_a, label_a, marks_a_prev, rows_a_prev, box_b, label_b, rows_b, marks_b, color_bar,
                 brackets_prev, formula, box_r, label_r, marks_r, resid_traces)
        self.wait(0.3)

        # ---- 6a: Region A jumps directly to n_a neurons (no incremental
        #      growth) -- re-laid out across the same tall, top-aligned panel
        #      (thinner slots, not a shorter/shrunk box), still individually
        #      scrolling with their own onset/boundary markers -------------
        vals_a = bundle.region_a.zscored[trials_demo][:, neurons_a, :].transpose(1, 0, 2).reshape(n_a, -1)
        rows_a = th.stacked_scrolling_traces(vals_a, times, t0, window, th.STEP3_SIDE_W,
                                              box_a.get_center(), usable_h1, colors_a, stroke_width=1.0)
        marks_a = th.stacked_scroll_markers(n_a, th.STEP3_SIDE_W, box_a.get_center(), usable_h1,
                                             t0, window, trial_duration=trial_duration,
                                             onset_offset=onset_offset)
        self.play(FadeOut(rows_a_prev), FadeIn(rows_a), FadeOut(marks_a_prev), FadeIn(marks_a), run_time=1.2)
        self.wait(0.2)

        # ---- 6b: drop the residual panel, beta matrix expands into its place
        #      -- sized to clear the other-region panel's own right edge
        #      (previously a fixed offset that predated the wider panel and
        #      ended up overlapping it) ------------------------------------
        _, coeff = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neurons_a, multi, np.arange(n_b))

        mat_left_bound = box_b.get_right()[0] + 0.15
        mat_right_bound = th.STEP3_RIGHT_X + (th.STEP3_SIDE_W + 0.3) / 2 - 0.2
        mat_center_x = (mat_left_bound + mat_right_bound) / 2
        big_cell_w = (mat_right_bound - mat_left_bound - 0.5) / n_a
        big_strip = th.beta_heatmap(coeff, cell_w=big_cell_w, cell_h=usable_h_mid / n_b)
        big_strip.move_to([mat_center_x, box_b.get_center()[1], 0])
        big_brackets = th.matrix_brackets(big_strip)
        top_box = th.panel_background(big_brackets.width + 0.2, panel_a_h)
        top_box.move_to([mat_center_x, th.box_center_y(panel_a_h), 0])
        mat_label = th.panel_title_for(top_box, f"beta · {n_b} × {n_a} coefficient matrix")

        self.play(
            FadeOut(label_r), FadeOut(box_r), FadeOut(resid_traces), FadeOut(marks_r),
            FadeOut(formula),
            ReplacementTransform(brackets_prev, big_brackets),
            FadeIn(mat_label),
            run_time=1.6,
        )
        self.wait(0.5)

        # ---- final slide: Region A and every other-region row (plus their
        #      own onset/boundary markers) scroll together -------------------
        self.play(t0.animate.set_value(max_t0), run_time=10, rate_func=linear)
        self.wait(1.0)

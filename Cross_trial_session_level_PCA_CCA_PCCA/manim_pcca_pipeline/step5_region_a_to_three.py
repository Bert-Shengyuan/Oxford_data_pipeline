"""
Step 5 -- residualize(): Region A grows from one neuron to three, one at a
time, each addition landing simultaneously: a new scrolling row in Panel 1,
a new column in the beta matrix, and a new scrolling residual trace. Region A
is residualized against every other remaining anatomical region at once;
Region B is held back for step 7.

Opens exactly where step 4 left off -- same full multi-region other-region
panel (wide, top-aligned with panels 1/3, tall enough to fill the window,
its narrow per-region colour bar to its left), every row still its own
scrolling trace sharing the scene's one `t0` -- reconstructed as a starting
frame rather than replayed. Panel 1 now uses the same "top-aligned,
window-filling" geometry as the other-region panel: its 3 eventual row
slots are pre-allocated across the panel's full height from the start, so
adding a neuron only fades a new row into an already-sized slot rather than
re-shrinking the existing ones. Panel 1 and the residual panel keep an
equal, shared width throughout; only the residual panel's own height grows
as Region A adds neurons (its own top edge staying fixed -- see
`residual_box_height` / `th.RESID_TOP_Y`), each of its rows carrying its own
onset/boundary markers.

    manim -pqh step5_region_a_to_three.py Step5RegionAToThree
"""

import numpy as np
from manim import FadeIn, FadeOut, ReplacementTransform, Scene, ValueTracker, VGroup, linear

import data_prep as dp
import theme as th


class Step5RegionAToThree(Scene):
    def construct(self):
        bundle = dp.load_pipeline_data()
        T, fs = bundle.T, bundle.fs
        n_trials = dp.N_TRIALS_DEMO
        trials_demo = bundle.trials_demo
        n_a = bundle.region_a.step_counts[1]

        neurons_a = bundle.region_a.neurons(n_a)
        multi = bundle.region_other_multi
        n_b = multi.n_neurons
        block_colors = th.region_qualitative_colors(multi.names, dp.ANATOMICAL_ORDER)
        colors_b = np.repeat(block_colors, [hi - lo for lo, hi in multi.boundaries])
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

        # ---- Panel 1 (Region A): wide-panel geometry -- top-aligned with the
        #      other panels, tall enough to fill the window, its `n_a`
        #      eventual row slots pre-allocated across that full height from
        #      the start, so a smaller per-row shrink is enough and adding a
        #      neuron never re-shrinks the rows already on screen -----------
        panel_a_h = th.CONTENT_H
        usable_h1 = panel_a_h * 0.88
        offsets_a, row_h_a = th.stacked_row_offsets(n_a, usable_h1)
        vals_a_all = bundle.region_a.zscored[trials_demo][:, neurons_a, :].transpose(1, 0, 2)  # (n_a, n_trials, T)

        def panel1_rows(n, center):
            """The first `n` of Region A's `n_a` pre-allocated row slots."""
            rows = VGroup()
            for k in range(n):
                vals_k = vals_a_all[k].reshape(-1)
                row_center = np.array(center) + np.array([0.0, offsets_a[k], 0.0])
                rows.add(th.make_scrolling_trace(vals_k, times, t0, window, th.STEP3_SIDE_W, row_h_a * 0.88,
                                                  row_center, color=colors_a[k]))
            return rows

        def panel1_marks(n, center):
            marks = VGroup()
            for k in range(n):
                row_center = np.array(center) + np.array([0.0, offsets_a[k], 0.0])
                marks.add(markers_for(th.STEP3_SIDE_W, row_center, height=row_h_a * 0.88))
            return marks

        resid_row_trace_h = row_h_a * 0.88  # exact height used for Region A's own rows

        def residual_box_height(n):
            """1.7 while aligned with panels 1/2 (n == 1); once neurons start
            being added, the box is exactly Panel 1's own height, so its rows
            share Panel 1's row slots (`offsets_a`/`row_h_a`) one-for-one."""
            return 1.7 if n <= 1 else panel_a_h

        def residual_rows(n, resid, center):
            """n scrolling residual traces. While n == 1 the single trace
            stays aligned with panels 1/2's old geometry; from n == 2 on, each
            row reuses Region A's own pre-allocated slot `offsets_a[k]` and
            row height `resid_row_trace_h`, so the two panels' rows match
            exactly rather than the residual box being independently sized."""
            if n == 1:
                tr = th.make_scrolling_trace(dp.flat_to_trial_major(resid, n_trials, T)[:, :, 0].reshape(-1),
                                              times, t0, window, th.STEP3_SIDE_W, 1.2, center, color=colors_a[0])
                return VGroup(tr)
            rows = VGroup()
            for k in range(n):
                vals_k = dp.flat_to_trial_major(resid, n_trials, T)[:, :, k].reshape(-1)
                row_center = np.array(center) + np.array([0.0, offsets_a[k], 0.0])
                rows.add(th.make_scrolling_trace(vals_k, times, t0, window, th.STEP3_SIDE_W, resid_row_trace_h,
                                                  row_center, color=colors_a[k]))
            return rows

        def residual_marks(n, center):
            """Per-row onset/boundary markers matching `residual_rows`'s own
            geometry -- every residual row gets its own dashed onset line and
            trial-boundary separators, not just the panel as a whole."""
            if n == 1:
                return VGroup(markers_for(th.STEP3_SIDE_W, center))
            marks = VGroup()
            for k in range(n):
                row_center = np.array(center) + np.array([0.0, offsets_a[k], 0.0])
                marks.add(markers_for(th.STEP3_SIDE_W, row_center, height=resid_row_trace_h))
            return marks

        # ---- already at step 4's end state -- no entrance animation -----------
        box_a = th.panel_background(th.STEP3_SIDE_W + 0.3, panel_a_h)
        box_a.move_to([th.STEP3_LEFT_X, th.box_center_y(panel_a_h), 0])
        label_a = th.panel_title_for(box_a, f"Region A · {dp.display_name(bundle.region_a.name)} · {n_trials} trials")
        rows_a = panel1_rows(1, box_a.get_center())
        marks_a = panel1_marks(1, box_a.get_center())

        # ---- other-region panel: wide, top-aligned with panels 1/3, tall
        #      enough to fill the window, every remaining anatomical region
        #      concatenated cortical -> subcortical with its own colour bar --
        #      same geometry/content step 4 ended on -----------------------
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

        resid1, coeff1 = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neurons_a[:1], multi, np.arange(n_b))
        strip = th.beta_heatmap(coeff1, cell_w=th.matrix_cell_width(1), cell_h=usable_h_mid / n_b)
        strip.move_to([th.STEP3_BETA_X, box_b.get_center()[1], 0])
        brackets = th.matrix_brackets(strip)

        formula = th.residual_formula_tex(n_b)
        formula.move_to([th.STEP3_RIGHT_X, th.FORMULA_Y, 0])

        box_r = th.panel_background(th.STEP3_SIDE_W + 0.3, residual_box_height(1))
        box_r.move_to([th.STEP3_RIGHT_X, panel_y, 0])
        label_r = th.panel_title_for(box_r, f"residual · Region A · {n_trials} trials")
        marks_r = residual_marks(1, box_r.get_center())
        rows_r = residual_rows(1, resid1, box_r.get_center())

        self.add(box_a, label_a, marks_a, rows_a, box_b, label_b, rows_b, marks_b, color_bar,
                 brackets, formula, box_r, label_r, marks_r, rows_r)
        self.wait(0.3)

        # ---- 5a: two more neurons, one at a time -- each addition is one
        #      combined frame: a new Panel-1 row fading into its already-sized
        #      slot + a new beta-matrix column + a new residual row (with its
        #      own onset/boundary markers), all at once ---------------------
        for n in range(2, n_a + 1):
            new_row_a = panel1_rows(n, box_a.get_center())[n - 1]
            new_mark_a = panel1_marks(n, box_a.get_center())[n - 1]

            resid_n, coeff_n = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neurons_a[:n], multi,
                                                              np.arange(n_b))
            new_strip = th.beta_heatmap(coeff_n, cell_w=th.matrix_cell_width(n), cell_h=usable_h_mid / n_b)
            new_strip.move_to([th.STEP3_BETA_X, box_b.get_center()[1], 0])
            new_brackets = th.matrix_brackets(new_strip)

            # n >= 2 here (this loop starts at 2), so the residual panel has
            # already left the steps-1-4 "aligned with a/b" regime for good
            new_box_r = th.panel_background(th.STEP3_SIDE_W + 0.3, residual_box_height(n))
            new_box_r.move_to([th.STEP3_RIGHT_X, th.box_center_y(residual_box_height(n), top=th.RESID_TOP_Y), 0])
            new_label_r = th.panel_title_for(new_box_r, f"residual · Region A ({n} neurons) · {n_trials} trials")
            new_rows_r = residual_rows(n, resid_n, new_box_r.get_center())
            new_marks_r = residual_marks(n, new_box_r.get_center())

            self.play(
                FadeIn(new_row_a), FadeIn(new_mark_a),
                ReplacementTransform(brackets, new_brackets),
                ReplacementTransform(box_r, new_box_r),
                ReplacementTransform(label_r, new_label_r),
                FadeOut(rows_r), FadeIn(new_rows_r),
                FadeOut(marks_r), FadeIn(new_marks_r),
                run_time=1.3,
            )
            brackets, box_r, label_r, rows_r, marks_r = new_brackets, new_box_r, new_label_r, new_rows_r, new_marks_r
            self.wait(0.3)

        # ---- final slide: Region A, the residual, AND every other-region row
        #      (plus its onset/boundary overlay) scroll together ------------
        self.play(t0.animate.set_value(max_t0), run_time=10, rate_func=linear)
        self.wait(1.0)

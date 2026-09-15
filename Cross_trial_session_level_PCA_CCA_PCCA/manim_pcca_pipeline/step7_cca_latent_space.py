"""
Step 7 -- pcca(): Region A and Region B, each independently residualized
against every other remaining anatomical region at once (the SAME
region_other_multi pool step 5 grows Region A against -- not against each
other, and not against the smaller single OTHER_REGION steps 2-3 use), then
ridge_cca() finds the most-aligned 1D direction between their two 3-neuron
residual spaces.

7a/7b -- opens on two frozen copies of step 5's own final frame (Region A's
3-neuron panel, the shared other-regions panel + colour bar, the beta
matrix, the residual panel), one for Region A and one for Region B, each
squeezed to half the usual content height at the SAME width step 5 used
(STEP3_LEFT_X/MID_X/BETA_X/RIGHT_X are reused unchanged) -- reconstructed
directly in its already-scrolled-to-t0=0 state (`t0` is never animated until
the final scroll below), which is exactly the frame step 5 holds immediately
before its own `t0.animate.set_value(max_t0)` line, so no growth animation
is replayed here.

7c/7d -- each half's residual panel (3 stacked scrolling rows) is replaced
by its own 3D axes: not the real 3-neuron residual (which, viewed from one
fixed camera angle, tends to read as a flat scribble rather than a clear
path through space), but a simulated Lissajous-style loop
(`theme.simulated_trajectory`, one fixed shape per region) with its own
multi-stop colour gradient, chosen purely so the loop's three-dimensionality
and per-region distinctness are unambiguous on screen -- with a point
looping along it for the rest of the scene. A projection plane per region
still searches for, then settles on, the REAL ridge_cca() direction (Wx[:,
0] / Wy[:, 0]) fit on the TRAIN half of a fixed session-wide train/test
split -- never on the small demo trial set being drawn, matching steps 2-6's
own beta-fitting convention (`train_test_trial_split` /
`fit_beta_train_apply_demo`); only the loop the plane rotates in front of is
illustrative, not the CCA direction itself or Panel 4's latent trace.

7e -- every visual from 7a-7d is left exactly as it is; Panel 1, Panel 2,
Panel 3 (the settled 3D latent space -- never rotated or rescaled again from
here on) and the beta matrix/formula between them simply scale down together,
uniformly, about each half's own top-left corner, until that whole assembly
is 2/3 of the row's width. That frees the remaining third on the right for
the new Panel 4, the 1D CCA-component-0 latent (resid_demo @ Wx[:, 0] /
resid_demo @ Wy[:, 0], i.e. each region's fixed TRAIN-fit projection applied
to its own 3-neuron residual dynamics), scrolling with its own onset/boundary
markers like every other trace in this walkthrough.

    manim -pqh step7_cca_latent_space.py Step7CCALatentSpace
"""

import numpy as np
from manim import (
    DEGREES,
    RIGHT,
    UP,
    Create,
    Dot3D,
    FadeIn,
    FadeOut,
    Group,
    Rotate,
    Square,
    Text,
    ThreeDAxes,
    ThreeDScene,
    ValueTracker,
    VGroup,
    VMobject,
    linear,
)
from manim.utils.space_ops import rotation_about_z
from manim.utils.space_ops import rotation_matrix as axis_angle_rotation_matrix

import data_prep as dp
import theme as th


class Step7CCALatentSpace(ThreeDScene):
    def construct(self):
        bundle = dp.load_pipeline_data()
        T, fs = bundle.T, bundle.fs
        n_trials = dp.N_TRIALS_DEMO
        trials_demo = bundle.trials_demo
        n_lat = dp.N_LATENT_NEURONS

        neurons_a3 = bundle.region_a.neurons(n_lat)
        neurons_b3 = bundle.region_b.neurons(n_lat)
        multi = bundle.region_other_multi
        n_b = multi.n_neurons
        colors_a = th.ramp(th.REGION_A_COLOR, n_lat)
        colors_b = th.ramp(th.REGION_B_COLOR, n_lat)
        block_colors = th.region_qualitative_colors(multi.names, dp.ANATOMICAL_ORDER)
        colors_mid = np.repeat(block_colors, [hi - lo for lo, hi in multi.boundaries])

        # ---- residualize both regions' 3 "hero" neurons against the SAME
        #      region_other_multi pool step 5 uses for Region A -- beta is
        #      fit on the session-wide TRAIN split (never on trials_demo),
        #      exactly like `fit_beta_train_apply_demo` elsewhere -----------
        train_idx, _test_idx = dp.train_test_trial_split(bundle.n_trials_full)

        resid_a_train, coeff_a = dp.residualize_with_beta(
            dp.select_flat(bundle.region_a.zscored, train_idx, neurons_a3),
            dp.select_flat(multi.zscored, train_idx, np.arange(n_b)),
        )
        resid_b_train, coeff_b = dp.residualize_with_beta(
            dp.select_flat(bundle.region_b.zscored, train_idx, neurons_b3),
            dp.select_flat(multi.zscored, train_idx, np.arange(n_b)),
        )
        resid_a_demo, _ = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neurons_a3, multi, np.arange(n_b))
        resid_b_demo, _ = dp.fit_beta_train_apply_demo(bundle, bundle.region_b, neurons_b3, multi, np.arange(n_b))

        Wx, Wy, rho = dp.ridge_cca(resid_a_train, resid_b_train, dp.LAMBDA_CCA, dp.N_COMPONENTS_CCA)
        proj_a = resid_a_demo @ Wx[:, 0]
        proj_b = resid_b_demo @ Wy[:, 0]

        self.set_camera_orientation(phi=68 * DEGREES, theta=-50 * DEGREES)

        caption = th.hyperparam_caption(
            bundle, extra=f"ρ (canonical corr., comp. 0) = {float(rho[0]):.2f}", show_region_b=True
        )

        window = th.scroll_window_seconds(n_trials, T, fs)
        times = th.concat_time_axis(n_trials, T, fs)
        max_t0 = max(float(times[-1]) - window, 0.01)
        t0 = ValueTracker(0.0)
        trial_duration = window
        onset_offset = -dp.DISPLAY_WINDOW_S[0]

        def markers_for(width, center, height):
            return th.make_scroll_markers(t0, window, width, height, center,
                                           trial_duration=trial_duration, onset_offset=onset_offset)

        # ---- non-fixed-in-frame mobjects (the 3D axes/curves/planes) DO get
        #      the ThreeDCamera's phi/theta rotation every frame, unlike the
        #      flat panels above (which bypass it via
        #      `add_fixed_in_frame_mobjects`) -- so placing a 3D mobject at a
        #      raw [x, y, 0] world coordinate does NOT make it appear at
        #      screen position (x, y) the way it would for a flat panel.
        #      `screen_to_world` inverts the camera's own rotation (read live,
        #      so this stays correct even after the ambient rotation in 7d)
        #      to find the world position that projects back to a given
        #      screen (x, y) at zero depth -- letting 3D content sit exactly
        #      where a same-position flat panel would.
        def camera_rotation_matrix() -> np.ndarray:
            phi = self.camera.get_phi()
            theta = self.camera.get_theta()
            gamma = self.camera.get_gamma()
            R = np.identity(3)
            for m in (rotation_about_z(-theta - 90 * DEGREES),
                      axis_angle_rotation_matrix(-phi, RIGHT),
                      rotation_about_z(gamma)):
                R = np.dot(m, R)
            return R

        def screen_to_world(x: float, y: float) -> np.ndarray:
            return camera_rotation_matrix().T @ np.array([x, y, 0.0])

        # ---- 7a/7b: step 5's own frame, at half height, once per region ----
        #      width is untouched -- STEP3_LEFT_X/MID_X/BETA_X/RIGHT_X are the
        #      exact x-positions step 5 uses; only `panel_h`/`top_y` (a half
        #      of CONTENT_H each, split around CONTENT_MID_Y) differ per call.
        half_gap = 0.5
        half_h = (th.CONTENT_H - half_gap) / 2
        top_of_top = th.CONTENT_TOP
        top_of_bottom = th.CONTENT_TOP - half_h - half_gap

        def build_region_frame(region_obj, neuron_idx, region_label, colors_target, coeff, resid_demo,
                                top_y, panel_h):
            n_t = len(neuron_idx)
            # a smaller fraction than steps 2-6's usual 0.88 -- these halves
            # get shrunk again in 7e, and 0.88 leaves too little clearance
            # between a row's own title (just above the box) and a trace
            # peak near the top of that row once everything is that much
            # smaller on screen.
            usable_h = panel_h * 0.8
            offsets_t, row_h_t = th.stacked_row_offsets(n_t, usable_h)
            vals_t_all = region_obj.zscored[trials_demo][:, neuron_idx, :].transpose(1, 0, 2)

            box_t = th.panel_background(th.STEP3_SIDE_W + 0.3, panel_h)
            box_t.move_to([th.STEP3_LEFT_X, th.box_center_y(panel_h, top=top_y), 0])
            label_t = th.panel_title_for(
                box_t, f"{region_label} · {dp.display_name(region_obj.name)} · {n_trials} trials")
            rows_t, marks_t = VGroup(), VGroup()
            for k in range(n_t):
                row_center = np.array(box_t.get_center()) + np.array([0.0, offsets_t[k], 0.0])
                rows_t.add(th.make_scrolling_trace(vals_t_all[k].reshape(-1), times, t0, window,
                                                    th.STEP3_SIDE_W, row_h_t * 0.88, row_center,
                                                    color=colors_target[k]))
                marks_t.add(markers_for(th.STEP3_SIDE_W, row_center, row_h_t * 0.88))

            box_mid = th.panel_background(th.STEP3_MID_W + 0.3, panel_h)
            box_mid.move_to([th.STEP3_MID_X, th.box_center_y(panel_h, top=top_y), 0])
            label_mid = th.panel_title_for(box_mid, f"Other regions ({len(multi.names)}) · {n_trials} trials")
            usable_h_mid = panel_h * 0.8
            vals_mid = multi.zscored[trials_demo].transpose(1, 0, 2).reshape(n_b, -1)
            rows_mid = th.stacked_scrolling_traces(vals_mid, times, t0, window, th.STEP3_MID_W,
                                                    box_mid.get_center(), usable_h_mid, colors_mid,
                                                    stroke_width=0.4)
            marks_mid = markers_for(th.STEP3_MID_W, box_mid.get_center(), usable_h_mid)
            color_bar = th.region_color_bar(multi.names, multi.boundaries, block_colors, n_b,
                                             box_mid.get_center(), usable_h_mid, box_mid.get_left()[0] - 0.12)

            strip = th.beta_heatmap(coeff, cell_w=th.matrix_cell_width(n_t), cell_h=usable_h_mid / n_b)
            strip.move_to([th.STEP3_BETA_X, box_mid.get_center()[1], 0])
            brackets = th.matrix_brackets(strip)

            formula = th.residual_formula_tex(n_b)
            formula.move_to([th.STEP3_RIGHT_X, top_y - 0.35, 0])

            # residual panel -- same height/row geometry as the region's own
            # panel (`row_h_t`), not step 5's independently-grown box, so its
            # rows start out exactly as tall as the neuron rows they came
            # from (this is the frozen n=3 end state, not a growth replay).
            box_r = th.panel_background(th.STEP3_SIDE_W + 0.3, panel_h)
            box_r.move_to([th.STEP3_RIGHT_X, th.box_center_y(panel_h, top=top_y), 0])
            label_r = th.panel_title_for(box_r, f"residual · {region_label} ({n_t} neurons) · {n_trials} trials")
            resid_trial_major = dp.flat_to_trial_major(resid_demo, n_trials, T)
            rows_r, marks_r = VGroup(), VGroup()
            for k in range(n_t):
                row_center = np.array(box_r.get_center()) + np.array([0.0, offsets_t[k], 0.0])
                rows_r.add(th.make_scrolling_trace(resid_trial_major[:, :, k].reshape(-1), times, t0, window,
                                                    th.STEP3_SIDE_W, row_h_t * 0.88, row_center,
                                                    color=colors_target[k]))
                marks_r.add(markers_for(th.STEP3_SIDE_W, row_center, row_h_t * 0.88))

            return {
                "panel_mobs": [box_t, label_t, rows_t, marks_t, box_mid, label_mid, rows_mid, marks_mid,
                               color_bar, strip, brackets, formula],
                "resid_mobs": [box_r, label_r, rows_r, marks_r],
                "resid_top_y": top_y,
                "panel_h": panel_h,
            }

        frame_a = build_region_frame(bundle.region_a, neurons_a3, "Region A", colors_a, coeff_a, resid_a_demo,
                                      top_of_top, half_h)
        frame_b = build_region_frame(bundle.region_b, neurons_b3, "Region B", colors_b, coeff_b, resid_b_demo,
                                      top_of_bottom, half_h)

        self.add_fixed_in_frame_mobjects(
            *frame_a["panel_mobs"], *frame_a["resid_mobs"],
            *frame_b["panel_mobs"], *frame_b["resid_mobs"],
        )
        self.wait(0.6)

        # ---- 7c: replace each half's residual panel with its own 3D space -
        self.play(
            *[FadeOut(m) for m in frame_a["resid_mobs"] + frame_b["resid_mobs"]],
            run_time=0.6,
        )

        trial_a_3d = th.simulated_trajectory(**th.SIM_TRAJ_A)
        trial_b_3d = th.simulated_trajectory(**th.SIM_TRAJ_B)

        axes_len = 2.3

        def make_axes(screen_x, screen_y):
            ax = ThreeDAxes(
                x_range=[-2.6, 2.6, 1], y_range=[-2.6, 2.6, 1], z_range=[-2.6, 2.6, 1],
                x_length=axes_len, y_length=axes_len, z_length=axes_len * 0.9,
            )
            ax.move_to(screen_to_world(screen_x, screen_y))
            return ax

        axes_a = make_axes(th.STEP3_RIGHT_X, th.box_center_y(half_h, top=top_of_top))
        axes_b = make_axes(th.STEP3_RIGHT_X, th.box_center_y(half_h, top=top_of_bottom))

        label_3d_a = th.panel_label("Region A · residual 3D trajectory", font_size=16)
        label_3d_a.move_to([th.STEP3_RIGHT_X, top_of_top + 0.2, 0])
        label_3d_b = th.panel_label("Region B · residual 3D trajectory", font_size=16)
        label_3d_b.move_to([th.STEP3_RIGHT_X, top_of_bottom + 0.2, 0])
        self.add_fixed_in_frame_mobjects(label_3d_a, label_3d_b)

        self.play(Create(axes_a), Create(axes_b), FadeIn(label_3d_a), FadeIn(label_3d_b), run_time=0.8)

        curve_a = VMobject(stroke_width=5, stroke_opacity=0.95)
        curve_a.set_points_as_corners([axes_a.c2p(*p) for p in trial_a_3d])
        curve_a.set_color_by_gradient(*th.GRADIENT_A)
        curve_b = VMobject(stroke_width=5, stroke_opacity=0.95)
        curve_b.set_points_as_corners([axes_b.c2p(*p) for p in trial_b_3d])
        curve_b.set_color_by_gradient(*th.GRADIENT_B)
        self.play(Create(curve_a), Create(curve_b), run_time=2.0)

        # ---- a point looping along each trial's trajectory, from here to
        #      the end of the scene ("dynamics over time") -----------------
        loop_seconds = 3.0
        phase = ValueTracker(0.0)
        phase.add_updater(lambda m, dt: m.set_value((m.get_value() + dt / loop_seconds) % 1.0))
        self.add(phase)

        def traj_point(traj, alpha):
            idx = min(int(alpha * (len(traj) - 1)), len(traj) - 1)
            return traj[idx]

        dot_a = Dot3D(axes_a.c2p(*trial_a_3d[0]), radius=0.08, color=th.INK)
        dot_a.add_updater(lambda m: m.move_to(axes_a.c2p(*traj_point(trial_a_3d, phase.get_value()))))
        dot_b = Dot3D(axes_b.c2p(*trial_b_3d[0]), radius=0.08, color=th.INK)
        dot_b.add_updater(lambda m: m.move_to(axes_b.c2p(*traj_point(trial_b_3d, phase.get_value()))))
        self.play(FadeIn(dot_a), FadeIn(dot_b), run_time=0.4)
        self.wait(0.2)

        # ---- 7d: two projection planes, each searching its own 3D space
        #      for the ridge_cca() direction fit on the TRAIN split --------
        def make_plane(axes_obj, color):
            sq = Square(side_length=1.4, fill_color=color, fill_opacity=0.22, stroke_color=color, stroke_width=2)
            sq.move_to(axes_obj.c2p(0, 0, 0))
            return sq

        plane_a = make_plane(axes_a, colors_a[0])
        plane_b = make_plane(axes_b, colors_b[0])
        R_a = th.rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), Wx[:, 0])
        R_b = th.rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), Wy[:, 0])
        # `apply_matrix` rotates about the WORLD origin by default (not the
        # mobject's own center) -- with axes_a/axes_b sitting well away from
        # world origin (see `screen_to_world` above), that would swing the
        # plane away from its own axes instead of spinning it in place, so
        # `about_point` is pinned to each axes' own origin explicitly.
        plane_a_final = make_plane(axes_a, colors_a[0]).apply_matrix(R_a, about_point=axes_a.c2p(0, 0, 0))
        plane_b_final = make_plane(axes_b, colors_b[0]).apply_matrix(R_b, about_point=axes_b.c2p(0, 0, 0))

        self.play(FadeIn(plane_a), FadeIn(plane_b), run_time=0.6)
        # No ambient camera rotation here -- the camera (and so axes_a/b,
        # curve_a/b, dot_a/b, which are NOT fixed-in-frame) must stay
        # completely still. Orbiting the camera while the plane also rotates
        # made the whole 3D panel look like it was translating across the
        # screen, when the axes themselves never move -- only the plane
        # rotates in place (about its own axes' origin) and then settles.
        self.play(
            Rotate(plane_a, angle=2 * np.pi, axis=RIGHT, about_point=axes_a.c2p(0, 0, 0)),
            Rotate(plane_b, angle=-2 * np.pi, axis=UP, about_point=axes_b.c2p(0, 0, 0)),
            run_time=2.5,
        )
        self.play(
            plane_a.animate.become(plane_a_final),
            plane_b.animate.become(plane_b_final),
            run_time=1.5,
        )
        self.wait(0.3)

        # ---- 7e: every visual from 7a-7d stays exactly as it is -- Panel
        #      1/2/3 (plus the beta matrix/formula/colour bar between them)
        #      simply scale down together, uniformly, about each half's own
        #      top-left corner (so nothing shifts or re-flows, it just gets
        #      smaller toward that fixed corner) until the group is 2/3 of
        #      the row's width, which frees the remaining third on the right
        #      for the new Panel 4 (the CCA component-0 latent).
        #
        #      The STATIC content (boxes, labels, beta matrix, colour bar,
        #      the 3D axes/curve/plane) scales cleanly with `.animate.scale`.
        #      The SCROLLING traces/markers (`rows_t`/`marks_t`/`rows_mid`/
        #      `marks_mid`) are `always_redraw` (theme.py's
        #      `make_scrolling_trace`/`make_scroll_markers`): their
        #      `center`/`width`/`height` are plain values captured once at
        #      construction time, not a live reference to their box, so
        #      scaling the already-built mobject only "sticks" for the
        #      frames the scale animation is actively interpolating -- the
        #      very next frame, their own redraw updater regenerates them
        #      back at the ORIGINAL pre-scale geometry (which is why
        #      everything looked right immediately after the scale, then
        #      went misaligned once the final scroll kept ticking the frame
        #      clock afterward). They're rebuilt fresh at the scaled-down
        #      geometry instead and cross-faded in alongside the scale.
        #      `dot_a`/`dot_b` are unaffected -- their updater reads
        #      `axes_a`/`axes_b`'s CURRENT `c2p` fresh every frame, not a
        #      captured value, so they track the scaled axes correctly. ----
        (box_t_a, label_t_a, rows_t_a, marks_t_a, box_mid_a, label_mid_a, rows_mid_a, marks_mid_a,
         color_bar_a, strip_a, brackets_a, formula_a) = frame_a["panel_mobs"]
        (box_t_b, label_t_b, rows_t_b, marks_t_b, box_mid_b, label_mid_b, rows_mid_b, marks_mid_b,
         color_bar_b, strip_b, brackets_b, formula_b) = frame_b["panel_mobs"]

        row_w = 14.1                                    # same total row width the STEP3 layout fills
        row_left = -row_w / 2
        k = 2 / 3                                        # Panel 1/2/3 together end up 2/3 of row_w

        anchor_a_screen = np.array([row_left, top_of_top, 0.0])
        anchor_b_screen = np.array([row_left, top_of_bottom, 0.0])
        # The 3D content's own "screen position" only exists after the
        # camera's rotation is applied (see `screen_to_world` above); scaling
        # it by the same factor about the SAME on-screen corner as the flat
        # panels means scaling its WORLD points about that corner's WORLD
        # equivalent -- valid here because the camera is a pure rotation (no
        # translation/perspective distortion at zero depth), so scaling
        # commutes through it: R(anchor + k*(p-anchor)) = R(anchor) +
        # k*(R(p)-R(anchor)), i.e. scaling in world space about
        # screen_to_world(...) is exactly scaling on screen about the
        # original screen point.
        anchor_a_world = screen_to_world(*anchor_a_screen[:2])
        anchor_b_world = screen_to_world(*anchor_b_screen[:2])

        def scaled_pt(old_point, anchor):
            return anchor + k * (np.asarray(old_point) - anchor)

        static_group_a = Group(box_t_a, label_t_a, box_mid_a, label_mid_a, color_bar_a,
                                strip_a, brackets_a, formula_a, label_3d_a)
        static_group_b = Group(box_t_b, label_t_b, box_mid_b, label_mid_b, color_bar_b,
                                strip_b, brackets_b, formula_b, label_3d_b)
        space_a = Group(axes_a, curve_a, plane_a)
        space_b = Group(axes_b, curve_b, plane_b)

        def rebuild_dynamic(region_obj, neuron_idx, colors_target, box_t_center, box_mid_center, anchor):
            n_t = len(neuron_idx)
            usable_h = k * (half_h * 0.8)
            offsets_t, row_h_t = th.stacked_row_offsets(n_t, usable_h)
            vals_t_all = region_obj.zscored[trials_demo][:, neuron_idx, :].transpose(1, 0, 2)
            new_box_t_center = scaled_pt(box_t_center, anchor)
            new_side_w = k * th.STEP3_SIDE_W

            rows_t, marks_t = VGroup(), VGroup()
            for i in range(n_t):
                row_center = new_box_t_center + np.array([0.0, offsets_t[i], 0.0])
                rows_t.add(th.make_scrolling_trace(vals_t_all[i].reshape(-1), times, t0, window,
                                                    new_side_w, row_h_t * 0.88, row_center,
                                                    color=colors_target[i]))
                marks_t.add(markers_for(new_side_w, row_center, row_h_t * 0.88))

            new_box_mid_center = scaled_pt(box_mid_center, anchor)
            new_mid_w = k * th.STEP3_MID_W
            usable_h_mid = k * (half_h * 0.8)
            vals_mid = multi.zscored[trials_demo].transpose(1, 0, 2).reshape(n_b, -1)
            rows_mid = th.stacked_scrolling_traces(vals_mid, times, t0, window, new_mid_w,
                                                    new_box_mid_center, usable_h_mid, colors_mid,
                                                    stroke_width=0.4)
            marks_mid = markers_for(new_mid_w, new_box_mid_center, usable_h_mid)
            return rows_t, marks_t, rows_mid, marks_mid

        new_rows_t_a, new_marks_t_a, new_rows_mid_a, new_marks_mid_a = rebuild_dynamic(
            bundle.region_a, neurons_a3, colors_a, box_t_a.get_center(), box_mid_a.get_center(), anchor_a_screen)
        new_rows_t_b, new_marks_t_b, new_rows_mid_b, new_marks_mid_b = rebuild_dynamic(
            bundle.region_b, neurons_b3, colors_b, box_t_b.get_center(), box_mid_b.get_center(), anchor_b_screen)

        # Panel 4 (CCA component-0 latent), one per half, sized to match the
        # scaled group's own new height and top edge, sitting in the freed
        # right third -- with the same onset/boundary-marker convention
        # (`markers_for`) every scrolling trace in this walkthrough uses.
        panel4_h = k * half_h
        panel4_w = row_w * (1 - k) - 0.3
        panel4_x = row_left + row_w * (1 + k) / 2
        pa_vals = dp.flat_to_trial_major(proj_a.reshape(-1, 1), n_trials, T)[:, :, 0].reshape(-1)
        pb_vals = dp.flat_to_trial_major(proj_b.reshape(-1, 1), n_trials, T)[:, :, 0].reshape(-1)

        def make_cca_panel(vals, region_label, color, top_y):
            box = th.panel_background(panel4_w, panel4_h)
            box.move_to([panel4_x, th.box_center_y(panel4_h, top=top_y), 0])
            label = th.panel_title_for(box, f"{region_label} · CCA comp. 0")
            trace = th.make_scrolling_trace(vals, times, t0, window, panel4_w, panel4_h * 0.7,
                                             box.get_center(), color=color)
            marks = markers_for(panel4_w, box.get_center(), panel4_h * 0.7)
            return box, label, trace, marks

        box_pa, label_pa, trace_pa, marks_pa = make_cca_panel(pa_vals, "Region A", colors_a[-1], top_of_top)
        box_pb, label_pb, trace_pb, marks_pb = make_cca_panel(pb_vals, "Region B", colors_b[-1], top_of_bottom)

        self.add_fixed_in_frame_mobjects(
            new_rows_t_a, new_marks_t_a, new_rows_mid_a, new_marks_mid_a,
            new_rows_t_b, new_marks_t_b, new_rows_mid_b, new_marks_mid_b,
            box_pa, label_pa, trace_pa, marks_pa, box_pb, label_pb, trace_pb, marks_pb,
        )
        self.play(
            static_group_a.animate.scale(k, about_point=anchor_a_screen),
            static_group_b.animate.scale(k, about_point=anchor_b_screen),
            space_a.animate.scale(k, about_point=anchor_a_world),
            space_b.animate.scale(k, about_point=anchor_b_world),
            FadeOut(rows_t_a), FadeOut(marks_t_a), FadeOut(rows_mid_a), FadeOut(marks_mid_a),
            FadeOut(rows_t_b), FadeOut(marks_t_b), FadeOut(rows_mid_b), FadeOut(marks_mid_b),
            FadeIn(new_rows_t_a), FadeIn(new_marks_t_a), FadeIn(new_rows_mid_a), FadeIn(new_marks_mid_a),
            FadeIn(new_rows_t_b), FadeIn(new_marks_t_b), FadeIn(new_rows_mid_b), FadeIn(new_marks_mid_b),
            FadeIn(box_pa), FadeIn(label_pa), FadeIn(trace_pa), FadeIn(marks_pa),
            FadeIn(box_pb), FadeIn(label_pb), FadeIn(trace_pb), FadeIn(marks_pb),
            run_time=1.5,
        )
        self.wait(0.2)

        self.play(t0.animate.set_value(max_t0), run_time=10, rate_func=linear)
        self.wait(1.0)

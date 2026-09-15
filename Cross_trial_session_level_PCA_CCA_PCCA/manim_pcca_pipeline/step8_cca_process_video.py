"""
Step 8 -- a standalone two-panel video isolating the pcca() CCA step, built
on step 7's revised Panel 3 (the simulated 3D trajectory, one per region,
`theme.simulated_trajectory` + its own colour gradient) and Panel 4 (the CCA
component-0 latent, resid_demo @ Wx[:, 0] / Wy[:, 0]) -- without step 7's
Panel 1/2 (raw + other-region scrolling traces) or the beta matrix/formula
between them, since this video's only subject is the CCA step itself.

Region A (top half) and Region B (bottom half) each carry TWO columns, side
by side, for the whole scene -- the 3D trajectory (left) and its CCA latent
trace (right). Both columns are laid out once at fixed x-positions, but they
appear in sequence, not together, and neither ever fades out for the other
to fade in -- each stays exactly where it appeared once it's on screen:
  1. left column only -- the 3D trajectory (simulated, for visual clarity)
     appears and plays;
  2. the plane-rotation segment, still left column only -- a projection
     plane per region searches for, then settles on, the REAL ridge_cca()
     direction (Wx[:, 0] / Wy[:, 0]) fit on the TRAIN half of a fixed
     session-wide train/test split, exactly as step 7's 7d (only the loop
     the plane rotates in front of is illustrative -- the CCA direction
     itself is real);
  3. only once the plane has settled does the right column appear, fading
     in alongside the now-still left column, followed by the scroll
     (`t0`) that plays both columns' remaining dynamics to the end.

    manim -pqh step8_cca_process_video.py Step8CCAProcessVideo
"""

import numpy as np
from manim import (
    DEGREES,
    RIGHT,
    UP,
    Create,
    Dot3D,
    FadeIn,
    Rotate,
    Square,
    ThreeDAxes,
    ThreeDScene,
    ValueTracker,
    VMobject,
    config,
    linear,
)
from manim.utils.space_ops import rotation_about_z
from manim.utils.space_ops import rotation_matrix as axis_angle_rotation_matrix

# ---- this video's own canvas: 4:3 (width:height) -- requested as "3:4
#      height:width" -- rather than the other steps' 16:9, so each row (a
#      3D trajectory beside its own latent panel) gets more vertical room.
#      Must run before `theme` is imported: theme.py reads config.frame_width
#      /frame_height once, at import time, to set its own layout constants
#      (CONTENT_TOP/BOTTOM etc.), and ThreeDScene's own camera is built from
#      config.pixel_width/frame_width at Scene-construction time -- both
#      earlier than anything in `construct()` below. Only pixel_width/
#      frame_width change; pixel_height/frame_height (and so the CLI's own
#      -ql/-qm/-qh/-qk vertical resolution) are left exactly as requested.
_ASPECT_W, _ASPECT_H = 4, 3
config.pixel_width = round(config.pixel_height * _ASPECT_W / _ASPECT_H)
config.frame_width = config.frame_height * _ASPECT_W / _ASPECT_H

import data_prep as dp
import theme as th


class Step8CCAProcessVideo(ThreeDScene):
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

        # ---- same real train/test-split CCA fit as step 7 -- only the 3D
        #      loop the plane rotates in front of (below) is simulated;
        #      Wx[:, 0] / Wy[:, 0] and the final latent traces are real. ----
        train_idx, _test_idx = dp.train_test_trial_split(bundle.n_trials_full)
        resid_a_train, _coeff_a = dp.residualize_with_beta(
            dp.select_flat(bundle.region_a.zscored, train_idx, neurons_a3),
            dp.select_flat(multi.zscored, train_idx, np.arange(n_b)),
        )
        resid_b_train, _coeff_b = dp.residualize_with_beta(
            dp.select_flat(bundle.region_b.zscored, train_idx, neurons_b3),
            dp.select_flat(multi.zscored, train_idx, np.arange(n_b)),
        )
        resid_a_demo, _ = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neurons_a3, multi, np.arange(n_b))
        resid_b_demo, _ = dp.fit_beta_train_apply_demo(bundle, bundle.region_b, neurons_b3, multi, np.arange(n_b))

        Wx, Wy, rho = dp.ridge_cca(resid_a_train, resid_b_train, dp.LAMBDA_CCA, dp.N_COMPONENTS_CCA)
        proj_a = resid_a_demo @ Wx[:, 0]
        proj_b = resid_b_demo @ Wy[:, 0]

        self.set_camera_orientation(phi=68 * DEGREES, theta=-50 * DEGREES)

        # caption = th.hyperparam_caption(
        #     bundle, extra=f"CCA step only · ρ (canonical corr., comp. 0) = {float(rho[0]):.2f}",
        #     show_region_b=True,
        # )
        # self.add_fixed_in_frame_mobjects(caption)

        window = th.scroll_window_seconds(n_trials, T, fs)
        times = th.concat_time_axis(n_trials, T, fs)
        max_t0 = max(float(times[-1]) - window, 0.01)
        t0 = ValueTracker(0.0)
        trial_duration = window
        onset_offset = -dp.DISPLAY_WINDOW_S[0]

        def markers_for(width, center, height):
            return th.make_scroll_markers(t0, window, width, height, center,
                                           trial_duration=trial_duration, onset_offset=onset_offset)

        # ---- same screen<->world trick step 7 uses so 3D content can sit at
        #      an exact screen (x, y), matching a same-position flat panel --
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

        half_gap = 0.6
        half_h = (th.CONTENT_H - half_gap) / 2
        top_of_top = th.CONTENT_TOP
        top_of_bottom = th.CONTENT_TOP - half_h - half_gap

        # ---- two side-by-side columns per row: the 3D trajectory (left)
        #      and its CCA latent trace (right) -- both columns' x-centres
        #      are fixed for the whole scene, so nothing ever re-flows into
        #      the other's spot. ------------------------------------------
        row_w = th.FRAME_W - 0.7             # derived from FRAME_W, not hard-coded --
                                              # this canvas is 4:3, narrower than the
                                              # other steps' 16:9
        col_gap = 0.5
        left_col_w = row_w * 0.48            # 3D column (axes + margin)
        right_col_w = row_w - left_col_w - col_gap   # latent-panel column
        row_left = -row_w / 2
        left_x = row_left + left_col_w / 2
        right_x = row_left + left_col_w + col_gap + right_col_w / 2

        # ---- left column: the 3D trajectory, same simulated loop + colour
        #      gradient as step 7's revised Panel 3. ------------------------
        axes_len = left_col_w * 0.5

        def make_axes(screen_y):
            ax = ThreeDAxes(
                x_range=[-2.6, 2.6, 1], y_range=[-2.6, 2.6, 1], z_range=[-2.6, 2.6, 1],
                x_length=axes_len, y_length=axes_len, z_length=axes_len * 0.9,
            )
            ax.move_to(screen_to_world(left_x, screen_y))
            return ax

        axes_a = make_axes(th.box_center_y(half_h, top=top_of_top))
        axes_b = make_axes(th.box_center_y(half_h, top=top_of_bottom))

        # this canvas's left column is narrower than the other steps' own 3D
        # panel column, so these titles are shrunk to fit it explicitly
        # (`panel_label` alone doesn't -- that's `panel_title_for`'s job for
        # a box-anchored label, but there's no box here to anchor to).
        label_a = th.panel_label(f"Region A · {dp.display_name(bundle.region_a.name)} · 3D trajectory",
                                  font_size=16)
        if label_a.width > left_col_w:
            label_a.scale_to_fit_width(left_col_w)
        label_a.move_to([left_x, top_of_top + 0.25, 0])
        label_b = th.panel_label(f"Region B · {dp.display_name(bundle.region_b.name)} · 3D trajectory",
                                  font_size=16)
        if label_b.width > left_col_w:
            label_b.scale_to_fit_width(left_col_w)
        label_b.move_to([left_x, top_of_bottom + 0.25, 0])

        trial_a_3d = th.simulated_trajectory(**th.SIM_TRAJ_A)
        trial_b_3d = th.simulated_trajectory(**th.SIM_TRAJ_B)

        curve_a = VMobject(stroke_width=6, stroke_opacity=0.95)
        curve_a.set_points_as_corners([axes_a.c2p(*p) for p in trial_a_3d])
        curve_a.set_color_by_gradient(*th.GRADIENT_A)
        curve_b = VMobject(stroke_width=6, stroke_opacity=0.95)
        curve_b.set_points_as_corners([axes_b.c2p(*p) for p in trial_b_3d])
        curve_b.set_color_by_gradient(*th.GRADIENT_B)

        # ---- right column: the CCA component-0 latent trace, built and
        #      placed now so it is on screen from the very first beat,
        #      alongside the 3D column -- not introduced later in its
        #      place. ------------------------------------------------------
        pa_vals = dp.flat_to_trial_major(proj_a.reshape(-1, 1), n_trials, T)[:, :, 0].reshape(-1)
        pb_vals = dp.flat_to_trial_major(proj_b.reshape(-1, 1), n_trials, T)[:, :, 0].reshape(-1)

        panel_w = right_col_w - 0.3
        panel_h = half_h * 0.8

        def make_latent_panel(vals, region_label, color, top_y):
            box = th.panel_background(panel_w, panel_h)
            box.move_to([right_x, th.box_center_y(panel_h, top=top_y), 0])
            label = th.panel_title_for(box, f"{region_label} · CCA component 0 (latent)")
            trace = th.make_scrolling_trace(vals, times, t0, window, panel_w, panel_h * 0.75,
                                             box.get_center(), color=color, stroke_width=2.5)
            marks = markers_for(panel_w, box.get_center(), panel_h * 0.75)
            return box, label, trace, marks

        box_a, label_pa, trace_pa, marks_pa = make_latent_panel(pa_vals, "Region A", th.GRADIENT_A[0], top_of_top)
        box_b, label_pb, trace_pb, marks_pb = make_latent_panel(pb_vals, "Region B", th.GRADIENT_B[0], top_of_bottom)
        # `box`/`label`/`trace`/`marks` above are only CONSTRUCTED here (kept
        # off the scene) -- the right column doesn't actually appear until
        # after the plane-rotation segment settles, below.

        self.add_fixed_in_frame_mobjects(label_a, label_b)

        # ---- left column only: axes, then the trajectory itself ----------
        self.play(Create(axes_a), Create(axes_b), FadeIn(label_a), FadeIn(label_b), run_time=0.8)
        self.play(Create(curve_a), Create(curve_b), run_time=2.2)

        # ---- a point looping along each trajectory, "dynamics over time",
        #      exactly as step 7's own dot_a/dot_b. --------------------
        loop_seconds = 3.0
        phase = ValueTracker(0.0)
        phase.add_updater(lambda m, dt: m.set_value((m.get_value() + dt / loop_seconds) % 1.0))
        self.add(phase)

        def traj_point(traj, alpha):
            idx = min(int(alpha * (len(traj) - 1)), len(traj) - 1)
            return traj[idx]

        dot_a = Dot3D(axes_a.c2p(*trial_a_3d[0]), radius=0.09, color=th.INK)
        dot_a.add_updater(lambda m: m.move_to(axes_a.c2p(*traj_point(trial_a_3d, phase.get_value()))))
        dot_b = Dot3D(axes_b.c2p(*trial_b_3d[0]), radius=0.09, color=th.INK)
        dot_b.add_updater(lambda m: m.move_to(axes_b.c2p(*traj_point(trial_b_3d, phase.get_value()))))
        self.play(FadeIn(dot_a), FadeIn(dot_b), run_time=0.4)
        self.wait(0.6)

        # ---- plane-rotation segment, left column only -- identical
        #      mechanics to step 7's 7d (searching, then settling on, the
        #      REAL ridge_cca() direction fit on the TRAIN split). No
        #      ambient camera rotation here either, for the same reason
        #      step 7 avoids it: the axes/curve/dot must stay completely
        #      still while only the plane rotates in place. The right
        #      column doesn't exist on screen yet at all. -----------------
        def make_plane(axes_obj, color):
            sq = Square(side_length=axes_len * 0.6, fill_color=color, fill_opacity=0.22, stroke_color=color,
                        stroke_width=2)
            sq.move_to(axes_obj.c2p(0, 0, 0))
            return sq

        plane_a = make_plane(axes_a, th.GRADIENT_A[0])
        plane_b = make_plane(axes_b, th.GRADIENT_B[0])
        R_a = th.rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), Wx[:, 0])
        R_b = th.rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), Wy[:, 0])
        plane_a_final = make_plane(axes_a, th.GRADIENT_A[0]).apply_matrix(R_a, about_point=axes_a.c2p(0, 0, 0))
        plane_b_final = make_plane(axes_b, th.GRADIENT_B[0]).apply_matrix(R_b, about_point=axes_b.c2p(0, 0, 0))

        self.play(FadeIn(plane_a), FadeIn(plane_b), run_time=0.6)
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

        # ---- only now, once the plane has finished searching and settled
        #      onto the real CCA direction, does the right column appear --
        #      the left column (axes/curve/dot/plane) stays exactly as it
        #      is; the latent panel is added alongside it, not in place of
        #      it. ----------------------------------------------------------
        self.add_fixed_in_frame_mobjects(box_a, label_pa, trace_pa, marks_pa, box_b, label_pb, trace_pb, marks_pb)
        self.play(
            FadeIn(box_a), FadeIn(label_pa), FadeIn(trace_pa), FadeIn(marks_pa),
            FadeIn(box_b), FadeIn(label_pb), FadeIn(trace_pb), FadeIn(marks_pb),
            run_time=0.8,
        )
        self.wait(0.3)

        # ---- both columns keep running to the end together: the dot loops
        #      on its own independent clock (`phase`, already updating)
        #      while the right column's latent trace scrolls through its
        #      real timeline via `t0` -- side by side, still nothing fades
        #      out for the other. ------------------------------------------
        self.play(t0.animate.set_value(max_t0), run_time=10, rate_func=linear)
        self.wait(1.0)

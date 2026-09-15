"""
Step 1 -- residualize(): one neuron, one predictor, one trial. Simulated
data (see data_prep.simulate_step1_traces): Region A's example neuron has
five separated Gaussian peaks; the other region's example neuron repeats two
of them, so the residual keeps exactly the other three.

    manim -pqh step1_single_trial_residual.py Step1SingleTrialResidual
"""

from manim import DOWN, Create, FadeIn, Scene, Text, Write

import data_prep as dp
import theme as th


class Step1SingleTrialResidual(Scene):
    def construct(self):
        t, x_vals, z_vals = dp.simulate_step1_traces()
        t_lo, t_hi = float(t[0]), float(t[-1])

        # caption_txt = ("simulated single-trial data (not session data)  ·  "
        #                 f"window [{t_lo:.0f}, {t_hi:.0f}] s, 0 = align time")
        # caption = Text(caption_txt, font=th.FONT, font_size=16, color=th.INK_MUTED)
        # if caption.width > th.FRAME_W - 0.4:
        #     caption.scale_to_fit_width(th.FRAME_W - 0.4)
        # caption.to_edge(DOWN, buff=0.22)
        # self.play(FadeIn(caption), run_time=0.5)

        panel_y = th.box_center_y(1.7)
        color_a = th.ramp(th.REGION_A_COLOR, 1)[0]
        color_z = th.ramp(th.REGION_B_COLOR, 1)[0]

        # ---- a: Region A example neuron, upper left ----------------------------
        box_a = th.panel_background(th.EQ_PANEL_W + 0.3, 1.7)
        box_a.move_to([th.EQ_LEFT_X, panel_y, 0])
        label_a = th.panel_title_for(box_a, "Region A · example neuron")
        trace_a = th.make_trace(x_vals, width=th.EQ_PANEL_W, height=1.2, color=color_a)
        trace_a.move_to(box_a.get_center())
        onset_a = th.onset_marker(th.EQ_PANEL_W, 1.2, t_lo, t_hi, center=box_a.get_center())

        self.play(FadeIn(label_a), Create(box_a), Create(onset_a), Create(trace_a), run_time=1.5)
        self.wait(4.5)

        # ---- b: the other region's example neuron, middle ----------------------
        box_b = th.panel_background(th.EQ_PANEL_W + 0.3, 1.7)
        box_b.move_to([th.EQ_MID_X, panel_y, 0])
        label_b = th.panel_title_for(box_b, "Other region · example neuron")
        trace_b = th.make_trace(z_vals, width=th.EQ_PANEL_W, height=1.2, color=color_z)
        trace_b.move_to(box_b.get_center())
        onset_b = th.onset_marker(th.EQ_PANEL_W, 1.2, t_lo, t_hi, center=box_b.get_center())

        self.play(FadeIn(label_b), Create(box_b), Create(onset_b), Create(trace_b), run_time=1.5)
        self.wait(8.5)

        # ---- c: beta_1, inset in the other region panel's top-right corner
        #      (no longer its own column, so a/b/residual stay equal width) -----
        X = x_vals.reshape(-1, 1)
        Z = z_vals.reshape(-1, 1)
        resid, coeff = dp.residualize_with_beta(X, Z)
        beta_1 = float(coeff[0, 0])

        beta_label = th.beta_label_tex(beta_1, idx=1)
        th.inset_top_right(box_b, beta_label)
        self.play(Write(beta_label), run_time=0.3)
        self.wait(0.1)

        # ---- d: formula, top right (no title this step) ------------------------
        formula = th.residual_formula_tex(None)
        formula.move_to([th.EQ_RIGHT_X, th.FORMULA_Y, 0])
        self.play(Write(formula), run_time=0.3)
        self.wait(1)

        # ---- e: residual of the Region A neuron, first row, right -- same
        #      height as a/b ------------------------------------------------------
        resid_vals = resid[:, 0]
        box_r = th.panel_background(th.EQ_PANEL_W + 0.3, 1.7)
        box_r.move_to([th.EQ_RIGHT_X, panel_y, 0])
        label_r = th.panel_title_for(box_r, "Region A example neuron · residual")
        trace_r = th.make_trace(resid_vals, width=th.EQ_PANEL_W, height=1.2, color=color_a)
        trace_r.move_to(box_r.get_center())
        onset_r = th.onset_marker(th.EQ_PANEL_W, 1.2, t_lo, t_hi, center=box_r.get_center())

        self.play(FadeIn(label_r), Create(box_r), Create(onset_r), Create(trace_r), run_time=0.5)
        self.wait(2.0)

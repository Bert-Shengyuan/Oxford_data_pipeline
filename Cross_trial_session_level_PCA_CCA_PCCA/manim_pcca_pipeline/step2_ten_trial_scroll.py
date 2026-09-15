"""
Step 2 -- residualize(): the same neuron pair, ten trials concatenated.
Region A vs. the other region (see data_prep's OTHER_REGION) -- Region B is
held back for step 7.

One video, two back-to-back segments that share Region A's panel: the other
region's neuron with a relatively small |beta_1| against Region A's example
neuron scrolls through first, then panels 2/3 are swapped for the other
region's neuron with a relatively large |beta_1| and the scroll replays.
Candidates are restricted to the other region's "active" neuron pool
(data_prep's own ACTIVITY_TOP_FRACTION ordering) so the picks are never a
near-silent unit whose ridge fit is numerically unstable (tiny variance in
the denominator inflating |beta| without any real signal to show on screen).

    manim -pqh step2_ten_trial_scroll.py Step2TenTrialScroll
"""

import numpy as np
from manim import Create, FadeIn, FadeOut, Scene, ValueTracker, Write, linear

import data_prep as dp
import theme as th


def _pick_beta_extreme_neurons(bundle, neuron_a_idx: int):
    """Regress Region A's example neuron onto every candidate other-region
    neuron one at a time -- the same ridge hat-matrix solve
    `residualize_with_beta` uses for a single-column Z, vectorized across
    neurons -- and return ((idx, beta) for the largest |beta_1|, (idx, beta)
    for the smallest |beta_1|) among the other region's *active* neurons
    (the top `ACTIVITY_TOP_FRACTION` of `neuron_order`, per data_prep's own
    dead/near-silent-neuron guard -- without this filter a near-zero-variance
    predictor can win "largest |beta_1|" purely from a tiny ridge denominator,
    with nothing visible on screen to justify it). Fit on the TRAIN split of
    the full session (`train_test_trial_split`), matching
    `fit_beta_train_apply_demo`'s convention -- never on just the
    N_TRIALS_DEMO trials displayed -- so the pick isn't itself tuned to the
    handful of trials on screen."""
    n_avail_o = bundle.region_other.n_neurons_available
    n_active = max(1, int(np.ceil(n_avail_o * dp.ACTIVITY_TOP_FRACTION)))
    candidates = bundle.region_other.neuron_order[:n_active]

    train_idx, _test_idx = dp.train_test_trial_split(bundle.n_trials_full)
    X = dp.select_flat(bundle.region_a.zscored, train_idx, np.array([neuron_a_idx]))
    Zs = dp.select_flat(bundle.region_other.zscored, train_idx, candidates)
    n = Zs.shape[0]
    num = Zs.T @ X                                              # (n_active, 1)
    den = (Zs ** 2).sum(axis=0, keepdims=True).T + dp.LAMBDA_HAT * n
    betas = (num / den).ravel()
    idx_high = int(candidates[np.argmax(np.abs(betas))])
    idx_low = int(candidates[np.argmin(np.abs(betas))])
    beta_high = float(betas[np.argmax(np.abs(betas))])
    beta_low = float(betas[np.argmin(np.abs(betas))])
    return (idx_high, beta_high), (idx_low, beta_low)


class Step2TenTrialScroll(Scene):
    def construct(self):
        bundle = dp.load_pipeline_data()
        T, fs = bundle.T, bundle.fs
        n_trials = dp.N_TRIALS_DEMO
        trials_demo = bundle.trials_demo

        neuron_a = bundle.region_a.neurons(1)
        (idx_high, _beta_high), (idx_low, _beta_low) = _pick_beta_extreme_neurons(bundle, neuron_a[0])

        color_a = th.ramp(th.REGION_A_COLOR, 1)[0]
        color_o = th.ramp(th.REGION_B_COLOR, 1)[0]

        # ---- Region A panel -- built once, stays on screen for both segments
        panel_y = th.box_center_y(1.7)
        box_a = th.panel_background(th.EQ_PANEL_W + 0.3, 1.7)
        box_a.move_to([th.EQ_LEFT_X, panel_y, 0])
        label_a = th.panel_title_for(box_a, f"Region A · {dp.display_name(bundle.region_a.name)} · neuron 1 · {n_trials} trials")
        self.play(FadeIn(label_a), Create(box_a), run_time=0.4)

        window = th.scroll_window_seconds(n_trials, T, fs)
        times = th.concat_time_axis(n_trials, T, fs)
        max_t0 = max(float(times[-1]) - window, 0.01)
        t0 = ValueTracker(0.0)
        trial_duration = window                       # one trial == one scroll-window width
        onset_offset = -dp.DISPLAY_WINDOW_S[0]         # seconds from a trial's start to its own t=0

        def markers_for(width, center):
            return th.make_scroll_markers(t0, window, width, 1.2, center,
                                           trial_duration=trial_duration, onset_offset=onset_offset)

        vals_a = bundle.region_a.zscored[trials_demo][:, neuron_a[0], :].reshape(-1)
        trace_a = th.make_scrolling_trace(vals_a, times, t0, window, th.EQ_PANEL_W, 1.2,
                                           box_a.get_center(), color=color_a)
        marks_a = markers_for(th.EQ_PANEL_W, box_a.get_center())
        self.play(FadeIn(marks_a), Create(trace_a), run_time=1.5)
        self.wait(1.5)
        def run_segment(neuron_o_idx: int, tag: str):
            neuron_o = np.array([neuron_o_idx])

            box_o = th.panel_background(th.EQ_PANEL_W + 0.3, 1.7)
            box_o.move_to([th.EQ_MID_X, panel_y, 0])
            label_o = th.panel_title_for(
                box_o,
                f"Other region · {dp.display_name(bundle.region_other.name)} · neuron {neuron_o_idx + 1} · {n_trials} trials",
            )

            vals_o = bundle.region_other.zscored[trials_demo][:, neuron_o[0], :].reshape(-1)
            trace_o = th.make_scrolling_trace(vals_o, times, t0, window, th.EQ_PANEL_W, 1.2,
                                               box_o.get_center(), color=color_o)
            marks_o = markers_for(th.EQ_PANEL_W, box_o.get_center())
            self.play(FadeIn(label_o), Create(box_o),FadeIn(marks_o), Create(trace_o), run_time=1.5)
            self.wait(1)

            # ---- beta_1 + formula, fixed top right -- Beta fit on the TRAIN
            #      split of all n_trials_full trials, applied to the demo
            #      trials for the residual actually scrolled on screen -----
            resid, coeff = dp.fit_beta_train_apply_demo(bundle, bundle.region_a, neuron_a,
                                                          bundle.region_other, neuron_o)
            beta_1 = float(coeff[0, 0])

            beta_label = th.beta_label_tex(beta_1, idx=1)
            th.inset_top_right(box_o, beta_label)

            formula = th.residual_formula_tex(None)
            formula.move_to([th.EQ_RIGHT_X, th.FORMULA_Y, 0])
            self.play(Write(beta_label), Write(formula), run_time=0.5)
            self.wait(0.1)

            # ---- residual, ten trials, scrolling -----------------------------
            resid_vals = dp.flat_to_trial_major(resid, n_trials, T)[:, :, 0].reshape(-1)
            box_r = th.panel_background(th.EQ_PANEL_W + 0.3, 1.7)
            box_r.move_to([th.EQ_RIGHT_X, panel_y, 0])
            label_r = th.panel_title_for(box_r, f"residual · Region A neuron 1 · {n_trials} trials")
            trace_r = th.make_scrolling_trace(resid_vals, times, t0, window, th.EQ_PANEL_W, 1.2,
                                               box_r.get_center(), color=color_a)
            marks_r = markers_for(th.EQ_PANEL_W, box_r.get_center())

            self.play(FadeIn(label_r), Create(box_r), FadeIn(marks_r),  Create(trace_r), run_time=0.6)
            self.wait(0.5)

            # ---- scroll through all ten trials, everything moving together --
            self.play(t0.animate.set_value(max_t0), run_time=6, rate_func=linear)
            self.wait(1.0)

            return [box_o, label_o, marks_o, trace_o, beta_label, formula, box_r, label_r, marks_r, trace_r]

        # # ---- segment 1: small |beta_1| ----------------------------------------
        # segment_low = run_segment(idx_low, "small |beta|")
        #
        # # ---- swap panels 2/3 for the large-|beta_1| neuron, Region A stays put
        # self.play(FadeOut(*segment_low), run_time=0.4)
        # t0.set_value(0.0)

        segment_high = run_segment(idx_high, "")
        self.wait(0.5)

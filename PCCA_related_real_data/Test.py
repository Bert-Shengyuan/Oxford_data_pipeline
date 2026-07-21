 # =============================================================================
 # 0d.  A0 poster scaling — shared by every session-wide heatmap / bar
 #      figure below (create_session_subspace_angles_single,
 #      create_session_gini_panel, create_session_gini_panel_full_ablation,
 #      create_session_mi_bar).  Figures sized for on-screen/paper viewing
 #      become illegible once printed at A0 (33.1 x 46.8 in); POSTER_SCALE
 #      multiplies canvas size, tick/annotation/label fontsize, and cell-
 #      grid linewidth relative to the screen-resolution defaults used
 #      elsewhere in this module.  _FS_FLOOR guards against per-cell
 #      annotation text (e.g. "0.73" in a large R x R heatmap) shrinking
 #      below legibility as R grows, since figsize scales with R but a
 #      fixed base fontsize does not.
 #
 #      This is a single tunable knob, not a fixed physical mapping: the
 #      "correct" POSTER_SCALE depends on the final placement width of
 #      each panel inside your A0 layout (whatever page-layout tool you're
 #      assembling the poster in), which this script has no visibility
 #      into.  Render once, compare against the panel's placeholder size
 #      in your layout, and adjust POSTER_SCALE up/down accordingly —
 #      raising dpi below is a separate, orthogonal knob for print
 #      sharpness once the physical size is right.
 # =============================================================================


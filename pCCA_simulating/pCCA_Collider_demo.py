"""
collider_bias_demo_v2.py
========================
Key change from v1
------------------
Panel 1 is now split into two sub-panels:

  1a  Raster  — stochastic Poisson spike ticks (unchanged)
  1b  Rate strip — λ per bin per region shown as 4 side-by-side bar charts
      (Z, A1, A2, B), each bar annotated with the numeric λ value.

The rate-strip bars have the SAME temporal pattern as the stem plots in
Panel 2 (source signals s_ZG, s_A2B and their sum B), making the chain

    source signal  →  λ (Poisson rate)  →  observed spikes

visually explicit and directly comparable across both panels.

Causal structure
    s_ZG  → Z, A1, B
    s_A2B → A2, B
B is a collider.  corr(A1, A2) = 0 before conditioning → -1 after.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# 0.  SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
plt.rcParams.update({'font.family': 'sans-serif', 'mathtext.fontset': 'stix'})

C = dict(
    Z    = '#2ca02c',   # green
    A1   = '#1f77b4',   # blue
    A2   = '#ff7f0e',   # orange
    B    = '#d62728',   # red
    sZG  = '#8c564b',   # brown   — source s_ZG
    sA2B = '#9467bd',   # purple  — source s_A2B
    proj = '#bcbd22',   # olive   — β·B component
    r1   = '#17becf',   # cyan    — A1 residual
    r2   = '#e377c2',   # pink    — A2 residual
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LATENT SOURCES  +  POISSON RATES  +  SPIKE COUNTS
# ─────────────────────────────────────────────────────────────────────────────
t = np.arange(1, 5)                          # time-bin indices [1,2,3,4]

s_ZG  = np.array([ 1.,  1., -2.5, -2.5])      # latent source driving Z, A1, B
s_A2B = np.array([ -2.5, -2.5,  1., 1.])      # latent source driving A2, B

BASE, AMP = 2.5, 1.5                         # λ = BASE + AMP * source_signal
lam_ZG  = BASE + AMP * s_ZG                 # [4.0, 4.0, 1.0, 1.0]
lam_A2B = BASE + AMP * s_A2B               # [4.0, 1.0, 4.0, 1.0]
lam_B   = lam_ZG + lam_A2B                 # [8.0, 5.0, 5.0, 2.0]

# λ lookup used in the rate strip — order matches raster rows (bottom→top)
LAM_LOOKUP = {
    'Z':  lam_ZG,
    'A1': lam_ZG,
    'A2': lam_A2B,
    'B':  lam_B,
}

# Idealized counts (exact lambda values to match later panels visually)
Z_ct = lam_ZG.astype(int)
A1_ct = lam_ZG.astype(int)
A2_ct = lam_A2B.astype(int)
B_ct = lam_B.astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ANALYTICAL RESIDUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
A1 = s_ZG.copy()
A2 = s_A2B.copy()
B  = s_ZG + s_A2B                            # [+2, 0, 0, -2]

beta1 = float(A1 @ B) / float(B @ B)         # 0.5
beta2 = float(A2 @ B) / float(B @ B)         # 0.5

proj1  = beta1 * B                            # [+1,  0,  0, -1]
proj2  = beta2 * B                            # [+1,  0,  0, -1]
A1_res = A1 - proj1                           # [ 0, +1, -1,  0]
A2_res = A2 - proj2                           # [ 0, -1, +1,  0]

r_before = float(A1 @ A2) / (np.linalg.norm(A1) * np.linalg.norm(A2))
r_after  = (float(A1_res @ A2_res) /
            (np.linalg.norm(A1_res) * np.linalg.norm(A2_res)))

# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def draw_raster(ax, counts_list, labels, colors, t):
    """
    Multi-region spike raster.  Each region occupies a horizontal band;
    within each bin, spikes are spread evenly as vertical ticks.
    """
    for row, (cnt, col) in enumerate(zip(counts_list, colors)):
        for bi, sc in enumerate(cnt):
            if sc > 0:
                xs = np.linspace(t[bi] - 0.32, t[bi] + 0.32, max(sc, 1))
                ax.vlines(xs,
                          ymin=row + 0.08, ymax=row + 0.52,
                          colors=col, linewidths=1.8)

    ax.set_yticks(np.arange(len(labels)) + 0.30)
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    for lbl, col in zip(ax.get_yticklabels(), colors):
        lbl.set_color(col)
    ax.set_xticks(t)
    ax.set_xticklabels([f'Bin {i}' for i in t], fontsize=9)
    ax.set_xlim(0.5, t[-1] + 0.5)
    ax.set_ylim(-0.05, len(labels))
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)


def draw_rate_strip(axes_list, lam_list, labels, colors, t,
                    source_labels, source_colors):
    """
    Draw one bar-chart per region showing Poisson rate λ per time bin.
    """
    lam_max_global = max(lam.max() for lam in lam_list)

    for ax, lam, lbl, col, src_lbl, src_col in zip(
            axes_list, lam_list, labels, colors,
            source_labels, source_colors):

        bars = ax.bar(t, lam, color=col, alpha=0.75,
                      edgecolor='white', linewidth=0.8, width=0.6, zorder=3)

        # Numeric λ annotation above each bar
        for xi, lv in zip(t, lam):
            ax.text(xi, lv + 0.15, f'{lv:.0f}',
                    ha='center', va='bottom',
                    fontsize=9, color=col, fontweight='bold')

        ax.set_ylim(0, lam_max_global + 1.8)
        ax.set_xticks(t)
        ax.set_xticklabels([f'B{i}' for i in t], fontsize=8)
        ax.set_xlim(0.4, t[-1] + 0.6)
        ax.set_ylabel(r'$\lambda$', fontsize=10, rotation=0, labelpad=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)

        # Region name + driving-source label inside the axes
        ax.text(0.04, 0.94, lbl,
                transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=col, va='top')
        ax.text(0.04, 0.76, src_lbl,
                transform=ax.transAxes,
                fontsize=8.5, color=src_col, va='top',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#ffffff', edgecolor=src_col,
                          linewidth=0.8, alpha=0.9))


def draw_stem(ax, t, vals, color, title='', ylim=(-2.6, 2.6)):
    """Styled stem plot with floating value labels."""
    ml, sl, bl = ax.stem(t, vals)
    plt.setp(sl, color=color,     linewidth=2.5, zorder=3)
    plt.setp(ml, color=color,     markersize=9,  zorder=4,
             markeredgewidth=0.5, markeredgecolor='white')
    plt.setp(bl, color='#555555', linewidth=1.0)

    def _lbl(v):
        if v == 0:       return '0'
        if v == int(v):  return f'{int(v):+d}'
        return f'{v:+.2f}'

    for xi, vi in zip(t, vals):
        pad = 0.25 if vi >= 0 else -0.25
        ax.text(xi, vi + pad, _lbl(vi),
                ha='center', va='bottom' if vi >= 0 else 'top',
                fontsize=9.5, color=color, fontweight='bold')

    ax.set_ylim(ylim)
    ax.set_xticks(t)
    ax.set_xticklabels([f'B{i}' for i in t], fontsize=8)
    ax.axhline(0, color='#555555', linewidth=0.8, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)


def op_cell(ax, sym):
    ax.axis('off')
    # Adjusted font size and vertical alignment for a cleaner look
    ax.text(0.5, 0.45, sym, ha='center', va='center',
            fontsize=30, fontweight='bold', color='#444444',
            transform=ax.transAxes)


def sec_label(ax, txt):
    ax.text(-0.02, 1.30, txt,
            transform=ax.transAxes, clip_on=False,
            fontsize=11, fontweight='bold', color='#222222', va='bottom')


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIGURE  —  outer grid
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.5, 28))
fig.patch.set_facecolor('white')

# Increased hspace for better breathing room between major sections
outer = gridspec.GridSpec(
    6, 1, figure=fig,
    hspace=0.65,
    top=0.955, bottom=0.030, left=0.08, right=0.96,
    height_ratios=[0.52, 1.85, 0.88, 1.16, 1.16, 1.10],
)

# ── Panel 0: header card ─────────────────────────────────────────────────────
ax0 = fig.add_subplot(outer[0])
ax0.axis('off')
card = (
    r"Causal structure:  $s\_ZG \rightarrow Z,\ A_1,\ B$"
    r"     $s\_A2B \rightarrow A_2,\ B$" + "\n\n"
    r"$B$ is a collider.  $s\_ZG \perp s\_A2B$  "
    r"$\Rightarrow$  $\mathrm{corr}(A_1,A_2)=0$ before conditioning." + "\n"
    r"After conditioning on $B$:  $\mathrm{corr}(A_{1,res},A_{2,res})=-1$."
)
# Increased padding and softer border for the header card
ax0.text(0.5, 0.50, card,
         ha='center', va='center', transform=ax0.transAxes, fontsize=12, linespacing=1.4,
         bbox=dict(boxstyle='round,pad=0.8',
                   facecolor='#f4f7ff', edgecolor='#baccff', linewidth=1.0))

# ── Panel 1: raster (top) + rate strip (bottom) ──────────────────────────────
gs1 = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[1],
    hspace=0.45,
    height_ratios=[2, 1],
)

# ── 1a  Raster ────────────────────────────────────────────────────────────────
ax_rast = fig.add_subplot(gs1[0])
draw_raster(
    ax_rast,
    counts_list=[Z_ct, A1_ct, A2_ct, B_ct],
    labels=['Z', 'A1', 'A2', 'B'],
    colors=[C['Z'], C['A1'], C['A2'], C['B']],
    t=t,
)
ax_rast.set_title(
    r'1a — Idealized spike raster (Count = $\lambda$)',
    fontsize=11, pad=10)

ylim_lo, ylim_hi = ax_rast.get_ylim()
ax_rast.text(1.02, 1.04, 'Bin:   1    2    3    4',
             transform=ax_rast.transAxes, fontsize=9,
             color='#666666', clip_on=False, va='bottom',
             fontfamily='monospace')
for row, (cnt, lbl, col) in enumerate(
        zip([Z_ct, A1_ct, A2_ct, B_ct],
            ['Z ', 'A1', 'A2', 'B '],
            [C['Z'], C['A1'], C['A2'], C['B']])):
    yf = (row + 0.30 - ylim_lo) / (ylim_hi - ylim_lo)
    ax_rast.text(1.02, yf,
                 f'{lbl}:  ' + '   '.join(f'{c:3d}' for c in cnt),
                 transform=ax_rast.transAxes, fontsize=9,
                 color=col, clip_on=False, va='center',
                 fontfamily='monospace')

# ── 1b  Rate strip ────────────────────────────────────────────────────────────
gs1b = gridspec.GridSpecFromSubplotSpec(
    1, 4, subplot_spec=gs1[1], wspace=0.45)

ax_rZ  = fig.add_subplot(gs1b[0])
ax_rA1 = fig.add_subplot(gs1b[1])
ax_rA2 = fig.add_subplot(gs1b[2])
ax_rB  = fig.add_subplot(gs1b[3])

draw_rate_strip(
    axes_list    = [ax_rZ,   ax_rA1,  ax_rA2,  ax_rB],
    lam_list     = [lam_ZG,  lam_ZG,  lam_A2B, lam_B],
    labels       = ['Z',     'A1',    'A2',    'B'],
    colors       = [C['Z'],  C['A1'], C['A2'], C['B']],
    t            = t,
    source_labels= [r'$\propto s\_ZG$',
                    r'$\propto s\_ZG$',
                    r'$\propto s\_A2B$',
                    r'$= \lambda_{sZG}+\lambda_{sA2B}$'],
    source_colors= [C['sZG'], C['sZG'], C['sA2B'], C['B']],
)

ax_rZ.set_title('1b — Poisson rate  ' + r'$\lambda$' + '  per bin',
                fontsize=11, fontweight='bold', pad=10)

# Softer bridge annotation box
ax_rZ.text(-0.25, -0.25,
           r'$\lambda = 2.5 + 1.5 \times \mathrm{source\ signal}$'
           '\n'
           r'Rate pattern below $\Longleftrightarrow$ source-signal stem in Panel 2',
           transform=ax_rZ.transAxes,
           fontsize=9.5, color='#444444', va='top',
           style='italic',
           bbox=dict(boxstyle='round,pad=0.6',
                     facecolor='#f9f9f2', edgecolor='#d1d1b4', linewidth=1.0))

# ── Panel 2: source stems + collider ─────────────────────────────────────────
gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.55)
ax_sZG  = fig.add_subplot(gs2[0])
ax_sA2B = fig.add_subplot(gs2[1])
ax_Bsum = fig.add_subplot(gs2[2])

draw_stem(ax_sZG,  t, s_ZG,  C['sZG'],
          title=r'Latent source  $s\_ZG$',  ylim=(-1.8, 1.8))
draw_stem(ax_sA2B, t, s_A2B, C['sA2B'],
          title=r'Latent source  $s\_A2B$', ylim=(-1.8, 1.8))
draw_stem(ax_Bsum, t, B,     C['B'],
          title=r'Collider  $B = s\_ZG + s\_A2B$', ylim=(-2.8, 2.8))

sec_label(ax_sZG, '── STEP 1   Independent sources and their collider sum')
ax_sZG.text(1.30, -0.10, r'$s\_ZG^{T} s\_A2B = 0$  (orthogonal)',
            ha='center', va='top', fontsize=12, color='#666666',
            transform=ax_sZG.transAxes)
ax_Bsum.text(0.50, -0.30, r'$B^T B = 8$',
             ha='center', va='top', fontsize=10, color=C['B'],
             transform=ax_Bsum.transAxes)

ax_sZG.text(0.50, 1.20,
            'Z & A1 rate: [4,4,1,1] matches',
            ha='center', va='bottom', fontsize=8.5, color=C['sZG'],
            transform=ax_sZG.transAxes, clip_on=False, style='italic')
ax_sA2B.text(0.50, 1.20,
             'A2 rate: [4,1,4,1] matches',
             ha='center', va='bottom', fontsize=8.5, color=C['sA2B'],
             transform=ax_sA2B.transAxes, clip_on=False, style='italic')
ax_Bsum.text(0.50, 1.20,
             r'B rate: [8,5,5,2] $\propto$ same ordering',
             ha='center', va='bottom', fontsize=8.5, color=C['B'],
             transform=ax_Bsum.transAxes, clip_on=False, style='italic')

# ── Panel 3: A1 residualization  (A1 − β₁·B = A1_res) ───────────────────────
# Adjusted width_ratios to give the rightmost text box slightly more room
gs3 = gridspec.GridSpecFromSubplotSpec(
    1, 7, subplot_spec=outer[3],
    width_ratios=[3.0, 0.45, 3.0, 0.45, 3.0, 0.35, 0.20], wspace=0.10)
ax3 = [fig.add_subplot(gs3[i]) for i in range(7)]

draw_stem(ax3[0], t, A1,    C['A1'],  title=r'$A_1$')
draw_stem(ax3[2], t, proj1, C['proj'],title=r'$\hat{A}_1 = \beta_1 B$')
draw_stem(ax3[4], t, A1_res,C['r1'],  title=r'$A_{1,res}$')
op_cell(ax3[1], '−')
op_cell(ax3[3], '=')
ax3[5].axis('off')

ax3[2].text(0.50, -0.30,
            r'$\beta_1 = \frac{A_1^T B}{B^T B} = \frac{4}{8} = 0.5$',
            ha='center', va='top', fontsize=10, color=C['proj'],
            transform=ax3[2].transAxes)

ax3[6].axis('off')
for yf, txt, col in [(0.76, r'$A_1^T B = 4$', C['A1']),
                     (0.48, r'$B^T B = 8$',    C['B']),
                     (0.18, r'$\beta_1 = 0.5$', C['proj'])]:
    ax3[6].text(0.10, yf, txt, ha='center', va='center', fontsize=11,
                color=col, fontweight='bold' if yf == 0.18 else 'normal',
                transform=ax3[6].transAxes)
ax3[6].set_title('Projection\ncoefficient', fontsize=10, pad=12)



sec_label(ax3[0], '── STEP 2a   Regress B out of A1')

# ── Panel 4: A2 residualization  (A2 − β₂·B = A2_res) ───────────────────────
gs4 = gridspec.GridSpecFromSubplotSpec(
    1, 7, subplot_spec=outer[4],
    width_ratios=[3.0, 0.45, 3.0, 0.45, 3.0, 0.35, 0.20], wspace=0.10)
ax4 = [fig.add_subplot(gs4[i]) for i in range(7)]

draw_stem(ax4[0], t, A2,    C['A2'],  title=r'$A_2$')
draw_stem(ax4[2], t, proj2, C['proj'],title=r'$\hat{A}_2 = \beta_2 B$')
draw_stem(ax4[4], t, A2_res,C['r2'],  title=r'$A_{2,res}$')
op_cell(ax4[1], '−')
op_cell(ax4[3], '=')
ax4[5].axis('off')

ax4[2].text(0.50, -0.30,
            r'$\beta_2 = \frac{A_2^T B}{B^T B} = \frac{4}{8} = 0.5$',
            ha='center', va='top', fontsize=10, color=C['proj'],
            transform=ax4[2].transAxes)

ax4[6].axis('off')
for yf, txt, col in [(0.76, r'$A_2^T B = 4$', C['A2']),
                     (0.48, r'$B^T B = 8$',    C['B']),
                     (0.18, r'$\beta_2 = 0.5$', C['proj'])]:
    ax4[6].text(0.10, yf, txt, ha='center', va='center', fontsize=11,
                color=col, fontweight='bold' if yf == 0.18 else 'normal',
                transform=ax4[6].transAxes)
ax4[6].set_title('Projection\ncoefficient', fontsize=10, pad=12)
sec_label(ax4[0], '── STEP 2b   Regress B out of A2')

# ── Panel 5: outcome ──────────────────────────────────────────────────────────
gs5 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[5], wspace=0.55)
ax5_ov  = fig.add_subplot(gs5[0])
ax5_bar = fig.add_subplot(gs5[1])
ax5_exp = fig.add_subplot(gs5[2])

# Residual overlay
draw_stem(ax5_ov, t, A1_res, C['r1'],
          title=r'$A_{1,res}$ and $A_{2,res}$ overlaid', ylim=(-1.7, 1.7))
ml2, sl2, _ = ax5_ov.stem(t + 0.14, A2_res)
plt.setp(sl2, color=C['r2'], linewidth=2.5, alpha=0.85, zorder=3)
plt.setp(ml2, color=C['r2'], markersize=9,  alpha=0.85, zorder=4,
         markeredgewidth=0.5, markeredgecolor='white')
ax5_ov.legend(handles=[
    plt.Line2D([0],[0], color=C['r1'], lw=2.5, marker='o', markersize=7,
               label=r'$A_{1,res} = [0,+1,-1,0]$'),
    plt.Line2D([0],[0], color=C['r2'], lw=2.5, marker='o', markersize=7,
               label=r'$A_{2,res} = [0,-1,+1,0]$'),
], fontsize=9, loc='upper right', framealpha=0.9, edgecolor='#dddddd')

# Correlation bar chart
cats  = [r'Before:  corr$(A_1, A_2)$',
         r'After:  corr$(A_{1,res}, A_{2,res})$']
bars  = ax5_bar.bar(cats, [r_before, r_after],
                    color=['#bccadd', '#e66a6a'],
                    edgecolor='#444444', linewidth=1.0, width=0.5, zorder=3)
for bar, v in zip(bars, [r_before, r_after]):
    y   = v + (0.05 if v >= 0 else -0.09)
    ax5_bar.text(bar.get_x() + bar.get_width()/2, y,
                 f'{v:.2f}', ha='center',
                 va='bottom' if v >= 0 else 'top',
                 fontsize=15, fontweight='bold', color='#222222')
ax5_bar.axhline(0, color='#333333', linewidth=1.0, zorder=2)
ax5_bar.set_ylim(-1.38, 0.55)
ax5_bar.set_ylabel('Pearson  r', fontsize=11, labelpad=8)
ax5_bar.set_title('Collider-bias outcome',
                  fontsize=11, fontweight='bold', pad=10)
ax5_bar.spines[['top', 'right']].set_visible(False)
ax5_bar.tick_params(axis='x', labelsize=9.5)

# Algebra explanation box - refined typography and padding
ax5_exp.axis('off')
exp_lines = [
    r"Why $A_{2,res} = -A_{1,res}$?",
    "",
    r"$A_{1,res} = \frac{1}{2}s\_ZG - \frac{1}{2}s\_A2B$",
    "",
    r"$A_{2,res} = -\frac{1}{2}s\_ZG + \frac{1}{2}s\_A2B$",
    "",
    r"$\Rightarrow A_{2,res} = -A_{1,res}$",
    "",
    "Conditioning fixes",
    r"$B = s\_ZG + s\_A2B$, creating",
    "a seesaw: any rise in one",
    r"source forces a fall in the other.",
    r"$\Rightarrow$ perfect anti-correlation.",
]
ax5_exp.text(0.02, 0.98, "\n".join(exp_lines),
             ha='left', va='top', fontsize=10,
             transform=ax5_exp.transAxes, linespacing=1.65,
             bbox=dict(boxstyle='round,pad=0.8',
                       facecolor='#fffaf2', edgecolor='#dcb578', linewidth=1.2))

sec_label(ax5_ov, '── STEP 3   Consequence: perfect spurious anti-correlation')

# ── Super-title  +  save ──────────────────────────────────────────────────────
fig.suptitle(
    'Collider Bias — Conditioning on B drives\n'
    r'$\mathrm{corr}(A_1, A_2):\ 0\ \longrightarrow\ -1$'
    "  (Berkson's paradox / pCCA failure mode)",
    fontsize=15, fontweight='bold', y=0.985)

plt.savefig('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/collider_bias_demo_v2.png',
            dpi=150, bbox_inches='tight', facecolor='white')
#plt.show()
print("Done.")
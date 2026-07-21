

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import truncnorm
# ══════════════════════════════════════════════════════════════════════════════
# 1.  PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(0)

T         = 10     # number of trials
tau       = 30       # time bins per trial
alpha_A   = 0.85     # region A coupling strength to shared state z(t)
alpha_B   = -0.85     # region B coupling strength to shared state z(t)
sigma_z   = 1.0      # std of internal state z(t)
sigma_eps = 0.40     # std of private noise ε(t,s) — independent per region

# Colour palette (matches the pipeline convention)
CMAP    = 'RdBu_r'   # diverging: warm = high z, cool = low z
C_B     = '#2166ac'  # blue  — region A lines
C_A     = '#d6604d'  # red   — region B lines
C_STIM  = '#4393c3'  # steel — Σ^stim  bar segment
C_NOISE = '#d73027'  # warm  — Σ^noise bar segment

# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# s    = np.arange(tau)
# mu_A = 2.0 * np.sin(np.pi * s / tau)
# mu_B = 2.0 * np.sin(np.pi * s / tau )   # half-sine bell
# #mu_B = 2.0 * np.sin(np.pi * s / tau + np.pi / 5)   # phase-shifted variant
#
#
# a1, b1 =  0, np.inf,
#
# # 采样 size=T
# z = truncnorm.rvs(a1, b1, loc=0, scale=sigma_z, size=T, random_state=rng)
#
#
# z_raw     = np.sort(z)
# z1 = z_raw
# z2 = z_raw  # shared internal state
# z_all = np.concatenate([z2,z1])
# eps_A = rng.normal(0.0, sigma_eps, size=(T, tau))    # private noise, region A
# eps_B = rng.normal(0.0, sigma_eps, size=(T, tau))    # private noise, region B
#
#
# vmin_z, vmax_z = z2.min(), z1.max()
# z_01     = (z_raw - vmin_z) / (vmax_z - vmin_z)   # z normalised to [0,1] for colours
#
# # Full spike trains x(t, s) — shape (T, tau)
# x_A = mu_A[None, :] + alpha_A * z1[:, None] + eps_A
# x_B = mu_B[None, :] + alpha_B * z2[:, None] + eps_B
#
# # ── PSTH subtraction (sample mean across trials) ──────────────────────────
# # In practice we subtract the sample PSTH, not the true μ.
# # The zero-sum constraint Σ_t δ(t,s) = 0 introduces a finite-T bias
# # in the shuffled cross-covariance of exactly  -Σ_AB^noise / (T-1).
# mu_A_hat = x_A.mean(axis=0)             # (tau,)
# mu_B_hat = x_B.mean(axis=0)             # (tau,)
# delta_A  = x_A - mu_A_hat[None, :]     # (T, tau)  noise residuals of A
# delta_B  = x_B - mu_B_hat[None, :]     # (T, tau)  noise residuals of B
#
# # ── Trial-level shuffle of region B (A stays intact) ─────────────────────
# # This mimics cross-session recording: the PSTH component is preserved but
# # the trial-by-trial pairing between A and B is destroyed.
# perm         = rng.permutation(T)
# delta_B_shuf = delta_B[perm]           # (T, tau)
#
# # ── Flatten for scatter plots: each point = one (trial, time-bin) pair ────
# z_flat            = np.repeat(z_raw, tau)  # (T·tau,)  — colour coding
# x_A_flat          = x_A.ravel()
# x_B_flat          = x_B.ravel()
# delta_A_flat      = delta_A.ravel()
# delta_B_flat      = delta_B.ravel()
# delta_B_shuf_flat = delta_B_shuf.ravel()


s    = np.arange(tau)
mu_A = 2.0 * np.sin(np.pi * s / tau)               # half-sine bell
# mu_B = 1.5 * np.sin(np.pi * s / tau + np.pi / 5)   # phase-shifted variant
mu_B = 2.0 * np.sin(np.pi * s / tau)

z     = np.sort(rng.normal(0.0, sigma_z,  size=T))         # shared internal state
eps_A = rng.normal(0.0, sigma_eps, size=(T, tau))    # private noise, region A
eps_B = rng.normal(0.0, sigma_eps, size=(T, tau))    # private noise, region B


vmin_z, vmax_z = z.min(), z.max()
z_01     = (z - vmin_z) / (vmax_z - vmin_z)
z_02     = 1-z_01
# Full spike trains x(t, s) — shape (T, tau)
x_A = mu_A[None, :] + alpha_A * z[:, None] + eps_A
x_B = mu_B[None, :] + alpha_B * z[:, None] + eps_B

# ── PSTH subtraction (sample mean across trials) ──────────────────────────
# In practice we subtract the sample PSTH, not the true μ.
# The zero-sum constraint Σ_t δ(t,s) = 0 introduces a finite-T bias
# in the shuffled cross-covariance of exactly  -Σ_AB^noise / (T-1).
mu_A_hat = x_A.mean(axis=0)             # (tau,)
mu_B_hat = x_B.mean(axis=0)             # (tau,)
delta_A  = x_A - mu_A_hat[None, :]     # (T, tau)  noise residuals of A
delta_B  = x_B - mu_B_hat[None, :]     # (T, tau)  noise residuals of B

# ── Trial-level shuffle of region B (A stays intact) ─────────────────────
# This mimics cross-session recording: the PSTH component is preserved but
# the trial-by-trial pairing between A and B is destroyed.
perm         = rng.permutation(T)
delta_B_shuf = delta_B[perm]           # (T, tau)

# ── Flatten for scatter plots: each point = one (trial, time-bin) pair ────
z_flat            = np.repeat(z, tau)  # (T·tau,)  — colour coding
x_A_flat          = x_A.ravel()

x_B_flat          = x_B.ravel()
x_B_flat_shuf     = x_B[perm].ravel()

delta_A_flat      = delta_A.ravel()
delta_B_flat      = delta_B.ravel()
delta_B_shuf_flat = delta_B_shuf.ravel()




# ══════════════════════════════════════════════════════════════════════════════
# 3.  STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

# # Theoretical population values (known from generative model)
# Sig_stim_theory  = float(np.mean(mu_A * mu_B))       # Σ_AB^stim  = (1/τ)Σ_s μ_A μ_B
# Sig_noise_theory = alpha_A * alpha_B * sigma_z ** 2  # Σ_AB^noise = α_A α_B σ_z²
#
# # Sample estimates
# Sig_stim_hat  = float(np.mean(mu_A_hat * mu_B_hat))  # (1/τ) Σ_s μ̂_A μ̂_B
# Sig_noise_hat = float(np.mean(delta_A * delta_B))    # Σ̂_AB^noise  (unshuffled)
# Sig_shuf_hat  = float(np.mean(delta_A * delta_B_shuf))  # after shuffle
#
# # Exact finite-T derangement expectation (from previous derivation):
# #   E_π[Σ̂_AB^π] = -Σ̂_AB^noise / (T-1)
# exact_shuf_bias = -Sig_noise_hat / (T - 1)
#
# # Pearson r  ≡  CCA canonical correlation for p_A = p_B = 1
# r_full  = float(np.corrcoef(x_A_flat,     x_B_flat)[0, 1])
# r_noise = float(np.corrcoef(delta_A_flat, delta_B_flat)[0, 1])
# r_shuf  = float(np.corrcoef(delta_A_flat, delta_B_shuf_flat)[0, 1])
# ══════════════════════════════════════════════════════════════════════════════
# 3.  STATISTICS  —  bias-corrected decomposition
# ══════════════════════════════════════════════════════════════════════════════

# ── Theoretical population values (from generative model) ────────────────
Sig_stim_theory  = float(np.mean(mu_A * mu_B))
Sig_noise_theory = alpha_A * alpha_B * sigma_z ** 2

# ── Raw (biased) sample estimates ─────────────────────────────────────────
# E[Sig_stim_raw]  = Σ_stim  + α_A α_B σ_z² / T     (biased low when α_Aα_B < 0)
# E[Sig_noise_raw] = Σ_noise × (T-1)/T               (biased toward zero by 1/T)
_Sig_stim_raw  = float(np.mean(mu_A_hat * mu_B_hat))
_Sig_noise_raw = float(np.mean(delta_A * delta_B))

# ── Unbiased estimates (Bessel-style correction) ──────────────────────────
# The noise bias absorbed into Sig_stim_raw equals Sig_noise_raw / (T-1):
#   E[Sig_stim_raw - Sig_noise_raw/(T-1)]
#     = (Σ_stim + Σ_noise/T) - Σ_noise(T-1)/T / (T-1)
#     = Σ_stim + Σ_noise/T  - Σ_noise/T  = Σ_stim  ✓
#
# The Bessel-corrected noise estimate:
#   E[T/(T-1) × Sig_noise_raw] = T/(T-1) × Σ_noise(T-1)/T = Σ_noise  ✓
Sig_noise_hat = _Sig_noise_raw * T / (T - 1)            # Bessel correction
Sig_stim_hat  = _Sig_stim_raw  - _Sig_noise_raw / (T - 1)  # de-absorb noise bias

# Sanity check: corrected total must equal raw total (total is always unbiased)
assert abs((Sig_stim_hat + Sig_noise_hat)
           - (_Sig_stim_raw + _Sig_noise_raw)) < 1e-10, "total not conserved"

# ── Shuffled cross-covariance ─────────────────────────────────────────────
# The derangement-expectation formula uses the RAW noise (denominator T):
#   E_π[Σ̂_AB^π] = Σ̂_stim - Σ̂_noise_raw / (T-1)
# Equivalently, using the Bessel-corrected noise (denominator T-1):
#   E_π[Σ̂_AB^π] = Σ̂_stim - Σ̂_noise_unbiased / T
Sig_shuf_hat    = float(np.mean(delta_A * delta_B_shuf))
exact_shuf_bias = -_Sig_noise_raw / (T - 1)   # ← always use RAW noise here

# ── Pearson r ─────────────────────────────────────────────────────────────
r_full  = float(np.corrcoef(x_A_flat,     x_B_flat)[0, 1])
r_shuf_signal  = float(np.corrcoef(x_A_flat, x_B_flat_shuf)[0, 1])
r_noise = float(np.corrcoef(delta_A_flat, delta_B_flat)[0, 1])
r_shuf_noise  = float(np.corrcoef(delta_A_flat, delta_B_shuf_flat)[0, 1])
# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    2, 4, figure=fig,
    hspace=0.52, wspace=0.70,
    left=0.06, right=0.97,
    top=0.92, bottom=0.08,
)
axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

cmap_obj = plt.get_cmap(CMAP)


# Shared ScalarMappable for panels without a PathCollection colourbar
sm1 = plt.cm.ScalarMappable(
    cmap=CMAP,
    norm=plt.Normalize(vmin=vmax_z, vmax=vmin_z)   # inverted: large z → blue
)

sm2 = plt.cm.ScalarMappable(
    cmap=CMAP,
    norm=plt.Normalize(vmin=vmax_z, vmax=vmin_z)   # inverted: large z → blue
)

sm2.set_array([])

# ── Small helper functions ─────────────────────────────────────────────────
def _cbar(mappable, ax, label):
    cb = plt.colorbar(mappable, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb

def _ols_line(ax, xv, yv, color='k', lw=1.6, alpha=0.85):
    m, b = np.polyfit(xv, yv, 1)
    xl = np.array([xv.min(), xv.max()])
    ax.plot(xl, m * xl + b, color=color, lw=lw, alpha=alpha)

def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
# (0,0)  PSTH + example single-trial traces coloured by z(t)
#
# Visual purpose: show that individual trials deviate from the PSTH in
# amplitude, and that the deviation magnitude is controlled by z(t).
# Warm colours (high z) → larger-amplitude trials; cool → smaller.
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0][0]

n_show   = 5
start = int(T / 2)
end = T

# rng 是你之前定义的 np.random.default_rng()
# replace=False 确保选出的 20 个 trial 不重复
idx_show = rng.choice(np.arange(start, end), size=n_show, replace=False)
for i in idx_show:
    ax.plot(s, x_A[i], color=cmap_obj(z_01[i]), alpha=0.38, lw=0.9)
    #ax.plot(s, x_B[i], color=cmap_obj(z_01[i]), alpha=0.38, lw=0.9)

ax.plot(s, mu_A_hat, color=C_A, lw=2.5, label=r'$\hat{\mu}_A(s)$  (A PSTH)')
#ax.plot(s, mu_B_hat, color=C_B, lw=2.5, ls='--', label=r'$\hat{\mu}_B(s)$  (B PSTH)')

_cbar(sm1, ax, '$z(t)$  internal state of A')
ax.legend(fontsize=8, frameon=False)
ax.set_xlabel('Time bin $s$', fontsize=10)
ax.set_ylabel('Firing rate', fontsize=10)
ax.set_title(
    'PSTH + single-trial traces  (region A)\n'
    r'Warm $z(t) \Rightarrow$ amplitude $\uparrow$',
    fontsize=10,
)
_clean(ax)

ax = axes[0][1]

for i in idx_show:
    #ax.plot(s, x_A[i], color=cmap_obj(z_01[i]), alpha=0.38, lw=0.9)
    ax.plot(s, x_B[i], color=cmap_obj(z_02[i]), alpha=0.38, lw=0.9)

#ax.plot(s, mu_A_hat, color=C_A, lw=2.5, label=r'$\hat{\mu}_A(s)$  (A PSTH)')
ax.plot(s, mu_B_hat, color=C_B, lw=2.5, label=r'$\hat{\mu}_B(s)$  (B PSTH)')

#_cbar(sm2, ax, '$z(t)$  internal state of B')
ax.legend(fontsize=8, frameon=False)
ax.set_xlabel('Time bin $s$', fontsize=10)
ax.set_ylabel('Firing rate', fontsize=10)
ax.set_title(
    'PSTH + single-trial traces  (region B)\n'
    r'cool $z(t) \Rightarrow$ amplitude $\downarrow$',
    fontsize=10,
)
_clean(ax)


# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0][2]

bar_x      = np.array([0, 1, 2])
# ── bar chart ─────────────────────────────────────────────────────────────
bar_labels = ['Theory\n(pop.)', 'Sample\n(unbiased)', 'Sample\n(shuf.)']
stim_v     = [Sig_stim_theory,  Sig_stim_hat,   Sig_stim_hat]
noise_v    = [np.abs(Sig_noise_theory), np.abs(Sig_noise_hat) ,Sig_shuf_hat]

ax.bar(bar_x, stim_v,
       color=C_STIM,  alpha=0.85, label=r'$\hat{\Sigma}_{AB}^{\rm stim}$')
ax.bar(bar_x, noise_v, bottom=stim_v,
       color=C_NOISE, alpha=0.85, label=r'$\hat{\Sigma}_{AB}^{\rm noise}$')

# Reference lines
ax.axhline(Sig_stim_theory + np.abs(Sig_noise_theory),
           color='k',    ls=':',  lw=1.3, alpha=0.65,
           label=r'$\Sigma_{AB}^{\rm total}$ (theory)')
ax.axhline(Sig_stim_hat + exact_shuf_bias,
           color='r', ls='--', lw=1.2, alpha=0.80,
           label=r'$\hat{\Sigma}^{\rm stim} - \hat{\Sigma}^{\rm noise}/(T{-}1)$')

ax.set_xticks(bar_x)
ax.set_xticklabels(bar_labels, fontsize=9)
ax.set_ylabel('Cross-covariance', fontsize=10)
ax.set_title(
    r'$\hat{\Sigma}_{AB} = \hat{\Sigma}^{\rm stim} + \hat{\Sigma}^{\rm noise}$'
    '\nShuffle kills $\\hat{\\Sigma}^{\\rm noise}$, '
    'leaves $\\hat{\\Sigma}^{\\rm stim}$ intact',
    fontsize=10,
)
ax.set_ylim(bottom=min(0.0, Sig_stim_hat + Sig_shuf_hat) * 1.3)
ax.legend(fontsize=7.5, frameon=False, loc='lower right')
_clean(ax)

# ─────────────────────────────────────────────────────────────────────────────
# (1,2)  Summary formulae and numerical results
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0][3]
ax.axis('off')

# (text, bold) pairs; empty string = vertical gap
lines = [
    ('Generative model  ' + r'($p_A = p_B = 1$)', True),
    (r'$x_A = \mu_A(s) + \alpha_A z(t) + \varepsilon_A$', False),
    (r'$x_B = \mu_B(s) + \alpha_B z(t) + \varepsilon_B$', False),
    ('', False),
    ('Exact decomposition', True),
    (r'$\Sigma_{AB} = \Sigma_{AB}^{\rm stim} + \Sigma_{AB}^{\rm noise}$', False),
    ('', False),
    (r'$\hat{\Sigma}^{\rm stim} = \overline{\hat\mu_A\hat\mu_B}'
     r'- \hat{\Sigma}^{\rm noise}_{\rm raw}/(T{-}1)$', False),
    (f'$\\quad = {Sig_stim_hat:.4f}$  (raw: {_Sig_stim_raw:.4f})', False),
    ('', False),
    (r'$\hat{\Sigma}^{\rm noise} = \frac{T}{T-1}\,\overline{\delta_A\delta_B}$', False),
    (f'$\\quad = {Sig_noise_hat:.4f}$  (raw: {_Sig_noise_raw:.4f})', False),
    ('Finite-T bias under derangements (exact)', True),
    (r'$\mathbb{E}_\pi[\hat{\Sigma}_{AB}^\pi]$'
     r'$= \hat{\Sigma}^{\rm stim} - \hat{\Sigma}^{\rm noise}/(T-1)$', False),
    ('  theory: '
     + f'{Sig_stim_hat + exact_shuf_bias:.4f}'
     + '   sample: '
     + f'{Sig_stim_hat + Sig_shuf_hat:.4f}', False),
    ('', False),
    ('Pearson r  (= CCA canonical corr.)', True),
    (f'  Full signal, unshuf.:       $r = {r_full:.3f}$', False),
    (f'  PSTH-sub,   unshuf.:        $r = {r_noise:.3f}$', False),
    (f'  PSTH-sub,   shuf.:          $r = {r_shuf_noise:.3f}$', False),
    (f'  PSTH-sub,   shuf.signal:          $r = {r_shuf_signal:.3f}$', False),
]

y0      = 0.985
dy_text = 0.055
dy_gap  = 0.022
for text, bold in lines:
    if text == '':
        y0 -= dy_gap
        continue
    ax.text(
        0.03, y0, text,
        transform=ax.transAxes,
        fontsize=8.5, va='top', ha='left',
        fontweight='bold' if bold else 'normal',
    )
    y0 -= dy_text


# ─────────────────────────────────────────────────────────────────────────────
# (1,0)  Scatter δ_A vs δ_B — PSTH-subtracted, UNSHUFFLED   ← KEY PANEL
#
# After removing the PSTH, the cross-covariance is purely Σ^noise.
# The colour coding reveals WHY: high-z trials (warm) sit in the
# top-right quadrant (δ_A > 0, δ_B > 0) and low-z trials (cool) in
# the bottom-left.  The shared internal state z(t) is the mediator:
#   δ_A(t,s) = α_A z(t) + ε_A(t,s)
#   δ_B(t,s) = α_B z(t) + ε_B(t,s)
# Σ^noise = α_A α_B σ_z² — recoverable only from simultaneous recording.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1][0]

sc = ax.scatter(x_A_flat, x_B_flat,
                c=z_flat, cmap=CMAP, vmin=vmin_z, vmax=vmax_z,
                s=4, alpha=0.35, linewidths=0)
_ols_line(ax, x_A_flat, x_B_flat)

_cbar(sc, ax, '$z(t)$')

ax.set_xlabel('$x_A(t,s)$', fontsize=10)
ax.set_ylabel('$x_B(t,s)$', fontsize=10)
ax.set_title(
    'Full signal  —  unshuffled\n'
    r'$r = $' + f'{r_full:.3f}'
    r'   $[\hat{\Sigma}^{\rm stim} + \hat{\Sigma}^{\rm noise}]$',
    fontsize=10,
)
_clean(ax)


ax = axes[1][1]

sc2 = ax.scatter(delta_A_flat, delta_B_flat,
                 c=z_flat, cmap=CMAP, vmin=vmin_z, vmax=vmax_z,
                 s=4, alpha=0.45, linewidths=0)
_ols_line(ax, delta_A_flat, delta_B_flat)
ax.axvline(0, color='grey', lw=0.7, alpha=0.45)
ax.axhline(0, color='grey', lw=0.7, alpha=0.45)
_cbar(sc2, ax, '$z(t)$')

# Quadrant annotations: the colour-to-quadrant alignment IS the proof
ax.text(0.74, 0.92,
        'high $z(t)$\n'
        r'$\delta_A \uparrow,\;\delta_B \uparrow$',
        transform=ax.transAxes, fontsize=8, ha='center', va='top',
        color=cmap_obj(0.92),
        bbox=dict(fc='white', alpha=0.82, ec='none', pad=2))
ax.text(0.24, 0.08,
        'low $z(t)$\n'
        r'$\delta_A \downarrow,\;\delta_B \downarrow$',
        transform=ax.transAxes, fontsize=8, ha='center', va='bottom',
        color=cmap_obj(0.08),
        bbox=dict(fc='white', alpha=0.82, ec='none', pad=2))

ax.set_xlabel(r'$\delta_A(t,s)$', fontsize=10)
ax.set_ylabel(r'$\delta_B(t,s)$', fontsize=10)
ax.set_title(
    'Noise residuals  —  unshuffled\n'
    r'$r = $' + f'{r_noise:.3f}'
    r'[only $\hat{\Sigma}_{AB}^{\rm noise}$]',
    fontsize=10,
)
_clean(ax)

# ─────────────────────────────────────────────────────────────────────────────
# (1,1)  Scatter δ_A vs δ_B^π — after shuffling B's trial blocks
#
# The z(t) colour gradient is still visible on the x-axis (region A is
# unchanged), but the y-axis now reflects a permuted trial's z — completely
# unrelated.  The OLS slope collapses and r → -Σ^noise/(T-1) ≈ 0.
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1][2]

sc3 = ax.scatter(delta_A_flat, delta_B_shuf_flat,
                 c=z_flat, cmap=CMAP, vmin=vmin_z, vmax=vmax_z,
                 s=4, alpha=0.35, linewidths=0)
_ols_line(ax, delta_A_flat, delta_B_shuf_flat)
ax.axvline(0, color='grey', lw=0.7, alpha=0.45)
ax.axhline(0, color='grey', lw=0.7, alpha=0.45)
_cbar(sc3, ax, r'$z_A(t)$  [A only; B permuted]')

# ax.text(0.50, 0.06,
#         r'$z(t)$ gradient on $x$-axis no longer aligns' '\n'
#         r'with $y$-axis  $\Rightarrow$  coupling destroyed',
#         transform=ax.transAxes, fontsize=8, ha='center', va='bottom',
#         color='#444444',
#         bbox=dict(fc='white', alpha=0.85, ec='none', pad=2))

ax.set_xlabel(r'$\delta_A(t,s)$', fontsize=10)
ax.set_ylabel(r'$\delta_B(\pi(t),\,s)$', fontsize=10)
ax.set_title(
    r'Noise residuals  —  shuffled  $(\pi(t) \neq t)$' '\n'
    r'$r = $' + f'{r_shuf_noise:.3f}'
    r'   $[\hat{\Sigma}^{\rm noise} \to -\hat{\Sigma}^{\rm noise}/(T{-}1)]$',
    fontsize=10,
)
_clean(ax)


# ─────────────────────────────────────────────────────────────────────────────
# (1,1)  Scatter δ_A vs δ_B^π — after shuffling B's trial blocks
#
# The z(t) colour gradient is still visible on the x-axis (region A is
# unchanged), but the y-axis now reflects a permuted trial's z — completely
# unrelated.  The OLS slope collapses and r → -Σ^noise/(T-1) ≈ 0.
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1][3]

sc3 = ax.scatter(x_A_flat, x_B_flat_shuf,
                 c=z_flat, cmap=CMAP, vmin=vmin_z, vmax=vmax_z,
                 s=4, alpha=0.35, linewidths=0)
_ols_line(ax, x_A_flat, x_B_flat_shuf)
ax.axvline(0, color='grey', lw=0.7, alpha=0.45)
ax.axhline(0, color='grey', lw=0.7, alpha=0.45)
_cbar(sc3, ax, r'$z_A(t)$  [A only; B permuted]')

# ax.text(0.50, 0.06,
#         r'$z(t)$ gradient on $x$-axis no longer aligns' '\n'
#         r'with $y$-axis  $\Rightarrow$  coupling destroyed',
#         transform=ax.transAxes, fontsize=8, ha='center', va='bottom',
#         color='#444444',
#         bbox=dict(fc='white', alpha=0.85, ec='none', pad=2))

ax.set_xlabel(r'$X_A(t,s)$', fontsize=10)
ax.set_ylabel(r'$X_B(\pi(t),\,s)$', fontsize=10)
ax.set_title(
    r'Full signall  —  shuffled  $(\pi(t) \neq t)$' '\n'
    r'$r = $' + f'{r_shuf_signal:.3f}'
    r'   $[\hat{\Sigma}^{\rm noise} \to -\hat{\Sigma}^{\rm noise}/(T{-}1)]$',
    fontsize=10,
)
_clean(ax)


# ── Main title ────────────────────────────────────────────────────────────
fig.suptitle(
    r'$T = $' + f'{T}' + r' trials  ·  $\tau = $' + f'{tau}'
    + r' time bins  ·  $\alpha_A = $'+ f'{alpha_A}' + r' $\alpha_B = $' + f'{alpha_B}'
    + r'  ·  $\sigma_z = $' + f'{sigma_z}'
    + r'  ·  $\sigma_\varepsilon = $' + f'{sigma_eps}',
    fontsize=11, fontweight='bold', y=0.996,
)

plt.savefig('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pCCA_simulation/cca_simultaneous_recording_demo.png', dpi=150, bbox_inches='tight')
# plt.show()
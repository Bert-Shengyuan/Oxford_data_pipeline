%% =========================================================
%  Analysis_1_Kinematics.m
%
%  Analysis 1.1  —  Trial-by-trial kinematic variability
%  Analysis 1.2  —  Reach phase φ(t) via Hilbert transform
%  Analysis 1.3  —  Miss-trial kinematics as boundary condition
%
%  Dataset    : tapproach_global_oyku.mat
%  Fs         : 50 Hz  |  t = sample/50 − 1  (seconds)
%  Convention : t = 0 is the kinematically-defined reach onset
% ==========================================================
close all; clc; clear all

%% ── Load data ──────────────────────────────────────────────────────────
load('/Users/shengyuancai/Downloads/Oxford_dataset/tapproach_global_oyku.mat')

%% ── Output directory ───────────────────────────────────────────────────
outDir = '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/Behaviour_phase/Phase_Analysis_1';
if ~exist(outDir, 'dir'); mkdir(outDir); end

%% ── Metadata ────────────────────────────────────────────────────────────
labels     = tapproach.label;
sessions   = tapproach.session_name;
%pos        = {tapproach.pos_x, tapproach.pos_y, tapproach.pos_z};
posLbls    = {'x (mm)', 'y (mm)', 'z (mm)'};
posNames   = {'x', 'y', 'z'};
trialTypes = {'cued hit long', 'spont hit long', 'spont miss long'};
trialNames = {'Cued Hit Long', 'Spont Hit Long', 'Spont Miss Long'};

% Colours: blue / red / green  (consistent with reference code)
rowColours = {[0.18 0.46 0.71], [0.17 0.63 0.17],[0.84 0.19 0.15], };

% ── Strip trailing all-NaN time samples ─────────────────────────────────
%  The final sample of every trial is recorded as NaN in this dataset.
%  Find the last column that contains at least one finite value.
lastValid = find( ~all(isnan(tapproach.pos_x), 1) & ...
                  ~all(isnan(tapproach.pos_y), 1) & ...
                  ~all(isnan(tapproach.pos_z), 1), 1, 'last' );

pos = { tapproach.pos_x(:, 1:lastValid), ...
        tapproach.pos_y(:, 1:lastValid), ...
        tapproach.pos_z(:, 1:lastValid) };

%% ── Time axis ───────────────────────────────────────────────────────────
Fs      = 50;                           % Hz
% nT      = size(tapproach.pos_x, 2);
nT = lastValid;
t  = (0:nT-1) / Fs - 1;
t       = (0:nT-1) / Fs - 1;           % −1 s … (nT−1)/50 − 1 s
t0_idx  = find(t >= 0, 1, 'first');    % sample index of reach onset
postIdx = find(t >= 0);                 % all post-onset samples
t_post  = t(postIdx);


%% ── Global trial indices ────────────────────────────────────────────────
idx_c  = find(strcmp(labels, 'cued hit long'));
idx_s  = find(strcmp(labels, 'spont hit long'));
idx_m  = find(strcmp(labels, 'spont miss long'));
idxAll = {idx_c, idx_s, idx_m};

%% ── Shared filter: 4th-order Butterworth band-pass (1–15 Hz) ────────────
%  This bandwidth captures the dominant reach-cycle frequency while
%  suppressing DC drift and high-frequency noise prior to Hilbert analysis.
[b_bp, a_bp] = butter(4, [1 15] / (Fs / 2), 'bandpass');

%  Speed-drop threshold used to define movement abort in miss trials
abortThresh = 0.20;   % fraction of peak speed


%% ================================================================
%  ANALYSIS 1.1
%  Trial-by-trial kinematic variability  σ²(t)
%
%  For each condition c and Cartesian coordinate d ∈ {x, y, z},
%  compute the cross-trial variance at every time point:
%
%    σ²_{c,d}(t) = Var_k [ r_{k,d}(t) ]
%
%  The scalar summary is the post-onset trace-variance:
%
%    Σ_c = Σ_d  ⟨ σ²_{c,d}(t) ⟩_{t ≥ 0}
%
%  Key contrast:
%    Σ_{cued}  vs.  Σ_{spont hit}  vs.  Σ_{spont miss}
% ================================================================
fprintf('\n=== Analysis 1.1: Trial-by-trial Kinematic Variability ===\n')

fig11 = figure('Name', 'Analysis 1.1 – σ²(t) per condition', ...
               'Position', [50 200 1400 750], 'Visible', 'off');
tl11  = tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

traceVar = zeros(1, 3);   % integrated trace-variance for each condition

for row = 1:3
    idx    = idxAll{row};
    cumVar = 0;

    for col = 1:3
        P  = pos{col}(idx, :);        % [nTrials × nT]  position matrix
        mu = nanmean(P, 1);            % trial-mean trajectory
        v2 = nanvar(P, 0, 1);          % σ²(t) across trials  [1 × nT]
        cumVar = cumVar + nanmean(v2(postIdx));

        ax = nexttile;
        hold on

        % Individual trials (faint, demeaned for visibility)
        for k = 1:size(P, 1)
            plot(ax, t, P(k,:) - mu, ...
                 'Color', [rowColours{row}, 0.08], 'LineWidth', 0.5)
        end

        % Mean trajectory on left y-axis
        yyaxis left
        plot(ax, t, mu, 'Color', rowColours{row}, 'LineWidth', 2.0)
        ylabel(sprintf('Mean %s', posLbls{col}))
        ax.YAxis(1).Color = rowColours{row} * 0.7;

        % Variance time-series on right y-axis
        yyaxis right
        plot(ax, t, v2, 'Color', rowColours{row} * 0.5, ...
             'LineWidth', 1.6, 'LineStyle', '--')
        ylabel('\sigma^2 (mm^2)')
        ax.YAxis(2).Color = rowColours{row} * 0.5;

        xline(ax, 0, '--k', 'LineWidth', 0.9)
        xlabel('Time (s)')
        if col == 1
            title(sprintf('%s   (n = %d)', trialNames{row}, numel(idx)), ...
                  'FontSize', 11, 'Interpreter', 'none')
        else
            title(posLbls{col}, 'FontSize', 10)
        end
        box off; grid on; hold off
    end

    traceVar(row) = cumVar;
    fprintf('  %-20s : post-onset trace-σ² = %.4e mm²\n', ...
            trialNames{row}, cumVar)
end

title(tl11, ...
    'Analysis 1.1 — Trial-by-trial variance \sigma^2(t) per condition', ...
    'FontWeight', 'bold', 'FontSize', 13, 'Interpreter', 'tex')

% ── Summary bar figure ──────────────────────────────────────────────────
fig11b = figure('Name', 'Analysis 1.1 – Summary', ...
                'Position', [100 150 600 380], 'Visible', 'off');
hold on
for row = 1:3
    bar(row, traceVar(row), 'FaceColor', rowColours{row}, 'EdgeColor', 'w')
end
set(gca, 'XTick', 1:3, 'XTickLabel', trialNames, 'FontSize', 10)
ylabel('\Sigma_d \langle\sigma^2_d(t)\rangle_{t \geq 0}   (mm^2)', ...
       'Interpreter', 'tex')
title('Post-onset trace-variance per condition', 'FontSize', 12)
grid on; box off; hold off


%% ================================================================
%  ANALYSIS 1.2
%  Reach phase  φ(t)  and per-trial phase-speed profiles
%
%  The instantaneous phase of the reach cycle is estimated as:
%
%    φ(t) = ∠ H[ y_filt(t) ],    φ ∈ [−π, +π)
%
%  where H[·] denotes the Hilbert transform and y_filt is the
%  anterior-posterior (AP) paw displacement band-passed at 1–15 Hz.
%  The analytic signal amplitude is discarded; only the phase angle
%  is retained.  Instantaneous 3-D speed is computed as:
%
%    v(t) = ‖ dr/dt ‖ = √( ẋ² + ẏ² + ż² )
%
%  Phase-speed pairs are then characterised per condition.
% ================================================================
fprintf('\n=== Analysis 1.2: Reach Phase via Hilbert Transform ===\n')

fig12a = figure('Name', 'Analysis 1.2 – Phase & Speed time-series', ...
                'Position', [50 200 1300 750], 'Visible', 'off');
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact')

fig12b = figure('Name', 'Analysis 1.2 – Phase–Speed joint density', ...
                'Position', [100 100 1050 340], 'Visible', 'off');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact')

for row = 1:3
    idx   = idxAll{row};
    nTr   = numel(idx);
    Y     = pos{2}(idx, :);   % anterior-posterior displacement

    phi_all   = nan(nTr, nT);
    speed_all = nan(nTr, nT);

    for k = 1:nTr
        y_k = Y(k, :);
        if any(isnan(y_k)); continue; end

        % Band-pass and Hilbert transform
        y_filt        = filtfilt(b_bp, a_bp, y_k);
        analytic      = hilbert(y_filt);
        phi_all(k, :) = angle(analytic);   % instantaneous phase

        % 3-D instantaneous speed  (central finite differences via gradient)
        dx = gradient(pos{1}(idx(k), :)) * Fs;
        dy = gradient(pos{2}(idx(k), :)) * Fs;
        dz = gradient(pos{3}(idx(k), :)) * Fs;
        speed_all(k, :) = sqrt(dx.^2 + dy.^2 + dz.^2);
    end

    mu_ph = nanmean(phi_all,   1);
    se_ph = nanstd(phi_all,  0, 1);
    mu_sp = nanmean(speed_all, 1);
    se_sp = nanstd(speed_all,0, 1);

    % ── Phase time-series panel ──────────────────────────────────────────
    figure(fig12a)
    ax1 = nexttile;
    hold on
    for k = 1:nTr
        plot(ax1, t, phi_all(k,:), ...
             'Color', [rowColours{row}, 0.09], 'LineWidth', 0.5)
    end
    shadePlot2D_v(ax1, t, mu_ph, se_ph, rowColours{row}, 2.2)
    xline(ax1, 0, '--k', 'LineWidth', 0.9)
    yline(ax1, 0, ':',   'Color', [0.5 0.5 0.5], 'LineWidth', 0.7)
    ylabel('\phi(t)  (rad)')
    xlabel('Time (s)')
    ylim([-pi pi])
    yticks([-pi -pi/2 0 pi/2 pi])
    yticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
    title(sprintf('%s — \\phi(t)   (n = %d)', trialNames{row}, nTr), ...
          'FontSize', 10, 'Interpreter', 'tex')
    box off; grid on; hold off

    % ── Speed time-series panel ──────────────────────────────────────────
    ax2 = nexttile;
    hold on
    for k = 1:nTr
        plot(ax2, t, speed_all(k,:), ...
             'Color', [rowColours{row}, 0.09], 'LineWidth', 0.5)
    end
    shadePlot2D_v(ax2, t, mu_sp, se_sp, rowColours{row}, 2.2)
    xline(ax2, 0, '--k', 'LineWidth', 0.9)
    ylabel('Speed  (mm/s)')
    xlabel('Time (s)')
    title(sprintf('%s — ||dr/dt||', trialNames{row}), ...
          'FontSize', 10, 'Interpreter', 'none')
    box off; grid on; hold off

% ── Post-onset phase–speed joint density (2-D histogram) ────────────
    figure(fig12b)
    phi_vec   = reshape(phi_all(:, postIdx),   1, []);
    speed_vec = reshape(speed_all(:, postIdx), 1, []);
    valid     = ~isnan(phi_vec) & ~isnan(speed_vec);

    ax3 = nexttile;

    if sum(valid) < 2
        % Insufficient data — leave tile blank with a notice
        text(ax3, 0.5, 0.5, 'Insufficient data', ...
             'Units', 'normalized', 'HorizontalAlignment', 'center', ...
             'FontSize', 10)
        title(trialNames{row}, 'FontSize', 10, 'Interpreter', 'none')
        axis(ax3, 'off')
    else
        % Explicit bin edges guarantee ≥ 2 edges regardless of data range
        phi_edges   = linspace(-pi,  pi,               41);   % 40 bins
        speed_max   = max(speed_vec(valid));
        speed_edges = linspace(0,    max(speed_max, eps), 41); % 40 bins

        histogram2(ax3, phi_vec(valid), speed_vec(valid), ...
                   phi_edges, speed_edges, ...
                   'DisplayStyle', 'tile', ...
                   'ShowEmptyBins', 'off', ...
                   'EdgeColor', 'none')
        colormap(ax3, 'hot')
        xlabel('\phi(t)  (rad)');  ylabel('Speed  (mm/s)')
        xlim([-pi pi])
        xticks([-pi -pi/2 0 pi/2 pi])
        xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
        title(trialNames{row}, 'FontSize', 10, 'Interpreter', 'none')
        colorbar(ax3)
        box off
    end
end

figure(fig12a)
sgtitle('phi(t) = angle H[y_{filt}(t)]  and instantaneous speed per condition', ...
        'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'none')

figure(fig12b)
sgtitle('Post-onset phase–speed joint density', ...
        'FontWeight', 'bold', 'FontSize', 12)


%% ================================================================
%  ANALYSIS 1.3
%  Miss-trial kinematics as a boundary condition
%
%  For spontaneous miss trials, t = 0 is defined by the kinematic
%  onset event. We characterise the post-onset reach trajectory:
%
%    (i)   3-D displacement from onset:
%              d(t) = ‖ r(t) − r(t₀) ‖
%
%    (ii)  Instantaneous speed and its time-derivative (deceleration):
%              v(t) = ‖ dr/dt ‖,    a(t) = dv/dt
%
%    (iii) Abort phase φ_abort: the phase value at which the movement
%          is deemed to have terminated, defined as the first sample
%          after the speed peak at which v(t) < abortThresh · v_peak.
%
%  These metrics establish the kinematic "signature" of a failed reach,
%  providing a behavioural boundary condition for neural state analysis.
% ================================================================
fprintf('\n=== Analysis 1.3: Miss-trial Kinematic Boundary Condition ===\n')

nMiss = numel(idx_m);
nPost = numel(postIdx);

% Pre-allocate trial-wise summary matrices
maxDisp_v    = nan(nMiss, 1);
peakSpeed_v  = nan(nMiss, 1);
abortPhase_v = nan(nMiss, 1);
abortTime_v  = nan(nMiss, 1);
dispMat      = nan(nMiss, nPost);
speedMat     = nan(nMiss, nPost);
decelMat     = nan(nMiss, nPost);
phiMat       = nan(nMiss, nPost);

for k = 1:nMiss
    tr = idx_m(k);
    xv = pos{1}(tr, :);
    yv = pos{2}(tr, :);
    zv = pos{3}(tr, :);

    if any(isnan([xv, yv, zv])); continue; end

    % (i) 3-D displacement from reach-onset position
    x0 = xv(t0_idx);  y0 = yv(t0_idx);  z0 = zv(t0_idx);
    d3 = sqrt((xv - x0).^2 + (yv - y0).^2 + (zv - z0).^2);
    dispMat(k, :) = d3(postIdx);

    % (ii) Speed and deceleration
    dx = gradient(xv) * Fs;
    dy = gradient(yv) * Fs;
    dz = gradient(zv) * Fs;
    sp = sqrt(dx.^2 + dy.^2 + dz.^2);
    speedMat(k, :) = sp(postIdx);
    decelMat(k, :) = gradient(sp(postIdx)) * Fs;   % mm/s²

    % (iii) Instantaneous reach phase via Hilbert transform on AP component
    y_filt = filtfilt(b_bp, a_bp, yv);
    phi_k  = angle(hilbert(y_filt));
    phiMat(k, :) = phi_k(postIdx);

    % Scalar summaries
    maxDisp_v(k)   = max(d3(postIdx));
    peakSpeed_v(k) = max(sp(postIdx));

    % Abort: first post-peak sample where speed falls below threshold
    sp_post  = sp(postIdx);
    [~, pkL] = max(sp_post);
    ab_rel   = find(sp_post(pkL:end) < abortThresh * peakSpeed_v(k), 1);
    if ~isempty(ab_rel)
        aIdx              = pkL + ab_rel - 1;
        abortPhase_v(k)   = phiMat(k, aIdx);
        abortTime_v(k)    = t_post(aIdx);
    end
end

% ── Console report ───────────────────────────────────────────────────────
fprintf('  Spont Miss Long (n = %d):\n', nMiss)
fprintf('    Max 3-D displacement : %.3f ± %.3f mm\n', ...
        nanmean(maxDisp_v),   nanstd(maxDisp_v))
fprintf('    Peak speed           : %.3f ± %.3f mm/s\n', ...
        nanmean(peakSpeed_v), nanstd(peakSpeed_v))
fprintf('    Abort phase φ_abort  : %.3f ± %.3f rad\n', ...
        nanmean(abortPhase_v), nanstd(abortPhase_v))
fprintf('    Time to abort        : %.3f ± %.3f s post-onset\n', ...
        nanmean(abortTime_v), nanstd(abortTime_v))

% ── Fig 1.3a: displacement, speed, deceleration ─────────────────────────
fig13a = figure('Name', 'Analysis 1.3 – Miss kinematics', ...
                'Position', [50 200 1350 450], 'Visible', 'off');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact')

% Panel 1 — displacement
ax1 = nexttile;
hold on
for k = 1:nMiss
    plot(ax1, t_post, dispMat(k,:), ...
         'Color', [rowColours{3}, 0.13], 'LineWidth', 0.6)
end
shadePlot2D_v(ax1, t_post, nanmean(dispMat,1), nanstd(dispMat,0,1), ...
              rowColours{3}, 2.5)
xlabel('Time from onset (s)');  ylabel('Displacement (mm)')
title('3-D Displacement from Onset', 'FontSize', 11)
box off; grid on; hold off

% Panel 2 — speed
ax2 = nexttile;
hold on
for k = 1:nMiss
    plot(ax2, t_post, speedMat(k,:), ...
         'Color', [rowColours{3}, 0.13], 'LineWidth', 0.6)
end
shadePlot2D_v(ax2, t_post, nanmean(speedMat,1), nanstd(speedMat,0,1), ...
              rowColours{3}, 2.5)
xlabel('Time from onset (s)');  ylabel('Speed (mm/s)')
title('Instantaneous Speed  ||dr/dt||', 'FontSize', 11, 'Interpreter', 'none')
box off; grid on; hold off

% Panel 3 — deceleration
ax3 = nexttile;
hold on
for k = 1:nMiss
    plot(ax3, t_post, decelMat(k,:), ...
         'Color', [rowColours{3}, 0.13], 'LineWidth', 0.6)
end
shadePlot2D_v(ax3, t_post, nanmean(decelMat,1), nanstd(decelMat,0,1), ...
              rowColours{3}, 2.5)
yline(ax3, 0, '--k', 'LineWidth', 0.9)
xlabel('Time from onset (s)');  ylabel('Acceleration (mm/s^2)')
title('Speed Derivative  (deceleration)', 'FontSize', 11)
box off; grid on; hold off

sgtitle(['Analysis 1.3 — Spont Miss Long: kinematic boundary condition' ...
         sprintf('   (n = %d trials)', nMiss)], ...
        'FontWeight', 'bold', 'FontSize', 12)

% ── Fig 1.3b: abort-phase distribution & comparison across conditions ────
fig13b = figure('Name', 'Analysis 1.3 – Abort phase & cross-condition', ...
                'Position', [100 100 1000 440], 'Visible', 'off');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact')

% Abort phase histogram for miss trials
ax4 = nexttile;
validAb = abortPhase_v(~isnan(abortPhase_v));
histogram(ax4, validAb, 20, ...
          'FaceColor', rowColours{3}, 'EdgeColor', 'w', 'FaceAlpha', 0.85)
xline(ax4, nanmean(validAb), '--k', 'LineWidth', 2.0, ...
      'Label', sprintf('\\mu = %.2f rad', nanmean(validAb)), ...
      'LabelHorizontalAlignment', 'left', 'Interpreter', 'tex')
xlim([-pi pi])
xticks([-pi -pi/2 0 pi/2 pi])
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
xlabel('\phi_{abort}  (rad)',  'Interpreter', 'tex')
ylabel('Trial count')
title(sprintf('Phase at Movement Abort\n(n_{valid} = %d)', numel(validAb)), ...
      'FontSize', 11)
grid on; box off

% Cross-condition violin/box: max displacement
ax5 = nexttile;
hold on
for row = 1:3
    idx = idxAll{row};
    dAll = nan(numel(idx), 1);
    for k = 1:numel(idx)
        tr = idx(k);
        xv = pos{1}(tr,:); yv = pos{2}(tr,:); zv = pos{3}(tr,:);
        if any(isnan([xv,yv,zv])); continue; end
        x0 = xv(t0_idx); y0 = yv(t0_idx); z0 = zv(t0_idx);
        d3 = sqrt((xv-x0).^2+(yv-y0).^2+(zv-z0).^2);
        dAll(k) = max(d3(postIdx));
    end
    boxchart(ax5, row * ones(sum(~isnan(dAll)),1), ...
             dAll(~isnan(dAll)), ...
             'BoxFaceColor', rowColours{row}, ...
             'MarkerColor',  rowColours{row}, ...
             'WhiskerLineColor', rowColours{row})
end
set(ax5, 'XTick', 1:3, 'XTickLabel', trialNames, 'FontSize', 9)
ylabel('Max post-onset displacement (mm)')
title('Peak Reach Extent by Condition', 'FontSize', 11)
grid on; box off; hold off

sgtitle('Analysis 1.3 — Abort-phase distribution and cross-condition displacement', ...
        'FontWeight', 'bold', 'FontSize', 12)


%% ── Save all figures ────────────────────────────────────────────────────
figs  = {fig11,                         fig11b, ...
         fig12a,                        fig12b, ...
         fig13a,                        fig13b};
names = {'Analysis_1_1_Variance.png',   'Analysis_1_1_Summary.png', ...
         'Analysis_1_2_PhaseSpeed.png', 'Analysis_1_2_JointDensity.png', ...
         'Analysis_1_3_MissKin.png',    'Analysis_1_3_AbortPhase.png'};

for f = 1:numel(figs)
    exportgraphics(figs{f}, fullfile(outDir, names{f}), 'Resolution', 300)
end
close all
fprintf('\nAll %d figures saved to:\n  %s\n', numel(figs), outDir)


%% ================================================================
%  LOCAL FUNCTIONS
% ================================================================

function shadePlot2D_v(ax, t, mu, se, col, lw)
% SHADEPLOT2D_V  Plot mean ± SE band on axis ax.
%   mu and se must be 1×nT row vectors. NaN-safe.
    valid = ~isnan(mu) & ~isnan(se);
    t_v   = t(valid);   mu_v = mu(valid);   se_v = se(valid);
    if ~isempty(t_v)
        hF = fill(ax, [t_v, fliplr(t_v)], ...
                      [mu_v + se_v, fliplr(mu_v - se_v)], ...
                  col, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
        uistack(hF, 'bottom');
    end
    plot(ax, t, mu, 'Color', col, 'LineWidth', lw)
end
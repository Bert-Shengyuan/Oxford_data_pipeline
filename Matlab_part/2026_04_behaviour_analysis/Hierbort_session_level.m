%% =========================================================
%  Analysis_1_Kinematics_SessionLevel.m
%
%  DESIGN PRINCIPLE
%  ────────────────
%  The fundamental unit of observation is the SESSION-MEAN
%  trajectory, not the individual trial. For each valid session s
%  and condition c, we first compute:
%
%    r̄^{(s,c)}(t) = (1/n_c^s) Σ_k  r_k^{(s,c)}(t)
%
%  All downstream analyses (variance, phase, abort kinematics)
%  operate exclusively on { r̄^{(s,c)} }_{s ∈ S_valid}.
%
%  VALIDITY CRITERION
%  ──────────────────
%  Session s is retained iff:
%    numSpont(s) > 10  AND  numCued(s) > 10  AND  numMiss(s) > 10
%
%  ANALYSES
%  ────────
%  1.1  Cross-session variance  Σ²_{c,d}(t) = Var_s[ r̄^{(s,c)}_d(t) ]
%  1.2  Session-mean reach phase  φ̄^{(s,c)}(t) via Hilbert transform
%  1.3  Session-mean miss-trial kinematics as boundary condition
%
%  Dataset : tapproach_global_oyku.mat
%  Fs      : 50 Hz  |  t = sample/50 − 1  (s)
%  t = 0   : kinematically-defined reach onset
% =========================================================
close all; clc; clear all

%% ── Load data ──────────────────────────────────────────────────────────
load('/Users/shengyuancai/Downloads/Oxford_dataset/tapproach_global_oyku.mat')

%% ── Output directory ───────────────────────────────────────────────────
outDir = '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/Behaviour_phase/Phase_Analysis_1_SessionLevel';
if ~exist(outDir, 'dir'); mkdir(outDir); end

%% ── Metadata ────────────────────────────────────────────────────────────
labels        = tapproach.label;          % [nTrials × 1] condition strings
sessions      = tapproach.session_name;   % [nTrials × 1] session IDs
posLbls       = {'x (mm)', 'y (mm)', 'z (mm)'};
posNames      = {'x', 'y', 'z'};
condStr       = {'cued hit long', 'spont hit long', 'spont miss long'};
condNames     = {'Cued Hit Long', 'Spont Hit Long', 'Spont Miss Long'};
rowColours    = {[0.18 0.46 0.71], [0.17 0.63 0.17], [0.84 0.19 0.15]};
abortThresh   = 0.20;   % fraction of peak speed for abort detection

%% ── Strip trailing all-NaN time samples ────────────────────────────────
lastValid = find( ~all(isnan(tapproach.pos_x), 1) & ...
                  ~all(isnan(tapproach.pos_y), 1) & ...
                  ~all(isnan(tapproach.pos_z), 1), 1, 'last');

pos = { tapproach.pos_x(:, 1:lastValid), ...
        tapproach.pos_y(:, 1:lastValid), ...
        tapproach.pos_z(:, 1:lastValid) };

%% ── Time axis ───────────────────────────────────────────────────────────
Fs      = 50;
nT      = lastValid;
t       = (0:nT-1) / Fs - 1;
t0_idx  = find(t >= 0, 1, 'first');
postIdx = find(t >= 0);
t_post  = t(postIdx);

%% ── Shared filter: 4th-order Butterworth band-pass (1–15 Hz) ────────────
[b_bp, a_bp] = butter(4, [1 15] / (Fs / 2), 'bandpass');

%% ================================================================
%  SESSION IDENTIFICATION & VALIDITY FILTERING
%
%  For each unique session, count trials per condition.
%  Retain session s only when:
%    n_spont(s) > 10  AND  n_cued(s) > 10  AND  n_miss(s) > 10
% ================================================================
fprintf('\n=== Session validity screening ===\n')

uniqueSessions = unique(sessions);
nSess          = numel(uniqueSessions);

% Pre-allocate counts table
nCued_v  = zeros(nSess,1);
nSpont_v = zeros(nSess,1);
nMiss_v  = zeros(nSess,1);

for s = 1:nSess
    inSess       = strcmp(sessions, uniqueSessions{s});
    nCued_v(s)   = sum(inSess & strcmp(labels, condStr{1}));
    nSpont_v(s)  = sum(inSess & strcmp(labels, condStr{2}));
    nMiss_v(s)   = sum(inSess & strcmp(labels, condStr{3}));
end

% Validity mask
validMask = (nSpont_v > 10) & (nCued_v > 10) & (nMiss_v > 10);
validSess = uniqueSessions(validMask);
nValid    = numel(validSess);

fprintf('  Total sessions   : %d\n', nSess)
fprintf('  Valid sessions   : %d  (criterion: n_cued>10, n_spont>10, n_miss>10)\n', nValid)
fprintf('\n  %-30s  %6s  %6s  %6s  %5s\n', ...
        'Session', 'nCued', 'nSpont', 'nMiss', 'Valid')
fprintf('  %s\n', repmat('-',1,60))
for s = 1:nSess
    fprintf('  %-30s  %6d  %6d  %6d  %5s\n', ...
            uniqueSessions{s}, nCued_v(s), nSpont_v(s), nMiss_v(s), ...
            mat2str(validMask(s)))
end

if nValid == 0
    error('No sessions satisfy the validity criterion. Cannot proceed.')
end

%% ================================================================
%  COMPUTE SESSION-MEAN TRAJECTORIES  (revised phase section)
%
%  Phase is now estimated independently on the pre-touch [t < 0]
%  and post-touch [t >= 0] segments, then concatenated:
%
%    phi_pre  = angle( H[ filtfilt(y_pre)  ] )
%    phi_post = angle( H[ filtfilt(y_post) ] )
%    sessPhiMean{c}(si,:) = [phi_pre , phi_post]
%
%  Minimum-length guard: filtfilt on a 4th-order bandpass
%  (8th-order after bilinear expansion) requires > 3*8 = 24 samples.
%  At Fs=50 Hz the pre-touch epoch is 50 samples — acceptable,
%  but flagged if somehow shorter.
% ================================================================

% ── Index vectors (derived from t, already defined above) ───────
preIdx  = find(t <  0);          % indices for t ∈ [-1, 0)
postIdx = find(t >= 0);          % indices for t ∈ [0,  2]
minLen  = 3 * (max(length(a_bp), length(b_bp)) - 1);   % filtfilt minimum

fprintf('\n=== Computing session-mean trajectories ===\n')

for c = 1:3
    sessMean{c}      = nan(nValid, nT, 3);
    sessPhiMean{c}   = nan(nValid, nT);
    sessSpeedMean{c} = nan(nValid, nT);
end

for si = 1:nValid
    sName  = validSess{si};
    inSess = strcmp(sessions, sName);

    for c = 1:3
        trIdx = find(inSess & strcmp(labels, condStr{c}));
        if isempty(trIdx); continue; end

        % ── Session-mean position trajectory ────────────────────────────
        for d = 1:3
            P = pos{d}(trIdx, :);
            sessMean{c}(si, :, d) = nanmean(P, 1);
        end

        % ── Session-mean AP signal ───────────────────────────────────────
        yMean = squeeze(sessMean{c}(si, :, 2));   % [1 × nT]

        % ── Segment-wise instantaneous phase ────────────────────────────
        %
        %  We filter and transform each epoch independently so that
        %  post-touch transients cannot leak into the pre-touch phase
        %  estimate through the Hilbert convolution kernel or filtfilt
        %  edge padding.

        phi = nan(1, nT);   % accumulator for this session × condition

        for seg = 1:2
            if seg == 1
                idx = preIdx;   label_seg = 'pre-touch';
            else
                idx = postIdx;  label_seg = 'post-touch';
            end

            y_seg = yMean(idx);

            % ── Validity checks ─────────────────────────────────────────
            if any(isnan(y_seg))
                warning('Session %s | cond %d | %s: NaNs present — skipping segment.', ...
                        sName, c, label_seg);
                continue
            end

            if numel(y_seg) <= minLen
                warning('Session %s | cond %d | %s: segment too short (%d samples, need >%d).', ...
                        sName, c, label_seg, numel(y_seg), minLen);
                continue
            end

            % ── Filter then transform ────────────────────────────────────
            y_filt      = filtfilt(b_bp, a_bp, y_seg);
            phi(idx)    = angle(hilbert(y_filt));
        end

        sessPhiMean{c}(si, :) = phi;

        % ── Session-mean 3-D speed (unchanged) ──────────────────────────
        dx = gradient(squeeze(sessMean{c}(si, :, 1))) * Fs;
        dy = gradient(squeeze(sessMean{c}(si, :, 2))) * Fs;
        dz = gradient(squeeze(sessMean{c}(si, :, 3))) * Fs;
        sessSpeedMean{c}(si, :) = sqrt(dx.^2 + dy.^2 + dz.^2);
    end
end

fprintf('  Done. Session-mean arrays: [%d sessions × %d time-points × 3 axes]\n', ...
        nValid, nT)


%% ================================================================
%  ANALYSIS 1.1
%  Cross-session variance  Σ²_{c,d}(t)
%
%  Operating exclusively on session-mean trajectories:
%
%    Σ²_{c,d}(t) = Var_s [ r̄^{(s,c)}_d(t) ]
%
%  Scalar summary (post-onset trace-variance):
%
%    Σ_c = Σ_d  〈 Σ²_{c,d}(t) 〉_{t≥0}
%
%  This quantifies how much the *session-average* reach profile
%  varies across animals / sessions, not within-session noise.
% ================================================================
fprintf('\n=== Analysis 1.1: Cross-session variance Σ²(t) ===\n')
 
fig11 = figure('Name', 'Analysis 1.1 – Cross-session Σ²(t)', ...
               'Position', [50 200 1400 750], 'Visible', 'off');
tl11  = tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
 
traceVar = zeros(1, 3);
 
for c = 1:3
    cumVar = 0;
 
    for d = 1:3
        % R: [nValid × nT]  session-mean position for condition c, axis d
        R  = squeeze(sessMean{c}(:, :, d));
        mu = nanmean(R, 1);       % grand mean across sessions  [1 × nT]
        v2 = nanvar(R,  0, 1);    % cross-session variance       [1 × nT]
        cumVar = cumVar + nanmean(v2(postIdx));
 
        ax = nexttile;
        hold on
 
        % Individual session-mean trajectories (faint)
        for si = 1:nValid
            plot(ax, t, R(si,:), ...
                 'Color', [rowColours{c}, 0.18], 'LineWidth', 0.8)
        end
 
        % Grand mean on left y-axis
        yyaxis left
        plot(ax, t, mu, 'Color', rowColours{c}, 'LineWidth', 2.2)
        ylabel(sprintf('Grand-mean %s', posLbls{d}))
        ax.YAxis(1).Color = rowColours{c} * 0.7;
 
        % Cross-session variance on right y-axis
        yyaxis right
        plot(ax, t, v2, 'Color', rowColours{c} * 0.5, ...
             'LineWidth', 1.6, 'LineStyle', '--')
        ylabel('\Sigma^2 (mm^2)')
        ax.YAxis(2).Color = rowColours{c} * 0.5;
 
        xline(ax, 0, '--k', 'LineWidth', 0.9)
        xlabel('Time (s)')
        if d == 1
            title(sprintf('%s   (N_{sess} = %d)', condNames{c}, nValid), ...
                  'FontSize', 11, 'Interpreter', 'none')
        else
            title(posLbls{d}, 'FontSize', 10)
        end
        box off; grid on; hold off
    end
 
    traceVar(c) = cumVar;
    fprintf('  %-22s : post-onset trace-Σ² = %.4e mm²\n', condNames{c}, cumVar)
end
 
title(tl11, ...
    sprintf('Analysis 1.1 — Cross-session variance \\Sigma^2(t)  [N_{valid} = %d sessions]', nValid), ...
    'FontWeight', 'bold', 'FontSize', 13, 'Interpreter', 'tex')
 
% ── Per-session within-session trace-variance ─────────────────────────
%  For each valid session s and condition c, compute the post-onset
%  trace-variance across trials within that session:
%
%    V^{(s,c)} = Σ_d  〈 Var_k[ r_{k,d}^{(s,c)}(t) ] 〉_{t≥0}
%
%  Result: sessTraceVar  [nValid × 3]  — one value per (session, condition)
sessTraceVar = nan(nValid, 3);
 
for c = 1:3
    for si = 1:nValid
        sName  = validSess{si};
        inSess = strcmp(sessions, sName);
        trIdx  = find(inSess & strcmp(labels, condStr{c}));
        if numel(trIdx) < 2; continue; end
 
        cumV = 0;
        for d = 1:3
            P    = pos{d}(trIdx, :);           % [nTrials × nT] within session
            v2   = nanvar(P, 0, 1);            % cross-trial variance  [1 × nT]
            cumV = cumV + nanmean(v2(postIdx));
        end
        sessTraceVar(si, c) = cumV;
    end
end
 
% ── Boxchart: distribution of V^{(s,c)} across sessions ──────────────
fig11b = figure('Name', 'Analysis 1.1 – Per-session variance distribution', ...
                'Position', [100 150 350 420], 'Visible', 'on');
hold on
for c = 1:3
    vals = sessTraceVar(:, c);
    vals = vals(~isnan(vals));
    if isempty(vals); continue; end
    boxchart(c * ones(numel(vals), 1), vals, ...
             'BoxFaceColor',    rowColours{c}, ...
             'MarkerColor',     rowColours{c}, ...
             'WhiskerLineColor', rowColours{c}, ...
             'BoxFaceAlpha',    0.75)
end
set(gca, 'XTick', 1:3, 'XTickLabel', condNames, 'FontSize', 10)
ylabel('\Sigma_d \langle\sigma^2_{k,d}(t)\rangle_{t\geq0}   (mm^2)', ...
       'Interpreter', 'tex')
ylim([0,max(max(sessTraceVar(:, :)))*1.1])
title(sprintf(['Within-session cross-trial trace-variance per condition\n[N_{valid} = %d sessions]'], nValid), ...
      'FontSize', 11, 'Interpreter', 'tex')
grid on; box off; hold off


%% ================================================================
%  ANALYSIS 1.2
%  Session-mean reach phase  φ̄^{(s,c)}(t)  and speed profiles
%
%  The instantaneous phase of the session-mean reach is:
%
%    φ̄^{(s,c)}(t) = ∠ H[ ȳ^{(s,c)}_filt(t) ],   φ ∈ [−π, +π)
%
%  where H[·] is the Hilbert transform and ȳ_filt is the
%  band-passed (1–15 Hz) session-mean AP position.
%  Speed is derived from the session-mean trajectory:
%
%    v̄^{(s,c)}(t) = ‖ d r̄^{(s,c)} / dt ‖
%
%  Cross-session statistics are then:
%    μ_φ(t) = mean_s [ φ̄^{(s,c)}(t) ]  (circular mean advisable)
%    μ_v(t) = mean_s [ v̄^{(s,c)}(t) ]
% ================================================================
fprintf('\n=== Analysis 1.2: Session-mean phase and speed profiles ===\n')

fig12a = figure('Name', 'Analysis 1.2 – Session-mean Phase & Speed', ...
                'Position', [50 200 1300 750], 'Visible', 'off');
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact')

fig12b = figure('Name', 'Analysis 1.2 – Phase–Speed joint density (session-mean)', ...
                'Position', [100 100 1050 340], 'Visible', 'off');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact')

for c = 1:3
    PHI   = sessPhiMean{c};     % [nValid × nT]
    SPD   = sessSpeedMean{c};   % [nValid × nT]

    mu_ph = nanmean(PHI, 1);
    se_ph = nanstd(PHI,  0, 1);
    mu_sp = nanmean(SPD, 1);
    se_sp = nanstd(SPD,  0, 1);

    % ── Phase time-series ────────────────────────────────────────────────
    figure(fig12a)
    ax1 = nexttile;
    hold on
    for si = 1:nValid
        plot(ax1, t, PHI(si,:), ...
             'Color', [rowColours{c}, 0.20], 'LineWidth', 0.8)
    end
    shadePlot2D_v(ax1, t, mu_ph, se_ph, rowColours{c}, 2.2)
    xline(ax1, 0, '--k', 'LineWidth', 0.9)
    yline(ax1, 0, ':',   'Color', [0.5 0.5 0.5], 'LineWidth', 0.7)
    ylabel('\phi(t)  (rad)')
    xlabel('Time (s)')
    ylim([-pi pi])
    yticks([-pi -pi/2 0 pi/2 pi])
    yticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
    title(sprintf('%s — \\phi(t)   (N = %d sessions)', condNames{c}, nValid), ...
          'FontSize', 10, 'Interpreter', 'tex')
    box off; grid on; hold off

    % ── Speed time-series ────────────────────────────────────────────────
    ax2 = nexttile;
    hold on
    for si = 1:nValid
        plot(ax2, t, SPD(si,:), ...
             'Color', [rowColours{c}, 0.20], 'LineWidth', 0.8)
    end
    shadePlot2D_v(ax2, t, mu_sp, se_sp, rowColours{c}, 2.2)
    xline(ax2, 0, '--k', 'LineWidth', 0.9)
    ylabel('Speed  (mm/s)')
    xlabel('Time (s)')
    ylim([0, 0.2])
    title(sprintf('%s — ||dr̄/dt||', condNames{c}), ...
          'FontSize', 10, 'Interpreter', 'none')
    box off; grid on; hold off

    % ── Post-onset phase–speed joint density across sessions ─────────────
    figure(fig12b)
    phi_vec   = reshape(PHI(:, postIdx),   1, []);
    speed_vec = reshape(SPD(:, postIdx),   1, []);
    valid     = ~isnan(phi_vec) & ~isnan(speed_vec);

    ax3 = nexttile;
    if sum(valid) < 2
        text(ax3, 0.5, 0.5, 'Insufficient data', ...
             'Units', 'normalized', 'HorizontalAlignment', 'center', ...
             'FontSize', 10)
        title(condNames{c}, 'FontSize', 10, 'Interpreter', 'none')
        axis(ax3, 'off')
    else
        phi_edges   = linspace(-pi, pi, 41);
        speed_max   = max(speed_vec(valid));
        speed_edges = linspace(0, max(speed_max, eps), 41);
        histogram2(ax3, phi_vec(valid), speed_vec(valid), ...
                   phi_edges, speed_edges, ...
                   'DisplayStyle', 'tile', 'ShowEmptyBins', 'off', ...
                   'EdgeColor', 'none')
        colormap(ax3, 'hot')
        xlabel('\phi(t)  (rad)');  ylabel('Speed  (mm/s)')
        xlim([-pi pi])
        xticks([-pi -pi/2 0 pi/2 pi])
         
        xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
        title(condNames{c}, 'FontSize', 10, 'Interpreter', 'none')
        colorbar(ax3); box off
    end
end

figure(fig12a)
sgtitle(sprintf('Session-mean phase \\phi(t) and speed  [N = %d sessions]', nValid), ...
        'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'tex')
figure(fig12b)
sgtitle(sprintf('Post-onset phase–speed density (session-mean, N = %d)', nValid), ...
        'FontWeight', 'bold', 'FontSize', 12)


%% ================================================================
%  ANALYSIS 1.3
%  Session-mean miss kinematics as a boundary condition
%
%  For condition "spont miss long", all computations operate on
%  the session-mean trajectory  r̄^{(s,miss)}(t).
%
%    (i)   Session-mean 3-D displacement from onset:
%              d̄^{(s)}(t) = ‖ r̄^{(s)}(t) − r̄^{(s)}(t₀) ‖
%
%    (ii)  Session-mean speed and its derivative:
%              v̄^{(s)}(t) = ‖ d r̄^{(s)} / dt ‖
%              ā^{(s)}(t) = d v̄^{(s)} / dt
%
%    (iii) Abort phase φ̄_abort^{(s)}: phase of the session-mean
%          speed profile at the first post-peak sample where
%          v̄(t) < abortThresh · v̄_peak.
%
%  Session-level scalars are then summarised across valid sessions.
% ================================================================
fprintf('\n=== Analysis 1.3: Session-mean miss kinematics ===\n')

nPost  = numel(postIdx);

maxDisp_v    = nan(nValid, 1);
peakSpeed_v  = nan(nValid, 1);
abortPhase_v = nan(nValid, 1);
abortTime_v  = nan(nValid, 1);
dispMat      = nan(nValid, nPost);
speedMat     = nan(nValid, nPost);
decelMat     = nan(nValid, nPost);

% Miss condition index = 3
for si = 1:nValid
    xMean = squeeze(sessMean{3}(si, :, 1));
    yMean = squeeze(sessMean{3}(si, :, 2));
    zMean = squeeze(sessMean{3}(si, :, 3));

    if any(isnan([xMean, yMean, zMean])); continue; end

    % (i) 3-D displacement of session-mean from onset position
    x0 = xMean(t0_idx);  y0 = yMean(t0_idx);  z0 = zMean(t0_idx);
    d3 = sqrt((xMean - x0).^2 + (yMean - y0).^2 + (zMean - z0).^2);
    dispMat(si, :) = d3(postIdx);

    % (ii) Speed and deceleration of session-mean trajectory
    dx = gradient(xMean) * Fs;
    dy = gradient(yMean) * Fs;
    dz = gradient(zMean) * Fs;
    sp = sqrt(dx.^2 + dy.^2 + dz.^2);
    speedMat(si, :)  = sp(postIdx);
    decelMat(si, :)  = gradient(sp(postIdx)) * Fs;

    maxDisp_v(si)    = max(d3(postIdx));
    peakSpeed_v(si)  = max(sp(postIdx));

    % (iii) Abort phase on session-mean speed profile
    sp_post  = sp(postIdx);
    [~, pkL] = max(sp_post);
    ab_rel   = find(sp_post(pkL:end) < abortThresh * peakSpeed_v(si), 1);
    if ~isempty(ab_rel)
        aIdx             = pkL + ab_rel - 1;
        % Phase from session-mean Hilbert signal
        abortPhase_v(si) = sessPhiMean{3}(si, postIdx(aIdx));
        abortTime_v(si)  = t_post(aIdx);
    end
end

% ── Console report ───────────────────────────────────────────────────────
fprintf('  Spont Miss Long (N_valid = %d sessions):\n', nValid)
fprintf('    Max 3-D displacement : %.3f ± %.3f mm\n', ...
        nanmean(maxDisp_v),    nanstd(maxDisp_v))
fprintf('    Peak speed           : %.3f ± %.3f mm/s\n', ...
        nanmean(peakSpeed_v),  nanstd(peakSpeed_v))
fprintf('    Abort phase φ_abort  : %.3f ± %.3f rad\n', ...
        nanmean(abortPhase_v), nanstd(abortPhase_v))
fprintf('    Time to abort        : %.3f ± %.3f s post-onset\n', ...
        nanmean(abortTime_v),  nanstd(abortTime_v))

% ── Fig 1.3a: displacement, speed, deceleration ─────────────────────────
fig13a = figure('Name', 'Analysis 1.3 – Session-mean miss kinematics', ...
                'Position', [50 200 1350 450], 'Visible', 'off');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact')

ax1 = nexttile; hold on
for si = 1:nValid
    plot(ax1, t_post, dispMat(si,:), ...
         'Color', [rowColours{3}, 0.20], 'LineWidth', 0.8)
end
shadePlot2D_v(ax1, t_post, nanmean(dispMat,1), nanstd(dispMat,0,1), ...
              rowColours{3}, 2.5)
xlabel('Time from onset (s)');  ylabel('Displacement (mm)')
title('Session-mean 3-D Displacement from Onset', 'FontSize', 11)
box off; grid on; hold off

ax2 = nexttile; hold on
for si = 1:nValid
    plot(ax2, t_post, speedMat(si,:), ...
         'Color', [rowColours{3}, 0.20], 'LineWidth', 0.8)
end
shadePlot2D_v(ax2, t_post, nanmean(speedMat,1), nanstd(speedMat,0,1), ...
              rowColours{3}, 2.5)
xlabel('Time from onset (s)');  ylabel('Speed (mm/s)')
title('Session-mean Speed  ||dr̄/dt||', 'FontSize', 11, 'Interpreter', 'none')
box off; grid on; hold off

ax3 = nexttile; hold on
for si = 1:nValid
    plot(ax3, t_post, decelMat(si,:), ...
         'Color', [rowColours{3}, 0.20], 'LineWidth', 0.8)
end
shadePlot2D_v(ax3, t_post, nanmean(decelMat,1), nanstd(decelMat,0,1), ...
              rowColours{3}, 2.5)
yline(ax3, 0, '--k', 'LineWidth', 0.9)
xlabel('Time from onset (s)');  ylabel('Acceleration (mm/s^2)')
title('Session-mean Deceleration Profile', 'FontSize', 11)
box off; grid on; hold off

sgtitle(sprintf(['Analysis 1.3 — Session-mean Spont Miss kinematics  ' ...
                 '[N_{valid} = %d sessions]'], nValid), ...
        'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'tex')

% ── Fig 1.3b: abort-phase histogram & cross-condition displacement ────────
fig13b = figure('Name', 'Analysis 1.3 – Abort phase & cross-condition displacement', ...
                'Position', [100 150 700 420], 'Visible', 'off');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact')

% Abort-phase histogram (one data point per valid session)
ax4 = nexttile;
validAb = abortPhase_v(~isnan(abortPhase_v));
histogram(ax4, validAb, min(20, max(5, numel(validAb))), ...
          'FaceColor', rowColours{3}, 'EdgeColor', 'w', 'FaceAlpha', 0.85)
if ~isempty(validAb)
    xline(ax4, nanmean(validAb), '--k', 'LineWidth', 2.0, ...
          'Label', sprintf('\\mu = %.2f rad', nanmean(validAb)), ...
          'LabelHorizontalAlignment', 'left', 'Interpreter', 'tex')
end
xlim([-pi pi])
xticks([-pi -pi/2 0 pi/2 pi])
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
xlabel('\phi_{abort}  (rad)', 'Interpreter', 'tex')
ylabel('Session count')
title(sprintf('Session-mean Abort Phase\n(N_{valid} = %d)', numel(validAb)), ...
      'FontSize', 11, 'Interpreter', 'tex')
grid on; box off

% Cross-condition session-mean max displacement
ax5 = nexttile; hold on
for c = 1:3
    dAll = nan(nValid, 1);
    for si = 1:nValid
        xMean = squeeze(sessMean{c}(si, :, 1));
        yMean = squeeze(sessMean{c}(si, :, 2));
        zMean = squeeze(sessMean{c}(si, :, 3));
        if any(isnan([xMean, yMean, zMean])); continue; end
        x0 = xMean(t0_idx);  y0 = yMean(t0_idx);  z0 = zMean(t0_idx);
        d3 = sqrt((xMean-x0).^2 + (yMean-y0).^2 + (zMean-z0).^2);
        dAll(si) = max(d3(postIdx));
    end
    validD = dAll(~isnan(dAll));
    if ~isempty(validD)
        boxchart(ax5, c * ones(numel(validD),1), validD, ...
                 'BoxFaceColor', rowColours{c}, ...
                 'MarkerColor',  rowColours{c}, ...
                 'WhiskerLineColor', rowColours{c})
    end
end
ylim([0 0.021])
set(ax5, 'XTick', 1:3, 'XTickLabel', condNames, 'FontSize', 9)
ylabel('Max post-onset session-mean displacement (mm)')
title('Session-mean Peak Reach Extent by Condition', 'FontSize', 11)
grid on; box off; hold off

sgtitle(sprintf(['Session-mean abort-phase and ' ...
                 'cross-condition displacement  [N = %d]'], nValid), ...
        'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'tex')


%% ── Save all figures ────────────────────────────────────────────────────
figs  = {fig11,                               fig11b, ...
         fig12a,                              fig12b, ... 
         fig13a,                              fig13b};
names = {'Analysis_1_1_CrossSessVariance.png','Analysis_1_1_Summary.png', ...
         'Analysis_1_2_SessionMeanPhase.png', 'Analysis_1_2_JointDensity.png', ...
         'Analysis_1_3_SessionMissKin.png',   'Analysis_1_3_AbortPhase.png'};

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
    valid = ~isnan(mu) & ~isnan(se);
    t_v   = t(valid);  mu_v = mu(valid);  se_v = se(valid);
    if ~isempty(t_v)
        hF = fill(ax, [t_v, fliplr(t_v)], ...
                      [mu_v + se_v, fliplr(mu_v - se_v)], ...
                  col, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
        uistack(hF, 'bottom');
    end
    plot(ax, t, mu, 'Color', col, 'LineWidth', lw)
end
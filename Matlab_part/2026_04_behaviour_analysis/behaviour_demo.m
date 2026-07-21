%% =========================================================
%  Position Trajectory Analysis — tapproach dataset
%  Parts A, B, C  (2-D traces + 3-D trajectories)
%  Sampling rate: 50 Hz   |   t = sample/50 - 1  (seconds)
% ==========================================================
close all; clc; clear all
load('/Users/shengyuancai/Downloads/Oxford_dataset/tapproach_global_oyku.mat')

% ── SEM helper (NaN-safe, SD used as band width per original code) ──────
sem = @(M) nanstd(M, 0, 1);

% ── Metadata ────────────────────────────────────────────────────────────
labels     = tapproach.label;
sessions   = tapproach.session_name;
pos        = {tapproach.pos_x, tapproach.pos_y, tapproach.pos_z};
posLbls    = {'x (mm)', 'y (mm)', 'z (mm)'};
trialTypes = {'cued hit long', 'spont hit long',  'spont miss long'};
trialNames = {'Cued hit long','Spont hit long',  'Spont miss long'};

% Colours: blue / red / green
rowColours = {[0.18 0.46 0.71], [0.17 0.63 0.17],[0.84 0.19 0.15]};   
lightC     = {rowColours{1}.*0.6 + 0.4, rowColours{2}.*0.6 + 0.4, rowColours{3}.*0.6 + 0.4}; 

% ── Time axis (seconds) ─────────────────────────────────────────────────
Fs = 50;                                % Hz
nT = size(tapproach.pos_x, 2);
t  = (0:nT-1) / Fs - 1;                 % −1 s … (nT−1)/50 − 1 s

% ── Global trial indices ─────────────────────────────────────────────────
idx_c = find(strcmp(labels, 'cued hit long'));
idx_s = find(strcmp(labels, 'spont hit long'));
idx_m = find(strcmp(labels, 'spont miss long'));

% ── Valid sessions (>10 spont, ≥1 cued, ≥1 miss) ────────────────────────
allSess   = unique(sessions);
validSess = {};
for k = 1:numel(allSess)
    inSess   = strcmp(sessions, allSess{k});
    numCued  = sum(strcmp(labels(inSess), 'cued hit long'));
    numSpont = sum(strcmp(labels(inSess), 'spont hit long'));
    numMiss  = sum(strcmp(labels(inSess), 'spont miss long'));
    
    if numSpont > 10 && numCued > 10 && numMiss > 10
        validSess{end+1} = allSess{k}; %#ok<AGROW>
    end
end
fprintf('Sessions meeting criteria: %d\n', numel(validSess));

% ── Random picks ────────────────────────────────────────────────────────
rng('shuffle')
pick_s = idx_s(randi(numel(idx_s)));
pick_c = idx_c(randi(numel(idx_c)));
pick_m = idx_m(randi(numel(idx_m)));
picks  = [pick_c, pick_s,  pick_m];

chosenSess = validSess{randi(numel(validSess))};
inSess     = strcmp(sessions, chosenSess);
fprintf('Part B session: %s\n', chosenSess);

% ================================================================
%  ░░░  SHARED UTILITY FUNCTIONS  ░░░
% ================================================================
% Shade mean ± band on a 2-D axis
function shadePlot2D(ax, t, mu, se, col, lw)
    validIdx = ~isnan(mu) & ~isnan(se);
    t_v = t(validIdx); mu_v = mu(validIdx); se_v = se(validIdx);
    if ~isempty(t_v)
        hF = fill(ax, [t_v, fliplr(t_v)], ...
                      [mu_v+se_v, fliplr(mu_v-se_v)], ...
                  col, 'FaceAlpha', 0.30, 'EdgeColor', 'none');
        uistack(hF, 'bottom');
    end
    plot(ax, t, mu, 'Color', col, 'LineWidth', lw)
end

%%
% Plot 3-D mean ± SD tube (approximated as ribbon along max-variance axis)
function plot3DmeanTube(ax, X, Y, Z, col, lw)
    if size(X,1) > 1
        mx = nanmean(X,1);  my = nanmean(Y,1);  mz = nanmean(Z,1);
        sx = nanstd(X,0,1); sy = nanstd(Y,0,1); sz = nanstd(Z,0,1);
        xu = mx+sx; xl = mx-sx;
        nT_ = numel(mx);
        Xs  = [xu; xl];   Ys = [my; my];   Zs = [mz; mz];
        surf(ax, Xs, Ys, Zs, 'FaceColor', col, 'FaceAlpha', 0.20, ...
             'EdgeColor', 'none')
    else
        mx = X; my = Y; mz = Z;
    end
    plot3(ax, mx, my, mz, 'Color', col, 'LineWidth', lw)
    plot3(ax, mx(1),   my(1),   mz(1),   'o', 'MarkerSize', 10, ...
          'MarkerFaceColor','k', 'MarkerEdgeColor','k')
    plot3(ax, mx(end-1), my(end-1), mz(end-1), 's', 'MarkerSize', 10, ...
          'MarkerFaceColor', 'k', 'MarkerEdgeColor','k')
end

%%
% ================================================================
%  PART A — 2-D: one random trial per type
% ================================================================
figA = figure('Name','Part A – 2D Single Trials', ...
              'Position',[50 200 1300 750], 'Visible','off');
tiledlayout(3, 3, 'TileSpacing','compact','Padding','compact')
for row = 1:3
    tr = picks(row);
    for col = 1:3
        nexttile
        plot(t, pos{col}(tr,:), 'Color', rowColours{row}, 'LineWidth', 1.6)
        xlabel('Time (s)'); ylabel(posLbls{col})
        ylim([-0.015, 0.015]) % Y-axis constraint
        xline(0,'--k','LineWidth',0.8)
        title(sprintf('%s\nTrial %d', trialNames{row}, tr), ...
              'FontSize',9,'Interpreter','none')
        box off; grid on
    end
end
sgtitle('Part A — Single-trial position traces', 'FontWeight','bold')

% ── Part A — 3-D trajectory ─────────────────────────────────────────────
figA3D = figure('Name','Part A – 3D Single Trials', ...
                'Position',[50 50 1200 420], 'Visible','off');
tiledlayout(1, 3, 'TileSpacing','compact','Padding','compact')
for row = 1:3
    tr  = picks(row);
    ax3 = nexttile;
    X   = pos{1}(tr,:);
    Y   = pos{2}(tr,:);
    Z   = pos{3}(tr,:);
    nSeg = nT - 1;
    cmap = interp1([0;1], [rowColours{row}; rowColours{row}.*0.3+0.7], ...
                   linspace(0,1,nSeg));
    for s = 1:nSeg
        plot3(ax3, X(s:s+1), Y(s:s+1), Z(s:s+1), ...
              'Color', cmap(s,:), 'LineWidth', 1.8)
        hold on
    end
    plot3(ax3, X(1),   Y(1),   Z(1),   'o', 'MarkerSize', 10, ...
          'MarkerFaceColor','k', 'MarkerEdgeColor','k')
    plot3(ax3, X(end-1), Y(end-1), Z(end-1), 's', 'MarkerSize', 10, ...
          'MarkerFaceColor','k', 'MarkerEdgeColor','k')
    xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)')
    title(sprintf('%s — Trial %d', trialNames{row}, tr), ...
          'Interpreter','none','FontSize',9)
    grid on; view(45, 25); hold off
end
sgtitle('Part A — Single-trial 3-D trajectories  (● start  ■ end)', ...
        'FontWeight','bold')

%%
% ================================================================
%  PART B — 2-D: within-session individual + mean ± SD
% ================================================================
figB = figure('Name','Part B – 2D Within-session', ...
              'Position',[50 200 1300 750], 'Visible','off');
tiledlayout(3, 3, 'TileSpacing','compact','Padding','compact')
for row = 1:3 
    inType = strcmp(labels, trialTypes{row}) & inSess;
    idxB   = find(inType);
    for col = 1:3
        ax = nexttile;
        P  = pos{col}(idxB, :);
        hold on
        for k = 1:size(P,1)
            plot(ax, t, P(k,:), 'Color', [rowColours{row} 0.15], ...
                 'LineWidth', 0.7)
        end
        xline(ax, 0,'--k','LineWidth',0.8)
        shadePlot2D(ax, t, nanmean(P,1), sem(P), rowColours{row}, 2.5)
        xlabel('Time (s)'); ylabel(posLbls{col})
        ylim([-0.015, 0.015]) % Y-axis constraint
        if col == 1
            title(sprintf('%s %s  (n=%d)', trialNames{row}, chosenSess, ...
                  numel(idxB)),'FontSize',13,'Interpreter','none')
        end
        box off; grid on; hold off
    end
end
sgtitle('Within-session: individual trials + mean ± SD', ...
        'FontWeight','normal','FontSize',15)

% ── Part B — 3-D trajectory ─────────────────────────────────────────────
figB3D = figure('Name','Part B – 3D Within-session', ...
                'Position',[50 50 1200 420], 'Visible','off');
tiledlayout(1, 3, 'TileSpacing','compact','Padding','compact')
for row = 1:3
    inType = strcmp(labels, trialTypes{row}) & inSess;
    idxB   = find(inType);
    ax3    = nexttile;
    hold on
    X = pos{1}(idxB,:);
    Y = pos{2}(idxB,:);
    Z = pos{3}(idxB,:);
    for k = 1:size(X,1)
        plot3(ax3, X(k,:), Y(k,:), Z(k,:), ...
              'Color', [rowColours{row} 0.15], 'LineWidth', 0.7)
    end
    plot3DmeanTube(ax3, X, Y, Z, rowColours{row}, 2.8)
    xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)')
    title(sprintf('%s\n%s  (n=%d)', trialNames{row}, chosenSess, numel(idxB)), ...
          'Interpreter','none','FontSize',9)
    grid on; view(45,25); hold off
end
sgtitle('Within-session 3-D trajectories (bold = mean)', ...
        'FontWeight','bold')

% ================================================================
%  PART C — 2-D: cross-session means + grand mean ± SD
% ================================================================
figC = figure('Name','Part C – 2D Cross-session', ...
              'Position',[50 200 1300 750], 'Visible','off');
tiledlayout(3, 3, 'TileSpacing','compact','Padding','compact')

sessMeansAll = cell(3, numel(validSess), 3);  % {type, sess, dim}
nValidArr    = zeros(1, 3);
for row = 1:3
    nValid = 0;
    for k = 1:numel(validSess)
        inType = strcmp(labels, trialTypes{row}) & ...
                 strcmp(sessions, validSess{k});
        if sum(inType) < 1; continue; end
        nValid = nValid + 1;
        for col = 1:3
            sessMeansAll{row, nValid, col} = nanmean(pos{col}(inType,:), 1);
        end
    end
    nValidArr(row) = nValid;
    for col = 1:3
        M  = cell2mat(sessMeansAll(row, 1:nValid, col)');  % [nSess x nT]
        ax = nexttile;
        hold on
        for k = 1:nValid
            plot(ax, t, M(k,:), 'Color', [rowColours{row} 0.20], ...
                 'LineWidth', 0.8)
        end
        xline(ax, 0,'--k','LineWidth',0.8)
        shadePlot2D(ax, t, nanmean(M,1), sem(M), rowColours{row}, 2.5)
        xlabel('Time (s)'); ylabel(posLbls{col})
        ylim([-0.015, 0.015]) % Y-axis constraint
        if col == 1
            title(sprintf('%s (N=%d sessions)', trialNames{row}, nValid), ...
                  'FontSize',13,'Interpreter','none') 
        end
        box off; grid on; hold off
    end
end
sgtitle('Cross-session: session means + grand mean ± SD', ...
        'FontWeight','normal','FontSize',15)

% ── Part C — 3-D trajectory ─────────────────────────────────────────────
figC3D = figure('Name','Part C – 3D Cross-session', ...
                'Position',[50 50 1200 420], 'Visible','off');
tiledlayout(1, 3, 'TileSpacing','compact','Padding','compact')
for row = 1:3
    nValid = nValidArr(row);
    ax3    = nexttile;
    hold on
    X = cell2mat(sessMeansAll(row, 1:nValid, 1)');  % [nSess x nT]
    Y = cell2mat(sessMeansAll(row, 1:nValid, 2)');
    Z = cell2mat(sessMeansAll(row, 1:nValid, 3)');
    
    for k = 1:nValid
        plot3(ax3, X(k,:), Y(k,:), Z(k,:), ...
              'Color', [rowColours{row} 0.20], 'LineWidth', 0.8)
    end
    plot3DmeanTube(ax3, X, Y, Z, rowColours{row}, 2.8)
    
    xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)')
    title(sprintf('%s\n(N=%d sessions)', trialNames{row}, nValid), ...
          'Interpreter','none','FontSize',9)
    grid on; view(45,25); hold off
end
sgtitle('Part C — Cross-session 3-D trajectories (bold = grand mean)', ...
        'FontWeight','bold')

% ================================================================
%  SAVE ALL FIGURES
% ================================================================
outDir = '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/Behaviour_phase/Test_demo';
if ~exist(outDir,'dir'); mkdir(outDir); end
%%
figs  = {figA,  figB,  figC,  figA3D,           figB3D,              figC3D};
names = {'Part_A_2D.png','Part_B_2D.png','Part_C_2D.png', ...
         'Part_A_3D.png','Part_B_3D.png','Part_C_3D.png'};
for f = 1:numel(figs)
    exportgraphics(figs{f}, fullfile(outDir, names{f}), 'Resolution', 300)
end
close all
fprintf('All 6 figures saved to:\n  %s\n', outDir)
%% Oxford Dataset GLM Analysis Pipeline - Complete Demonstration
%
% EDUCATIONAL OVERVIEW:
% This script demonstrates the complete workflow for analyzing General Linear
% Models in the context of Canonical Correlation Analysis using the Oxford
% neurophysiology dataset. The analysis addresses fundamental questions about
% how neural populations encode shared variance patterns across brain regions.
%
% THEORETICAL FOUNDATION:
% The GLM framework allows us to decompose the CCA latent variables into
% individual neuronal contributions. Mathematically, we model:
%
%   L(t) = β₀ + Σᵢ βᵢ · PSHTᵢ(t) + ε
%
% where L(t) is the canonical variate (CCA latent variable), PSHTᵢ(t) represents
% the peri-stimulus time histogram of neuron i, βᵢ are the GLM coefficients
% quantifying each neuron's contribution, and ε represents residual variance.
%
% SCIENTIFIC QUESTIONS ADDRESSED:
% 1. How well can we reconstruct canonical patterns from individual neurons?
% 2. Is neural information distributed across populations or concentrated in key neurons?
% 3. How does encoding strategy vary across different brain region pairs?
%
% PIPELINE STAGES:
% Stage 1: GLM Coefficient Extraction (oxford_GLM_explain.m)
%          - Fits GLM models relating PSTH to CCA latents
%          - Identifies significant neuronal contributors
%          - Quantifies explanatory power (R²) for each region
%
% Stage 2: Sensitivity Analysis (oxford_GLM_draw.m)
%          - Systematic neuron removal experiments
%          - Quantifies information distribution (sparse vs distributed)
%          - Generates publication-quality visualizations

clear; clc;
fprintf('====================================================\n');
fprintf('Oxford Dataset: GLM-CCA Analysis Pipeline\n');
fprintf('====================================================\n\n');

%% STAGE 1: GLM Coefficient Extraction
% =======================================
%
% METHODOLOGICAL RATIONALE:
% Before we can assess how information is distributed across neurons, we must
% first establish each neuron's baseline contribution to the canonical pattern.
% The GLM provides this through coefficient estimation via maximum likelihood.
%
% STATISTICAL CONSIDERATIONS:
% - We use normal distribution with identity link (linear regression)
% - Cross-validation from CCA ensures generalization
% - Minimum neuron threshold (default: 30) ensures statistical power
%
% INTERPRETATION OF OUTPUT:
% - GLM R²: Proportion of CCA latent variance explained by neural activity
% - β coefficients: Individual neuron contribution weights
% - p-values: Statistical significance of each neuron's contribution

fprintf('STAGE 1: Extracting GLM Coefficients\n');
fprintf('=====================================\n');
fprintf('This stage fits General Linear Models to relate neuronal activity\n');
fprintf('to the CCA latent variables, quantifying each neuron''s contribution\n');
fprintf('to the canonical correlation pattern.\n\n');

fprintf('Executing: oxford_GLM_explain.m\n');
fprintf('Expected outputs:\n');
fprintf('  - all_glm_coefficients.mat: β coefficients and statistics\n');
fprintf('  - glm_analysis_summary.txt: Human-readable report\n\n');

% Uncomment the following line to execute Stage 1
% oxford_glm_cca_coefficients_extraction();

fprintf('Stage 1 Complete ✓\n');
fprintf('Review the summary report to assess model quality (R² values)\n');
fprintf('before proceeding to sensitivity analysis.\n\n');

% CHECKPOINT: Quality Control
fprintf('QUALITY CONTROL CHECKPOINT:\n');
fprintf('---------------------------\n');
fprintf('Before proceeding to Stage 2, verify:\n');
fprintf('1. GLM R² values are reasonable (typically 0.2-0.8)\n');
fprintf('   - Low R² (<0.2) suggests weak neural-latent relationship\n');
fprintf('   - High R² (>0.8) indicates strong reconstruction capability\n');
fprintf('2. Sufficient sessions per region pair (≥5 recommended)\n');
fprintf('3. Adequate number of significant neurons (p < 0.05)\n\n');

pause(2);  % Brief pause for readability

%% STAGE 2: Sensitivity Analysis
% ================================
%
% THEORETICAL MOTIVATION:
% A fundamental question in systems neuroscience concerns the distribution
% of neural information. Two competing hypotheses exist:
%
% H₁ (Sparse Coding): Information concentrates in a few highly-informative neurons
%     Prediction: Removing top-ranked neurons causes dramatic R² degradation
%
% H₂ (Distributed Coding): Information distributes redundantly across population
%     Prediction: Random and top-ranked removal produce similar degradation
%
% EXPERIMENTAL DESIGN:
% We implement a factorial neuron removal paradigm:
% - Factor 1: Removal Strategy (Top-ranked vs Random)
% - Factor 2: Removal Proportion (0%, 5%, 10%, ..., 90%)
% - Dependent Variable: Explained Variance (R²)
%
% QUANTITATIVE METRICS:
% 1. Concentration Index: ΔR²_top(50%) - ΔR²_random(50%)
%    - Values > 0.15 suggest sparse coding
%    - Values < 0.05 suggest distributed coding
%    - Intermediate values indicate mixed strategies
%
% 2. Degradation Slope: Rate of R² decline with neuron removal
%    - Steep slope indicates vulnerability to neuron loss
%    - Shallow slope suggests robust, redundant encoding

fprintf('STAGE 2: Sensitivity Analysis\n');
fprintf('==============================\n');
fprintf('This stage performs systematic neuron removal experiments to\n');
fprintf('characterize whether information is sparsely or distributedly encoded.\n\n');

fprintf('Executing: oxford_GLM_draw.m\n');
fprintf('Expected outputs:\n');
fprintf('  - sensitivity_*.mat: Removal experiment data per region pair\n');
fprintf('  - sensitivity_*.png: Degradation curves visualizations\n');
fprintf('  - cross_pair_concentration_summary.png: Comparative analysis\n\n');

% Uncomment the following line to execute Stage 2
% oxford_glm_sensitivity_analysis();

fprintf('Stage 2 Complete ✓\n\n');

%% INTERPRETATION GUIDE
% ======================
fprintf('====================================================\n');
fprintf('RESULTS INTERPRETATION GUIDE\n');
fprintf('====================================================\n\n');

fprintf('UNDERSTANDING SENSITIVITY PLOTS:\n');
fprintf('--------------------------------\n');
fprintf('The sensitivity visualizations show two key curves:\n\n');

fprintf('1. BLUE CURVE (Top-Ranked Removal):\n');
fprintf('   Removes neurons with highest |β| coefficients first\n');
fprintf('   → Tests whether information concentrates in specific neurons\n\n');

fprintf('2. RED CURVE (Random Removal):\n');
fprintf('   Removes neurons randomly (Monte Carlo averaged)\n');
fprintf('   → Provides baseline expectation for distributed encoding\n\n');

fprintf('CURVE DIVERGENCE INTERPRETATION:\n');
fprintf('--------------------------------\n');
fprintf('Large Divergence (Blue drops faster than Red):\n');
fprintf('  → Sparse/Hierarchical Coding\n');
fprintf('  → Few neurons carry most information\n');
fprintf('  → Similar to "grandmother cell" hypothesis\n');
fprintf('  → Example: Visual cortex face cells\n\n');

fprintf('Small Divergence (Blue and Red similar):\n');
fprintf('  → Distributed/Democratic Coding\n');
fprintf('  → Information redundantly encoded\n');
fprintf('  → Robust to individual neuron loss\n');
fprintf('  → Example: Motor cortex reaching movements\n\n');

fprintf('CONCENTRATION INDEX INTERPRETATION:\n');
fprintf('-----------------------------------\n');
fprintf('This metric quantifies the divergence at 50%% removal:\n');
fprintf('  • > 0.15: Strongly concentrated (sparse coding)\n');
fprintf('  • 0.05-0.15: Moderately concentrated (mixed strategy)\n');
fprintf('  • < 0.05: Distributed coding\n\n');

%% STATISTICAL CONSIDERATIONS
fprintf('====================================================\n');
fprintf('STATISTICAL CONSIDERATIONS\n');
fprintf('====================================================\n\n');

fprintf('SAMPLE SIZE REQUIREMENTS:\n');
fprintf('-------------------------\n');
fprintf('• Minimum 5 sessions per region pair (for reliable means)\n');
fprintf('• Minimum 30 neurons per region (for stable GLM fitting)\n');
fprintf('• At least 50 trials per session (for cross-validation)\n\n');

fprintf('ERROR BARS AND UNCERTAINTY:\n');
fprintf('---------------------------\n');
fprintf('• Shaded regions represent ±1 SEM across sessions\n');
fprintf('• Random removal uses Monte Carlo with 10 iterations\n');
fprintf('• Session-to-session variability reflects biological heterogeneity\n\n');

fprintf('MULTIPLE COMPARISONS:\n');
fprintf('---------------------\n');
fprintf('When comparing across multiple region pairs, consider:\n');
fprintf('• Bonferroni correction for family-wise error rate\n');
fprintf('• FDR control for exploratory analyses\n');
fprintf('• Cross-validation helps prevent overfitting\n\n');

%% NEXT STEPS AND EXTENSIONS
fprintf('====================================================\n');
fprintf('NEXT STEPS AND POTENTIAL EXTENSIONS\n');
fprintf('====================================================\n\n');

fprintf('1. TEMPORAL DYNAMICS:\n');
fprintf('   Analyze how encoding concentration varies across time bins\n');
fprintf('   within trials (early vs late responses)\n\n');

fprintf('2. CROSS-COMPONENT ANALYSIS:\n');
fprintf('   Compare encoding strategies across different CCA components\n');
fprintf('   (Component 1 vs Component 2, etc.)\n\n');

fprintf('3. BEHAVIORAL CORRELATIONS:\n');
fprintf('   Relate concentration indices to task performance metrics\n');
fprintf('   (reaction time, accuracy, learning rate)\n\n');

fprintf('4. ANATOMICAL MAPPING:\n');
fprintf('   Investigate whether concentration index varies by cortical layer,\n');
fprintf('   cell type, or anatomical subregion\n\n');

fprintf('5. NETWORK ANALYSIS:\n');
fprintf('   Use functional connectivity to identify "hub" neurons that\n');
fprintf('   coordinate information across regions\n\n');

%% TROUBLESHOOTING
fprintf('====================================================\n');
fprintf('COMMON ISSUES AND SOLUTIONS\n');
fprintf('====================================================\n\n');

fprintf('ISSUE: Low GLM R² values (<0.2)\n');
fprintf('SOLUTION:\n');
fprintf('  • Check CCA component significance (use only significant components)\n');
fprintf('  • Verify neuron sampling matches CCA preprocessing\n');
fprintf('  • Ensure proper time alignment between PSTH and CCA\n\n');

fprintf('ISSUE: Flat sensitivity curves (no degradation with removal)\n');
fprintf('SOLUTION:\n');
fprintf('  • This may indicate overfitting in original GLM\n');
fprintf('  • Try regularization (Ridge/Lasso GLM)\n');
fprintf('  • Verify sufficient trials for stable estimates\n\n');

fprintf('ISSUE: Dimension mismatch errors\n');
fprintf('SOLUTION:\n');
fprintf('  • Confirm trial counts match between PSTH and CCA\n');
fprintf('  • Check for preprocessing differences (trial exclusion, etc.)\n');
fprintf('  • Verify time window consistency across analyses\n\n');

fprintf('====================================================\n');
fprintf('Pipeline Demonstration Complete\n');
fprintf('====================================================\n');
fprintf('\nFor questions or issues, consult the function documentation\n');
fprintf('within oxford_GLM_explain.m and oxford_GLM_draw.m\n\n');

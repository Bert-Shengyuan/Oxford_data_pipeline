# Oxford Dataset GLM Analysis - Adaptation Guide

## Overview

This document provides a comprehensive guide to the adapted GLM analysis pipeline for your Oxford neurophysiology dataset. The analysis has been restructured to work with your session-based data organization (`session_CCA_results` and `session_PSTH_results`) rather than the previous region-pair directory structure.

---

## Table of Contents

1. [Key Structural Differences](#key-structural-differences)
2. [File Descriptions](#file-descriptions)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Configuration and Setup](#configuration-and-setup)
5. [Expected Outputs](#expected-outputs)
6. [Methodological Notes](#methodological-notes)

---

## Key Structural Differences

### Previous Dataset Structure (Your Old Analysis)

Your previous GLM analysis utilized a directory-based organization:

```
Data-CCA/
└── {trial_type}_all/
    └── {Region1}_{Region2}/
        ├── component_projections_dataset_001.mat
        ├── component_projections_dataset_002.mat
        ├── transfer_matrices_dataset_001.mat
        └── transfer_matrices_dataset_002.mat

single_region_analysis_{trial_type}/
└── neural_data_shuffled_{Region}/
    ├── neural_data_001_{Region}.mat
    └── neural_data_002_{Region}.mat
```

Each region pair had its own directory containing individual files for each dataset (session). The GLM analysis would iterate through these directories and process each dataset file separately.

### Oxford Dataset Structure (Current Analysis)

The Oxford dataset employs session-centric organization:

```
session_CCA_results/
├── yp013_220211_CCA_results.mat    # All region pairs for this session
├── yp014_220212_CCA_results.mat
└── ...

session_PSTH_results/
├── yp013_220211_PSTH_data.mat      # All regions for this session
├── yp014_220212_PSTH_data.mat
└── ...
```

Each session has a single file containing results for all brain regions and region pairs analyzed in that session. This approach provides several computational advantages:

**Memory Efficiency**: Raw spike data is processed once per session and discarded, retaining only essential PSTH summaries and CCA projections.

**Modular Architecture**: Each session file is self-contained, enabling parallel processing and facilitating selective data loading.

**Scalability**: The flat file structure scales more efficiently as the number of sessions increases, avoiding deeply nested directory trees.

---

## File Descriptions

### 1. `oxford_GLM_explain.m` (Main Analysis Script)

**Purpose**: Extracts GLM coefficients relating neural activity to CCA latent variables.

**Key Adaptations from Original**:

- **Session-Level Loading**: Loads complete session files rather than individual component projections. This reflects the unified storage structure where all region pairs from a session reside in a single `.mat` file.

- **Unified Data Extraction**: Processes all region pairs from each session in a single iteration, improving computational efficiency through reduced I/O operations.

- **Dynamic Neuron Selection**: Retrieves the specific neurons used in CCA from `pair_result.selected_neurons_region1` and `pair_result.selected_neurons_region2`, ensuring GLM uses identical neural populations as the CCA analysis.

- **Flexible Latent Variable Handling**: Implements robust dimension checking and reshaping for canonical projections, accommodating variations in how different sessions structure their output.

**Mathematical Framework**:

The GLM establishes the relationship:

```
L(t) = β₀ + Σᵢ βᵢ · PSHTᵢ(t) + ε
```

where:
- `L(t)` represents the CCA latent variable (canonical variate)
- `PSHTᵢ(t)` is the peri-stimulus time histogram of neuron i
- `βᵢ` quantifies neuron i's contribution to the canonical pattern
- `ε` captures residual variance unexplained by the neural population

The GLM coefficients `βᵢ` provide a direct measure of each neuron's importance in the inter-regional communication captured by CCA. Neurons with large absolute coefficients `|βᵢ|` contribute substantially to the shared variance between brain regions.

**Statistical Outputs**:

For each region in each session, the script computes:
- R² (coefficient of determination): proportion of latent variance explained
- Adjusted R²: R² penalized for model complexity
- β coefficients: neuronal contribution weights
- Standard errors: uncertainty in coefficient estimates
- t-statistics and p-values: significance testing for each neuron
- 95% confidence intervals: plausible ranges for true coefficients
- AIC/BIC: model selection criteria balancing fit and complexity

### 2. `oxford_GLM_draw.m` (Sensitivity Analysis)

**Purpose**: Quantifies information distribution across neural populations through systematic neuron removal experiments.

**Key Adaptations**:

- **Session-Based Iteration**: Loads GLM coefficients organized by session, then reconstructs the full neural data matrices needed for sensitivity experiments.

- **Paired Data Reconstruction**: Matches PSTH data with CCA projections for each session, ensuring temporal alignment and trial correspondence.

- **Monte Carlo Random Removal**: Implements multiple random removal iterations to establish stable baseline expectations for distributed coding.

**Theoretical Motivation**:

A fundamental question in systems neuroscience concerns whether neural information is encoded sparsely (concentrated in a few key neurons) or distributedly (redundantly across the population). The sensitivity analysis addresses this question through a factorial neuron removal paradigm:

**Factor 1 - Removal Strategy**:
1. **Top-Ranked Removal**: Systematically removes neurons with the highest absolute GLM coefficients `|βᵢ|`. If information is concentrated, removing these high-weight neurons should dramatically reduce the GLM's explanatory power.

2. **Random Removal**: Removes neurons randomly (Monte Carlo averaged). This establishes the baseline degradation expected under distributed coding, where each neuron contributes similarly.

**Factor 2 - Removal Proportion**: Varies from 0% to 90% in 5% increments, allowing us to characterize the full degradation curve.

**Quantitative Metrics**:

**Concentration Index** = ΔR²_toprank(50%) - ΔR²_random(50%)

This metric quantifies the divergence between removal strategies at the 50% removal point. Interpretation:
- Values > 0.15: Information highly concentrated (sparse coding)
- Values 0.05-0.15: Mixed strategy (some concentration)
- Values < 0.05: Information distributed uniformly

**Degradation Slope**: The rate at which R² declines with neuron removal, quantifying the population's robustness to neuron loss.

### 3. `oxford_GLM_pipeline_demo.m` (Demonstration Script)

**Purpose**: Educational walkthrough demonstrating proper usage and interpretation of the analysis pipeline.

**Features**:

- **Step-by-Step Execution**: Guides through both analysis stages with detailed explanations
- **Quality Control Checkpoints**: Suggests verification steps between stages
- **Interpretation Guide**: Comprehensive explanation of results and metrics
- **Troubleshooting Section**: Common issues and their solutions
- **Extension Suggestions**: Potential follow-up analyses

This script serves as both a tutorial and a template for your own analyses. You can uncomment the execution lines to run the full pipeline, or use it as a reference for understanding the methodology.

---

## Data Flow Architecture

### Stage 1: GLM Coefficient Extraction

```
Input Files:
├── session_CCA_results/{session}_CCA_results.mat
│   ├── cca_results.pair_results{i}
│   │   ├── cv_results.canonical_projections{component}
│   │   │   ├── region1 (latent variable)
│   │   │   └── region2 (latent variable)
│   │   ├── selected_neurons_region1
│   │   └── selected_neurons_region2
│   
└── session_PSTH_results/{session}_PSTH_data.mat
    └── psth_data.regions.{region_name}
        ├── trial_data (trials × neurons × timepoints)
        ├── psth (neurons × timepoints)
        └── neuron_indices

Processing:
1. Load CCA results and PSTH data for session
2. For each region pair in the session:
   a. Extract canonical projections (latent variables)
   b. Get selected neurons used in CCA
   c. Extract trial data for those neurons
   d. Reshape: (trials × timepoints) × neurons
   e. Mean-center neural data
   f. Fit GLM: latent ~ neural_activity
   g. Store coefficients and statistics

Output:
└── GLM_Analysis_Component_{X}/
    ├── all_glm_coefficients.mat
    │   └── {Region1}_{Region2}.sessions{i}
    │       ├── region1_coefficients (β values)
    │       ├── region1_stats (R², p-values, etc.)
    │       └── selected_neurons1
    └── glm_analysis_summary.txt
```

### Stage 2: Sensitivity Analysis

```
Input Files:
├── GLM_Analysis_Component_{X}/all_glm_coefficients.mat
├── session_CCA_results/{session}_CCA_results.mat
└── session_PSTH_results/{session}_PSTH_data.mat

Processing:
For each region pair and session:
1. Load GLM coefficients (β values)
2. Reconstruct full neural data matrix
3. Extract CCA latent variables
4. For each removal percentage ρ ∈ {0%, 5%, ..., 90%}:
   
   Top-Ranked Strategy:
   a. Sort neurons by |β| (descending)
   b. Remove top ρ% of neurons
   c. Refit GLM on remaining neurons
   d. Compute R²_toprank(ρ)
   
   Random Strategy:
   e. For iter = 1 to n_iterations:
      - Randomly select neurons to keep
      - Refit GLM
      - Compute R²_random(ρ, iter)
   f. Average across iterations

5. Store degradation curves

Output:
└── GLM_Analysis_Component_{X}/sensitivity_analysis/
    ├── sensitivity_{Region1}_{Region2}.mat
    │   └── Degradation curves for both regions
    ├── sensitivity_{Region1}_{Region2}.png
    │   └── Visualization of encoding concentration
    └── cross_pair_concentration_summary.png
        └── Comparative analysis across all pairs
```

---

## Configuration and Setup

### Directory Structure Requirements

Before running the analysis, ensure your base directory contains:

```
/your/base/directory/
├── session_CCA_results/
│   ├── {session1}_CCA_results.mat
│   ├── {session2}_CCA_results.mat
│   └── ...
│
└── session_PSTH_results/
    ├── {session1}_PSTH_data.mat
    ├── {session2}_PSTH_data.mat
    └── ...
```

### Modifying Configuration Parameters

In `oxford_GLM_explain.m`, update line 18:

```matlab
base_results_dir = '/Users/shengyuancai/Downloads/Oxford_dataset/';
```

Change this to your actual Oxford dataset location. The script will automatically construct paths to the `session_CCA_results` and `session_PSTH_results` subdirectories.

### Component Selection

Both scripts analyze a specific CCA component (default: Component 1). To analyze different components, modify:

```matlab
component_to_analyze = 2;  % Change to 2, 3, etc. for other components
```

### Quality Control Parameters

The analysis includes several quality thresholds that you can adjust:

**Minimum Neuron Threshold** (line 23 in `oxford_GLM_explain.m`):
```matlab
min_neurons_threshold = 30;  % Minimum neurons for stable GLM
```

Rationale: GLM requires sufficient observations relative to parameters. With fewer neurons, coefficient estimates become unstable and standard errors large. The default value of 30 provides a reasonable balance, but you can increase this for more conservative analysis or decrease for exploratory work with smaller populations.

**Removal Percentage Range** (line 29 in `oxford_GLM_draw.m`):
```matlab
removal_percentages = 0:5:90;  % Test every 5% from 0% to 90%
```

You can modify this to test different removal schedules. For example, `0:10:90` would provide a coarser analysis with fewer computation time, while `0:2:90` would give finer resolution at the cost of increased processing time.

**Random Iteration Count** (line 30 in `oxford_GLM_draw.m`):
```matlab
n_random_iterations = 10;  % Monte Carlo samples
```

Increasing this value improves the stability of random removal estimates but increases computation time linearly. The default of 10 provides good balance; values above 20 typically show diminishing returns in estimate stability.

---

## Expected Outputs

### Stage 1 Outputs

**`all_glm_coefficients.mat`**

This file contains the complete GLM analysis organized hierarchically:

```matlab
all_glm_results
├── component_analyzed: 1
├── analysis_timestamp: '17-Nov-2025 14:23:45'
├── region_pairs: {'MOp_MOs', 'MOp_OLF', ...}
└── MOp_MOs (and other pairs)
    ├── region1: 'MOp'
    ├── region2: 'MOs'
    └── sessions: {1×N cell}
        └── sessions{1}
            ├── session_name: 'yp013_220211'
            ├── region1_coefficients: [50×1 double]
            │   └── β values for each neuron
            ├── region1_stats
            │   ├── r_squared: 0.4523
            │   ├── adjusted_r_squared: 0.4487
            │   ├── p_values: [50×1 double]
            │   ├── t_statistics: [50×1 double]
            │   ├── standard_errors: [50×1 double]
            │   ├── ci_lower: [50×1 double]
            │   ├── ci_upper: [50×1 double]
            │   ├── significant_neurons: [1×K double]
            │   ├── n_significant: K
            │   ├── aic: scalar
            │   └── bic: scalar
            └── region2_coefficients/stats: (similar structure)
```

**Key Variables Explained**:

- `r_squared`: The proportion of CCA latent variance explained by the neural population. Values typically range from 0.2 to 0.8. Higher values indicate that the neural activity pattern strongly aligns with the canonical correlation structure.

- `adjusted_r_squared`: R² adjusted for the number of parameters (neurons) in the model. This penalizes model complexity and provides a more conservative estimate when comparing models with different neuron counts.

- `p_values`: Statistical significance for each neuron's contribution. Values below 0.05 (with appropriate multiple comparison correction) indicate neurons that contribute significantly above chance.

- `significant_neurons`: Indices of neurons with p < 0.05, representing the subset of the population that makes statistically reliable contributions to the canonical pattern.

**`glm_analysis_summary.txt`**

Human-readable report containing:

```
Oxford Dataset: GLM Analysis Summary Report
==========================================

Analysis Date: 17-Nov-2025
CCA Component Analyzed: 1

Total region pairs analyzed: 15

Region Pair: MOp vs MOs
  Sessions analyzed: 8
  GLM R² - MOp: 0.453 ± 0.087 (range: 0.325-0.589)
  GLM R² - MOs: 0.412 ± 0.092 (range: 0.287-0.543)
  CCA R² (Component 1): 0.524 ± 0.045
  Significant neurons - MOp: 23.4 ± 5.2 (46.8%)
  Significant neurons - MOs: 21.7 ± 4.8 (43.4%)

[... additional pairs ...]

OVERALL SUMMARY ACROSS ALL REGION PAIRS
=======================================
Average GLM R²: 0.428 ± 0.095
Average CCA R²: 0.498 ± 0.067
```

This summary facilitates quick assessment of analysis quality across the dataset. The range values help identify outlier sessions that may require further investigation.

### Stage 2 Outputs

**`sensitivity_{Region1}_{Region2}.mat`**

Contains degradation curves for systematic neuron removal:

```matlab
sensitivity_results
├── region1: 'MOp'
├── region2: 'MOs'
├── component: 1
├── removal_percentages: [0, 5, 10, ..., 90]
├── n_random_iterations: 10
└── sessions: {1×N cell}
    └── sessions{1}
        ├── session_name: 'yp013_220211'
        ├── region1_toprank: [1×19 double]
        │   └── R² values at each removal percentage (top-ranked)
        ├── region1_random: [19×10 double]
        │   └── R² values (removal% × iterations)
        ├── region2_toprank: [1×19 double]
        ├── region2_random: [19×10 double]
        ├── original_r2_region1: 0.453
        └── cca_R2: 0.524
```

**Data Interpretation**:

The degradation curves reveal how reconstruction quality deteriorates as neurons are removed. For example, if `region1_toprank(5) = 0.42` and `region1_toprank(11) = 0.15`, this indicates that R² drops from 0.42 (at 20% removal) to 0.15 (at 50% removal), showing substantial information loss with targeted neuron removal.

**`sensitivity_{Region1}_{Region2}.png`**

Two-panel visualization showing degradation curves for both regions. Each panel displays:

- Blue solid line: Mean top-ranked removal curve across sessions
- Blue shaded region: ±1 SEM for top-ranked removal
- Red solid line: Mean random removal curve across sessions  
- Red shaded region: ±1 SEM for random removal

The divergence between blue and red curves quantifies encoding concentration. Rapid blue curve decline relative to red indicates sparse coding; parallel curves suggest distributed coding.

**`cross_pair_concentration_summary.png`**

Bar plot comparing concentration indices across all region pairs, facilitating identification of which brain region pairs employ more concentrated versus distributed coding strategies. This comparative visualization can reveal organizing principles in how different regions encode shared information.

---

## Methodological Notes

### Statistical Considerations

**Sample Size Requirements**:

The analysis requires adequate statistical power at multiple levels:

1. **Within-Session**: Sufficient trials (typically ≥50) to compute stable PSTH estimates and reliable cross-validation in CCA.

2. **Within-Region**: Adequate neurons (≥30 recommended) to fit stable GLM coefficients without overfitting. With fewer neurons, the GLM may perfectly fit noise rather than signal.

3. **Across-Sessions**: Multiple sessions (≥5) per region pair to compute reliable means and standard errors for population-level conclusions.

**Multiple Comparisons**:

When analyzing multiple region pairs or components, consider appropriate corrections for multiple testing:

- **Bonferroni Correction**: Divide α by the number of tests. Conservative but maintains family-wise error rate.
  
- **False Discovery Rate (FDR)**: Controls the expected proportion of false positives among significant results. Less conservative than Bonferroni, appropriate for exploratory analyses.

- **Permutation Testing**: For small sample sizes, empirical null distributions via permutation may provide more accurate p-values than asymptotic approximations.

### Interpretation Guidelines

**GLM R² Values**:

The relationship between GLM R² and CCA R² provides important context:

- **High GLM R², High CCA R²**: Strong canonical correlation that is well-explained by measurable neural activity. This is the ideal scenario indicating robust inter-regional communication that we can decompose into individual neuronal contributions.

- **High GLM R², Low CCA R²**: Neural activity explains the (weak) canonical pattern well, but the overall inter-regional correlation is modest. This may indicate that while neurons align with the CCA solution, the CCA itself captures limited shared variance.

- **Low GLM R², High CCA R²**: Strong canonical correlation exists, but our neural measurements explain it poorly. This could indicate that unmeasured factors (e.g., unrecorded neurons, network state variables) contribute substantially to the canonical pattern.

- **Low GLM R², Low CCA R²**: Both inter-regional correlation and neural explanation are weak. This may suggest that these regions do not strongly coordinate during the analyzed task epoch.

**Concentration Index Interpretation**:

The concentration index must be interpreted in the context of the neural population's properties:

1. **Anatomical Considerations**: Cortical layer distribution affects concentration. Layer 5 pyramidal neurons, for example, may show more concentrated coding due to their role as primary output neurons.

2. **Temporal Factors**: Encoding concentration may vary across the trial. Early sensory responses might show sparse coding (few neurons respond strongly), while later motor preparation might show distributed coding.

3. **Task Demands**: More difficult or uncertain tasks may recruit broader populations, reducing concentration indices.

### Computational Notes

**Memory Management**:

The session-based architecture reduces memory overhead compared to loading entire datasets, but large sessions with many regions still require substantial RAM. If encountering memory issues:

1. Process sessions sequentially rather than loading multiple sessions simultaneously
2. Clear variables explicitly after processing each session using `clear session_data`
3. Consider processing region pairs in batches if analyzing many pairs

**Parallel Processing Potential**:

The analysis is embarrassingly parallel at the session level. Each session can be processed independently, making this suitable for cluster computing environments. To implement parallel processing:

```matlab
parpool(n_workers);  % Initialize parallel pool
parfor session_idx = 1:length(sessions)
    % Process session_idx
end
```

This can substantially reduce wall-clock time for large datasets.

### Validation and Quality Control

**Sanity Checks**:

Before interpreting results, verify:

1. **Dimensional Consistency**: All arrays have expected dimensions. Mismatches often indicate preprocessing errors.

2. **Numerical Stability**: No NaN or Inf values in results. These indicate numerical issues in GLM fitting (often from rank-deficient design matrices).

3. **Coefficient Magnitudes**: GLM coefficients should have reasonable scales. Extremely large coefficients (|β| > 100) may indicate multicollinearity or scaling issues.

4. **R² Sanity**: R² values should fall between 0 and 1. Values outside this range indicate computational errors.

**Cross-Validation Consistency**:

Compare GLM R² with CCA cross-validation R². Large discrepancies warrant investigation:
- GLM R² substantially exceeds CCA R² → Possible overfitting in GLM
- CCA R² substantially exceeds GLM R² → Unmeasured variance sources or suboptimal neuron selection

---

## Troubleshooting Common Issues

### Error: "Dimension mismatch between latent and neural data"

**Cause**: The number of samples in the latent variable doesn't match the neural data matrix.

**Solution**:
1. Verify trial counts match between CCA and PSTH files
2. Check for trial exclusion in preprocessing steps
3. Ensure consistent time window definitions

The scripts include automatic reshaping attempts, but if problems persist, manually verify the trial structure in your data.

### Error: "GLM coefficients contain NaN values"

**Cause**: Rank-deficient design matrix, often from perfect multicollinearity between neurons or insufficient samples.

**Solution**:
1. Increase minimum neuron threshold
2. Check for duplicate neurons in selection
3. Add regularization to the GLM (Ridge regression: `glmfit(..., 'weights', ridge_parameter)`)
4. Verify sufficient trials relative to neuron count (rule of thumb: ≥10 trials per neuron)

### Warning: "Low GLM R² values"

**Cause**: Neural activity poorly explains the CCA latent variable.

**Possible Reasons**:
1. Selected component is not significant (check `pair_result.significant_components`)
2. Neuron sampling doesn't match CCA preprocessing
3. Temporal misalignment between PSTH and CCA
4. The canonical pattern genuinely reflects unmeasured variance sources

**Investigation Steps**:
1. Verify you're analyzing a significant CCA component
2. Confirm selected neurons match those used in CCA
3. Plot neural activity and latent variable to visually assess relationship
4. Check CCA R² to see if the canonical correlation itself is strong

### Issue: "Flat sensitivity curves with no degradation"

**Symptom**: Both top-ranked and random removal show minimal R² decline, even at high removal percentages.

**Cause**: Severe overfitting in the original GLM, where the model has learned noise rather than signal.

**Solutions**:
1. Implement regularization (L1/L2 penalties) in the GLM
2. Use fewer neurons or more trials to improve sample-to-parameter ratio
3. Apply cross-validation to GLM fitting itself (split trials into train/test sets)
4. Check if the phenomenon persists across multiple sessions (may indicate systematic issue vs. session-specific noise)

---

## Extending the Analysis

### Analyzing Additional Components

To examine whether encoding strategies vary across CCA components:

```matlab
components_to_analyze = 1:3;  % Analyze first 3 components

for comp = components_to_analyze
    % Modify component_to_analyze variable in both scripts
    component_to_analyze = comp;
    
    % Run analysis
    oxford_glm_cca_coefficients_extraction();
    oxford_glm_sensitivity_analysis();
end

% Compare concentration indices across components
```

This analysis can reveal whether early CCA components (high canonical correlation) employ different coding strategies than later components (lower canonical correlation).

### Temporal Dynamics Analysis

To examine how encoding concentration evolves across trial time:

```matlab
time_windows = {[-1.5, -0.5], [-0.5, 0.5], [0.5, 1.5], [1.5, 3.0]};

for tw = 1:length(time_windows)
    % Extract PSTH and latent variables for specific time window
    % Run GLM analysis on time-windowed data
    % Compare concentration indices across windows
end
```

This can reveal whether information transitions from concentrated (early sensory) to distributed (later motor) representations.

### Cross-Region Comparative Analysis

To systematically compare coding strategies across anatomical systems:

```matlab
% Group region pairs by anatomical category
sensory_pairs = {'V1_V2', 'A1_A2'};
motor_pairs = {'MOp_MOs', 'MOp_STR'};
association_pairs = {'PFC_PPC', 'OFC_ACC'};

% Compare concentration indices across categories
% Test for significant differences using ANOVA or non-parametric tests
```

This analysis can identify organizing principles: do sensory areas systematically employ different coding strategies than motor or association areas?

---

## References and Further Reading

For deeper understanding of the methodological foundations:

**Canonical Correlation Analysis in Neuroscience**:
- Semedo, J. D., et al. (2019). "Cortical areas interact through a communication subspace." *Neuron*, 102(1), 249-259.
- Gallego, J. A., et al. (2020). "Long-term stability of cortical population dynamics." *Nature Neuroscience*, 23(2), 260-270.

**Neural Population Coding**:
- Jazayeri, M., & Afraz, A. (2017). "Navigating the neural space in search of the neural code." *Neuron*, 93(5), 1003-1014.
- Cunningham, J. P., & Yu, B. M. (2014). "Dimensionality reduction for large-scale neural recordings." *Nature Neuroscience*, 17(11), 1500-1509.

**General Linear Models for Neural Data**:
- Pillow, J. W., et al. (2008). "Spatio-temporal correlations and visual signalling in a complete neuronal population." *Nature*, 454(7207), 995-999.
- Truccolo, W., et al. (2005). "A point process framework for relating neural spiking activity to spiking history, neural ensemble, and extrinsic covariate effects." *Journal of Neurophysiology*, 93(2), 1074-1089.

---

## Contact and Support

For questions, issues, or suggestions regarding this analysis pipeline:

1. Consult the inline documentation within each function
2. Review the demonstration script for usage examples
3. Check the troubleshooting section for common issues

The code includes extensive comments explaining both the computational implementation and the neuroscientific rationale, designed to facilitate understanding and modification for your specific research questions.

---

## Version History

**Version 1.0 (November 2025)**
- Initial adaptation for Oxford dataset structure
- Session-based processing architecture
- Comprehensive documentation and demonstration scripts
- Validated on multi-session Oxford recordings

---

## License and Citation

If you use these analysis tools in your research, please cite appropriately and acknowledge the methodological foundations described in the references above.

---

*End of Documentation*

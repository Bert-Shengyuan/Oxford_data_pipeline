#!/usr/bin/env python3
"""
Cross-Trial-Type PCA Analysis - Batch Processing Script
========================================================

This script demonstrates how to use the CrossTrialTypePCAAnalyzer for
multi-session batch analysis with cross-session aggregation.

Key Features:
-------------
1. Load neural data from multiple trial types (cued_hit_long, spont_hit_long, spont_miss_long)
2. Extract PCA weights from the reference condition (cued_hit_long)
3. Project neural activity from all conditions onto the same PCA subspace
4. Aggregate results across sessions for each region INDEPENDENTLY
5. Create summary figures following anatomical ordering

CRITICAL DIFFERENCE FROM CCA:
-----------------------------
Unlike CCA (which requires paired recordings), PCA aggregates each region
independently across ALL sessions that recorded it, regardless of what
other regions were co-recorded. For example:

- mPFC: aggregated from sessions {s1, s2, s5, s7} where mPFC was recorded
- HY: aggregated from sessions {s3, s5, s8} where HY was recorded

At position (mPFC, HY) in the matrix:
- Row shows mPFC data from all 4 sessions recording mPFC
- Column shows HY data from all 3 sessions recording HY

The session sets need NOT overlap.

Mathematical Framework:
-----------------------
The cross-session aggregated projection for region R is:
$$\\bar{\\mathbf{z}}_{c,\\text{pop}} = \\frac{1}{N_R} \\sum_{s \\in S_R} \\bar{\\mathbf{z}}_{c,s}$$

where $S_R$ is the set of all sessions recording region R.

Usage:
------
python run_cross_trial_type_pca_analysis.py

Configuration can be modified in the create_analysis_config() function.

Author: Oxford Neural Analysis Pipeline
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cross_trial_type_pca_analysis import (
    CrossTrialTypePCAAnalyzer,
    CrossTrialTypePCAPipeline,
    CrossSessionPCAAnalyzer,
    CrossTrialTypePCASummaryVisualizer,
    TRIAL_TYPES,
    TRIAL_TYPE_COLORS,
    ANATOMICAL_ORDER,
    HIERARCHICAL_GROUPING,
    HIERARCHICAL_ORDER,
    MIN_SESSIONS_THRESHOLD
)


def create_analysis_config() -> dict:
    """
    Create configuration for cross-trial-type PCA analysis.

    This function defines all parameters for the analysis pipeline.
    Modify this function to customize the analysis for your needs.

    Returns:
        Configuration dictionary with the following keys:
        - base_dir: Root directory containing all session results
        - sessions: List of session names to analyze
        - reference_type: Trial type for PCA weight extraction
        - n_components: Number of PCA components to analyze
        - output_base_dir: Base directory for output files
        - enable_cross_session: Whether to perform cross-session aggregation
        - min_sessions: Minimum sessions for cross-session analysis
    """
    config = {
        # Base directory containing all session results
        'base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset',

        # Sessions to analyze - full list for comprehensive batch analysis
        'sessions': [
            'yp010_220209',
            'yp010_220210',
            'yp010_220211',
            'yp010_220212',
            'yp012_220208',
            'yp012_220209',
            'yp012_220210',
            'yp012_220211',
            'yp012_220212',
            'yp013_220209',
            'yp013_220210',
            'yp013_220211',
            'yp013_220212',
            'yp014_220208',
            'yp014_220209',
            'yp014_220210',
            'yp014_220211',
            'yp014_220212',
            'yp020_220331',
            'yp020_220401',
            'yp020_220402',
            'yp020_220403',
            'yp020_220404',
            'yp020_220405',
            'yp020_220407',
            'yp021_220331',
            'yp021_220401',
            'yp021_220402',
            'yp021_220403',
            'yp021_220404',
            'yp021_220405',
            'yp021_220407',
        ],

        # Reference trial type - PCA weights are extracted from this condition
        'reference_type': 'cued_hit_long',

        # Number of PCA components to analyze
        'n_components': 5,

        # Output directory (will be created if doesn't exist)
        'output_base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/cross_trial_type_pca',

        # Enable cross-session aggregation
        # Note: For PCA, each region is aggregated INDEPENDENTLY
        'enable_cross_session': True,

        # Minimum number of sessions required for cross-session analysis
        # Applied per-region (not per-pair like CCA)
        'min_sessions': MIN_SESSIONS_THRESHOLD,

        # Enable hierarchical aggregation: when True, after region-level analysis,
        # aggregates regions into broader categories (e.g., STR+STRv -> Striatum,
        # MD+VALVM+LP+VPMPO -> Thalamus) and produces additional summary figures
        # with 9 hierarchical regions instead of 13 individual regions.
        # When False, only the original region-level analysis is performed.
        'use_hierarchical': True
    }

    return config


def create_single_session_config() -> dict:
    """
    Create configuration for single session demo analysis.

    Returns:
        Configuration dictionary for single session analysis
    """
    config = {
        'base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset',
        'sessions': ['yp021_220405'],  # Single session for demo
        'reference_type': 'cued_hit_long',
        'n_components': 5,
        'output_base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/cross_trial_type_pca',
        'enable_cross_session': False,
        'min_sessions': 1,
        'use_hierarchical': False
    }

    return config


def run_single_session_demo(config: dict) -> CrossTrialTypePCAAnalyzer:
    """
    Run detailed analysis for a single session.

    This function provides step-by-step execution with verbose output,
    useful for understanding the analysis pipeline.

    Parameters:
        config: Analysis configuration dictionary

    Returns:
        Configured analyzer object with results
    """
    session_name = config['sessions'][0]

    print("=" * 70)
    print("CROSS-TRIAL-TYPE PCA ANALYSIS - SINGLE SESSION DEMO")
    print("=" * 70)
    print(f"\nSession: {session_name}")
    print(f"Reference type: {config['reference_type']}")
    print(f"Components: {config['n_components']}")

    # Initialize analyzer
    analyzer = CrossTrialTypePCAAnalyzer(
        base_dir=config['base_dir'],
        session_name=session_name,
        reference_type=config['reference_type'],
        n_components=config['n_components']
    )

    # Step 1: Load all trial types
    print("\n" + "=" * 50)
    print("STEP 1: Loading Data from All Trial Types")
    print("=" * 50)

    if not analyzer.load_all_trial_types():
        print("ERROR: Failed to load trial types")
        return None

    print(f"\nLoaded trial types: {analyzer.available_trial_types}")
    print(f"Available regions: {analyzer.available_regions}")

    # Step 2: Analyze each region independently
    for region in analyzer.available_regions:
        print("\n" + "=" * 50)
        print(f"ANALYZING REGION: {region}")
        print("=" * 50)

        # Step 2a: Extract neural data
        print("\n--- Extracting Neural Data ---")
        if not analyzer.extract_neural_data_for_region(region):
            print(f"  Skipping region - insufficient data")
            continue

        # Step 2b: Extract PCA weights from reference
        print("\n--- Extracting PCA Weights ---")
        if not analyzer.extract_pca_weights(region):
            print(f"  Skipping region - PCA weights not available")
            continue

        # Step 2c: Compute projections across all trial types
        print("\n--- Computing Cross-Trial-Type Projections ---")
        if not analyzer.compute_projections_for_region(region):
            print(f"  Skipping region - projection failed")
            continue

        # Step 2d: Compute statistical comparisons
        print("\n--- Computing Statistics ---")
        analyzer.compute_statistics_for_region(region)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return analyzer


def run_batch_analysis(config: dict) -> CrossTrialTypePCAPipeline:
    """
    Run analysis for multiple sessions using the pipeline.

    This function performs comprehensive batch analysis including:
    1. Per-session analysis for all sessions
    2. Cross-session aggregation for regions with ≥min_sessions
    3. Summary figure generation with upper-triangle format

    Note: For PCA, cross-session aggregation is performed INDEPENDENTLY
    for each region, unlike CCA which requires paired regions.

    Parameters:
        config: Analysis configuration dictionary

    Returns:
        Pipeline object with all results
    """
    print("=" * 70)
    print("CROSS-TRIAL-TYPE PCA ANALYSIS - BATCH PROCESSING")
    print("=" * 70)
    print("\nIMPORTANT: PCA aggregates each region INDEPENDENTLY")
    print("Sessions need NOT include both regions simultaneously")

    pipeline = CrossTrialTypePCAPipeline(config)

    # Run all session analyses
    results = pipeline.run_all_sessions()

    # Run cross-session aggregation if enabled
    if config.get('enable_cross_session', True):
        pipeline.run_cross_session_aggregation()

    return pipeline


def create_aggregate_summary_figure(
    pipeline: CrossTrialTypePCAPipeline,
    output_dir: Path
) -> plt.Figure:
    """
    Create aggregate summary figure across all analyzed sessions.

    This figure shows peak amplitudes and temporal correlations
    aggregated across all sessions per region.

    Parameters:
        pipeline: Completed pipeline with analyzers
        output_dir: Directory to save figure

    Returns:
        Matplotlib figure object
    """
    if not pipeline.analyzers:
        print("No analyzers available for aggregate summary")
        return None

    # Collect statistics across sessions per region
    region_peak_diffs = {}
    region_correlations = {}

    for session_name, analyzer in pipeline.analyzers.items():
        for region, stats in analyzer.statistical_results.items():
            if region not in region_peak_diffs:
                region_peak_diffs[region] = {tt: [] for tt in TRIAL_TYPES.keys()}
                region_correlations[region] = {}

            # Collect peak amplitudes
            peaks = stats.get('peak_amplitudes', {})
            for trial_type, peak_values in peaks.items():
                if peak_values:
                    region_peak_diffs[region][trial_type].extend(peak_values)

            # Collect temporal correlations
            correlations = stats.get('temporal_correlations', {})
            for comp_key, corr_list in correlations.items():
                if comp_key not in region_correlations[region]:
                    region_correlations[region][comp_key] = []
                for corr in corr_list:
                    region_correlations[region][comp_key].append(corr['r2'])

    # Create aggregate figure
    n_regions = len(region_peak_diffs)
    if n_regions == 0:
        print("No regions with statistics available")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fontsize = 12

    # Plot 1: Peak amplitudes by region and trial type
    ax1 = axes[0]
    regions = list(region_peak_diffs.keys())
    n_trial_types = len(TRIAL_TYPES)
    x = np.arange(len(regions))
    width = 0.25

    for idx, (trial_type, color) in enumerate(TRIAL_TYPE_COLORS.items()):
        means = []
        stds = []
        for region in regions:
            values = region_peak_diffs[region].get(trial_type, [])
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(0)
                stds.append(0)

        offset = (idx - n_trial_types / 2 + 0.5) * width
        ax1.bar(x + offset, means, width, yerr=stds, label=trial_type.replace('_', ' '),
                color=color, alpha=0.8, capsize=3, edgecolor='black')

    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, fontsize=fontsize - 2, rotation=45, ha='right')
    ax1.set_ylabel('Peak Amplitude (mean ± std)', fontsize=fontsize)
    ax1.set_title('Peak Amplitudes Across Regions and Trial Types', fontsize=fontsize + 1)
    ax1.legend(loc='upper right', fontsize=fontsize - 2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot 2: Temporal correlations by region (stacked for comparisons)
    ax2 = axes[1]

    # Get all comparison keys
    all_comp_keys = set()
    for region_corr in region_correlations.values():
        all_comp_keys |= set(region_corr.keys())

    if all_comp_keys:
        comparison_keys = sorted(list(all_comp_keys))
        n_comparisons = len(comparison_keys)
        width = 0.8 / max(1, n_comparisons)

        for idx, comp_key in enumerate(comparison_keys):
            means = []
            stds = []
            for region in regions:
                values = region_correlations[region].get(comp_key, [])
                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(0)
                    stds.append(0)

            offset = (idx - n_comparisons / 2 + 0.5) * width
            label = comp_key.replace('cued_hit_long_vs_', 'vs ').replace('_long', '')
            ax2.bar(x + offset, means, width, yerr=stds, label=label,
                    alpha=0.8, capsize=3, edgecolor='black')

        ax2.set_xticks(x)
        ax2.set_xticklabels(regions, fontsize=fontsize - 2, rotation=45, ha='right')
        ax2.set_ylabel('Temporal R² (mean ± std)', fontsize=fontsize)
        ax2.set_title('Temporal Correlation with Reference', fontsize=fontsize + 1)
        ax2.set_ylim([0, 1])
        ax2.legend(loc='upper right', fontsize=fontsize - 2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    plt.suptitle(
        f'Aggregate Summary Across {len(pipeline.analyzers)} Sessions\n'
        f'(Each region aggregated independently)',
        fontsize=fontsize + 2, fontweight='bold'
    )

    plt.tight_layout()

    output_path = output_dir / 'pca_aggregate_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregate summary: {output_path}")

    return fig


def print_analysis_summary(analyzer: CrossTrialTypePCAAnalyzer) -> None:
    """
    Print detailed summary of analysis results.

    Parameters:
        analyzer: Completed analyzer object
    """
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nSession: {analyzer.session_name}")
    print(f"Reference type: {analyzer.reference_type}")
    print(f"Trial types analyzed: {', '.join(analyzer.available_trial_types)}")
    print(f"Regions analyzed: {', '.join(analyzer.available_regions)}")

    for region, stats in analyzer.statistical_results.items():
        print(f"\n--- {region} ---")

        if 'peak_amplitudes' in stats:
            print("  Peak Amplitudes (PC1):")
            for tt, peaks in stats['peak_amplitudes'].items():
                if peaks:
                    print(f"    {tt}: {peaks[0]:.4f}")

        if 'temporal_correlations' in stats:
            print("  Temporal Correlations:")
            for comp_key, corr_list in stats['temporal_correlations'].items():
                if corr_list:
                    print(f"    {comp_key}:")
                    for idx, corr in enumerate(corr_list[:3]):
                        print(f"      PC{idx+1}: R²={corr['r2']:.4f}")


def print_cross_session_summary(pipeline: CrossTrialTypePCAPipeline) -> None:
    """
    Print summary of cross-session analysis results.

    Parameters:
        pipeline: Completed pipeline with cross-session results
    """
    print("\n" + "=" * 70)
    print("CROSS-SESSION ANALYSIS SUMMARY")
    print("=" * 70)
    print("\nNote: Each region is aggregated INDEPENDENTLY across all")
    print("      sessions that recorded it (unlike CCA paired analysis)")

    min_sessions = pipeline.config.get('min_sessions', MIN_SESSIONS_THRESHOLD)

    print(f"\nMinimum sessions threshold: {min_sessions}")
    print(f"Total regions analyzed: {len(pipeline.cross_session_analyzers)}")

    # Count regions meeting threshold
    valid_regions = []
    insufficient_regions = []

    for region, cs_analyzer in pipeline.cross_session_analyzers.items():
        n_sessions = len(cs_analyzer.session_projections)
        if n_sessions >= min_sessions:
            valid_regions.append((region, n_sessions))
        else:
            insufficient_regions.append((region, n_sessions))

    print(f"\nRegions with ≥{min_sessions} sessions (valid for cross-session analysis):")
    for region, n in sorted(valid_regions, key=lambda x: x[1], reverse=True):
        print(f"  {region:8s}: {n:3d} sessions")

    if insufficient_regions:
        print(f"\nRegions with <{min_sessions} sessions (insufficient data):")
        for region, n in sorted(insufficient_regions, key=lambda x: x[1], reverse=True):
            print(f"  {region:8s}: {n:3d} sessions")

    # Summary statistics
    if valid_regions:
        print(f"\nSummary:")
        print(f"  Valid regions for cross-session analysis: {len(valid_regions)}")
        print(f"  Total sessions across valid regions: {sum(n for _, n in valid_regions)}")

        if pipeline.summary_visualizer:
            output_dir = pipeline.summary_visualizer.output_dir
            print(f"\nSummary figures saved to: {output_dir}")


def main():
    """Main entry point for cross-trial-type PCA analysis."""

    # Get configuration
    config = create_analysis_config()

    # Run analysis mode selection
    print("\n" + "=" * 70)
    print("CROSS-TRIAL-TYPE PCA ANALYSIS")
    print("=" * 70)
    print("\nThis analysis projects neural activity from different trial types")
    print("onto PCA subspaces trained on a reference condition.")
    print("\nIMPORTANT DIFFERENCE FROM CCA:")
    print("  - CCA requires paired recordings of both regions")
    print("  - PCA aggregates each region INDEPENDENTLY")
    print("  - Sessions need NOT include multiple regions simultaneously")
    print(f"\nReference condition: {config['reference_type']}")
    print(f"Comparison conditions: spont_hit_long, spont_miss_long")
    print(f"Minimum sessions for cross-session analysis: {config.get('min_sessions', MIN_SESSIONS_THRESHOLD)}")

    # Option 1: Single session detailed demo
    if len(config['sessions']) == 1:
        print("\nRunning single session analysis...")
        analyzer = run_single_session_demo(config)

        if analyzer:
            print_analysis_summary(analyzer)

    # Option 2: Batch processing multiple sessions
    else:
        print(f"\nRunning batch analysis for {len(config['sessions'])} sessions...")
        pipeline = run_batch_analysis(config)

        # Print cross-session summary
        if config.get('enable_cross_session', True):
            print_cross_session_summary(pipeline)

        # Create legacy aggregate summary (for backward compatibility)
        output_dir = Path(config.get('output_base_dir', '.'))
        output_dir.mkdir(parents=True, exist_ok=True)
        create_aggregate_summary_figure(pipeline, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

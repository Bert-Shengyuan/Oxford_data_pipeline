#!/usr/bin/env python3
"""
Cross-Trial-Type CCA Analysis - Batch Processing Script
========================================================

This script demonstrates how to use the CrossTrialTypeCCAAnalyzer for
multi-session batch analysis with cross-session aggregation.

Key Features:
-------------
1. Load neural data from multiple trial types (cued_hit_long, spont_hit_long, spont_miss_long)
2. Extract CCA weights from the reference condition (cued_hit_long)
3. Project neural activity from all conditions onto the same CCA subspace
4. Aggregate results across sessions for each region pair
5. Create summary figures following anatomical ordering

Cross-Session Aggregation:
--------------------------
For region pairs with ≥5 sessions, the pipeline:
- Computes session-averaged projections
- Aggregates across sessions with mean ± SEM
- Creates boxplots of temporal R² distributions
- Generates upper-triangle summary matrices

Mathematical Framework:
-----------------------
The cross-session aggregated projection is:
$$\bar{\mathbf{u}}_{c,\text{pop}} = \frac{1}{N_s} \sum_{s=1}^{N_s} \bar{\mathbf{u}}_{c,s}$$

with cross-session SEM:
$$\text{SEM}_{\text{pop}} = \frac{\sigma_{\text{sessions}}}{\sqrt{N_s}}$$

Usage:
------
python run_cross_trial_type_analysis.py

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

from cross_trial_type_cca_analysis import (
    CrossTrialTypeCCAAnalyzer,
    CrossTrialTypeCCAPipeline,
    CrossSessionCCAAnalyzer,
    CrossTrialTypeSummaryVisualizer,
    TRIAL_TYPES,
    TRIAL_TYPE_COLORS,
    ANATOMICAL_ORDER,
    MIN_SESSIONS_THRESHOLD,
    sort_pair_by_anatomy
)


def create_analysis_config() -> dict:
    """
    Create configuration for cross-trial-type CCA analysis.
    
    This function defines all parameters for the analysis pipeline.
    Modify this function to customize the analysis for your needs.
    
    Returns:
        Configuration dictionary with the following keys:
        - base_dir: Root directory containing all session results
        - sessions: List of session names to analyze
        - reference_type: Trial type for CCA weight training
        - n_components: Number of CCA components to analyze
        - region_pairs: List of region pairs to analyze (None for auto-detect)
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
        # 'sessions': [
        #     'yp010_220209',
        #     'yp010_220210',
        #     'yp010_220211',
        #     'yp010_220212',
        #     'yp012_220208',
        #     'yp012_220209',
        #     'yp012_220210',
        # ],
        # Reference trial type - CCA weights are extracted from this condition
        'reference_type': 'cued_hit_long',
        
        # Number of CCA components to analyze
        'n_components': 3,
        
        # Region pairs to analyze (set to None for auto-detection from each session)
        # Using anatomical ordering for consistent visualization
        'region_pairs': None,  # Auto-detect all available pairs
        
        # Output directory (will be created if doesn't exist)
        'output_base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/cross_trial_type_cca',
        
        # Enable cross-session aggregation for pairs with sufficient sessions
        'enable_cross_session': True,
        
        # Minimum number of sessions required for cross-session analysis
        'min_sessions': MIN_SESSIONS_THRESHOLD
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
        'region_pairs': [
            ('mPFC', 'STR'),
            ('MOp', 'STR'),
            ('MOs', 'STR'),
            ('ORB', 'STR'),
            ('MOp', 'MOs'),
        ],
        'output_base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/cross_trial_type_cca',
        'enable_cross_session': False,
        'min_sessions': 1
    }
    
    return config


def run_single_session_demo(config: dict) -> CrossTrialTypeCCAAnalyzer:
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
    print("CROSS-TRIAL-TYPE CCA ANALYSIS - SINGLE SESSION DEMO")
    print("=" * 70)
    print(f"\nSession: {session_name}")
    print(f"Reference type: {config['reference_type']}")
    print(f"Components: {config['n_components']}")
    
    # Initialize analyzer
    analyzer = CrossTrialTypeCCAAnalyzer(
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
    print(f"Available pairs: {len(analyzer.available_pairs)}")
    
    # Step 2: Analyze each region pair
    region_pairs = config.get('region_pairs')
    if region_pairs is None:
        region_pairs = [(r1, r2) for r1, r2, _ in analyzer.available_pairs]
    
    for region_pair in region_pairs:
        region_i, region_j = region_pair
        
        print("\n" + "=" * 50)
        print(f"ANALYZING REGION PAIR: {region_i} vs {region_j}")
        print("=" * 50)
        
        # Step 2a: Extract neural data
        print("\n--- Extracting Neural Data ---")
        if not analyzer.extract_neural_data(region_pair):
            print(f"  Skipping pair - insufficient data")
            continue
        
        # Step 2b: Extract CCA weights from reference
        print("\n--- Extracting CCA Weights ---")
        if not analyzer.extract_cca_weights(region_pair):
            print(f"  Skipping pair - CCA weights not available")
            continue
        
        # Step 2c: Compute projections across all trial types
        print("\n--- Computing Cross-Trial-Type Projections ---")
        if not analyzer.compute_projections():
            print(f"  Skipping pair - projection failed")
            continue
        
        # Step 2d: Compute statistical comparisons
        print("\n--- Computing Statistics ---")
        stats = analyzer.compute_statistics()
        
        # Step 2e: Create visualizations
        print("\n--- Creating Visualizations ---")
        
        # Main comparison figure
        analyzer.create_projection_comparison_figure(region_pair)
        
        # Statistical summary figure
        analyzer.create_statistical_summary_figure(region_pair)
        
        # Detailed temporal figures for each component
        for comp_idx in range(min(3, analyzer.n_components)):
            analyzer.create_detailed_temporal_figure(region_pair, comp_idx)
        
        # Step 2f: Generate reports
        print("\n--- Generating Reports ---")
        analyzer.generate_summary_report(region_pair)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {analyzer.output_dir}")
    
    return analyzer


def run_batch_analysis(config: dict) -> CrossTrialTypeCCAPipeline:
    """
    Run analysis for multiple sessions using the pipeline.
    
    This function performs comprehensive batch analysis including:
    1. Per-session analysis for all sessions
    2. Cross-session aggregation for region pairs with ≥min_sessions
    3. Summary figure generation with upper-triangle format
    
    Parameters:
        config: Analysis configuration dictionary
        
    Returns:
        Pipeline object with all results
    """
    print("=" * 70)
    print("CROSS-TRIAL-TYPE CCA ANALYSIS - BATCH PROCESSING")
    print("=" * 70)
    
    pipeline = CrossTrialTypeCCAPipeline(config)
    
    # Run all session analyses
    results = pipeline.run_all_sessions()
    
    # Run cross-session aggregation if enabled
    if config.get('enable_cross_session', True):
        pipeline.run_cross_session_aggregation()
    
    return pipeline


def create_aggregate_summary_figure(
    pipeline: CrossTrialTypeCCAPipeline,
    output_dir: Path
) -> plt.Figure:
    """
    Create aggregate summary figure across all analyzed sessions.
    
    This figure shows peak amplitudes and temporal correlations
    aggregated across all sessions and region pairs.
    
    Parameters:
        pipeline: Completed pipeline with analyzers
        output_dir: Directory to save figure
        
    Returns:
        Matplotlib figure object
    """
    if not pipeline.analyzers:
        print("No analyzers available for aggregate summary")
        return None
    
    # Collect statistics across sessions
    all_peak_diffs = {}
    all_correlations = {}
    
    for session_name, analyzer in pipeline.analyzers.items():
        if not analyzer.statistical_results:
            continue
        
        # Collect peak amplitude differences
        peaks = analyzer.statistical_results.get('peak_amplitudes', {})
        correlations = analyzer.statistical_results.get('temporal_correlations', {})
        
        for trial_type, peak_data in peaks.items():
            if trial_type not in all_peak_diffs:
                all_peak_diffs[trial_type] = {'region_i': [], 'region_j': []}
            all_peak_diffs[trial_type]['region_i'].extend(peak_data['region_i'])
            all_peak_diffs[trial_type]['region_j'].extend(peak_data['region_j'])
        
        for comp_name, corr_data in correlations.items():
            if comp_name not in all_correlations:
                all_correlations[comp_name] = {'region_i': [], 'region_j': []}
            for c in corr_data['region_i']:
                all_correlations[comp_name]['region_i'].append(c['r2'])
            for c in corr_data['region_j']:
                all_correlations[comp_name]['region_j'].append(c['r2'])
    
    # Create aggregate figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fontsize = 12
    
    # Plot 1: Peak amplitudes across trial types
    ax1 = axes[0]
    trial_types = list(all_peak_diffs.keys())
    x = np.arange(len(trial_types))
    
    means = [np.mean(all_peak_diffs[tt]['region_i'] + all_peak_diffs[tt]['region_j']) 
             for tt in trial_types]
    stds = [np.std(all_peak_diffs[tt]['region_i'] + all_peak_diffs[tt]['region_j']) 
            for tt in trial_types]
    
    colors = [TRIAL_TYPE_COLORS.get(tt, 'gray') for tt in trial_types]
    
    ax1.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([tt.replace('_', '\n') for tt in trial_types], fontsize=fontsize - 2)
    ax1.set_ylabel('Peak Amplitude (mean ± std)', fontsize=fontsize)
    ax1.set_title('Peak Amplitudes Across Trial Types', fontsize=fontsize + 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Temporal correlations
    ax2 = axes[1]
    comparisons = list(all_correlations.keys())
    
    if comparisons:
        x = np.arange(len(comparisons))
        means = [np.mean(all_correlations[c]['region_i'] + all_correlations[c]['region_j'])
                 for c in comparisons]
        stds = [np.std(all_correlations[c]['region_i'] + all_correlations[c]['region_j'])
                for c in comparisons]
        
        ax2.bar(x, means, yerr=stds, color='steelblue', alpha=0.8, capsize=5, edgecolor='black')
        ax2.set_xticks(x)
        ax2.set_xticklabels([c.replace('_', '\n') for c in comparisons], fontsize=fontsize - 3)
        ax2.set_ylabel('Temporal R² (mean ± std)', fontsize=fontsize)
        ax2.set_title('Temporal Correlation with Reference', fontsize=fontsize + 1)
        ax2.set_ylim([0, 1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    plt.suptitle(
        f'Aggregate Summary Across {len(pipeline.analyzers)} Sessions',
        fontsize=fontsize + 2, fontweight='bold'
    )
    
    plt.tight_layout()
    
    output_path = output_dir / 'aggregate_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregate summary: {output_path}")
    
    return fig


def print_analysis_summary(analyzer: CrossTrialTypeCCAAnalyzer) -> None:
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
    
    if analyzer.projections:
        print(f"\nProjections computed for {len(analyzer.projections)} trial types:")
        for tt, proj in analyzer.projections.items():
            print(f"  {tt}: {proj['n_trials']} trials")
    
    if analyzer.statistical_results:
        print("\nStatistical Results:")
        
        # Peak amplitudes
        print("\n  Peak Amplitudes (Component 1):")
        peaks = analyzer.statistical_results.get('peak_amplitudes', {})
        for tt, peak_data in peaks.items():
            print(f"    {tt}:")
            print(f"      Region i: {peak_data['region_i'][0]:.4f}")
            print(f"      Region j: {peak_data['region_j'][0]:.4f}")
        
        # Temporal correlations
        print("\n  Temporal Correlations (vs reference):")
        correlations = analyzer.statistical_results.get('temporal_correlations', {})
        for comp_name, corr_data in correlations.items():
            print(f"    {comp_name}:")
            if corr_data['region_i']:
                print(f"      Region i R²: {corr_data['region_i'][0]['r2']:.4f}")
            if corr_data['region_j']:
                print(f"      Region j R²: {corr_data['region_j'][0]['r2']:.4f}")
        
        # Statistical tests
        print("\n  Statistical Tests (Component 1):")
        pairwise = analyzer.statistical_results.get('pairwise_tests', {})
        for comp_name, test_data in pairwise.items():
            print(f"    {comp_name}:")
            if test_data['region_i']:
                t = test_data['region_i'][0]
                sig = '*' if t['wilcoxon_p'] < 0.05 else ''
                print(f"      Region i: p={t['wilcoxon_p']:.4f}{sig}, d={t['cohens_d']:.3f}")
            if test_data['region_j']:
                t = test_data['region_j'][0]
                sig = '*' if t['wilcoxon_p'] < 0.05 else ''
                print(f"      Region j: p={t['wilcoxon_p']:.4f}{sig}, d={t['cohens_d']:.3f}")
    
    print(f"\nOutput saved to: {analyzer.output_dir}")


def print_cross_session_summary(pipeline: CrossTrialTypeCCAPipeline) -> None:
    """
    Print summary of cross-session analysis results.
    
    Parameters:
        pipeline: Completed pipeline with cross-session results
    """
    print("\n" + "=" * 70)
    print("CROSS-SESSION ANALYSIS SUMMARY")
    print("=" * 70)
    
    min_sessions = pipeline.config.get('min_sessions', MIN_SESSIONS_THRESHOLD)
    
    print(f"\nMinimum sessions threshold: {min_sessions}")
    print(f"Total region pairs analyzed: {len(pipeline.cross_session_analyzers)}")
    
    # Count pairs meeting threshold
    valid_pairs = []
    insufficient_pairs = []
    
    for pair_key, cs_analyzer in pipeline.cross_session_analyzers.items():
        n_sessions = len(cs_analyzer.session_projections)
        if n_sessions >= min_sessions:
            valid_pairs.append((pair_key, n_sessions))
        else:
            insufficient_pairs.append((pair_key, n_sessions))
    
    print(f"\nPairs with ≥{min_sessions} sessions (valid for cross-session analysis):")
    for pair_key, n in sorted(valid_pairs, key=lambda x: x[1], reverse=True):
        print(f"  {pair_key[0]:6s} vs {pair_key[1]:6s}: {n:3d} sessions")
    
    if insufficient_pairs:
        print(f"\nPairs with <{min_sessions} sessions (insufficient data):")
        for pair_key, n in sorted(insufficient_pairs, key=lambda x: x[1], reverse=True):
            print(f"  {pair_key[0]:6s} vs {pair_key[1]:6s}: {n:3d} sessions")
    
    # Summary statistics
    if valid_pairs:
        print(f"\nSummary:")
        print(f"  Valid pairs for cross-session analysis: {len(valid_pairs)}")
        print(f"  Total sessions across valid pairs: {sum(n for _, n in valid_pairs)}")
        
        if pipeline.summary_visualizer:
            output_dir = pipeline.summary_visualizer.output_dir
            print(f"\nSummary figures saved to: {output_dir}")


def main():
    """Main entry point for cross-trial-type CCA analysis."""
    
    # Get configuration
    config = create_analysis_config()
    
    # Run analysis mode selection
    print("\n" + "=" * 70)
    print("CROSS-TRIAL-TYPE CCA ANALYSIS")
    print("=" * 70)
    print("\nThis analysis projects neural activity from different trial types")
    print("onto CCA subspaces trained on a reference condition.")
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

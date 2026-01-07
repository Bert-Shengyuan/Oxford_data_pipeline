#!/usr/bin/env python3
"""
Oxford Dataset PCA Visualization - Independent Region Analysis
==============================================================

This script implements PCA visualization where each region is analyzed
independently across ALL sessions, without requiring simultaneous
recordings of paired regions.

Scientific Rationale:
---------------------
PCA operates as an unsupervised method on each region's neural population
independently. Unlike CCA (which requires within-session pairing to identify
shared variance), PCA identifies intrinsic variance structure within each
region. Therefore, we can aggregate PCA results across all sessions that
recorded a given region, regardless of what other regions were co-recorded.

Key Features:
-------------
1. Region-independent loading: Extract PCA results for each region from
   all sessions where that region was recorded
2. Cumulative variance visualization: 3-row × 4-column layout showing
   mean cumulative variance ± standard deviation across sessions
3. Anatomical ordering: Consistent region ordering following cortical →
   subcortical → fiber hierarchy

Author: Oxford Neural Analysis Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mat73
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Configure visualization aesthetics
sns.set_style("white")
sns.set_context("paper", font_scale=1.0)

# =============================================================================
# CANONICAL ANATOMICAL ORDERING
# =============================================================================
# This ordering reflects the cortex → subcortex → fiber hierarchy and ensures
# consistent visualization across all figures in the manuscript.

# ANATOMICAL_ORDER = [
#     'mPFC', 'ORB', 'MOp', 'MOs', 'OLF',  # Cortical regions
#     'STR', 'STRv', 'HIPP',  # Striatal & limbic
#     'MD', 'LP', 'VALVM', 'VPMPO', 'ILM',  # Thalamic nuclei
#     'HY',  # Hypothalamic
#     'fiber',  # Fiber tracts
#     'other'  # Catch-all category
# ]

ANATOMICAL_ORDER = [
    'mPFC', 'ORB', 'MOp', 'MOs', 'OLF',  # Cortical regions
    'STR', 'STRv',  # Striatal & limbic
    'MD', 'LP', 'VALVM', 'VPMPO', 'ILM',  # Thalamic nuclei
    'HY',  # Hypothalamic
]

class OxfordPCAVisualizer:
    """
    Visualizer for PCA results aggregated independently per region.

    Unlike CCA which requires paired recordings, PCA is computed independently
    for each region. This class loads PCA results from all sessions and
    aggregates them per region (not per region-pair), enabling visualization
    of intrinsic variance structure across the entire dataset.

    Attributes:
        base_results_dir: Root directory containing session results
        pca_results_dir: Directory containing *_analysis_results.mat files
        n_components: Number of principal components to visualize
        region_data: Dictionary mapping region → list of session PCA results
        anatomical_order: Canonical ordering for visualization
    """

    def __init__(
            self,
            base_results_dir: str,
            results_subdir: str = "sessions_cued_hit_long_results",
            n_components: int = 10
    ):
        """
        Initialize the PCA visualizer.

        Parameters:
            base_results_dir: Root directory (e.g., '/path/to/Oxford_dataset')
            results_subdir: Subdirectory containing analysis results
                           Options: 'sessions_cued_hit_long_results' (cued state)
                                    'sessions_spont_short_results' (spontaneous)
            n_components: Number of PCs to include in cumulative variance
        """
        self.base_results_dir = Path(base_results_dir)
        self.pca_results_dir = self.base_results_dir / results_subdir
        self.n_components = n_components
        self.anatomical_order = ANATOMICAL_ORDER

        # Primary data container: region → [session_results]
        # Each session_result contains cumulative_variance, explained_variance, etc.
        self.region_data: Dict[str, List[Dict]] = {}

        # Track which sessions contributed to each region
        self.region_sessions: Dict[str, List[str]] = {}

        # Validate directory exists
        if not self.pca_results_dir.exists():
            raise ValueError(f"Results directory not found: {self.pca_results_dir}")

        print("=" * 70)
        print("Oxford PCA Visualizer - Independent Region Analysis")
        print("=" * 70)
        print(f"Results directory: {self.pca_results_dir}")
        print(f"Components to visualize: {self.n_components}")

    def load_all_sessions(self) -> None:
        """
        Load PCA results from all sessions, organizing by region.

        This method iterates through all *_analysis_results.mat files,
        extracts PCA results for each region present, and aggregates
        them independently. A single session may contribute data to
        multiple regions.
        """
        print("\nLoading PCA results from all sessions...")
        print("-" * 50)

        # Find all analysis result files
        result_files = list(self.pca_results_dir.glob("*_analysis_results.mat"))
        print(f"Found {len(result_files)} session files")

        for result_file in result_files:
            session_name = result_file.stem.replace("_analysis_results", "")
            self._process_session_file(result_file, session_name)

        # Report loading summary
        print("\n" + "-" * 50)
        print("Loading Summary:")
        for region in self._get_ordered_regions():
            n_sessions = len(self.region_data.get(region, []))
            print(f"  {region}: {n_sessions} sessions")

    def _process_session_file(self, file_path: Path, session_name: str) -> None:
        """
        Process a single session file, extracting PCA results per region.

        Parameters:
            file_path: Path to the *_analysis_results.mat file
            session_name: Identifier for this session
        """
        try:
            # Load using mat73 (handles MATLAB v7.3 files)
            data = mat73.loadmat(str(file_path))

            if 'pca_results' not in data:
                print(f"  {session_name}: No pca_results found, skipping")
                return

            pca_results = data['pca_results']

            # Identify all regions with valid PCA data in this session
            # PCA results are organized as: pca_results.REGION_NAME.projections, etc.
            available_regions = [
                key for key in pca_results.keys()
                if isinstance(pca_results[key], dict) and 'projections' in pca_results[key]
            ]

            if not available_regions:
                print(f"  {session_name}: No valid PCA regions found")
                return

            # Extract data for each region independently
            for region in available_regions:
                region_pca = pca_results[region]
                extracted_data = self._extract_region_pca_data(region_pca, region)

                if extracted_data is not None:
                    extracted_data['session_name'] = session_name

                    # Initialize container if this is first occurrence of region
                    if region not in self.region_data:
                        self.region_data[region] = []
                        self.region_sessions[region] = []

                    self.region_data[region].append(extracted_data)
                    self.region_sessions[region].append(session_name)

            print(f"  {session_name}: Extracted {len(available_regions)} regions")

        except Exception as e:
            print(f"  {session_name}: Error loading - {str(e)}")

    def _extract_region_pca_data(
            self,
            region_pca: dict,
            region_name: str
    ) -> Optional[Dict]:
        """
        Extract PCA metrics from a region's PCA results structure.

        Expected structure from MATLAB:
            region_pca.explained_variance    - Individual PC variance (%)
            region_pca.cumulative_variance   - Cumulative variance (%)
            region_pca.projections.mean      - Mean temporal projections
            region_pca.projections.std       - Std of temporal projections

        Parameters:
            region_pca: Dictionary containing PCA results for one region
            region_name: Name of the region (for error messages)

        Returns:
            Dictionary with cumulative_variance, explained_variance arrays,
            temporal projections, or None if extraction fails
        """
        try:
            extracted = {}

            # Extract cumulative variance (primary visualization metric)
            if 'cumulative_variance' in region_pca:
                cum_var = np.array(region_pca['cumulative_variance']).flatten()
                extracted['cumulative_variance'] = cum_var
            elif 'explained_variance' in region_pca:
                # Compute cumulative from individual explained variances
                exp_var = np.array(region_pca['explained_variance']).flatten()
                extracted['explained_variance'] = exp_var
                extracted['cumulative_variance'] = np.cumsum(exp_var)
            else:
                # Cannot proceed without variance information
                return None

            # Extract explained variance if available separately
            if 'explained_variance' in region_pca and 'explained_variance' not in extracted:
                extracted['explained_variance'] = np.array(
                    region_pca['explained_variance']
                ).flatten()

            # Extract number of neurons if available
            if 'n_neurons' in region_pca:
                extracted['n_neurons'] = int(region_pca['n_neurons'])

            # =================================================================
            # Extract temporal projections for visualization
            # Structure: projections.mean{comp_idx} -> 1xT array
            #            projections.std{comp_idx}  -> 1xT array
            # =================================================================
            if 'projections' in region_pca:
                projections = region_pca['projections']
                extracted['temporal_projections'] = {}

                # Extract mean projections per component
                if 'mean' in projections:
                    means = projections['mean']
                    for comp_idx in range(len(means)):
                        if comp_idx not in extracted['temporal_projections']:
                            extracted['temporal_projections'][comp_idx] = {}

                        mean_data = means[comp_idx]
                        if isinstance(mean_data, np.ndarray):
                            extracted['temporal_projections'][comp_idx]['mean'] = mean_data.flatten()
                        elif isinstance(mean_data, (list, tuple)):
                            extracted['temporal_projections'][comp_idx]['mean'] = np.array(mean_data).flatten()

                # Extract std projections per component
                if 'std' in projections:
                    stds = projections['std']
                    for comp_idx in range(len(stds)):
                        if comp_idx not in extracted['temporal_projections']:
                            extracted['temporal_projections'][comp_idx] = {}

                        std_data = stds[comp_idx]
                        if isinstance(std_data, np.ndarray):
                            extracted['temporal_projections'][comp_idx]['std'] = std_data.flatten()
                        elif isinstance(std_data, (list, tuple)):
                            extracted['temporal_projections'][comp_idx]['std'] = np.array(std_data).flatten()

            return extracted

        except Exception as e:
            print(f"    Warning: Could not extract PCA data for {region_name}: {e}")
            return None

    def _get_ordered_regions(self) -> List[str]:
        """
        Return available regions in anatomical order.

        Filters ANATOMICAL_ORDER to include only regions present in the
        loaded data, ensuring consistent ordering across visualizations.
        """
        return [r for r in self.anatomical_order if r in self.region_data]

    def create_cumulative_variance_figure(
            self,
            figsize: Tuple[float, float] = (16, 12),
            save_path: Optional[str] = None,
            n_rows: int = 3,
            n_cols: int = 4
    ) -> None:
        """
        Create a multi-panel figure showing cumulative variance per region.

        Layout: 3 rows × 4 columns = 12 subplots
        Each subplot shows:
            - Mean cumulative variance curve (averaged across sessions)
            - Shaded region indicating ± standard deviation across sessions

        Unused subplots are hidden with ax.axis('off').

        Parameters:
            figsize: Figure dimensions (width, height) in inches
            save_path: If provided, save figure to this path (without extension)
            n_rows: Number of subplot rows (default: 3)
            n_cols: Number of subplot columns (default: 4)
        """
        print("\nCreating cumulative variance figure...")

        ordered_regions = self._get_ordered_regions()
        n_regions = len(ordered_regions)
        max_panels = n_rows * n_cols

        print(f"  Regions to plot: {n_regions} (max panels: {max_panels})")

        # Create figure with specified layout
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()  # Enable linear indexing

        # Define consistent color palette
        line_color = '#2E86AB'  # Steel blue for mean line
        fill_color = '#2E86AB'  # Same color for shaded region

        for idx, region in enumerate(ordered_regions):
            if idx >= max_panels:
                print(f"  Warning: {region} exceeds panel limit, skipping")
                continue

            ax = axes[idx]
            self._plot_region_cumulative_variance(ax, region, line_color, fill_color)

        # Hide unused subplots
        for idx in range(n_regions, max_panels):
            axes[idx].axis('off')

        # Add overall title
        fig.suptitle(
            'PCA Cumulative Variance by Region\n'
            f'(Mean ± SD across {self._get_total_sessions()} sessions)',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure
        if save_path:
            output_file = f"{save_path}_pca_cumulative_variance.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_file}")

        plt.close(fig)
        print("  Cumulative variance figure complete")

    def _plot_region_cumulative_variance(
            self,
            ax: plt.Axes,
            region: str,
            line_color: str,
            fill_color: str
    ) -> None:
        """
        Plot cumulative variance for a single region.

        Computes mean and standard deviation across all sessions for this
        region, then displays as a line with shaded confidence region.

        Parameters:
            ax: Matplotlib axes object for this subplot
            region: Name of the region to plot
            line_color: Color for the mean line
            fill_color: Color for the shaded std region
        """
        region_sessions = self.region_data[region]
        n_sessions = len(region_sessions)

        if n_sessions == 0:
            ax.text(0.5, 0.5, f'{region}\nNo data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return

        # Collect cumulative variance arrays from all sessions
        # Note: Different sessions may have different numbers of PCs due to
        # varying neuron counts. We truncate to the minimum length.
        all_cum_var = [s['cumulative_variance'] for s in region_sessions]
        min_length = min(len(cv) for cv in all_cum_var)

        # Truncate all arrays to minimum length and stack
        truncated = np.array([cv[:min_length] for cv in all_cum_var])

        # Compute statistics across sessions (axis=0)
        mean_cum_var = np.mean(truncated, axis=0)
        std_cum_var = np.std(truncated, axis=0)

        # Limit to n_components for visualization
        n_plot = min(self.n_components, len(mean_cum_var))
        components = np.arange(1, n_plot + 1)
        mean_plot = mean_cum_var[:n_plot]
        std_plot = std_cum_var[:n_plot]

        # Plot mean line with shaded standard deviation
        ax.plot(components, mean_plot, color=line_color, linewidth=2.5,
                marker='o', markersize=4, label='Mean')
        ax.fill_between(
            components,
            mean_plot - std_plot,
            mean_plot + std_plot,
            alpha=0.25,
            color=fill_color,
            label='±1 SD'
        )

        # Add reference lines
        ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, linewidth=1)

        # Formatting
        ax.set_xlim(0.5, n_plot + 0.5)
        ax.set_ylim(0, 105)
        ax.set_xlabel('Principal Component', fontsize=10)
        ax.set_ylabel('Cumulative Var. (%)', fontsize=10)
        ax.set_title(f'{region} (n={n_sessions})', fontsize=12, fontweight='bold')

        # Set integer x-ticks
        ax.set_xticks(components[::2] if n_plot > 5 else components)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=':')

        # Add annotation for final cumulative variance
        final_var = mean_plot[-1]
        ax.annotate(
            f'{final_var:.1f}%',
            xy=(n_plot, final_var),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=9,
            color=line_color
        )

    def _get_total_sessions(self) -> int:
        """Return the total number of unique sessions loaded."""
        all_sessions = set()
        for sessions in self.region_sessions.values():
            all_sessions.update(sessions)
        return len(all_sessions)

    def create_variance_summary_table(
            self,
            save_path: Optional[str] = None
    ) -> None:
        """
        Create a summary figure showing variance metrics as a table/heatmap.

        Displays for each region:
            - Number of sessions
            - Mean variance explained by PC1
            - Mean cumulative variance at PC3, PC5, PC10

        Parameters:
            save_path: If provided, save figure to this path
        """
        print("\nCreating variance summary table...")

        ordered_regions = self._get_ordered_regions()

        # Prepare summary statistics
        summary_data = []
        for region in ordered_regions:
            region_sessions = self.region_data[region]
            n_sessions = len(region_sessions)

            if n_sessions == 0:
                continue

            # Compute mean cumulative variance at key points
            all_cum_var = [s['cumulative_variance'] for s in region_sessions]
            min_len = min(len(cv) for cv in all_cum_var)
            stacked = np.array([cv[:min_len] for cv in all_cum_var])
            mean_cv = np.mean(stacked, axis=0)

            # Extract variance at PC1, PC3, PC5, PC10 (if available)
            pc1_var = mean_cv[0] if len(mean_cv) >= 1 else np.nan
            pc3_cum = mean_cv[2] if len(mean_cv) >= 3 else np.nan
            pc5_cum = mean_cv[4] if len(mean_cv) >= 5 else np.nan
            pc10_cum = mean_cv[9] if len(mean_cv) >= 10 else np.nan

            summary_data.append({
                'Region': region,
                'Sessions': n_sessions,
                'PC1 (%)': pc1_var,
                'PC3 Cum (%)': pc3_cum,
                'PC5 Cum (%)': pc5_cum,
                'PC10 Cum (%)': pc10_cum
            })

        # Create figure with table
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        # Build table data
        col_labels = ['Region', 'n', 'PC1', 'PC3\n(cum)', 'PC5\n(cum)', 'PC10\n(cum)']
        table_data = []
        for row in summary_data:
            table_data.append([
                row['Region'],
                str(row['Sessions']),
                f"{row['PC1 (%)']:.1f}",
                f"{row['PC3 Cum (%)']:.1f}",
                f"{row['PC5 Cum (%)']:.1f}",
                f"{row['PC10 Cum (%)']:.1f}" if not np.isnan(row['PC10 Cum (%)']) else '-'
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colColours=['#E8E8E8'] * len(col_labels)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        plt.title(
            'PCA Variance Summary by Region',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        if save_path:
            output_file = f"{save_path}_pca_variance_summary.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_file}")

        plt.close(fig)
        print("  Variance summary table complete")

    def generate_report(self) -> None:
        """Print a comprehensive summary of the loaded PCA data."""
        print("\n" + "=" * 70)
        print("PCA Analysis Report")
        print("=" * 70)
        print(f"Results directory: {self.pca_results_dir}")
        print(f"Total unique sessions: {self._get_total_sessions()}")
        print(f"Regions with data: {len(self.region_data)}")

        print("\nPer-Region Statistics:")
        print("-" * 50)

        for region in self._get_ordered_regions():
            sessions = self.region_data[region]
            n = len(sessions)

            if n > 0:
                all_cv = [s['cumulative_variance'] for s in sessions]
                min_len = min(len(cv) for cv in all_cv)
                stacked = np.array([cv[:min_len] for cv in all_cv])
                mean_cv = np.mean(stacked, axis=0)

                pc1 = mean_cv[0] if len(mean_cv) >= 1 else np.nan
                pc5 = mean_cv[4] if len(mean_cv) >= 5 else np.nan

                print(f"  {region:8s}: {n:3d} sessions, "
                      f"PC1={pc1:5.1f}%, PC5_cum={pc5:5.1f}%")

    # =========================================================================
    # TEMPORAL PROJECTION FIGURES - INDEPENDENT REGION AGGREGATION
    # =========================================================================
    # The key distinction from CCA: for PCA temporal projections, each region's
    # data is aggregated across ALL sessions recording that region, regardless
    # of what other regions were co-recorded. This enables visualisation of
    # region pairs even when they were never recorded simultaneously.
    # =========================================================================

    def create_temporal_projection_figure(
            self,
            figsize: Tuple[float, float] = (60, 60),
            component_idx: int = 0,
            save_path: Optional[str] = None,
            min_sessions: int = 3
    ) -> None:
        """
        Create cross-region temporal projection matrix figure for PCA.

        CRITICAL DISTINCTION FROM CCA:
        ------------------------------
        For PCA, each region's temporal projection is computed independently
        from ALL sessions that recorded that region. When displaying region
        pair (A, B) in the matrix:

            - Region A projection: mean across all sessions in $S_A$
            - Region B projection: mean across all sessions in $S_B$

        where $S_A$ and $S_B$ are computed independently and need not overlap.
        This differs fundamentally from CCA, which requires $S_A \cap S_B$.

        Parameters:
            figsize: Figure dimensions (width, height) in inches
            component_idx: Which principal component to visualize (0-based)
            save_path: Output path (without extension)
            min_sessions: Minimum sessions required per region to display
        """
        print(f"\nCreating PCA temporal projection figure (Component {component_idx + 1})...")
        print(f"  Using INDEPENDENT region aggregation (differs from CCA)")

        # Pre-compute mean projections for each region across all its sessions
        # This is done once and reused for all pairs involving that region
        region_mean_projections = self._compute_region_mean_projections(
            component_idx, min_sessions
        )

        if not region_mean_projections:
            print("  No regions with sufficient data for temporal projections")
            return

        # Get regions that have valid projection data
        valid_regions = [r for r in self._get_ordered_regions()
                         if r in region_mean_projections]
        n_regions = len(valid_regions)

        print(f"  Regions with valid projections: {n_regions}")

        # Create figure grid
        fig, axes = plt.subplots(n_regions, n_regions, figsize=figsize)

        if n_regions == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)

        # Time vector for plotting
        time_vec = np.linspace(-1.5, 3.0, 226)

        for i, region_i in enumerate(valid_regions):
            for j, region_j in enumerate(valid_regions):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: display region name and session count
                    n_sessions = len(self.region_data.get(region_i, []))
                    ax.text(0.5, 0.5, f'{region_i}\n(n={n_sessions})',
                            ha='center', va='center',
                            fontsize=32, fontweight='bold')
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    ax.axis('off')

                elif i > j:
                    # Lower triangle: hide
                    ax.axis('off')

                else:
                    # Upper triangle: plot both regions' projections
                    self._plot_pca_temporal_projection_pair(
                        ax, region_i, region_j,
                        region_mean_projections, time_vec
                    )

        # Add figure title
        fig.suptitle(
            f'PCA Temporal Projections - Component {component_idx + 1}',
            fontsize=48, fontweight='bold', y=0.995
        )

        plt.tight_layout(rect=[0, 0.01, 1, 0.99])

        if save_path:
            output_file = f"{save_path}_pca_temporal_comp{component_idx + 1}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_file}")

        plt.close(fig)
        print("  Temporal projection figure complete")

    def _compute_region_mean_projections(
            self,
            component_idx: int,
            min_sessions: int = 3
    ) -> Dict[str, Dict]:
        """
        Pre-compute mean temporal projections for each region independently.

        This implements the core PCA logic: for each region, we aggregate
        across ALL sessions that recorded that region, regardless of what
        other regions were co-recorded in those sessions.

        Parameters:
            component_idx: Which PC to compute projections for
            min_sessions: Minimum sessions required

        Returns:
            Dictionary mapping region -> {
                'mean': mean projection array,
                'std': std projection array,
                'n_sessions': number of sessions used
            }
        """
        region_projections = {}

        for region in self._get_ordered_regions():
            sessions = self.region_data.get(region, [])

            # Collect projections from all sessions for this region
            proj_arrays = []

            for session in sessions:
                if 'temporal_projections' not in session:
                    continue

                temp_proj = session['temporal_projections']

                if component_idx not in temp_proj:
                    continue

                comp_data = temp_proj[component_idx]

                if 'mean' in comp_data and len(comp_data['mean']) > 0:
                    proj_arrays.append(np.abs(comp_data['mean']))

            # Require minimum number of sessions
            if len(proj_arrays) < min_sessions:
                continue

            # Compute mean and std across sessions
            # Handle potentially different lengths by truncating to minimum
            min_len = min(len(p) for p in proj_arrays)
            truncated = np.array([p[:min_len] for p in proj_arrays])

            region_projections[region] = {
                'mean': np.mean(truncated, axis=0),
                'std': np.std(truncated, axis=0),
                'n_sessions': len(proj_arrays),
                'length': min_len
            }

        return region_projections

    def _plot_pca_temporal_projection_pair(
            self,
            ax: plt.Axes,
            region_i: str,
            region_j: str,
            region_projections: Dict[str, Dict],
            time_vec: np.ndarray
    ) -> None:
        """
        Plot PCA temporal projections for a region pair.

        IMPORTANT: Each region's projection is computed independently from
        ALL sessions recording that region. The two lines shown may be
        derived from completely different (non-overlapping) session sets.

        Parameters:
            ax: Matplotlib axes object
            region_i: First region name
            region_j: Second region name
            region_projections: Pre-computed projections per region
            time_vec: Time vector for x-axis
        """
        # Check that both regions have valid data
        if region_i not in region_projections or region_j not in region_projections:
            ax.set_visible(False)
            return

        proj_i = region_projections[region_i]
        proj_j = region_projections[region_j]

        # Determine common length for plotting
        min_len = min(proj_i['length'], proj_j['length'], len(time_vec))
        t = time_vec[:min_len]

        mean_i = proj_i['mean'][:min_len]
        std_i = proj_i['std'][:min_len]
        mean_j = proj_j['mean'][:min_len]
        std_j = proj_j['std'][:min_len]

        # Plot region i (red)
        ax.plot(t, mean_i, color='red', linewidth=2, alpha=0.9,
                label=f'{region_i} (n={proj_i["n_sessions"]})')
        ax.fill_between(t, mean_i - std_i, mean_i + std_i,
                        alpha=0.15, color='red')

        # Plot region j (blue)
        ax.plot(t, mean_j, color='blue', linewidth=2, alpha=0.9,
                label=f'{region_j} (n={proj_j["n_sessions"]})')
        ax.fill_between(t, mean_j - std_j, mean_j + std_j,
                        alpha=0.15, color='blue')

        # Reference line at stimulus onset
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=3)

        # Formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.8, linestyle=':', linewidth=2)

        ax.set_yticks(np.arange(0, 10, 2))
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)
        ax.set_ylim([0, 3])
        ax.set_xticks([-1.5, 0, 2, 3])
        ax.set_xticklabels(['-1.5', '0', '2', '3'], fontsize=20)
        ax.tick_params(axis='both', which='major', width=2, length=8)

        for spine in ax.spines.values():
            spine.set_linewidth(3)

    def create_all_component_figures(
            self,
            n_components_to_plot: int = 5,
            figsize: Tuple[float, float] = (40, 40),
            save_path: Optional[str] = None,
            min_sessions: int = 3
    ) -> None:
        """
        Create temporal projection figures for multiple components.

        Convenience method that generates figures for the first n components.

        Parameters:
            n_components_to_plot: Number of components to visualize
            figsize: Figure dimensions
            save_path: Base output path
            min_sessions: Minimum sessions per region
        """
        print(f"\nGenerating temporal projection figures for {n_components_to_plot} components...")

        for comp_idx in range(n_components_to_plot):
            self.create_temporal_projection_figure(
                figsize=figsize,
                component_idx=comp_idx,
                save_path=save_path,
                min_sessions=min_sessions
            )


def main():
    """Demonstration of PCA visualization pipeline."""

    # Configuration - modify these paths for your system
    base_dir = '/Users/shengyuancai/Downloads/Oxford_dataset'
    output_dir = Path(base_dir) / 'Paper_output' / 'figures_pca'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Oxford PCA Visualization Pipeline")
    print("=" * 70)

    # Option 1: Cued state analysis
    print("\n[1] Processing CUED state sessions...")
    pca_viz_cued = OxfordPCAVisualizer(
        base_results_dir=base_dir,
        results_subdir="sessions_cued_hit_long_results",
        n_components=10
    )
    pca_viz_cued.load_all_sessions()
    pca_viz_cued.generate_report()

    # Create cumulative variance figure (3x4 layout)
    pca_viz_cued.create_cumulative_variance_figure(
        figsize=(16, 12),
        save_path=str(output_dir / "cued_long"),
        n_rows=3,
        n_cols=4
    )

    # Create variance summary table
    pca_viz_cued.create_variance_summary_table(
        save_path=str(output_dir / "cued_long")
    )

    # Create temporal projection figures (INDEPENDENT region aggregation)
    # Each region is averaged across ALL sessions recording that region,
    # regardless of what other regions were co-recorded
    pca_viz_cued.create_all_component_figures(
        n_components_to_plot=3,
        figsize=(40, 40),
        save_path=str(output_dir / "cued_long"),
        min_sessions=3
    )
    # Option 2: Spontaneous state analysis
    print("\n[2] Processing SPONTANEOUS state sessions...")
    pca_viz_spont = OxfordPCAVisualizer(
        base_results_dir=base_dir,
        results_subdir="sessions_spont_miss_long_results",
        n_components=10
    )
    pca_viz_spont.load_all_sessions()
    pca_viz_spont.generate_report()

    # Create cumulative variance figure
    pca_viz_spont.create_cumulative_variance_figure(
        figsize=(16, 12),
        save_path=str(output_dir / "spont_long"),
        n_rows=3,
        n_cols=4
    )

    # Create variance summary table
    pca_viz_spont.create_variance_summary_table(
        save_path=str(output_dir / "spont_long")
    )

    # Create temporal projection figures
    pca_viz_spont.create_all_component_figures(
        n_components_to_plot=3,
        figsize=(40, 40),
        save_path=str(output_dir / "spont_long"),
        min_sessions=3
    )


    # Option 3: Spontaneous state analysis
    print("\n[2] Processing SPONTANEOUS state sessions...")
    pca_viz_spont_short = OxfordPCAVisualizer(
        base_results_dir=base_dir,
        results_subdir="sessions_spont_short_results",
        n_components=10
    )
    pca_viz_spont_short.load_all_sessions()
    pca_viz_spont_short.generate_report()

    # Create cumulative variance figure
    pca_viz_spont_short.create_cumulative_variance_figure(
        figsize=(16, 12),
        save_path=str(output_dir / "spont_short"),
        n_rows=3,
        n_cols=4
    )

    # Create variance summary table
    pca_viz_spont_short.create_variance_summary_table(
        save_path=str(output_dir / "spont_short")
    )

    # Create temporal projection figures
    pca_viz_spont_short.create_all_component_figures(
        n_components_to_plot=3,
        figsize=(40, 40),
        save_path=str(output_dir / "spont_short"),
        min_sessions=3
    )

    # Option 4: Spontaneous state analysis
    print("\n[2] Processing SPONTANEOUS state sessions...")
    pca_viz_spont_hit = OxfordPCAVisualizer(
        base_results_dir=base_dir,
        results_subdir="sessions_spont_hit_long_results",
        n_components=10
    )
    pca_viz_spont_hit.load_all_sessions()
    pca_viz_spont_hit.generate_report()

    # Create cumulative variance figure
    pca_viz_spont_hit.create_cumulative_variance_figure(
        figsize=(16, 12),
        save_path=str(output_dir / "spont_hit_long"),
        n_rows=3,
        n_cols=4
    )

    # Create variance summary table
    pca_viz_spont_hit.create_variance_summary_table(
        save_path=str(output_dir / "spont_hit_long")
    )

    # Create temporal projection figures
    pca_viz_spont_hit.create_all_component_figures(
        n_components_to_plot=3,
        figsize=(40, 40),
        save_path=str(output_dir / "spont_hit_long"),
        min_sessions=3
    )

    print("\n" + "=" * 70)
    print("PCA Visualization Pipeline Complete")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Oxford Dataset CCA Visualization - Cross-Regional Connectivity Matrices
========================================================================

This script implements CCA visualization with connectivity matrices displaying
the mean and standard deviation of cross-validated $R^2$ values across sessions.

Scientific Rationale:
---------------------
Canonical Correlation Analysis (CCA) inherently requires within-session pairing
of brain regions, as it identifies shared variance between two neural populations
recorded simultaneously. Unlike PCA, CCA results cannot be aggregated across
sessions where only one of the two regions was recorded.

Key Features:
-------------
1. Connectivity matrices: $n \times n$ region matrices where $n = $ number of regions
2. Component-wise visualization: Separate matrices for each CCA component
3. Rank-based significance: Components ordered by cross-validated $R^2$
4. Mean ± SD display: Upper row shows mean, lower row shows standard deviation
5. Anatomical ordering: Consistent cortical → subcortical → fiber hierarchy

Mathematical Framework:
-----------------------
For region pair $(i, j)$ and CCA component $k$:
$$\bar{R}^2_{ij,k} = \frac{1}{N_{ij}} \sum_{s \in S_{ij}} R^2_{ij,k,s}$$

where $S_{ij}$ is the set of sessions containing both regions $i$ and $j$,
and $N_{ij} = |S_{ij}|$ is the count of such sessions.

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
# ANATOMICAL_ORDER = [
#     'mPFC', 'ORB', 'MOp', 'MOs', 'OLF',   # Cortical regions
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
    'HY'
]


class OxfordCCAVisualizer:
    """
    Visualizer for CCA results organized by region pairs across sessions.

    CCA fundamentally requires within-session paired recordings. This class
    aggregates CCA results across all sessions where a given region pair
    was recorded simultaneously, computing summary statistics for each
    canonical component.

    Attributes:
        base_results_dir: Root directory containing session results
        cca_results_dir: Directory containing *_analysis_results.mat files
        n_components: Number of canonical components to visualize
        pair_data: Dictionary mapping (region_i, region_j) → list of session results
        anatomical_order: Canonical ordering for matrix visualization
    """

    def __init__(
            self,
            base_results_dir: str,
            results_subdir: str = "sessions_cued_hit_long_results",
            n_components: int = 5,
            min_sessions_threshold: int = 3
    ):
        """
        Initialize the CCA visualizer.

        Parameters:
            base_results_dir: Root directory (e.g., '/path/to/Oxford_dataset')
            results_subdir: Subdirectory containing analysis results
            n_components: Number of CCA components to visualize
            min_sessions_threshold: Minimum sessions required to display a pair
        """
        self.base_results_dir = Path(base_results_dir)
        self.cca_results_dir = self.base_results_dir / results_subdir
        self.n_components = n_components
        self.min_sessions = min_sessions_threshold
        self.anatomical_order = ANATOMICAL_ORDER

        # Primary data container: (region_i, region_j) → [session_results]
        # Each session_result contains mean_cv_R2, std_cv_R2, projections, etc.
        self.pair_data: Dict[Tuple[str, str], List[Dict]] = {}

        # Track all regions encountered
        self.all_regions: set = set()

        # Time vector for temporal projections
        self.time_vec = np.linspace(-1.5, 3.0, 226)

        # Validate directory
        if not self.cca_results_dir.exists():
            raise ValueError(f"Results directory not found: {self.cca_results_dir}")

        print("=" * 70)
        print("Oxford CCA Visualizer - Cross-Regional Connectivity")
        print("=" * 70)
        print(f"Results directory: {self.cca_results_dir}")
        print(f"Components to visualize: {self.n_components}")
        print(f"Minimum sessions threshold: {self.min_sessions}")

    def load_all_sessions(self) -> None:
        """
        Load CCA results from all sessions, organizing by region pairs.

        Iterates through all *_analysis_results.mat files and extracts
        CCA pair_results. Each pair_result is added to the corresponding
        (region_i, region_j) entry in self.pair_data.
        """
        print("\nLoading CCA results from all sessions...")
        print("-" * 50)

        result_files = list(self.cca_results_dir.glob("*_analysis_results.mat"))
        print(f"Found {len(result_files)} session files")

        for result_file in result_files:
            session_name = result_file.stem.replace("_analysis_results", "")
            self._process_session_file(result_file, session_name)

        # Report loading summary
        print("\n" + "-" * 50)
        print("Loading Summary:")
        print(f"  Total unique region pairs: {len(self.pair_data)}")
        print(f"  Total unique regions: {len(self.all_regions)}")

        # Count pairs meeting threshold
        valid_pairs = sum(1 for sessions in self.pair_data.values()
                          if len(sessions) >= self.min_sessions)
        print(f"  Pairs with ≥{self.min_sessions} sessions: {valid_pairs}")

    def _process_session_file(self, file_path: Path, session_name: str) -> None:
        """
        Process a single session file, extracting CCA results per region pair.

        Parameters:
            file_path: Path to the *_analysis_results.mat file
            session_name: Identifier for this session
        """
        try:
            data = mat73.loadmat(str(file_path))

            if 'cca_results' not in data:
                print(f"  {session_name}: No cca_results found, skipping")
                return

            cca_results = data['cca_results']

            if 'pair_results' not in cca_results:
                print(f"  {session_name}: No pair_results found")
                return

            pair_results = cca_results['pair_results']

            # Handle different data structures (list vs array)
            if isinstance(pair_results, np.ndarray):
                pair_results = pair_results.tolist() if pair_results.ndim == 1 else pair_results

            n_pairs_extracted = 0
            for pair_idx, pair_result in enumerate(pair_results):
                if self._process_pair_result(pair_result, session_name):
                    n_pairs_extracted += 1

            print(f"  {session_name}: Extracted {n_pairs_extracted} pairs")

        except Exception as e:
            print(f"  {session_name}: Error loading - {str(e)}")

    def _process_pair_result(self, pair_result: dict, session_name: str) -> bool:
        """
        Extract CCA metrics from a single pair result.

        Parameters:
            pair_result: Dictionary containing CCA results for one region pair
            session_name: Session identifier

        Returns:
            True if extraction successful, False otherwise
        """
        try:
            if not isinstance(pair_result, dict):
                return False

            # Extract region names
            region_i = self._extract_string(pair_result, 'region_i')
            region_j = self._extract_string(pair_result, 'region_j')

            if not region_i or not region_j:
                return False

            # Create canonical pair key (alphabetically ordered)
            pair_key = (region_i, region_j)

            # Track all regions
            self.all_regions.add(region_i)
            self.all_regions.add(region_j)

            # Extract cross-validated R² values
            if 'cv_results' not in pair_result:
                return False

            cv_results = pair_result['cv_results']

            if 'mean_cv_R2' not in cv_results:
                return False

            mean_cv_R2 = np.array(cv_results['mean_cv_R2']).flatten()

            # Extract standard deviation if available
            std_cv_R2 = None
            if 'std_cv_R2' in cv_results:
                std_cv_R2 = np.array(cv_results['std_cv_R2']).flatten()
            elif 'cv_R2' in cv_results:
                # Compute std from individual fold results
                cv_R2 = np.array(cv_results['cv_R2'])
                std_cv_R2 = np.std(cv_R2, axis=0)

            # Extract temporal projections if available
            projections = None
            if 'projections' in pair_result and 'components' in pair_result['projections']:
                projections = self._extract_projections(pair_result['projections'])

            # Create session entry
            session_data = {
                'session_name': session_name,
                'mean_cv_R2': mean_cv_R2,
                'std_cv_R2': std_cv_R2,
                'projections': projections,
                'n_components': len(mean_cv_R2)
            }

            # Store in pair_data
            if pair_key not in self.pair_data:
                self.pair_data[pair_key] = []

            self.pair_data[pair_key].append(session_data)
            return True

        except Exception as e:
            return False

    def _extract_projections(self, projections_data: dict) -> Optional[Dict]:
        """
        Extract temporal projection data from CCA results.

        Parameters:
            projections_data: Dictionary containing projection components

        Returns:
            Dictionary with region_i and region_j projections per component
        """
        try:
            components = projections_data['components']
            extracted = {}

            for comp_idx, comp_list in enumerate(components):
                if len(comp_list) > 0:
                    comp_data = comp_list[0]
                    if isinstance(comp_data, dict):
                        extracted[comp_idx] = {
                            'region_i_mean': np.array(comp_data.get('region_i_mean', [])).flatten(),
                            'region_j_mean': np.array(comp_data.get('region_j_mean', [])).flatten()
                        }

            return extracted if extracted else None

        except:
            return None

    def _extract_string(self, data: dict, field: str) -> Optional[str]:
        """Helper to extract string from various MATLAB data structures."""
        try:
            if field not in data:
                return None
            value = data[field]
            if isinstance(value, str):
                return value
            elif isinstance(value, np.ndarray):
                if value.dtype.kind in ['U', 'S']:
                    return str(value.item())
            elif isinstance(value, list) and len(value) > 0:
                return str(value[0])
            return None
        except:
            return None

    def _get_ordered_regions(self) -> List[str]:
        """Return available regions in anatomical order."""
        return [r for r in self.anatomical_order if r in self.all_regions]

    def _compute_rank_mask(
            self,
            pair_key: Tuple[str, str],
            component_idx: int
    ) -> bool:
        """
        Determine if a component should be displayed based on rank.

        The rank is determined by the mean $R^2$ value across sessions.
        Components are ordered by descending $R^2$, and sig1 corresponds
        to the highest, sig2 to the second highest, etc.

        Parameters:
            pair_key: Tuple of (region_i, region_j)
            component_idx: Target component index (0-based)

        Returns:
            True if this component has rank = component_idx + 1
        """
        sessions = self.pair_data.get(pair_key, [])
        if len(sessions) < self.min_sessions:
            return False

        # Compute mean R² for each component across sessions
        n_comps = min(s['n_components'] for s in sessions)
        mean_r2_per_comp = []

        for comp in range(n_comps):
            r2_values = [s['mean_cv_R2'][comp] for s in sessions
                         if comp < len(s['mean_cv_R2'])]
            mean_r2_per_comp.append(np.mean(r2_values))

        # Get rank ordering (descending by R²)
        rank_order = np.argsort(mean_r2_per_comp)[::-1]

        # Return True if this component has the requested rank
        return component_idx < len(rank_order) and rank_order[component_idx] == component_idx

    def create_connectivity_matrices_figure(
            self,
            figsize: Tuple[float, float] = (20, 12),
            save_path: Optional[str] = None
    ) -> None:
        """
        Create multi-panel connectivity matrix figure.

        Layout:
            Row 1: n_components connectivity matrices (mean $R^2$)
            Row 2: n_components standard deviation matrices

        Each matrix shows region × region values with:
            - Mean/Std computed across all sessions where that pair was recorded
            - Values ordered by component rank (sig1 = highest R², etc.)

        Parameters:
            figsize: Figure dimensions
            save_path: Output path (without extension)
        """
        print("\nCreating CCA connectivity matrices...")

        ordered_regions = self._get_ordered_regions()
        n_regions = len(ordered_regions)
        n_cols = self.n_components

        print(f"  Regions: {n_regions}")
        print(f"  Components: {n_cols}")

        # Create figure with 2 rows × n_components columns
        fig, axes = plt.subplots(2, n_cols, figsize=figsize)

        # Compute matrices for each component
        for comp_idx in range(n_cols):
            mean_matrix, std_matrix = self._compute_component_matrices(
                ordered_regions, comp_idx
            )

            # Plot mean matrix (top row)
            ax_mean = axes[0, comp_idx]
            self._plot_connectivity_matrix(
                ax_mean, mean_matrix, ordered_regions,
                title=f'Mean CV-$R^2$ [Comp {comp_idx + 1}]',
                cmap='viridis',
                vmin=0, vmax=0.6
            )

            # Plot std matrix (bottom row)
            ax_std = axes[1, comp_idx]
            self._plot_connectivity_matrix(
                ax_std, std_matrix, ordered_regions,
                title=f'Std CV-$R^2$ [Comp {comp_idx + 1}]',
                cmap='plasma',
                vmin=0, vmax=0.2
            )

        # Add row labels
        axes[0, 0].set_ylabel('Mean Across Sessions', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Std Across Sessions', fontsize=12, fontweight='bold')

        # Overall title
        # fig.suptitle(
        #     'CCA Cross-Regional Connectivity Matrices\n'
        #     f'(n ≥ {self.min_sessions} sessions per pair)',
        #     fontsize=14,
        #     fontweight='bold',
        #     y=0.98
        # )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            output_file = f"{save_path}_cca_connectivity_matrices.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_file}")

        plt.close(fig)
        print("  Connectivity matrices complete")

    def _compute_component_matrices(
            self,
            ordered_regions: List[str],
            component_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std matrices for a specific component.

        For each region pair, aggregates R² values across all sessions
        where that pair was recorded, then computes statistics.

        Parameters:
            ordered_regions: List of region names in display order
            component_idx: Which component to compute (0-based)

        Returns:
            Tuple of (mean_matrix, std_matrix) both of shape (n_regions, n_regions)
        """
        n = len(ordered_regions)
        mean_matrix = np.full((n, n), np.nan)
        std_matrix = np.full((n, n), np.nan)

        for i, region_i in enumerate(ordered_regions):
            for j, region_j in enumerate(ordered_regions):
                if i == j:
                    continue  # Diagonal remains NaN

                # Check both orderings of the pair
                pair_key = (region_i, region_j)
                alt_key = (region_j, region_i)

                sessions1 = self.pair_data.get(pair_key, [])
                sessions2 = self.pair_data.get(alt_key, [])
                sessions = sessions1+sessions2

                # if not sessions:
                #     sessions = self.pair_data.get(alt_key, [])

                if len(sessions) < self.min_sessions:
                    continue

                # Collect R² values for this component across sessions
                # Using rank-based extraction: comp 0 = highest R², etc.
                r2_values = []
                for session in sessions:
                    mean_cv_R2_O = session['mean_cv_R2']
                    mean_cv_R2 = mean_cv_R2_O[~np.isnan(mean_cv_R2_O)]
                    n_comps = len(mean_cv_R2)

                    if component_idx < n_comps:
                        # Get rank ordering for this session
                        rank_order = np.argsort(mean_cv_R2)[::-1]
                        # Extract the component with this rank
                        ranked_comp_idx = rank_order[component_idx]
                        r2_values.append(mean_cv_R2[ranked_comp_idx])

                if r2_values:
                    mean_matrix[i, j] = np.mean(r2_values)
                    std_matrix[i, j] = np.std(r2_values)

        return mean_matrix, std_matrix

    def _plot_connectivity_matrix(
            self,
            ax: plt.Axes,
            matrix: np.ndarray,
            region_labels: List[str],
            title: str,
            cmap: str = 'viridis',
            vmin: Optional[float] = None,
            vmax: Optional[float] = None
    ) -> None:
        """
        Plot a single connectivity matrix as a heatmap.

        Parameters:
            ax: Matplotlib axes object
            matrix: Square matrix to plot
            region_labels: Labels for rows/columns
            title: Subplot title
            cmap: Colormap name
            vmin, vmax: Color scale limits
        """
        # Mask NaN values for proper visualization
        masked_matrix = np.ma.masked_invalid(matrix)

        # Auto-scale if not specified
        if vmax is None:
            vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0

        # Plot heatmap
        im = ax.imshow(masked_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # Configure axes
        n = len(region_labels)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(region_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(region_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Add value annotations for non-NaN cells
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    val = matrix[i, j]
                    # Choose text color based on background
                    text_color = 'white' if val < (vmax * 0.35) else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color=text_color)

    def create_temporal_projection_figure(
            self,
            figsize: Tuple[float, float] = (60, 60),
            component_idx: int = 0,
            save_path: Optional[str] = None
    ) -> None:
        """
        Create cross-region temporal projection matrix figure.

        Similar to the original CCA_PCA_all_first_component_oxford.py
        but with improved organization and anatomical ordering.

        Parameters:
            figsize: Figure dimensions
            component_idx: Which component to visualize (0-based)
            save_path: Output path (without extension)
        """
        print(f"\nCreating temporal projection figure (Component {component_idx + 1})...")

        ordered_regions = self._get_ordered_regions()
        n_regions = len(ordered_regions)

        fig, axes = plt.subplots(n_regions, n_regions, figsize=figsize)

        for i, region_i in enumerate(ordered_regions):
            for j, region_j in enumerate(ordered_regions):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: region name
                    ax.text(0.5, 0.5, region_i, ha='center', va='center',
                            fontsize=32, fontweight='bold')
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    ax.axis('off')
                elif i > j:
                    # Lower triangle: hide
                    ax.axis('off')
                else:
                    # Upper triangle: plot projection
                    self._plot_temporal_projection(
                        ax, region_i, region_j, component_idx
                    )

        fig.suptitle(
            f'CCA Temporal Projections - Component {component_idx + 1}',
            fontsize=48, fontweight='bold', y=0.995
        )

        plt.tight_layout(rect=[0, 0.01, 1, 0.99])

        if save_path:
            output_file = f"{save_path}_cca_temporal_comp{component_idx + 1}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_file}")

        plt.close(fig)

    def _plot_temporal_projection(
            self,
            ax: plt.Axes,
            region_i: str,
            region_j: str,
            component_idx: int
    ) -> None:
        """
        Plot temporal projections for a single region pair.

        Sign alignment is performed by:
        - Identifying baseline latent from first session with positive peak
        - Computing correlation of each session with baseline
        - Flipping sessions with negative correlation

        Parameters:
            ax: Matplotlib axes
            region_i: First region name
            region_j: Second region name
            component_idx: Component to plot (0-based)
        """
        # Find pair data
        pair_key = (region_i, region_j)
        sessions = self.pair_data.get(pair_key, [])
        # Reset is_flipped flag for each pair
        is_flipped = False
        if not sessions:
            pair_key = (region_j, region_i)
            is_flipped = True
            sessions = self.pair_data.get(pair_key, [])

        if len(sessions) < self.min_sessions:
            ax.set_visible(False)
            return

        # Collect projections across sessions
        proj_i_all = []
        proj_j_all = []

        for session in sessions:
            if session['projections'] is None:
                continue
            if component_idx not in session['projections']:
                continue
            if is_flipped:
                proj_data = session['projections'][component_idx]
                proj_i = proj_data.get('region_j_mean', [])
                proj_j = proj_data.get('region_i_mean', [])

                if len(proj_i) > 0 and len(proj_j) > 0:
                    proj_i_all.append(proj_i)
                    proj_j_all.append(proj_j)
            else:
                proj_data = session['projections'][component_idx]
                proj_i = proj_data.get('region_i_mean', [])
                proj_j = proj_data.get('region_j_mean', [])

                if len(proj_i) > 0 and len(proj_j) > 0:
                    proj_i_all.append(proj_i)
                    proj_j_all.append(proj_j)


        if len(proj_i_all) < self.min_sessions:
            ax.set_visible(False)
            return

        # Truncate to minimum length
        min_len = min(min(len(p) for p in proj_i_all), min(len(p) for p in proj_j_all))
        proj_i_arr_raw = np.array([p[:min_len] for p in proj_i_all])
        proj_j_arr_raw = np.array([p[:min_len] for p in proj_j_all])

        # Align signs based on correlation with baseline
        # Find baseline for region_i (first session with positive peak)
        baseline_i_idx = None
        for idx, proj in enumerate(proj_i_arr_raw):
            peak_val = proj[np.argmax(np.abs(proj))]
            if peak_val > 0:
                baseline_i_idx = idx
                break

        if baseline_i_idx is None:
            baseline_i_idx = 0

        baseline_i = proj_i_arr_raw[baseline_i_idx]

        # Find baseline for region_j
        baseline_j_idx = None
        for idx, proj in enumerate(proj_j_arr_raw):
            peak_val = proj[np.argmax(np.abs(proj))]
            if peak_val > 0:
                baseline_j_idx = idx
                break

        if baseline_j_idx is None:
            baseline_j_idx = 0

        baseline_j = proj_j_arr_raw[baseline_j_idx]

        # Align all sessions based on correlation with baseline
        proj_i_arr = np.zeros_like(proj_i_arr_raw)
        proj_j_arr = np.zeros_like(proj_j_arr_raw)

        for idx in range(len(proj_i_arr_raw)):
            # Align region_i
            corr_i = np.corrcoef(baseline_i, proj_i_arr_raw[idx])[0, 1]
            proj_i_arr[idx] = proj_i_arr_raw[idx] if corr_i >= 0 else -proj_i_arr_raw[idx]

            # Align region_j
            corr_j = np.corrcoef(baseline_j, proj_j_arr_raw[idx])[0, 1]
            proj_j_arr[idx] = proj_j_arr_raw[idx] if corr_j >= 0 else -proj_j_arr_raw[idx]

        # Compute statistics
        mean_i = np.mean(proj_i_arr, axis=0)
        mean_j = np.mean(proj_j_arr, axis=0)
        std_i = np.std(proj_i_arr, axis=0)
        std_j = np.std(proj_j_arr, axis=0)

        # Adjust time vector
        time_vec = self.time_vec[:min_len] if len(self.time_vec) >= min_len else np.linspace(-1.5, 3.0, min_len)

        # Plot
        ax.plot(time_vec, mean_i, color='red', linewidth=2, alpha=0.9, label=region_i)
        ax.fill_between(time_vec, mean_i - std_i, mean_i + std_i, alpha=0.15, color='red')

        ax.plot(time_vec, mean_j, color='blue', linewidth=2, alpha=0.9, label=region_j)
        ax.fill_between(time_vec, mean_j - std_j, mean_j + std_j, alpha=0.15, color='blue')

        # Reference line at t=0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=3)

        # Formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.8, linestyle=':', linewidth=2)
        ax.set_yticks(np.arange(0, 10, 2))
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)
        ax.set_ylim([-2, 5])
        ax.set_xticks([-1.5, 0, 2, 3])
        ax.set_xticklabels(['-1.5', '0', '2', '3'], fontsize=20)
        ax.tick_params(axis='both', which='major', width=2, length=8)

        for spine in ax.spines.values():
            spine.set_linewidth(3)

    def create_population_summary_figure(
            self,
            figsize: Tuple[float, float] = (18, 10),
            save_path: Optional[str] = None
    ) -> None:
        """
        Create neural population summary with component $R^2$ heatmaps.

        Modelled after _create_population_summary from Single_paired_region_example_all_oxford.py.

        Layout:
            - Left column: Neural population sizes per region
            - Top right: Maximum $R^2$ connectivity matrix
            - Bottom right: Component-wise $R^2$ heatmap

        Parameters:
            figsize: Figure dimensions
            save_path: Output path
        """
        print("\nCreating population summary figure...")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 1],
                              hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[:, 0])  # Population sizes (full left column)
        ax2 = fig.add_subplot(gs[0, 1])  # Max R² connectivity matrix
        ax3 = fig.add_subplot(gs[1, 1])  # Components R² heatmap

        ordered_regions = self._get_ordered_regions()
        n_regions = len(ordered_regions)

        # =====================================================================
        # Plot 1: Region presence (number of sessions per region)
        # =====================================================================
        region_session_counts = []
        for region in ordered_regions:
            count = sum(1 for pair, sessions in self.pair_data.items()
                        if region in pair and len(sessions) >= self.min_sessions)
            region_session_counts.append(count)

        y_pos = np.arange(n_regions)
        ax1.barh(y_pos, region_session_counts, color='steelblue', alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(ordered_regions)
        ax1.set_xlabel('Number of Valid Pairs', fontsize=12)
        ax1.set_title('Region Pair Coverage', fontsize=14, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        for idx, count in enumerate(region_session_counts):
            ax1.text(count + 0.3, idx, str(count), va='center', fontsize=10)

        # =====================================================================
        # Plot 2: Maximum R² connectivity matrix
        # =====================================================================
        max_r2_matrix = np.full((n_regions, n_regions), np.nan)

        for i, region_i in enumerate(ordered_regions):
            for j, region_j in enumerate(ordered_regions):
                if i == j:
                    continue

                pair_key = (region_i, region_j)
                sessions = self.pair_data.get(pair_key, [])
                if not sessions:
                    sessions = self.pair_data.get((region_j, region_i), [])

                if len(sessions) >= self.min_sessions:
                    # Get max R² across components, averaged across sessions
                    max_r2_per_session = [np.max(s['mean_cv_R2']) for s in sessions]
                    max_r2_matrix[i, j] = np.mean(max_r2_per_session)

        im = ax2.imshow(max_r2_matrix, cmap='viridis', vmin=0, vmax=0.6,aspect='equal')
        ax2.set_xticks(np.arange(n_regions))
        ax2.set_yticks(np.arange(n_regions))
        ax2.set_xticklabels(ordered_regions, rotation=45, ha='right', fontsize=9)
        ax2.set_yticklabels(ordered_regions, fontsize=9)
        ax2.set_title('Maximum CCA $R^2$', fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Max $R^2$', fontsize=10)

        # Add annotations
        for i in range(n_regions):
            for j in range(n_regions):
                if not np.isnan(max_r2_matrix[i, j]):
                    val = max_r2_matrix[i, j]
                    text_color = 'white' if val < 0.3 else 'black'
                    ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=8, color=text_color)

        # =====================================================================
        # Plot 3: Component-wise R² heatmap (pairs × components)
        # =====================================================================
        pair_labels = []
        all_r2_values = []

        for pair_key, sessions in self.pair_data.items():
            if len(sessions) < self.min_sessions:
                continue

            # Compute mean R² per component across sessions
            n_comps = min(s['n_components'] for s in sessions)
            n_comps = min(n_comps, self.n_components)

            mean_r2_per_comp = []
            for comp in range(n_comps):
                r2_vals = [s['mean_cv_R2'][comp] for s in sessions
                           if comp < len(s['mean_cv_R2'])]
                mean_r2_per_comp.append(np.mean(r2_vals))

            # Pad if needed
            while len(mean_r2_per_comp) < self.n_components:
                mean_r2_per_comp.append(0.0)

            pair_labels.append(f"{pair_key[0]}-{pair_key[1]}")
            all_r2_values.append(mean_r2_per_comp)

        if all_r2_values:
            r2_matrix = np.array(all_r2_values).T  # Components × Pairs

            im3 = ax3.imshow(r2_matrix, aspect='auto', cmap='plasma', vmin=0,vmax=0.6)

            ax3.set_xticks(np.arange(len(pair_labels)))
            ax3.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
            ax3.set_yticks(np.arange(self.n_components))
            ax3.set_yticklabels([f'Comp {i + 1}' for i in range(self.n_components)], fontsize=10)
            ax3.set_xlabel('Region Pairs', fontsize=12)
            ax3.set_ylabel('CCA Components', fontsize=12)
            ax3.set_title('Cross-validated $R^2$ by Component', fontsize=14, fontweight='bold')

            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('$R^2$', fontsize=10)

            # # Add annotations
            # for i in range(r2_matrix.shape[0]):
            #     for j in range(r2_matrix.shape[1]):
            #         if r2_matrix[i, j] > 0:
            #             text_color = 'white' if r2_matrix[i, j] < 0.3 else 'black'
            #             ax3.text(j, i, f'{r2_matrix[i, j]:.2f}',
            #                      ha='center', va='center', fontsize=6, color=text_color)

        fig.suptitle('CCA Population Summary with Component Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            output_file = f"{save_path}_cca_population_summary.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_file}")

        plt.close(fig)
        print("  Population summary complete")

    def generate_report(self) -> None:
        """Print comprehensive summary of loaded CCA data."""
        print("\n" + "=" * 70)
        print("CCA Analysis Report")
        print("=" * 70)
        print(f"Results directory: {self.cca_results_dir}")
        print(f"Total region pairs: {len(self.pair_data)}")
        print(f"Minimum sessions threshold: {self.min_sessions}")

        print("\nPer-Pair Statistics:")
        print("-" * 50)

        for pair_key, sessions in sorted(self.pair_data.items()):
            n = len(sessions)
            if n >= self.min_sessions:
                # Get mean of max R² across sessions
                max_r2_vals = [np.max(s['mean_cv_R2']) for s in sessions]
                mean_max = np.mean(max_r2_vals)
                std_max = np.std(max_r2_vals)

                print(f"  {pair_key[0]:6s} - {pair_key[1]:6s}: "
                      f"{n:3d} sessions, "
                      f"max $R^2$ = {mean_max:.3f} ± {std_max:.3f}")


def main():
    """Demonstration of CCA visualization pipeline."""

    # Configuration
    base_dir = '/Users/shengyuancai/Downloads/Oxford_dataset'
    output_dir = Path(base_dir) / 'Paper_output' / 'figures_cca'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Oxford CCA Visualization Pipeline")
    print("=" * 70)

    # Option 1: Cued state analysis
    print("\n[1] Processing CUED state sessions...")
    cca_viz_cued = OxfordCCAVisualizer(
        base_results_dir=base_dir,
        results_subdir="sessions_cued_hit_long_results",
        n_components=5,
        min_sessions_threshold=3
    )
    cca_viz_cued.load_all_sessions()
    cca_viz_cued.generate_report()
    cca_viz_cued.create_connectivity_matrices_figure(
        figsize=(20, 12),
        save_path=str(output_dir / "cued_long")
    )
    cca_viz_cued.create_population_summary_figure(
        figsize=(18, 10),
        save_path=str(output_dir / "cued_long")
    )
    # Optional: Create temporal projection figures for each component
    for comp_idx in range(min(3, cca_viz_cued.n_components)):
        cca_viz_cued.create_temporal_projection_figure(
            figsize=(40, 40),
            component_idx=comp_idx,
            save_path=str(output_dir / "cued_long")
        )


    # Option 2: Spontaneous state analysis
    print("\n[2] Processing SPONTANEOUS state sessions...")
    cca_viz_spont_long = OxfordCCAVisualizer(
        base_results_dir=base_dir,
        results_subdir="sessions_spont_miss_long_results",
        n_components=5,
        min_sessions_threshold=3
    )
    cca_viz_spont_long.load_all_sessions()
    cca_viz_spont_long.generate_report()
    cca_viz_spont_long.create_connectivity_matrices_figure(
        figsize=(20, 12),
        save_path=str(output_dir / "spont_long")
    )
    cca_viz_spont_long.create_population_summary_figure(
        figsize=(18, 10),
        save_path=str(output_dir / "spont_long")
    )

    for comp_idx in range(min(3, cca_viz_spont_long.n_components)):
        cca_viz_spont_long.create_temporal_projection_figure(
            figsize=(40, 40),
            component_idx=comp_idx,
            save_path=str(output_dir / "spont_long")
        )



    # Option 3: Spontaneous state analysis
    print("\n[2] Processing SPONTANEOUS state sessions...")
    cca_viz_spont_short = OxfordCCAVisualizer(
        base_results_dir=base_dir,
        results_subdir="sessions_spont_short_results",
        n_components=5,
        min_sessions_threshold=3
    )
    cca_viz_spont_short.load_all_sessions()
    cca_viz_spont_short.generate_report()
    cca_viz_spont_short.create_connectivity_matrices_figure(
        figsize=(20, 12),
        save_path=str(output_dir / "spont_short")
    )
    cca_viz_spont_short.create_population_summary_figure(
        figsize=(18, 10),
        save_path=str(output_dir / "spont_short")
    )

    for comp_idx in range(min(3, cca_viz_spont_short.n_components)):
        cca_viz_spont_short.create_temporal_projection_figure(
            figsize=(40, 40),
            component_idx=comp_idx,
            save_path=str(output_dir / "spont_short")
        )


    print("\n[2] Processing SPONTANEOUS state sessions...")
    cca_viz_spont = OxfordCCAVisualizer(
        base_results_dir=base_dir,
        results_subdir="sessions_spont_hit_long_results",
        n_components=5,
        min_sessions_threshold=3
    )
    cca_viz_spont.load_all_sessions()
    cca_viz_spont.generate_report()
    cca_viz_spont.create_connectivity_matrices_figure(
        figsize=(20, 12),
        save_path=str(output_dir / "spont_long_hit")
    )
    cca_viz_spont.create_population_summary_figure(
        figsize=(18, 10),
        save_path=str(output_dir / "spont_long_hit")
    )
    for comp_idx in range(min(3, cca_viz_spont.n_components)):
        cca_viz_spont.create_temporal_projection_figure(
            figsize=(40, 40),
            component_idx=comp_idx,
            save_path=str(output_dir / "spont_long_hit")
        )

    print("\n" + "=" * 70)
    print("CCA Visualization Pipeline Complete")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Oxford Dataset CCA Connectivity Matrices Visualization
======================================================

ARCHITECTURAL DISTINCTION FROM PCA:
-----------------------------------
Canonical Correlation Analysis fundamentally requires paired observations from
two brain regions recorded simultaneously within the same session. This
constraint arises from the mathematical formulation of CCA, which seeks to
identify linear combinations of variables from two sets that maximise
their mutual correlation.

Formally, let $\mathbf{X}_1 \in \mathbb{R}^{T \times n_1}$ and
$\mathbf{X}_2 \in \mathbb{R}^{T \times n_2}$ denote the neural activity matrices
from regions 1 and 2, where $T$ represents the number of time points (which must
be identical for both regions within a session). CCA finds projection vectors
$\mathbf{a}$ and $\mathbf{b}$ such that:

$$\rho = \max_{\mathbf{a}, \mathbf{b}} \text{corr}(\mathbf{X}_1 \mathbf{a}, \mathbf{X}_2 \mathbf{b})$$

The requirement for simultaneous recording means that when aggregating CCA
results across sessions, we must only include sessions where BOTH regions
of a given pair were recorded.

FIGURE STRUCTURE:
-----------------
First row: n connectivity matrices showing mean cross-validated R² for each
           CCA component (sig1, sig2, sig3, ...), where sigX indicates the
           component ranked by R² value within each session.
Second row: n standard deviation matrices showing cross-session variability.

Author: Adapted for Oxford neurophysiology dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings
import mat73

warnings.filterwarnings('ignore')

# Configure publication-quality aesthetics
sns.set_style("white")
sns.set_context("paper", font_scale=1.0)


class OxfordCCAConnectivityVisualizer:
    """
    Visualizer for CCA connectivity matrices across brain regions.

    This class implements within-session pairing requirements where CCA
    results are only aggregated for sessions containing both regions of
    each region pair.

    Significance Ranking Strategy:
    ------------------------------
    For each session's CCA results, components are ranked by their cross-validated
    R² values in descending order. This yields 'sig1' (highest R²), 'sig2'
    (second highest), etc. This ranking normalises the comparison across sessions,
    as the raw component ordering from the CCA algorithm may vary based on
    initialisation conditions.

    Mathematically, for session $s$ with CCA results yielding $K$ components,
    we define the ranking function $\pi_s: \{1, ..., K\} \to \{1, ..., K\}$ such that:

    $$R^2_{\pi_s(1)} \geq R^2_{\pi_s(2)} \geq ... \geq R^2_{\pi_s(K)}$$

    The 'sigX' notation refers to the component with rank X under this ordering.
    """

    # Canonical anatomical ordering
    ANATOMICAL_ORDER = [
        'mPFC', 'ORB', 'MOp', 'MOs', 'OLF',  # Cortical regions
        'STR', 'STRv', 'HIPP',  # Striatal & limbic
        'MD', 'LP', 'VALVM', 'VPMPO', 'ILM',  # Thalamic nuclei
        'HY',  # Hypothalamic
        'fiber',  # Fiber tracts
        'other'  # Catch-all category
    ]

    def __init__(
            self,
            base_results_dir: str,
            session_subdir: str = 'sessions_cued_hit_long_results',
            n_components: int = 3
    ):
        """
        Initialise the CCA visualizer.

        Parameters
        ----------
        base_results_dir : str
            Base directory containing Oxford dataset results
        session_subdir : str
            Subdirectory containing session-level analysis results
        n_components : int
            Number of CCA components to display (ranked by R²)
        """
        self.base_results_dir = Path(base_results_dir)
        self.results_dir = self.base_results_dir / session_subdir
        self.n_components = n_components

        # Data containers: organised by region pair
        # pair_cca_data[(region1, region2)] = {'cv_R2': [session_values], 'session_names': [...]}
        self.pair_cca_data: Dict[Tuple[str, str], Dict[str, List]] = defaultdict(
            lambda: {'cv_R2': [], 'session_names': []}
        )
        self.available_regions: List[str] = []
        self.all_region_pairs: List[Tuple[str, str]] = []

    def load_all_session_cca_data(self) -> None:
        """
        Load CCA results from all sessions, maintaining within-session pairing.

        For each region pair, we accumulate the cross-validated R² values
        across all sessions where both regions were recorded. This respects
        the fundamental requirement that CCA operates on paired data.
        """
        print("=" * 70)
        print("Loading CCA Results: Within-Session Pairing Required")
        print("=" * 70)
        print(f"Results directory: {self.results_dir}")

        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        # Locate all session analysis files
        session_files = list(self.results_dir.glob("*_analysis_results.mat"))
        print(f"Found {len(session_files)} session files\n")

        all_regions = set()

        for session_file in session_files:
            session_name = session_file.stem.replace("_analysis_results", "")
            regions_in_session = self._process_session_cca(session_file, session_name)
            all_regions.update(regions_in_session)

        # Sort regions anatomically
        self.available_regions = self._sort_regions_anatomically(list(all_regions))

        # Extract unique region pairs from loaded data
        self.all_region_pairs = list(self.pair_cca_data.keys())

        print("\n" + "-" * 50)
        print("Summary of Loaded CCA Data:")
        print(f"Total unique regions: {len(self.available_regions)}")
        print(f"Total region pairs: {len(self.all_region_pairs)}")
        for pair, data in self.pair_cca_data.items():
            print(f"  {pair[0]} <-> {pair[1]}: {len(data['session_names'])} sessions")
        print("-" * 50)

    def _process_session_cca(self, session_file: Path, session_name: str) -> List[str]:
        """
        Extract CCA cross-validated R² for all region pairs in a single session.

        Parameters
        ----------
        session_file : Path
            Path to the session's analysis results .mat file
        session_name : str
            Identifier for the session

        Returns
        -------
        List[str]
            List of region names present in this session
        """
        regions_found = []

        try:
            data = mat73.loadmat(str(session_file))

            if 'cca_results' not in data:
                print(f"  [{session_name}] No cca_results found, skipping")
                return regions_found

            cca_results = data['cca_results']

            # The CCA results structure contains pair_results
            if 'pair_results' not in cca_results:
                print(f"  [{session_name}] No pair_results found, skipping")
                return regions_found

            pair_results = cca_results['pair_results']

            # Handle different possible data structures
            if isinstance(pair_results, list):
                pair_list = pair_results
            elif isinstance(pair_results, np.ndarray):
                pair_list = pair_results.tolist() if pair_results.ndim == 1 else [pair_results]
            else:
                pair_list = [pair_results]

            print(f"  [{session_name}] Processing {len(pair_list)} region pairs")

            for pair_result in pair_list:
                if pair_result is None:
                    continue

                # Extract region names and cv_R2
                region1 = self._extract_field(pair_result, 'region_i')
                region2 = self._extract_field(pair_result, 'region_j')
                cv_R2 = self._extract_field(pair_result[], 'mean_cv_R2')

                if region1 is None or region2 is None or cv_R2 is None:
                    continue

                # Ensure consistent ordering of region pair (alphabetical)
                if region1 > region2:
                    region1, region2 = region2, region1

                regions_found.extend([region1, region2])

                # Convert cv_R2 to array
                cv_R2 = np.asarray(cv_R2).flatten()

                # Store in pair_cca_data
                pair_key = (region1, region2)
                self.pair_cca_data[pair_key]['cv_R2'].append(cv_R2)
                self.pair_cca_data[pair_key]['session_names'].append(session_name)

        except Exception as e:
            print(f"  [{session_name}] Error processing: {str(e)}")

        return list(set(regions_found))

    def _extract_field(self, data: Any, field_name: str) -> Any:
        """
        Safely extract a field from various data structures.

        Parameters
        ----------
        data : Any
            Data structure (dict, object, etc.)
        field_name : str
            Name of the field to extract

        Returns
        -------
        Any
            The extracted value or None if not found
        """
        if isinstance(data, dict):
            return data.get(field_name)
        elif hasattr(data, field_name):
            return getattr(data, field_name)
        elif isinstance(data, np.ndarray) and data.dtype.names is not None:
            if field_name in data.dtype.names:
                return data[field_name]
        return None

    def _sort_regions_anatomically(self, regions: List[str]) -> List[str]:
        """
        Sort region names according to the canonical anatomical hierarchy.
        """
        known_regions = [r for r in self.ANATOMICAL_ORDER if r in regions]
        unknown_regions = sorted([r for r in regions if r not in self.ANATOMICAL_ORDER])
        return known_regions + unknown_regions

    def compute_connectivity_matrices(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute mean and std connectivity matrices for each CCA component.

        For each component (ranked by R² value), we construct:
        1. A mean connectivity matrix where element (i, j) represents the
           mean cross-validated R² between regions i and j across sessions.
        2. A standard deviation matrix quantifying cross-session variability.

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Dictionary with keys 'mean' and 'std', each containing matrices
            indexed by component number.
        """
        n_regions = len(self.available_regions)

        # Initialise storage for each component
        result = {
            'mean': {},
            'std': {},
            'n_sessions': np.zeros((n_regions, n_regions), dtype=int)
        }

        for comp_idx in range(self.n_components):
            result['mean'][comp_idx] = np.full((n_regions, n_regions), np.nan)
            result['std'][comp_idx] = np.full((n_regions, n_regions), np.nan)

        # Create region name to index mapping
        region_to_idx = {r: i for i, r in enumerate(self.available_regions)}

        # Populate matrices from pair data
        for (region1, region2), pair_data in self.pair_cca_data.items():
            if region1 not in region_to_idx or region2 not in region_to_idx:
                continue

            idx1 = region_to_idx[region1]
            idx2 = region_to_idx[region2]

            # Stack all sessions' cv_R2 values
            all_cv_R2 = pair_data['cv_R2']
            n_sessions = len(all_cv_R2)

            if n_sessions == 0:
                continue

            result['n_sessions'][idx1, idx2] = n_sessions
            result['n_sessions'][idx2, idx1] = n_sessions

            # For each component, rank by R² across sessions and compute statistics
            for comp_idx in range(self.n_components):
                comp_values = []

                for cv_R2 in all_cv_R2:
                    # Rank components by R² value (descending)
                    sorted_indices = np.argsort(cv_R2)[::-1]

                    if comp_idx < len(sorted_indices):
                        ranked_idx = sorted_indices[comp_idx]
                        comp_values.append(cv_R2[ranked_idx])

                if comp_values:
                    mean_val = np.mean(comp_values)
                    std_val = np.std(comp_values, ddof=1) if len(comp_values) > 1 else 0.0

                    # Fill both (i,j) and (j,i) for symmetric matrix
                    result['mean'][comp_idx][idx1, idx2] = mean_val
                    result['mean'][comp_idx][idx2, idx1] = mean_val
                    result['std'][comp_idx][idx1, idx2] = std_val
                    result['std'][comp_idx][idx2, idx1] = std_val

        return result

    def create_connectivity_figure(
            self,
            figsize: Optional[Tuple[float, float]] = None,
            save_path: Optional[str] = None,
            dpi: int = 300,
            cmap_mean: str = 'viridis',
            cmap_std: str = 'Reds'
    ) -> plt.Figure:
        """
        Create multi-panel figure with mean and std connectivity matrices.

        Structure:
        - First row: n_components mean R² connectivity matrices (sig1, sig2, ...)
        - Second row: n_components standard deviation matrices

        Parameters
        ----------
        figsize : Optional[Tuple[float, float]]
            Figure dimensions; if None, computed automatically
        save_path : Optional[str]
            If provided, save the figure to this path
        dpi : int
            Resolution for saved figure
        cmap_mean : str
            Colormap for mean R² matrices
        cmap_std : str
            Colormap for standard deviation matrices

        Returns
        -------
        plt.Figure
            The generated matplotlib figure
        """
        # Compute connectivity matrices
        matrices = self.compute_connectivity_matrices()

        n_comp = self.n_components
        n_regions = len(self.available_regions)

        # Auto-compute figure size if not provided
        if figsize is None:
            width_per_panel = max(6, n_regions * 0.5)
            figsize = (width_per_panel * n_comp, width_per_panel * 2.2)

        # Create figure: 2 rows (mean, std) × n_comp columns
        fig, axes = plt.subplots(2, n_comp, figsize=figsize)

        # Handle single component case
        if n_comp == 1:
            axes = axes.reshape(2, 1)

        print(f"\nCreating connectivity matrices for {n_comp} components")
        print(f"Regions ({n_regions}): {self.available_regions}")

        # Determine global color limits for consistent scaling
        all_mean_vals = np.concatenate([
            matrices['mean'][c][~np.isnan(matrices['mean'][c])]
            for c in range(n_comp)
        ])
        all_std_vals = np.concatenate([
            matrices['std'][c][~np.isnan(matrices['std'][c])]
            for c in range(n_comp)
        ])

        mean_vmin, mean_vmax = 0, np.nanmax(all_mean_vals) if len(all_mean_vals) > 0 else 1
        std_vmin, std_vmax = 0, np.nanmax(all_std_vals) if len(all_std_vals) > 0 else 0.5

        # Plot mean matrices (first row)
        for comp_idx in range(n_comp):
            ax = axes[0, comp_idx]
            mean_matrix = matrices['mean'][comp_idx]

            # Mask the diagonal (self-connections are undefined for CCA)
            masked_matrix = np.ma.masked_where(np.eye(n_regions, dtype=bool), mean_matrix)

            im = ax.imshow(
                masked_matrix, cmap=cmap_mean, aspect='equal',
                vmin=mean_vmin, vmax=mean_vmax
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Mean cv-R²', fontsize=10)

            # Configure axis labels
            ax.set_xticks(np.arange(n_regions))
            ax.set_yticks(np.arange(n_regions))
            ax.set_xticklabels(self.available_regions, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(self.available_regions, fontsize=9)

            ax.set_title(f'Mean cv-R² (sig{comp_idx + 1})', fontsize=12, fontweight='bold')

            # Add value annotations for non-NaN cells
            for i in range(n_regions):
                for j in range(n_regions):
                    if i != j and not np.isnan(mean_matrix[i, j]):
                        val = mean_matrix[i, j]
                        text_color = 'white' if val > (mean_vmax - mean_vmin) * 0.6 + mean_vmin else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=7, color=text_color)

        # Plot std matrices (second row)
        for comp_idx in range(n_comp):
            ax = axes[1, comp_idx]
            std_matrix = matrices['std'][comp_idx]

            # Mask the diagonal
            masked_matrix = np.ma.masked_where(np.eye(n_regions, dtype=bool), std_matrix)

            im = ax.imshow(
                masked_matrix, cmap=cmap_std, aspect='equal',
                vmin=std_vmin, vmax=std_vmax
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Std cv-R²', fontsize=10)

            # Configure axis labels
            ax.set_xticks(np.arange(n_regions))
            ax.set_yticks(np.arange(n_regions))
            ax.set_xticklabels(self.available_regions, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(self.available_regions, fontsize=9)

            ax.set_title(f'Std cv-R² (sig{comp_idx + 1})', fontsize=12, fontweight='bold')

            # Add value annotations
            for i in range(n_regions):
                for j in range(n_regions):
                    if i != j and not np.isnan(std_matrix[i, j]):
                        val = std_matrix[i, j]
                        text_color = 'white' if val > (std_vmax - std_vmin) * 0.6 + std_vmin else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=7, color=text_color)

        # Add overall title
        fig.suptitle(
            'CCA Cross-Regional Connectivity: Cross-Validated R² Statistics\n'
            '(Components ranked by R² value within each session)',
            fontsize=14, fontweight='bold', y=1.02
        )

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            suffix = save_path.suffix.lower()
            if suffix == '.pdf':
                fig.savefig(save_path, format='pdf', dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

            print(f"\nFigure saved to: {save_path}")

        return fig

    def create_single_component_summary(
            self,
            component_idx: int = 0,
            figsize: Tuple[float, float] = (12, 10),
            save_path: Optional[str] = None,
            dpi: int = 300
    ) -> plt.Figure:
        """
        Create a detailed summary figure for a single CCA component.

        This provides a more detailed view of a specific component, showing
        the connectivity matrix with session counts annotated.

        Parameters
        ----------
        component_idx : int
            Which ranked component to visualize (0 = sig1, 1 = sig2, etc.)
        figsize : Tuple[float, float]
            Figure dimensions
        save_path : Optional[str]
            If provided, save the figure to this path
        dpi : int
            Resolution for saved figure

        Returns
        -------
        plt.Figure
            The generated matplotlib figure
        """
        matrices = self.compute_connectivity_matrices()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        mean_matrix = matrices['mean'][component_idx]
        std_matrix = matrices['std'][component_idx]
        n_sessions_matrix = matrices['n_sessions']
        n_regions = len(self.available_regions)

        # Left panel: Mean R²
        ax = axes[0]
        masked_matrix = np.ma.masked_where(np.eye(n_regions, dtype=bool), mean_matrix)
        im = ax.imshow(masked_matrix, cmap='viridis', aspect='equal', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Mean cv-R²', fontsize=11)

        ax.set_xticks(np.arange(n_regions))
        ax.set_yticks(np.arange(n_regions))
        ax.set_xticklabels(self.available_regions, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(self.available_regions, fontsize=10)
        ax.set_title(f'Mean cv-R² (Component {component_idx + 1})', fontsize=13, fontweight='bold')

        # Annotate with values and session counts
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j and not np.isnan(mean_matrix[i, j]):
                    val = mean_matrix[i, j]
                    n_sess = n_sessions_matrix[i, j]
                    text_color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}\n(n={n_sess})', ha='center', va='center',
                            fontsize=7, color=text_color)

        # Right panel: Standard deviation
        ax = axes[1]
        masked_matrix = np.ma.masked_where(np.eye(n_regions, dtype=bool), std_matrix)
        vmax_std = np.nanmax(std_matrix) if np.nansum(~np.isnan(std_matrix)) > 0 else 0.5
        im = ax.imshow(masked_matrix, cmap='Reds', aspect='equal', vmin=0, vmax=vmax_std)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Std cv-R²', fontsize=11)

        ax.set_xticks(np.arange(n_regions))
        ax.set_yticks(np.arange(n_regions))
        ax.set_xticklabels(self.available_regions, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(self.available_regions, fontsize=10)
        ax.set_title(f'Std cv-R² (Component {component_idx + 1})', fontsize=13, fontweight='bold')

        # Annotate
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j and not np.isnan(std_matrix[i, j]):
                    val = std_matrix[i, j]
                    text_color = 'white' if val > vmax_std * 0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=8, color=text_color)

        fig.suptitle(
            f'CCA Component {component_idx + 1} (Ranked by R²): Connectivity Statistics\n'
            f'(Values show mean R² and session count)',
            fontsize=14, fontweight='bold', y=1.02
        )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        return fig


def main():
    """
    Demonstration of the CCA connectivity matrices visualization.

    This function illustrates the proper usage of the visualizer class
    for generating publication-quality connectivity matrices showing
    cross-regional communication strength across brain regions.
    """
    # Configuration
    base_dir = '/Users/shengyuancai/Downloads/Oxford_dataset'

    # Select experimental condition
    session_subdir = 'sessions_cued_hit_long_results'  # or 'sessions_spont_short_results'

    output_dir = Path(base_dir) / 'Paper_output' / 'figures_cca'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Oxford Dataset CCA Connectivity Visualization")
    print("=" * 60)
    print(f"Session type: {session_subdir}")
    print(f"Output directory: {output_dir}")
    print()

    # Initialise and run visualizer
    visualizer = OxfordCCAConnectivityVisualizer(
        base_results_dir=base_dir,
        session_subdir=session_subdir,
        n_components=3  # Display top 3 ranked components
    )

    # Load all session data (within-session pairing required)
    visualizer.load_all_session_cca_data()

    # Generate multi-panel connectivity figure
    fig = visualizer.create_connectivity_figure(
        save_path=str(output_dir / 'cca_connectivity_matrices.png'),
        dpi=300
    )

    # Also save as PDF
    fig.savefig(
        output_dir / 'cca_connectivity_matrices.pdf',
        format='pdf',
        bbox_inches='tight'
    )
    plt.close(fig)

    # Generate detailed single-component summary for sig1
    fig_detail = visualizer.create_single_component_summary(
        component_idx=0,
        save_path=str(output_dir / 'cca_component1_detailed.png'),
        dpi=300
    )
    plt.close(fig_detail)

    print("\n" + "=" * 60)
    print("CCA Visualization Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
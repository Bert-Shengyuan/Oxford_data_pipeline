#!/usr/bin/env python3
"""
Oxford Dataset Multi-Component Visualization - Corrected PCA Handling
=====================================================================
This corrected version properly handles the fundamental structural difference
between CCA and PCA data organization:

- CCA: Organized by REGION PAIRS (supervised cross-regional analysis)
- PCA: Organized by INDIVIDUAL REGIONS (unsupervised within-region analysis)

Scientific Rationale:
--------------------
CCA inherently requires paired regions since it identifies shared variance.
PCA operates on each region independently, identifying intrinsic variance structure.

When visualizing PCA results for a region pair, we retrieve the independent
PCA projections from each region and display them together for comparison with
the CCA results which were computed jointly for that pair.

Author: Corrected Implementation for Region-Based PCA
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
import mat73
from datetime import datetime

warnings.filterwarnings('ignore')

sns.set_style("white")
sns.set_context("paper", font_scale=0.8)


class OxfordMultiComponentVisualizer:
    """
    Visualizer supporting both CCA (pair-based) and PCA (region-based) analysis.

    Key Architectural Insight:
    -------------------------
    CCA results: cca_results.pair_results[i] contains data for region pair (region_i, region_j)
    PCA results: pca_results.REGION.projections contains data for single region REGION

    To visualize a region pair in PCA mode, we retrieve independent PCA results
    from both regions and combine them for visualization.
    """

    def __init__(self, base_results_dir: str, n_components: int = 5,
                 region_order: Optional[List[str]] = None,
                 analysis_type: str = 'CCA'):
        self.base_results_dir = Path(base_results_dir)
        self.cca_results_dir = self.base_results_dir / "sessions_cued_hit_long_results"
        self.pca_results_dir = self.cca_results_dir
        self.n_components = n_components
        self.analysis_type = analysis_type.upper()
        self.region_data = {}
        self.region_names = set()
        self.time_vec = np.linspace(-1.5, 3.0, 226)
        self.region_order = region_order
        self.filtered_stats = {}
        self.threshold = 3

        if self.analysis_type not in ['CCA', 'PCA']:
            raise ValueError("analysis_type must be either 'CCA' or 'PCA'")

    def load_oxford_session_data(self, min_sessions: int = 5):
        """Load analysis results with proper handling of structural differences."""
        print(f"Loading Oxford {self.analysis_type} results...")

        if self.analysis_type == 'CCA':
            self._load_cca_results()
        else:
            self._load_pca_results()

    def _load_cca_results(self):
        """Load CCA results organized by region pairs."""
        print(f"Loading CCA results from: {self.cca_results_dir}")

        if not self.cca_results_dir.exists():
            raise ValueError(f"CCA results directory not found: {self.cca_results_dir}")

        cca_files = list(self.cca_results_dir.glob("*_cca_results.mat"))
        print(f"Found {len(cca_files)} session CCA result files")

        for cca_file in cca_files:
            session_name = cca_file.stem.replace("_cca_results", "")
            print(f"Processing CCA session: {session_name}")

            try:
                cca_data = mat73.loadmat(str(cca_file))

                if 'cca_results' in cca_data:
                    cca_results = cca_data['cca_results']

                    if 'pair_results' in cca_results:
                        pair_results = cca_results['pair_results']

                        if isinstance(pair_results, list):
                            pair_results_list = pair_results
                        elif isinstance(pair_results, np.ndarray):
                            pair_results_list = pair_results.tolist() if pair_results.ndim == 1 else pair_results[0]
                        else:
                            continue

                        for pair_idx, pair_result in enumerate(pair_results_list):
                            self._process_cca_pair_result(pair_result, pair_idx, session_name)

            except Exception as e:
                print(f"Error processing {cca_file}: {str(e)}")
                continue

    def _load_pca_results(self):
        """
        Load PCA results organized by individual regions.

        Structure from MATLAB:
        pca_results.MD.projections.mean{comp_idx} = 1x305 double
        pca_results.OLF.projections.mean{comp_idx} = 1x305 double
        """
        print(f"Loading PCA results from: {self.pca_results_dir}")

        if not self.pca_results_dir.exists():
            raise ValueError(f"PCA results directory not found: {self.pca_results_dir}")

        pca_files = list(self.pca_results_dir.glob("*_analysis_results.mat"))
        print(f"Found {len(pca_files)} session PCA result files")

        for pca_file in pca_files:
            session_name = pca_file.stem.replace("_analysis_results", "")
            print(f"Processing PCA session: {session_name}")

            try:
                pca_data = mat73.loadmat(str(pca_file))

                if 'pca_results' in pca_data:
                    pca_results = pca_data['pca_results']

                    # Extract all available regions
                    available_regions = [key for key in pca_results.keys()
                                         if isinstance(pca_results[key], dict) and
                                         'projections' in pca_results[key]]

                    print(f"  Found PCA for regions: {available_regions}")

                    for region in available_regions:
                        self.region_names.add(region)

                    # Create pseudo-pairs
                    for i, region1 in enumerate(available_regions):
                        for region2 in available_regions[i:]:
                            if region1 != region2:
                                self._process_pca_region_pair(
                                    pca_results, region1, region2, session_name
                                )

            except Exception as e:
                print(f"Error processing {pca_file}: {str(e)}")
                continue

    def _process_cca_pair_result(self, pair_result, pair_idx, session_name):
        """Process CCA pair results."""
        try:
            if isinstance(pair_result, dict):
                pair_data = pair_result
            elif isinstance(pair_result, np.ndarray) and pair_result.size > 0:
                pair_data = pair_result.item() if pair_result.ndim == 0 else pair_result
            else:
                return

            region1 = self._extract_string(pair_data, 'region_i')
            region2 = self._extract_string(pair_data, 'region_j')

            if region1 and region2:
                pair_key = (region1, region2)

                if pair_key not in self.region_data:
                    self.region_data[pair_key] = {
                        'projections': [[] for _ in range(self.n_components)],
                        'R2_values': [[] for _ in range(self.n_components)],
                        'session_info': []
                    }

                self._extract_cca_projections(pair_data, pair_key, session_name)
                self.region_names.add(region1)
                self.region_names.add(region2)

        except Exception as e:
            print(f"Error processing CCA pair: {str(e)}")

    def _process_pca_region_pair(self, pca_results, region1, region2, session_name):
        """Process PCA results for a region pair by combining independent regional analyses."""
        try:
            pair_key = (region1, region2)

            if pair_key not in self.region_data:
                self.region_data[pair_key] = {
                    'projections': [[] for _ in range(self.n_components)],
                    'R2_values': [[] for _ in range(self.n_components)],
                    'session_info': []
                }

            region1_data = pca_results[region1]
            region2_data = pca_results[region2]

            if 'projections' not in region1_data or 'projections' not in region2_data:
                return

            proj1 = region1_data['projections']
            proj2 = region2_data['projections']

            if 'mean' in proj1 and 'mean' in proj2:
                means1 = proj1['mean']
                means2 = proj2['mean']

                n_comps = min(self.n_components, len(means1), len(means2))

                for comp_idx in range(n_comps):
                    region1_proj = self._extract_array_from_cell(means1, comp_idx)
                    region2_proj = self._extract_array_from_cell(means2, comp_idx)

                    if region1_proj is not None and region2_proj is not None:
                        projection_data = {
                            'region1_mean': region1_proj,
                            'region2_mean': region2_proj,
                            'session': session_name
                        }

                        self.region_data[pair_key]['projections'][comp_idx].append(projection_data)

                        # Use explained variance as R² proxy for PCA
                        if 'explained' in region1_data and 'explained' in region2_data:
                            exp1 = region1_data['explained']
                            exp2 = region2_data['explained']
                            avg_exp = (float(exp1[comp_idx]) + float(exp2[comp_idx])) / 2.0
                            self.region_data[pair_key]['R2_values'][comp_idx].append(avg_exp / 100.0)
                        else:
                            self.region_data[pair_key]['R2_values'][comp_idx].append(0.0)

                self.region_data[pair_key]['session_info'].append(session_name)

        except Exception as e:
            print(f"Error processing PCA pair {region1}-{region2}: {str(e)}")

    def _extract_cca_projections(self, pair_data, pair_key, session_name):
        """Extract CCA canonical projections."""
        try:
            if 'canonical_projections' not in pair_data:
                return

            projections = pair_data['canonical_projections']

            if 'cv_results' in pair_data:
                cv_results = pair_data['cv_results']

                if 'mean_cv_R2' in cv_results:
                    mean_cv_R2 = cv_results['mean_cv_R2']
                    if isinstance(mean_cv_R2, np.ndarray):
                        mean_cv_R2 = mean_cv_R2.flatten()

                    for comp_idx in range(min(self.n_components, len(mean_cv_R2))):
                        self.region_data[pair_key]['R2_values'][comp_idx].append(
                            float(mean_cv_R2[comp_idx])
                        )

            for comp_idx in range(min(self.n_components, len(projections))):
                comp_proj = projections[comp_idx]

                if isinstance(comp_proj, dict):
                    region1_proj = self._extract_projection(comp_proj, 'region1_projection')
                    region2_proj = self._extract_projection(comp_proj, 'region2_projection')

                    if region1_proj is not None and region2_proj is not None:
                        region1_mean = np.mean(region1_proj, axis=0)
                        region2_mean = np.mean(region2_proj, axis=0)

                        projection_data = {
                            'region1_mean': region1_mean,
                            'region2_mean': region2_mean,
                            'session': session_name
                        }

                        self.region_data[pair_key]['projections'][comp_idx].append(projection_data)

            self.region_data[pair_key]['session_info'].append(session_name)

        except Exception as e:
            print(f"Error extracting CCA projections: {str(e)}")

    def _extract_array_from_cell(self, cell_array, index):
        """Extract numpy array from MATLAB cell array."""
        try:
            if isinstance(cell_array, list):
                data = cell_array[index]
            elif isinstance(cell_array, np.ndarray):
                data = cell_array[index]
            else:
                return None

            if isinstance(data, np.ndarray):
                return data.flatten()
            elif isinstance(data, (list, tuple)):
                return np.array(data).flatten()
            else:
                return None

        except Exception as e:
            return None

    def _extract_string(self, data_struct, field_name):
        """Extract string fields from mat73 structures."""
        try:
            if isinstance(data_struct, dict) and field_name in data_struct:
                field_value = data_struct[field_name]

                if isinstance(field_value, str):
                    return field_value
                elif isinstance(field_value, np.ndarray):
                    if field_value.dtype.kind in ['U', 'S']:
                        return str(field_value.item())
                elif isinstance(field_value, list) and len(field_value) > 0:
                    return str(field_value[0])

            return None
        except:
            return None

    def _extract_projection(self, comp_data, field_name):
        """Extract projection data arrays."""
        try:
            if field_name in comp_data:
                proj_data = comp_data[field_name]

                if isinstance(proj_data, np.ndarray):
                    return proj_data
                elif isinstance(proj_data, list):
                    return np.array(proj_data)

            return None
        except:
            return None

    def create_component_figures(self, figsize=(60, 60), save_path: str = None):
        """Create cross-region merged figures for all components."""
        print(f"\nCreating {self.analysis_type} component figures...")

        all_regions = sorted(list(self.region_names))

        if self.region_order is None:
            self.region_order = all_regions

        print(f"Regions: {self.region_order}")

        for comp_idx in range(self.n_components):
            print(f"Component {comp_idx + 1}...")

            n_regions = len(self.region_order)
            fig, axes = plt.subplots(n_regions, n_regions, figsize=figsize)

            if n_regions == 1:
                axes = np.array([[axes]])
            elif axes.ndim == 1:
                axes = axes.reshape(1, -1)

            for i, region1 in enumerate(self.region_order):
                for j, region2 in enumerate(self.region_order):
                    ax = axes[i, j]

                    if i == j:
                        ax.text(0.5, 0.5, region1, ha='center', va='center',
                                fontsize=32, fontweight='bold')
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.axis('off')
                    elif i>j:
                        ax.axis('off')
                    elif i < j:
                        self._plot_component_pair(ax, region1, region2, comp_idx)

            analysis_label = "Canonical Correlation" if self.analysis_type == 'CCA' else "Principal Component"
            fig.suptitle(f'{analysis_label} Analysis - Component {comp_idx + 1}',
                         fontsize=48, fontweight='bold', y=0.995)

            plt.tight_layout(rect=[0, 0.01, 1, 0.99])

            if save_path:
                output_file = f"{save_path}_{self.analysis_type.lower()}_component_{comp_idx + 1}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Saved: {output_file}")

            plt.close(fig)

    def _plot_component_pair(self, ax, region1, region2, comp_idx):
        """Plot component projections for region pair."""
        session_means_r1 = []
        session_means_r2 = []

        region1 = str(region1).strip()
        region2 = str(region2).strip()
        threshold = self.threshold
        if region1 in self.region_names and region2 in self.region_names:
            pair_data = self._get_pair_data(region1, region2)

            if (pair_data and comp_idx < len(pair_data['projections']) and
                    len(pair_data['projections'][comp_idx]) >= threshold):

                for proj in pair_data['projections'][comp_idx]:
                    session_means_r1.append(np.abs(proj['region1_mean']))
                    session_means_r2.append(np.abs(proj['region2_mean']))

        if len(session_means_r1) < threshold:
            ax.set_visible(False)
            return

        grand_mean1 = np.mean(np.array(session_means_r1), axis=0)
        grand_mean2 = np.mean(np.array(session_means_r2), axis=0)
        grand_std_1 = np.std(np.array(session_means_r1), axis=0)
        grand_std_2 = np.std(np.array(session_means_r2), axis=0)

        ax.plot(self.time_vec, grand_mean1, color='red', linewidth=2, alpha=0.9)
        ax.fill_between(self.time_vec, grand_mean1 - grand_std_1,
                        grand_mean1 + grand_std_1, alpha=0.15, color='red')

        ax.plot(self.time_vec, grand_mean2, color='blue', linewidth=2, alpha=0.9)
        ax.fill_between(self.time_vec, grand_mean2 - grand_std_2,
                        grand_mean2 + grand_std_2, color='blue', alpha=0.15)

        if -1.5 <= 0 <= 3.0:
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.8, linestyle=':', linewidth=2)

        ax.set_yticks(np.arange(0, 10, 2))
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)
        ax.set_ylim([0, 10])
        ax.set_xticks([-1.5, 0, 2, 3])
        ax.set_xticklabels(['-1.5', '0', '2', '3'], fontsize=20)
        ax.tick_params(axis='both', which='major', width=2, length=8)

        for spine in ax.spines.values():
            spine.set_linewidth(3)

    def _get_pair_data(self, region1, region2):
        """Retrieve pair data."""
        if (region1, region2) in self.region_data:
            return self.region_data[(region1, region2)]
        elif (region2, region1) in self.region_data:
            return self.region_data[(region2, region1)]
        return None


def main():
    """Demonstration of corrected CCA and PCA visualization."""
    base_dir = '/Users/shengyuancai/Downloads/Oxford_dataset'
    output_dir = Path(base_dir) / 'Paper_output' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Corrected CCA/PCA Visualization")
    print("=" * 60)

    # CCA
    # print("\nGenerating CCA...")
    # cca_viz = OxfordMultiComponentVisualizer(
    #     base_results_dir=base_dir,
    #     n_components=5,
    #     analysis_type='CCA'
    # )
    # cca_viz.load_oxford_session_data(min_sessions=2)
    # cca_viz.create_component_figures(
    #     figsize=(60, 24),
    #     save_path=str(output_dir / "oxford_merged")
    # )

    # PCA
    print("\nGenerating PCA...")
    pca_viz = OxfordMultiComponentVisualizer(
        base_results_dir=base_dir,
        n_components=5,
        analysis_type='PCA'
    )
    pca_viz.load_oxford_session_data(min_sessions=1)
    pca_viz.create_component_figures(
        figsize=(40, 40),
        save_path=str(output_dir / "oxford_merged")
    )

    print("\nComplete!")


if __name__ == "__main__":
    main()
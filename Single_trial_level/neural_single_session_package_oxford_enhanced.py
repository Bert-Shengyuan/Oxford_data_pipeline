#!/usr/bin/env python3
"""
Oxford Dataset Single Session Neural Analysis Package - CORRECTED PCA Handling
============================================================================

This corrected version properly handles the fundamental difference between:
- CCA: Pair-based results structure (cca_results.pair_results)
- PCA: Region-based results structure (pca_results.REGION_NAME)

Key Corrections:
---------------
1. Region pair extraction now uses CCA structure only (since pairs require paired analysis)
2. PCA weight extraction retrieves data from individual regions independently
3. Dual index bars display CCA weights (from pair analysis) and PCA weights (from region analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
import mat73
from scipy.stats import zscore

try:
    from rastermap import Rastermap

    RASTERMAP_AVAILABLE = True

except ImportError:
    RASTERMAP_AVAILABLE = False
    print("Warning: Rastermap not installed. Install with: pip install rastermap")


class OxfordSingleSessionAnalyzer:
    """
    CORRECTED analyzer supporting both CCA and PCA with proper structure handling.
    """

    def __init__(self, base_results_dir, session_name, region_pair=None):
        """
        Initialize the analyzer for a specific Oxford session.

        Note: We always load both CCA and PCA results regardless of specified
        analysis_type, since we want to compare them. Region pairs are defined
        by CCA structure (which inherently requires pairs).
        """
        self.base_results_dir = Path(base_results_dir)
        self.results_dir = self.base_results_dir / "session_analysis_results"
        self.session_name = session_name
        self.region_pair = region_pair

        # Initialize data containers
        self.session_data = None
        self.cca_results = None
        self.pca_results = None
        self.neural_data = None
        self.time_bins = None
        self.rastermap_results = {}
        self.global_rastermap_results = None
        self.available_regions = []

        # Create output directory
        self.output_dir = Path(f"oxford_single_session_analysis/{self.session_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initialized Oxford Session Analyzer")
        print(f"Session: {self.session_name}")

    # def load_session_data(self):
    #     """
    #     Load CCA, PCA results and neural data from session_analysis_results.
    #
    #     CRITICAL: Region pairs are extracted from CCA results only, since
    #     CCA inherently analyzes pairs. PCA data is then retrieved for those
    #     same regions to enable comparison.
    #     """
    #     # Load combined analysis results file
    #     results_file = self.results_dir / f"{self.session_name}_analysis_results.mat"
    #     if not results_file.exists():
    #         raise FileNotFoundError(f"Analysis results file not found: {results_file}")
    #
    #     print(f"Loading analysis results from: {results_file}")
    #     try:
    #         self.session_data = mat73.loadmat(str(results_file))
    #         print("Successfully loaded analysis results with mat73")
    #     except Exception as e:
    #         print(f"mat73 failed, trying scipy.io: {e}")
    #         self.session_data = sio.loadmat(str(results_file))
    #
    #     # Extract time bins from region_data
    #     if 'region_data' in self.session_data and 'time_axis' in self.session_data['region_data']:
    #         self.time_bins = np.array(self.session_data['region_data']['time_axis']).flatten()
    #     else:
    #         self.time_bins = np.linspace(-1.5, 3, 305)
    #
    #     # Extract available regions
    #     if 'region_data' in self.session_data and 'regions' in self.session_data['region_data']:
    #         regions_data = self.session_data['region_data']['regions']
    #         self.available_regions = list(regions_data.keys())
    #         print(f"Available regions: {', '.join(self.available_regions)}")
    #
    #     # Extract CCA results
    #     if 'cca_results' in self.session_data:
    #         self.cca_results = self.session_data['cca_results']
    #         print("Successfully loaded CCA results")
    #
    #     # Extract PCA results
    #     if 'pca_results' in self.session_data:
    #         self.pca_results = self.session_data['pca_results']
    #         print("Successfully loaded PCA results")
    #         print(f"Available PCA regions: {list(self.pca_results.keys())}")
    #
    #     print("Session data loaded successfully")
    #
    #     # Extract available region pairs (from CCA only)
    #     self._extract_region_pairs()

    def _extract_region_pairs(self):
        """
        Extract region pairs from CCA results.

        CRITICAL CORRECTION: We ONLY use CCA results to define pairs, since
        CCA inherently analyzes pairs while PCA analyzes individual regions.
        """
        self.available_pairs = []

        if self.cca_results is None:
            print("WARNING: No CCA results available - cannot define region pairs")
            return

        if 'pair_results' not in self.cca_results:
            print("WARNING: No pair_results in CCA structure")
            return

        pair_results = self.cca_results['pair_results']

        for pair_idx, pair_result in enumerate(pair_results):
            if isinstance(pair_result, dict):
                region_i = self._extract_string(pair_result, 'region_i')
                region_j = self._extract_string(pair_result, 'region_j')

                if region_i and region_j:
                    self.available_pairs.append((region_i, region_j, pair_idx))

        print(f"Found {len(self.available_pairs)} region pairs (from CCA):")
        for r1, r2, idx in self.available_pairs:
            print(f"  {idx}: {r1} vs {r2}")

    def select_region_pair(self, region_i=None, region_j=None):
        """Select a specific region pair for analysis."""
        if region_i and region_j:
            for r1, r2, idx in self.available_pairs:
                if (r1 == region_i and r2 == region_j) or (r1 == region_j and r2 == region_i):
                    self.region_pair = (r1, r2)
                    self.pair_idx = idx
                    print(f"Selected region pair: {r1} vs {r2}")
                    return True
            print(f"Region pair {region_i} vs {region_j} not found")
            return False
        else:
            if self.available_pairs:
                r1, r2, idx = self.available_pairs[0]
                self.region_pair = (r1, r2)
                self.pair_idx = idx
                print(f"Using first available pair: {r1} vs {r2}")
                return True
            return False

    # def extract_neural_activity_matrices(self):
    #     """Extract neural activity matrices for the selected region pair from session data."""
    #     if not self.region_pair:
    #         raise ValueError("No region pair selected. Call select_region_pair() first.")
    #
    #     region1, region2 = self.region_pair
    #     print(f"\nExtracting neural activity for {region1} vs {region2}")
    #
    #     if 'region_data' not in self.session_data or 'regions' not in self.session_data['region_data']:
    #         raise ValueError("No region data found in session results")
    #
    #     regions_data = self.session_data['region_data']['regions']
    #
    #     region_data = {}
    #     for region_name in [region1, region2]:
    #         if region_name in regions_data:
    #             region_info = regions_data[region_name]
    #             if 'spike_data' in region_info:
    #                 spike_data = region_info['spike_data']
    #                 region_data[region_name] = spike_data
    #                 print(f"{region_name}: {spike_data.shape} (neurons x time x trials)")
    #             else:
    #                 print(f"WARNING: No spike_data found for {region_name}")
    #         else:
    #             print(f"WARNING: Region {region_name} not found in session data")
    #
    #     self.neural_data = region_data
    #     print("Neural activity matrices extracted successfully")

    # def compute_global_rastermap(self, n_clusters=10, n_PCs=200, locality=0.3):
    #     """
    #     Compute global rastermap sorting by pooling all viable regions.
    #     Uses trial-level firing rates reshaped as (neurons, time*trials).
    #     """
    #     if not RASTERMAP_AVAILABLE:
    #         print("Rastermap not available")
    #         return None
    #
    #     if not self.session_data or 'region_data' not in self.session_data:
    #         raise ValueError("Session data not loaded. Call load_session_data() first.")
    #
    #     print("\n" + "=" * 70)
    #     print("COMPUTING GLOBAL RASTERMAP SORTING (ALL REGIONS)")
    #     print("=" * 70)
    #
    #     all_neural_data = []
    #     neuron_region_mapping = []
    #     region_neuron_indices = {}
    #     cumulative_idx = 0
    #
    #     # Extract neural data from all available regions
    #     regions_data = self.session_data['region_data']['regions']
    #
    #     for region_name in sorted(self.available_regions):
    #         if region_name in regions_data and 'spike_data' in regions_data[region_name]:
    #             spike_data = regions_data[region_name]['spike_data']
    #             n_neurons, n_time, n_trials = spike_data.shape
    #
    #             # Reshape to (neurons, time*trials) for Rastermap
    #             reshaped_data = spike_data.reshape(n_neurons, n_time * n_trials)
    #
    #             all_neural_data.append(reshaped_data)
    #
    #             # Store region information
    #             region_neuron_indices[region_name] = {
    #                 'start': cumulative_idx,
    #                 'end': cumulative_idx + n_neurons,
    #                 'n_neurons': n_neurons
    #             }
    #             cumulative_idx += n_neurons
    #
    #             neuron_region_mapping.extend([region_name] * n_neurons)
    #             print(f"  {region_name}: {n_neurons} neurons, {n_time} time bins, {n_trials} trials")
    #
    #     if not all_neural_data:
    #         raise ValueError("No neural data found in any region")
    #
    #     pooled_data = np.vstack(all_neural_data)
    #     print(f"\nPooled data: {pooled_data.shape} (total_neurons x time*trials)")
    #
    #     # Normalize data
    #     pooled_data_norm = zscore(pooled_data, axis=1, nan_policy='omit')
    #     pooled_data_norm = np.nan_to_num(pooled_data_norm, nan=0.0)
    #
    #     # Fit Rastermap model
    #     model = Rastermap(n_clusters=n_clusters, n_PCs=n_PCs, locality=locality, time_lag_window=5)
    #     model.fit(pooled_data_norm)
    #
    #     # Store global results
    #     self.global_rastermap_results = {
    #         'sorting_idx': model.isort,
    #         'embedding': model.embedding,
    #         'pooled_data': pooled_data,
    #         'pooled_data_norm': pooled_data_norm,
    #         'neuron_region_mapping': neuron_region_mapping,
    #         'region_neuron_indices': region_neuron_indices,
    #         'n_total_neurons': pooled_data.shape[0],
    #         'n_time': n_time,
    #         'n_trials': n_trials
    #     }
    #
    #     print(f"\nGlobal rastermap complete: {len(model.isort)} neurons sorted")
    #     return self.global_rastermap_results

    # def perform_rastermap_analysis(self, n_clusters=10, n_PCs=200, locality=0.3):
    def perform_rastermap_analysis(self):
        """
        Extract region-specific rastermap data from global sorting.
        For each region in the current pair, extracts the relevant indices
        and data from the global rastermap results.
        """
        if not RASTERMAP_AVAILABLE:
            print("Rastermap not available")
            return

        if self.neural_data is None:
            raise ValueError("Neural data not extracted")

        if self.global_rastermap_results is None:
            raise ValueError("Global rastermap not computed. Call compute_global_rastermap() first.")

        print("\n" + "=" * 70)
        print("EXTRACTING REGION-SPECIFIC DATA FROM GLOBAL RASTERMAP")
        print("=" * 70)

        global_sort = self.global_rastermap_results['sorting_idx']
        region_indices = self.global_rastermap_results['region_start_indices']
        n_time = self.global_rastermap_results['n_time']
        n_trials = self.global_rastermap_results['n_trials']

        for region_name, spike_data in self.neural_data.items():
            print(f"\nProcessing: {region_name}")

            if region_name not in region_indices:
                print(f"  WARNING: {region_name} not found in global rastermap results")
                continue

            # Get indices for this region in the global sorting
            region_info = region_indices[region_name]
            start_idx = region_info['start']
            end_idx = region_info['end']
            n_neurons = region_info['n_neurons']

            # Find where this region's neurons appear in the global sorting
            region_mask = (global_sort >= start_idx) & (global_sort < end_idx)
            region_positions_in_global = np.where(region_mask)[0]

            # Map back to original neuron indices within this region
            region_neurons_sorted = global_sort[region_mask] - start_idx

            # Prepare trial-averaged data for visualization

            transpose_spike_data = np.transpose(spike_data, (1, 2, 0))
            # reshaped_data = transpose_spike_data.reshape(-1, transpose_spike_data.shape[2]).T
            # reshaped_data = zscore(reshaped_data, axis=1)


            #trial_averaged_test = np.mean(spike_data, axis=0)

            trial_averaged = np.mean(transpose_spike_data, axis=2)


            # test = transpose_spike_data[1,:,:]
            # test_mean = np.mean(test, axis=1)


            # Average across trials
            trial_averaged_norm = zscore(trial_averaged, axis=1)
            trial_averaged_norm = np.nan_to_num(trial_averaged_norm, nan=0.0)
            trial_averaged_norm = trial_averaged_norm[region_neurons_sorted,:]
            # Store results
            self.rastermap_results[region_name] = {
                'sorting_idx': region_neurons_sorted,  # Indices within this region
                'global_positions': region_positions_in_global,  # Positions in global sorting
                'trial_averaged': trial_averaged,
                'trial_averaged_norm': trial_averaged_norm,
                'spike_data': spike_data,  # Keep full trial-level data
                'n_neurons': n_neurons,
                'n_time': n_time,
                'n_trials': n_trials
            }

            print(f"  Region neurons: {n_neurons}")
            print(f"  Positions in global sorting: {len(region_positions_in_global)} neurons")

    def create_dual_analysis_visualization(self, component_idx=0, save_fig=True,
                                           n_neurons_show=50, n_trials_to_show=5):
        """
        Create enhanced visualization with new layout:
        - Column 1: Both regions' Rastermap-ordered PSTHs (trial-averaged)
        - Columns 2-3: PCA and CCA index bars
        - Column 4: Example trials from both regions
        """
        if self.neural_data is None or not self.rastermap_results:
            raise ValueError("Run extract_neural_activity_matrices() and perform_rastermap_analysis() first")

        if self.global_rastermap_results is None:
            raise ValueError("Run compute_global_rastermap() first")

        print(f"\nCreating enhanced dual visualization for component {component_idx + 1}")

        # Extract CCA weights (from pair structure)
        cca_weights = self._extract_cca_weights(component_idx) if self.cca_results else None

        # Extract PCA weights (from individual region structures)
        pca_weights = self._extract_pca_weights(component_idx) if self.pca_results else None

        # Create figure with new layout
        fig = plt.figure(figsize=(28, 16))
        fontsize = 18

        # Define grid: 2 rows for 2 regions, 4 columns
        gs = fig.add_gridspec(2, 4, width_ratios=[3, 1, 1, 2],
                              hspace=0.25, wspace=0.3)

        region_names = list(self.neural_data.keys())

        for region_idx, region_name in enumerate(region_names):
            # Column 1: Rastermap-ordered PSTH
            ax_psth = fig.add_subplot(gs[region_idx, 0])

            # Column 2: CCA weights bar
            ax_cca = fig.add_subplot(gs[region_idx, 1])

            # Column 3: PCA weights bar
            ax_pca = fig.add_subplot(gs[region_idx, 2])

            # Column 4: Example trials
            ax_trials = fig.add_subplot(gs[region_idx, 3])

            # Get rastermap data for this region
            if region_name not in self.rastermap_results:
                print(f"No rastermap results for {region_name}")
                continue

            raster_data = self.rastermap_results[region_name]

            # # Extract regional indices from global sorting
            # global_positions = raster_data['global_positions']
            selected_regional_neurons = raster_data['sorting_idx']
            #
            # # Select neurons to show based on global positions
            # n_available = len(global_positions)
            # if n_available < n_neurons_show:
            #     selected_global_positions = global_positions
            #     selected_regional_neurons = regional_sort_idx
            # else:
            #     # Sample evenly from global positions
            #     step = n_available // n_neurons_show
            #     indices = np.arange(0, n_available, step)[:n_neurons_show]
            #     selected_global_positions = global_positions[indices]
            #     selected_regional_neurons = regional_sort_idx[indices]

            # Plot 1: Rastermap-ordered PSTH (trial-averaged)
            sorted_psth = raster_data['trial_averaged_norm']
            vmax = np.percentile(np.abs(sorted_psth), 98)
            im = ax_psth.imshow(sorted_psth, aspect='auto', cmap='RdBu_r',vmin=-vmax, vmax=vmax,
                                extent=[self.time_bins[0], self.time_bins[-1],
                                        len(sorted_psth), 0])
            cbar = plt.colorbar(im, ax=ax_psth, fraction=0.046, pad=0.04)
            cbar.set_label('Z-scored firing rate', fontsize=fontsize)
            cbar.ax.tick_params(axis='both', labelsize=fontsize)
            if region_idx == 1:
                ax_psth.set_xlabel('Time (s)', fontsize=fontsize)
            ax_psth.set_ylabel('Neurons (Global sorted)', fontsize=fontsize)
            ax_psth.set_title(f'{region_name} - Rastermap Ordered Activity',
                              fontsize=fontsize, fontweight='normal')
            ax_psth.tick_params(axis='both', labelsize=fontsize - 2)
            ax_psth.axvline(x=0, color='red', linestyle='--', linewidth=3,
                            label='Bar Off' if region_idx == 0 else '')
            if region_idx == 0:
                ax_psth.legend(fontsize=fontsize - 4)

            # Plot 2: CCA weights bar
            if cca_weights and region_name in cca_weights:
                weights = cca_weights[region_name]
                # Get weights for selected neurons
                selected_weights = weights[selected_regional_neurons]

                ax_cca.barh(np.arange(len(selected_weights)), selected_weights,
                            color='steelblue', alpha=0.8)
                ax_cca.set_ylim([len(selected_weights) - 0.5, -0.5])
                ax_cca.set_xlabel('CCA Weight', fontsize=fontsize - 2)
                ax_cca.set_title('CCA', fontsize=fontsize)
                ax_cca.tick_params(axis='both', labelsize=fontsize - 4)
                ax_cca.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
                ax_cca.spines['top'].set_visible(False)
                ax_cca.spines['right'].set_visible(False)
            else:
                ax_cca.text(0.5, 0.5, 'No CCA\nweights',
                            transform=ax_cca.transAxes,
                            ha='center', va='center', fontsize=fontsize - 2)
                ax_cca.set_xticks([])
                ax_cca.set_yticks([])

            # Plot 3: PCA weights bar
            if pca_weights and region_name in pca_weights:
                weights = pca_weights[region_name]
                # Get weights for selected neurons
                selected_weights = weights[selected_regional_neurons]

                ax_pca.barh(np.arange(len(selected_weights)), selected_weights,
                            color='darkgreen', alpha=0.8)
                ax_pca.set_ylim([len(selected_weights) - 0.5, -0.5])
                ax_pca.set_xlabel('PCA Weight', fontsize=fontsize - 2)
                ax_pca.set_title('PCA', fontsize=fontsize)
                ax_pca.tick_params(axis='both', labelsize=fontsize - 4)
                ax_pca.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
                ax_pca.spines['top'].set_visible(False)
                ax_pca.spines['right'].set_visible(False)
            else:
                ax_pca.text(0.5, 0.5, 'No PCA\nweights',
                            transform=ax_pca.transAxes,
                            ha='center', va='center', fontsize=fontsize - 2)
                ax_pca.set_xticks([])
                ax_pca.set_yticks([])

            # Plot 4: Example trials
            spike_data = raster_data['spike_data']
            n_trials = spike_data.shape[0]

            np.random.seed(42)
            # Select random trials to show
            if n_trials < n_trials_to_show:
                trial_indices = np.arange(n_trials)
            else:
                trial_indices = np.random.choice(n_trials, n_trials_to_show, replace=False)
                trial_indices = np.sort(trial_indices)

            # Create subplot grid for trials
            trial_data = []

            for trial_idx in trial_indices:
                trial_activity = spike_data[trial_idx,selected_regional_neurons, : ]
                trial_data.append(trial_activity)

            # Stack trials vertically with small gap
            combined_trials = np.vstack([trial_data[i] for i in range(len(trial_data))])

            im_trials = ax_trials.imshow(combined_trials, aspect='auto', cmap='hot',
                                         extent=[self.time_bins[0], self.time_bins[-1],
                                                 len(combined_trials), 0])

            ax_trials.set_xlabel('Time (s)', fontsize=fontsize)
            ax_trials.set_title(f'Example Trials', fontsize=fontsize)
            ax_trials.tick_params(axis='both', labelsize=fontsize - 4)
            ax_trials.axvline(x=0, color='cyan', linestyle='--', linewidth=2)

            # Add trial boundaries
            for i in range(1, len(trial_data)):
                y_pos = i * len(selected_regional_neurons)
                ax_trials.axhline(y=y_pos, color='white', linestyle='-', linewidth=1)

            # Label trials
            ax_trials.set_ylabel('Trials', fontsize=fontsize)
            trial_positions = [(i + 0.5) * len(selected_regional_neurons)
                               for i in range(len(trial_data))]
            ax_trials.set_yticks(trial_positions)
            ax_trials.set_yticklabels([f'T{trial_indices[i] + 1}'
                                       for i in range(len(trial_data))],
                                      fontsize=fontsize - 4)

        # # Add overall title
        # plt.suptitle(f'Dual Analysis Visualization - Component {component_idx + 1}\n'
        #              f'{self.region_pair[0]} vs {self.region_pair[1]}',
        #              fontsize=fontsize + 2, fontweight='bold')

        if save_fig:
            fig_path = self.output_dir / f"enhanced_dual_viz_comp{component_idx + 1}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {fig_path}")

        return fig

    def _extract_cca_weights(self, component_idx):
        """
        Extract CCA weights from PAIR structure.

        Returns: {region_name: weight_vector}
        """
        if self.cca_results is None:
            return None

        weights = {}

        try:
            pair_result = self.cca_results['pair_results'][self.pair_idx]

            if 'mean_A_matrix' not in pair_result and  'mean_B_matrix' not in pair_result:
                print("No canonical_weights in CCA pair result")
                return None

            weight_matrices_A = pair_result['mean_A_matrix']
            weight_matrices_B = pair_result['mean_B_matrix']

            if component_idx >= len(weight_matrices_A):
                print(f"Component {component_idx} not available in CCA")
                return None

            comp_weights_A = weight_matrices_A[:,component_idx].flatten()
            comp_weights_B = weight_matrices_B[:,component_idx].flatten()
            # Extract weights for each region in the pair
            region1_name = self.region_pair[0]
            region2_name = self.region_pair[1]

            # weights[region1_name] = np.abs(comp_weights_A)
            # weights[region2_name] = np.abs(comp_weights_B)

            weights[region1_name] = comp_weights_A
            weights[region2_name] = comp_weights_B

        except Exception as e:
            print(f"Error extracting CCA weights: {e}")
            return None

        return weights

    def _extract_pca_weights(self, component_idx):
        """
        Extract PCA weights from REGION structure.

        CRITICAL: PCA is organized by region, not pairs!
        pca_results.REGION.components{comp_idx}.coefficients

        Returns: {region_name: weight_vector}
        """
        if self.pca_results is None:
            return None

        weights = {}

        try:
            for region_name in self.region_pair:
                if region_name not in self.pca_results:
                    print(f"Region {region_name} not in PCA results")
                    continue

                region_pca = self.pca_results[region_name]

                if 'coefficients' not in region_pca:
                    print(f"No components in PCA for {region_name}")
                    continue

                components = region_pca['coefficients']

                if component_idx >= len(components):
                    print(f"Component {component_idx} not available in PCA for {region_name}")
                    continue

                # weights[region_name]  = np.abs(components[:,component_idx].flatten())
                weights[region_name] = components[:, component_idx].flatten()

        except Exception as e:
            print(f"Error extracting PCA weights: {e}")
            import traceback
            traceback.print_exc()
            return None

        return weights if weights else None

    # def _get_intersected_neurons(self, region_name, cca_weights, pca_weights, n_neurons_show):
    #     """Get neuron indices using intersection of global rastermap and CCA/PCA weights."""
    #     global_sort = self.global_rastermap_results['sorting_idx']
    #     region_map = self.global_rastermap_results['neuron_region_mapping']
    #
    #     region_neurons_in_global = np.array([i for i, r in enumerate(region_map) if r == region_name])
    #     global_top = region_neurons_in_global[:n_neurons_show * 3]
    #
    #     cca_top = None
    #     if cca_weights and region_name in cca_weights:
    #         cca_w = cca_weights[region_name]
    #         cca_top_idx = np.argsort(cca_w)[::-1][:n_neurons_show * 3]
    #         cca_top = region_neurons_in_global[cca_top_idx]
    #
    #     pca_top = None
    #     if pca_weights and region_name in pca_weights:
    #         pca_w = pca_weights[region_name]
    #         pca_top_idx = np.argsort(pca_w)[::-1][:n_neurons_show * 3]
    #         pca_top = region_neurons_in_global[pca_top_idx]
    #
    #     sets_to_intersect = [set(global_top)]
    #     if cca_top is not None:
    #         sets_to_intersect.append(set(cca_top))
    #     if pca_top is not None:
    #         sets_to_intersect.append(set(pca_top))
    #
    #     intersected = set.intersection(*sets_to_intersect)
    #     intersected = np.array(sorted(intersected))
    #
    #     if len(intersected) < n_neurons_show // 2:
    #         print(f"  Small intersection ({len(intersected)}), using global top")
    #         return global_top[:n_neurons_show]
    #
    #     return intersected[:n_neurons_show]

    # def _plot_index_bar(self, ax, weights, selected_neurons, global_sort, title, foresize):
    #     """Plot index bar showing neuron weights."""
    #     sorted_weights = weights[global_sort]
    #     selected_weights = sorted_weights[selected_neurons]
    #
    #     ax.barh(range(len(selected_weights)), selected_weights, color='steelblue', edgecolor='black')
    #     ax.set_ylabel('Neuron', fontsize=foresize - 4)
    #     ax.set_xlabel('|Weight|', fontsize=foresize - 4)
    #     ax.set_title(title, fontsize=foresize - 2, fontweight='bold')
    #     ax.tick_params(axis='both', labelsize=foresize - 6)
    #     ax.set_ylim([-0.5, len(selected_weights) - 0.5])

    def _plot_example_trials(self, ax, region_name, selected_neurons, global_sort,
                             n_trials_to_show, foresize):
        """Plot example trial rasters."""
        psth_matrix = self.neural_data[region_name]
        n_trials = min(n_trials_to_show, psth_matrix.shape[2])

        trial_raster = []
        for trial_idx in range(n_trials):
            trial_data = psth_matrix[:, :, trial_idx]
            sorted_trial = trial_data[global_sort]
            selected_trial = sorted_trial[selected_neurons]

            trial_data_norm = (selected_trial - selected_trial.min(axis=1, keepdims=True)) / \
                              (selected_trial.max(axis=1, keepdims=True) -
                               selected_trial.min(axis=1, keepdims=True) + 1e-8)
            trial_raster.append(trial_data_norm)

        combined_raster = np.vstack([np.vstack([r, np.ones((5, r.shape[1])) / 0])
                                     for r in trial_raster])[:-5]

        ax.imshow(combined_raster, aspect='auto', cmap='plasma',
                  extent=[self.time_bins[0], self.time_bins[-1], 0, combined_raster.shape[0]])

        ax.set_xlabel('Time (s)', fontsize=foresize)
        ax.set_title(f'Example Trials ({n_trials})', fontsize=foresize)
        ax.set_yticks([])
        ax.tick_params(axis='both', labelsize=foresize)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=7)

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

    def generate_summary_report(self):
        """Generate summary report."""
        report_path = self.output_dir / "analysis_summary.txt"

        with open(report_path, 'w') as f:
            f.write(f"Oxford Single Session Analysis Report\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Region Pair: {self.region_pair[0]} vs {self.region_pair[1]}\n")
            f.write(f"CCA Available: {'Yes' if self.cca_results else 'No'}\n")
            f.write(f"PCA Available: {'Yes' if self.pca_results else 'No'}\n")

            if self.global_rastermap_results:
                f.write(f"\nGlobal Rastermap:\n")
                f.write(f"  Total neurons: {self.global_rastermap_results['n_total_neurons']}\n")

            f.write(f"\nGenerated Figures:\n")
            for fig_file in self.output_dir.glob("*.png"):
                f.write(f"  - {fig_file.name}\n")

        print(f"Summary saved: {report_path}")
        return report_path


def main_enhanced_oxford_analysis():
    """Demonstration with corrected PCA handling."""
    BASE_RESULTS_DIR = "/Users/shengyuancai/Downloads/Oxford_dataset"
    SESSION_NAME = "yp013_220211"

    print("=" * 70)
    print("OXFORD ANALYSIS - CORRECTED PCA HANDLING")
    print("=" * 70)

    analyzer = OxfordSingleSessionAnalyzer(
        base_results_dir=BASE_RESULTS_DIR,
        session_name=SESSION_NAME
    )

    # print("\nLoading session data...")
    # analyzer.load_session_data()

    print("\nSelecting region pair...")
    if not analyzer.select_region_pair():
        print("No region pairs available")
        return

    print("\nExtracting neural matrices...")
    analyzer.extract_neural_activity_matrices()

    if RASTERMAP_AVAILABLE:
        # print("\nComputing global rastermap...")
        # analyzer.compute_global_rastermap(n_clusters=10, n_PCs=200, locality=0.3)

        print("\nComputing region-specific rastermap...")
        analyzer.perform_rastermap_analysis()

        print("\nCreating dual CCA/PCA visualization...")
        for comp_idx in range(3):
            try:
                analyzer.create_dual_analysis_visualization(
                    component_idx=comp_idx,
                    save_fig=True,
                    n_neurons_show=50
                )
            except Exception as e:
                print(f"Could not visualize component {comp_idx + 1}: {e}")
                import traceback
                traceback.print_exc()

    analyzer.generate_summary_report()

    print(f"\nâœ“ Analysis complete!")
    print(f"Results: {analyzer.output_dir}")

    return analyzer


if __name__ == "__main__":
    analyzer = main_enhanced_oxford_analysis()
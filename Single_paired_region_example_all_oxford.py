#!/usr/bin/env python3
"""
Oxford Dataset Complete Single Session Integration - CORRECTED PCA Handling
==========================================================================

This corrected version properly handles the architectural difference:
- CCA: Defines region pairs (pair-based analysis)
- PCA: Provides independent region analyses (region-based)

Key Correction:
--------------
Region pairs are ALWAYS extracted from CCA results (since pairs require
paired analysis). For those pairs, we overlay PCA information by retrieving
independent analyses from each region.

This enables comparison of supervised (CCA) versus unsupervised (PCA)
dimensionality reduction for the same neural populations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Import corrected analyzer
from neural_single_session_package_oxford_enhanced import (
    OxfordSingleSessionAnalyzer, RASTERMAP_AVAILABLE
)


class OxfordIntegratedAnalysisPipeline:
    """
    CORRECTED pipeline that properly handles CCA pair-based and PCA region-based structures.
    """

    def __init__(self, config):
        """Initialize the integrated analysis pipeline."""
        self.config = config
        self.analyzers = {}
        self.session_summary = {}

        # Add containers for loaded session data
        self.session_data = None
        self.available_regions = []
        self.region_neural_data = {}

        required_keys = ['base_results_dir', 'session_name']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")

        self.config.setdefault('n_components_visualize', 3)
        self.config.setdefault('output_base_dir', 'oxford_integrated_analysis')
        self.config.setdefault('region_pairs', None)

        print("Oxford Integrated Analysis Pipeline Initialized")
        print("=" * 60)
        self.print_configuration()

    def print_configuration(self):
        """Display configuration."""
        print("Analysis Configuration:")
        print("-" * 30)
        for key, value in self.config.items():
            if isinstance(value, (str, int, float, bool)):
                print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) < 5:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: <{type(value).__name__}>")
        print()

    def run_complete_analysis(self):
        """Execute complete analysis pipeline."""
        print("\nStarting Oxford Dataset Analysis Pipeline")
        print("=" * 60)

        analysis_start_time = datetime.now()

        success = self.load_and_validate_session()
        if not success:
            print("Session validation failed")
            return False

        success = self.analyze_region_pairs
        if not success:
            print("Region pair analysis failed")
            return False

        success = self.create_comprehensive_visualizations()
        if not success:
            print("Visualization creation failed")
            return False

        success = self.generate_integrated_summary()
        if not success:
            print("Summary generation failed, but analysis is complete")

        analysis_duration = (datetime.now() - analysis_start_time).total_seconds()

        print(f"\nComplete Analysis Pipeline Successful!")
        print(f"Total analysis time: {analysis_duration:.1f} seconds")
        print("=" * 60)
        self.print_results_summary()

        return True

    def load_and_validate_session(self):
        """Load and validate session data - loads once for entire pipeline."""
        print("\nStep 1: Loading and Validating Session Data")
        print("-" * 40)

        try:
            # Load the session analysis results file
            results_file = Path(self.config['base_results_dir']) / \
                           f"{self.config['session_name']}_analysis_results.mat"

            if not results_file.exists():
                raise FileNotFoundError(f"Analysis results file not found: {results_file}")

            print(f"Loading session data from: {results_file}")

            # Load using mat73
            try:
                import mat73
                self.session_data = mat73.loadmat(str(results_file))
                print("Successfully loaded session data with mat73")
            except Exception as e:
                print(f"mat73 failed, trying scipy.io: {e}")
                import scipy.io as sio
                self.session_data = sio.loadmat(str(results_file))

            # Extract available regions from the data structure
            if 'region_data' in self.session_data and 'regions' in self.session_data['region_data']:
                regions_data = self.session_data['region_data']['regions']
                self.available_regions = list(regions_data.keys())
                print(f"Found {len(self.available_regions)} regions: {', '.join(self.available_regions)}")

                # Extract neural data for each region
                for region in self.available_regions:
                    if 'spike_data' in regions_data[region]:
                        #spike_data = regions_data[region]['spike_data']
                        select_neurons = regions_data[region]['selected_neurons'].reshape(-1).astype(int) - 1
                        # spike_data = regions_data[region_name]['spike_data']
                        spike_data = regions_data[region]['spike_data'][:, select_neurons, :]
                        self.region_neural_data[region] = spike_data
                        print(f"  {region}: {spike_data.shape} (trials x neurons x time)")
                    else:
                        print(f"  WARNING: No spike_data found for region {region}")

            # Extract CCA results for available pairs
            if 'cca_results' in self.session_data and 'pair_results' in self.session_data['cca_results']:
                cca_pair_results = self.session_data['cca_results']['pair_results']
                available_pairs = []

                for idx, pair_result in enumerate(cca_pair_results):
                    if isinstance(pair_result, dict):
                        region_i = self._extract_string(pair_result, 'region_i')
                        region_j = self._extract_string(pair_result, 'region_j')
                        if region_i and region_j:
                            available_pairs.append((region_i, region_j, idx))

                self.session_summary['available_pairs'] = available_pairs
                print(f"\nFound {len(available_pairs)} region pairs in CCA results")

            else:
                print("WARNING: No CCA results found in session data")
                self.session_summary['available_pairs'] = []

            # Set up region pairs to analyze
            self.session_summary['session_name'] = self.config['session_name']
            self.session_summary['n_region_pairs'] = len(self.session_summary['available_pairs'])

            if self.config['region_pairs'] is None:
                self.config['region_pairs'] = [(r1, r2) for r1, r2, _ in self.session_summary['available_pairs']]
                print(f"  Will analyze all {len(self.config['region_pairs'])} available pairs")
            else:
                print(f"  Will analyze specified {len(self.config['region_pairs'])} pairs")

            if RASTERMAP_AVAILABLE:
                print("  Running Rastermap analysis...")
                # First compute global rastermap if not already done
                if not hasattr(self, 'global_rastermap_results') or self.global_rastermap_results is None:
                    print("  Computing global Rastermap for all regions...")
                    self.global_rastermap_results = self._compute_global_rastermap_all_regions()

            return True

        except Exception as e:
            print(f"Failed to load session data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_string(self, data_dict, field_name):
        """Helper to extract string from MATLAB data structures."""
        try:
            if field_name in data_dict:
                field_value = data_dict[field_name]
                if isinstance(field_value, str):
                    return field_value
                elif isinstance(field_value, np.ndarray):
                    if field_value.size == 1:
                        return str(field_value.item())
                    elif field_value.dtype.char == 'U':
                        return str(field_value[0])
                elif isinstance(field_value, list) and len(field_value) > 0:
                    return str(field_value[0])
            return None
        except:
            return None

    @property
    def analyze_region_pairs(self):
        """Analyze each specified region pair using already-loaded data."""
        print("\nStep 2: Analyzing Region Pairs")
        print("-" * 40)

        successful_pairs = 0

        for pair_idx, (region1, region2) in enumerate(self.config['region_pairs']):
            print(f"\nAnalyzing pair {pair_idx + 1}/{len(self.config['region_pairs'])}: "
                  f"{region1} vs {region2}")

            try:
                # Create analyzer with pre-loaded data
                analyzer = OxfordSingleSessionAnalyzer(
                    base_results_dir=self.config['base_results_dir'],
                    session_name=self.config['session_name']
                )

                # Pass the already-loaded session data to the analyzer
                analyzer.session_data = self.session_data
                analyzer.cca_results = self.session_data.get('cca_results', None)
                analyzer.pca_results = self.session_data.get('pca_results', None)

                # Extract available pairs from the session data
                analyzer.available_pairs = self.session_summary['available_pairs']

                # Set time bins if available
                if 'timepoints' in self.session_data.get('region_data', {}):
                    analyzer.time_bins = np.linspace(-1.5, 3,int(self.session_data['region_data']['timepoints'].flatten()[0]))
                else:
                    analyzer.time_bins = np.linspace(-1.5, 3, 226)

                pair_output_dir = Path(self.config['output_base_dir']) / \
                                  self.config['session_name'] / \
                                  f"{region1}_vs_{region2}"
                analyzer.output_dir = pair_output_dir
                analyzer.output_dir.mkdir(parents=True, exist_ok=True)

                # Select the region pair
                if not analyzer.select_region_pair(region1, region2):
                    print(f"Could not find pair {region1} vs {region2}")
                    continue

                # Extract neural data from already-loaded data
                analyzer.neural_data = {}
                regions_data = self.session_data['region_data']['regions']

                for region_name in [region1, region2]:
                    if region_name in regions_data and 'spike_data' in regions_data[region_name]:
                        #pike_data = regions_data[region_name]['spike_data']
                        select_neurons = regions_data[region_name]['selected_neurons'].reshape(-1).astype(int) - 1
                        # spike_data = regions_data[region_name]['spike_data']
                        spike_data = regions_data[region_name]['spike_data'][:, select_neurons, :]
                        analyzer.neural_data[region_name] = spike_data
                        print(f"  {region_name}: {spike_data.shape} (neurons x time x trials)")
                    else:
                        print(f"  WARNING: No spike data found for {region_name}")
                        continue
                test = len(analyzer.neural_data)
                if len(analyzer.neural_data) != 2:
                    print(f"  Skipping pair - insufficient data")
                    continue

                # Pass global rastermap results to analyzer
                analyzer.global_rastermap_results = self.global_rastermap_results

                # Perform region-specific analysis
                analyzer.perform_rastermap_analysis()

                for comp_idx in range(self.config['n_components_visualize']):
                    try:
                        analyzer.create_dual_analysis_visualization(
                            component_idx=comp_idx,
                            save_fig=True
                        )
                    except Exception as e:
                        print(f"  Could not visualize component {comp_idx + 1}: {e}")


                analyzer.generate_summary_report()

                pair_key = f"{region1}_vs_{region2}"
                self.analyzers[pair_key] = analyzer
                successful_pairs += 1

                print(f"  Analysis complete for {region1} vs {region2}")

            except Exception as e:
                print(f"Failed to analyze {region1} vs {region2}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\nSuccessfully analyzed {successful_pairs}/{len(self.config['region_pairs'])} pairs")
        return successful_pairs > 0

    def _compute_global_rastermap_all_regions(self):
        """Compute global rastermap using all viable regions' trial-level firing rates."""
        try:
            from rastermap import Rastermap
            from scipy.stats import zscore
        except ImportError:
            print("Rastermap not available")
            return None

        print("\n" + "=" * 70)
        print("COMPUTING GLOBAL RASTERMAP FOR ALL REGIONS")
        print("=" * 70)

        all_neural_data = []
        neuron_region_mapping = []
        region_start_indices = {}
        current_idx = 0

        # Collect data from all regions
        regions_data = self.session_data['region_data']['regions']
        for region_name in sorted(self.available_regions):
            if region_name in regions_data and 'spike_data' in regions_data[region_name]:
                select_neurons = regions_data[region_name]['selected_neurons'].reshape(-1).astype(int)-1
                #spike_data = regions_data[region_name]['spike_data']
                spike_data = regions_data[region_name]['spike_data'][:,select_neurons,:]
                # Reshape to (neurons, time*trials) for Rastermap

                n_trials, n_neurons,n_time  = spike_data.shape
                transpose_spike_data = np.transpose(spike_data, (0, 2, 1))
                reshaped_data = transpose_spike_data.reshape(-1, transpose_spike_data.shape[2]).T

                all_neural_data.append(reshaped_data)
                if region_name not in region_start_indices:
                    region_start_indices[region_name] = {}
                region_start_indices[region_name]['start'] = current_idx

                current_idx += n_neurons
                region_start_indices[region_name]['end'] = current_idx
                region_start_indices[region_name]['n_neurons'] = n_neurons
                neuron_region_mapping.extend([region_name] * n_neurons)
                print(f"  {region_name}: {n_neurons} neurons")

        # Pool all data
        pooled_data = np.vstack(all_neural_data)
        print(f"\nPooled data shape: {pooled_data.shape} (total_neurons x time*trials)")

        # Normalize
        pooled_data_norm = zscore(pooled_data, axis=1, nan_policy='omit')
        pooled_data_norm = np.nan_to_num(pooled_data_norm, nan=0.0)

        # Fit Rastermap
        # model = Rastermap(n_clusters=10, n_PCs=200, locality=0.3, time_lag_window=5)

        # model = Rastermap(n_PCs=pooled_data.shape[0],
        #                   locality=0.1,
        #                   grid_upsample=5,
        #                   time_lag_window=5)
        model = Rastermap(n_PCs=pooled_data.shape[0],
                          locality=0,
                          grid_upsample=5)
        model.fit(pooled_data_norm)

        global_results = {
            'sorting_idx': model.isort,
            'embedding': model.embedding,
            'pooled_data': pooled_data,
            'pooled_data_norm': pooled_data_norm,
            'neuron_region_mapping': neuron_region_mapping,
            'region_start_indices': region_start_indices,
            'n_total_neurons': pooled_data.shape[0],
            'n_time': n_time,
            'n_trials': n_trials
        }

        print(f"\nGlobal rastermap complete: {len(model.isort)} neurons sorted")
        return global_results

    def create_comprehensive_visualizations(self):
        """Create integrated visualizations across region pairs."""
        print("\nStep 3: Creating Comprehensive Visualizations")
        print("-" * 40)

        if not self.analyzers:
            print("No analyzers available for visualization")
            return False

        try:
            integrated_output = Path(self.config['output_base_dir']) / \
                                self.config['session_name'] / "integrated_visualizations"
            integrated_output.mkdir(parents=True, exist_ok=True)

            self._create_cca_strength_comparison(integrated_output)
            self._create_temporal_dynamics_comparison(integrated_output)
            self._create_population_summary(integrated_output)

            print("Comprehensive visualizations created")
            return True

        except Exception as e:
            print(f"Visualization creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_pca_projection_for_region(self, pca_results, region_name, component_idx):
        """
        Extract PCA temporal projection for a specific region and component.

        CORRECTED: Handles region-based PCA structure properly.
        pca_results.REGION.projections.mean{component_idx} â†’ [1Ã—305 double]
        """
        try:
            if region_name not in pca_results:
                print(f"  Warning: Region '{region_name}' not in PCA results")
                return None, None

            region_data = pca_results[region_name]

            if 'projections' not in region_data:
                print(f"  Warning: No projections for region '{region_name}'")
                return None, None

            projections = region_data['projections']

            mean_proj = None
            if 'mean' in projections:
                means = projections['mean']
                if isinstance(means, (list, np.ndarray)) and component_idx < len(means):
                    mean_data = means[component_idx]
                    if isinstance(mean_data, np.ndarray):
                        mean_proj = mean_data.flatten()
                    elif isinstance(mean_data, (list, tuple)):
                        mean_proj = np.array(mean_data).flatten()

            std_proj = None
            if 'std' in projections:
                stds = projections['std']
                if isinstance(stds, (list, np.ndarray)) and component_idx < len(stds):
                    std_data = stds[component_idx]
                    if isinstance(std_data, np.ndarray):
                        std_proj = std_data.flatten()
                    elif isinstance(std_data, (list, tuple)):
                        std_proj = np.array(std_data).flatten()

            if 'components' in projections:
                trial_size = projections['components'][0][0].shape[0]
                std_proj =  std_proj / np.sqrt(trial_size)
            return mean_proj, std_proj

        except Exception as e:
            print(f"  Error extracting PCA projection for {region_name}, comp {component_idx}: {e}")
            return None, None

    def _create_temporal_dynamics_comparison(self, output_dir):
        """
        Create temporal dynamics comparison showing BOTH PCA and CCA in SAME figure.

        CORRECTED: Properly extracts PCA projections from region-based structure.
        """
        foresize = 14

        for comp_idx in range(self.config['n_components_visualize']):
            n_pairs = len(self.analyzers)
            if n_pairs == 0:
                return

            fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 5 * n_pairs),
                                     sharex=True, sharey=False)
            if n_pairs == 1:
                axes = [axes]

            for ax_idx, (pair_key, analyzer) in enumerate(self.analyzers.items()):
                ax = axes[ax_idx]

                region1_name = analyzer.region_pair[0]
                region2_name = analyzer.region_pair[1]

                # Extract CCA projections
                cca_r2 = np.nan
                if analyzer.cca_results is not None:
                    try:
                        pair_results = analyzer.cca_results['pair_results'][analyzer.pair_idx]
                        projections = pair_results['projections']
                        components = projections['components']

                        if comp_idx < len(components):
                            comp_data = components[comp_idx][0]

                            cca_region1_mean = np.abs(comp_data['region_i_mean'].flatten())
                            cca_region2_mean = np.abs(comp_data['region_j_mean'].flatten())

                            # cca_region1_mean = comp_data['region_i_mean'].flatten()
                            # cca_region2_mean = comp_data['region_j_mean'].flatten()

                            cca_region1_std = comp_data['region_i_std'].flatten()/ np.sqrt(comp_data['region_i_trials'].shape[0])
                            cca_region2_std = comp_data['region_j_std'].flatten()/ np.sqrt(comp_data['region_i_trials'].shape[0])

                            cca_r2 = comp_data['R2'].flatten()[0]

                            time_bins = analyzer.time_bins

                            ax.plot(time_bins, cca_region1_mean, 'b-', linewidth=2,
                                    label=f'{region1_name} (CCA)', alpha=0.8)
                            ax.plot(time_bins, cca_region2_mean, 'r-', linewidth=2,
                                    label=f'{region2_name} (CCA)', alpha=0.8)

                            # ax.fill_between(time_bins,
                            #                 cca_region1_mean - cca_region1_std / 2,
                            #                 cca_region1_mean + cca_region1_std / 2,
                            #                 alpha=0.15, color='blue')
                            # ax.fill_between(time_bins,
                            #                 cca_region2_mean - cca_region2_std / 2,
                            #                 cca_region2_mean + cca_region2_std / 2,
                            #                 alpha=0.15, color='red')

                            ax.fill_between(time_bins,
                                            cca_region1_mean - cca_region1_std,
                                            cca_region1_mean + cca_region1_std,
                                            alpha=0.15, color='blue')
                            ax.fill_between(time_bins,
                                            cca_region2_mean - cca_region2_std,
                                            cca_region2_mean + cca_region2_std,
                                            alpha=0.15, color='red')

                    except Exception as e:
                        print(f"  Error extracting CCA data for {pair_key}: {e}")

                # Extract PCA projections
                if analyzer.pca_results is not None:
                    try:
                        pca_region1_mean, pca_region1_std = self._extract_pca_projection_for_region(
                            analyzer.pca_results, region1_name, comp_idx
                        )

                        pca_region2_mean, pca_region2_std = self._extract_pca_projection_for_region(
                            analyzer.pca_results, region2_name, comp_idx
                        )
                        # First, compute the absolute value as you have done
                        pca_abs1 = np.abs(pca_region1_mean)
                        pca_abs2 = np.abs(pca_region2_mean)

                        # pca_abs1 = pca_region1_mean
                        # pca_abs2 = pca_region2_mean

                        # test = min(pca_abs1.min(), pca_abs2.min())
                        # Then, apply min-max normalization
                        # pca_normalized1 = (pca_abs1 - min(pca_abs1.min(), pca_abs2.min())) / (
                        #             max(pca_abs1.max(), pca_abs2.max()) - min(pca_abs1.min(), pca_abs2.min()))
                        #
                        # # Then, apply min-max normalization
                        # pca_normalized2 = (pca_abs2 - min(pca_abs1.min(), pca_abs2.min())) / (
                        #             max(pca_abs1.max(), pca_abs2.max()) - min(pca_abs1.min(), pca_abs2.min()))

                        if pca_region1_mean is not None:
                            ax.plot(time_bins, pca_abs1, 'b--', linewidth=2,
                                    label=f'{region1_name} (PCA)', alpha=0.7)
                            if pca_region1_std is not None:
                                ax.fill_between(time_bins,
                                                pca_abs1 - pca_region1_std,
                                                pca_abs1 + pca_region1_std,
                                                alpha=0.1, color='blue')

                        if pca_region2_mean is not None:
                            ax.plot(time_bins, pca_abs2, 'r--', linewidth=2,
                                    label=f'{region2_name} (PCA)', alpha=0.7)
                            if pca_region2_std is not None:
                                ax.fill_between(time_bins,
                                                pca_abs2 - pca_region2_std,
                                                pca_abs2 + pca_region2_std,
                                                alpha=0.1, color='red')

                        # if pca_region1_mean is not None:
                        #     ax.plot(time_bins, pca_normalized1, 'b--', linewidth=2,
                        #             label=f'{region1_name} (PCA)', alpha=0.7)
                        #     if pca_region1_std is not None:
                        #         ax.fill_between(time_bins,
                        #                         pca_normalized1 - pca_region1_std / 2.5,
                        #                         pca_normalized1 + pca_region1_std / 2.5,
                        #                         alpha=0.1, color='blue')
                        #
                        # if pca_region2_mean is not None:
                        #     ax.plot(time_bins, pca_normalized2, 'r--', linewidth=2,
                        #             label=f'{region2_name} (PCA)', alpha=0.7)
                        #     if pca_region2_std is not None:
                        #         ax.fill_between(time_bins,
                        #                         pca_normalized2 - pca_region2_std / 2.5,
                        #                         pca_normalized2 + pca_region2_std / 2.5,
                        #                         alpha=0.1, color='red')

                    except Exception as e:
                        print(f"  Error extracting PCA data for {pair_key}: {e}")

                # Behavioral markers
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.6, linewidth=2)
                ax.axvspan(0, 3, alpha=0.1, color='grey')
                #ax.axvspan(0, 3, alpha=0.1, color='grey', label='After bar off')
                #ax.axvspan(0.5, 1.5, alpha=0.1, color='blue', label='Reward Consumption')

                # Formatting
                ax.set_ylabel('Projection Magnitude', fontsize=foresize)
                title_str = f'{pair_key.replace("_", " ")} - Component {comp_idx + 1}'
                if not np.isnan(cca_r2):
                    title_str += f' - CCA R2: {cca_r2:.3f}'
                ax.set_title(title_str, fontsize=foresize + 2, fontweight='normal')

                ax.legend(loc='upper right', fontsize=foresize - 2, ncol=2)
                #ax.legend(loc='lower right', fontsize=foresize - 2, ncol=2)


                ax.grid(True, alpha=0.3)
                ax.set_xlim(-1.5, 3)

                # ax.set_ylim(-1, 10)
                # ax.set_ylim(-5, 5)

                ax.tick_params(axis='both', labelsize=foresize)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            axes[-1].set_xlabel('Time (s)', fontsize=foresize + 2)

            plt.tight_layout()

            save_path = output_dir / f"temporal_dynamics_CCA_PCA_comp{comp_idx + 1}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  Created CCA/PCA temporal dynamics: {save_path.name}")

    def _create_cca_strength_comparison(self, output_dir):
        """
        Create SEPARATE figures for CCA strength and PCA variance.

        CORRECTED: Properly extracts PCA explained variance from region-based structure.
        """
        # ========================================================================
        # FIGURE 1: CCA STRENGTH
        # ========================================================================

        fig_cca, ax_cca = plt.subplots(figsize=(14, 8))

        pair_names = []
        cca_r2_values = []
        cca_std_values = []
        n_sig_components = []

        for pair_key, analyzer in self.analyzers.items():
            if analyzer.cca_results is not None:
                try:
                    pair_results = analyzer.cca_results['pair_results'][analyzer.pair_idx]
                    cv_results = pair_results['cv_results']

                    mean_cv_R2 = cv_results['mean_cv_R2'].flatten()
                    std_cv_R2 = np.std(cv_results['cv_R2'], axis=0)

                    sig_components = pair_results['significant_components'].flatten()

                    pair_names.append(pair_key.replace('_vs_', '\nvs\n'))
                    cca_r2_values.append(mean_cv_R2[:self.config['n_components_visualize']])
                    cca_std_values.append(std_cv_R2[:self.config['n_components_visualize']])
                    n_sig_components.append(len(sig_components))
                except Exception as e:
                    print(f"  Error extracting CCA strength for {pair_key}: {e}")

        if len(pair_names) > 0:
            x = np.arange(len(pair_names))
            width = 0.8 / self.config['n_components_visualize']

            for comp_idx in range(self.config['n_components_visualize']):
                r2_comp = [r2[comp_idx] if comp_idx < len(r2) else 0 for r2 in cca_r2_values]
                std_comp = [std[comp_idx] if comp_idx < len(std) else 0 for std in cca_std_values]
                offset = (comp_idx - self.config['n_components_visualize'] / 2 + 0.5) * width

                ax_cca.bar(x + offset, r2_comp, width,
                           yerr=std_comp,
                           capsize=4,
                           label=f'Component {comp_idx + 1}',
                           alpha=0.8)

            ax_cca.set_xlabel('Region Pair', fontsize=14, fontweight='bold')
            ax_cca.set_ylabel('Cross-validated R', fontsize=14, fontweight='bold')
            ax_cca.set_title(f'CCA Strength Across Region Pairs\n'
                             f'Session: {self.config["session_name"]}',
                             fontsize=16, fontweight='bold')
            ax_cca.set_xticks(x)
            ax_cca.set_xticklabels(pair_names, fontsize=11)
            ax_cca.legend(fontsize=12, loc='upper right', framealpha=0.9)
            ax_cca.spines['top'].set_visible(False)
            ax_cca.spines['right'].set_visible(False)
            ax_cca.grid(True, axis='y', alpha=0.3)
            ax_cca.set_ylim(bottom=0)

            for idx, n_sig in enumerate(n_sig_components):
                ax_cca.text(idx, ax_cca.get_ylim()[1] * 0.95,
                            f'n_sig={n_sig}',
                            ha='center', va='top',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            save_path_cca = output_dir / "cca_strength_comparison.png"
            plt.savefig(save_path_cca, dpi=300, bbox_inches='tight')
            plt.close(fig_cca)

            print(f"  Created CCA strength comparison: {save_path_cca.name}")

        # ========================================================================
        # FIGURE 2: PCA EXPLAINED VARIANCE
        # ========================================================================

        fig_pca, ax_pca = plt.subplots(figsize=(14, 8))

        pair_names_pca = []
        pca_variance_values = []

        for pair_key, analyzer in self.analyzers.items():
            if analyzer.pca_results is not None:
                try:
                    region1_name = analyzer.region_pair[0]
                    region2_name = analyzer.region_pair[1]

                    variance_region1 = None
                    variance_region2 = None

                    if region1_name in analyzer.pca_results:
                        region1_data = analyzer.pca_results[region1_name]
                        if 'explained' in region1_data:
                            variance_region1 = region1_data['explained'][:self.config['n_components_visualize']]

                    if region2_name in analyzer.pca_results:
                        region2_data = analyzer.pca_results[region2_name]
                        if 'explained' in region2_data:
                            variance_region2 = region2_data['explained'][:self.config['n_components_visualize']]

                    if variance_region1 is not None and variance_region2 is not None:
                        avg_variance = (np.array(variance_region1) + np.array(variance_region2)) / 2.0
                        pair_names_pca.append(pair_key.replace('_vs_', '\nvs\n'))
                        pca_variance_values.append(avg_variance)

                except Exception as e:
                    print(f"  Error extracting PCA variance for {pair_key}: {e}")

        if len(pair_names_pca) > 0:
            x_pca = np.arange(len(pair_names_pca))
            width_pca = 0.8 / self.config['n_components_visualize']

            for comp_idx in range(self.config['n_components_visualize']):
                var_comp = [var[comp_idx] if comp_idx < len(var) else 0 for var in pca_variance_values]
                offset = (comp_idx - self.config['n_components_visualize'] / 2 + 0.5) * width_pca

                ax_pca.bar(x_pca + offset, var_comp, width_pca,
                           label=f'Component {comp_idx + 1}',
                           alpha=0.8)

            ax_pca.set_xlabel('Region Pair', fontsize=14, fontweight='bold')
            ax_pca.set_ylabel('Explained Variance (%)', fontsize=14, fontweight='bold')
            ax_pca.set_title(f'PCA Explained Variance Across Region Pairs\n'
                             f'(Average of Both Regions)\n'
                             f'Session: {self.config["session_name"]}',
                             fontsize=16, fontweight='bold')
            ax_pca.set_xticks(x_pca)
            ax_pca.set_xticklabels(pair_names_pca, fontsize=11)
            ax_pca.legend(fontsize=12, loc='upper right', framealpha=0.9)
            ax_pca.spines['top'].set_visible(False)
            ax_pca.spines['right'].set_visible(False)
            ax_pca.grid(True, axis='y', alpha=0.3)
            ax_pca.set_ylim(bottom=0)

            note_text = ('Note: Values represent average explained variance from both regions.\n'
                         'High values indicate strong intrinsic structure independent of coordination.')
            ax_pca.text(0.02, 0.98, note_text,
                        transform=ax_pca.transAxes,
                        fontsize=9, style='italic',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

            plt.tight_layout()
            save_path_pca = output_dir / "pca_variance_comparison.png"
            plt.savefig(save_path_pca, dpi=300, bbox_inches='tight')
            plt.close(fig_pca)

            print(f"  Created PCA variance comparison: {save_path_pca.name}")

    def _create_population_summary(self, output_dir):
        """Create neural population summary with component R² heatmaps."""
        fig = plt.figure(figsize=(18, 10))

        # Create grid: population sizes on left, heatmap on right
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1],
                              hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[:, 0])  # Population sizes (full left column)
        ax2 = fig.add_subplot(gs[0, 1])  # Max R² connectivity matrix
        ax3 = fig.add_subplot(gs[1, 1])  # Components R² heatmap

        # Plot 1: Population sizes
        region_names = []
        population_sizes = []

        for analyzer in self.analyzers.values():
            for region_name, neural_data in analyzer.neural_data.items():
                if region_name not in region_names:
                    region_names.append(region_name)
                    population_sizes.append(neural_data.shape[1])


        unique_regions = sorted(set(region_names))


        # Step 2: Define anatomical hierarchy
        # The ordering reflects cortex → subcortex → fiber
        anatomical_order = ['mPFC', 'ORB', 'MOp', 'MOs', 'OLF',  # Cortical regions
                            'STR', 'STRv', 'HIPP',  # Striatal & limbic
                            'MD', 'LP', 'VALVM', 'VPMPO', 'ILM',  # Thalamic
                            'HY',  # Hypothalamic
                            'fiber',  # Fiber tracts
                            'other']  # Catch-all

        # Step 3: Order your regions according to anatomical hierarchy
        # Only include regions that exist in your data
        unique_regions = [region for region in anatomical_order
                           if region in unique_regions]


        y_pos = np.arange(len(unique_regions))
        ax1.barh(y_pos, population_sizes, color='steelblue', alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(unique_regions)
        ax1.set_xlabel('Number of Neurons', fontsize=12)
        ax1.set_title('Neural Population Sizes', fontsize=14)
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        for idx, size in enumerate(population_sizes):
            ax1.text(size + 1, idx, str(size), va='center', fontsize=10)

        # Plot 2: Max R² connectivity matrix




        n_regions = len(unique_regions)
        connectivity_matrix = np.zeros((n_regions, n_regions))

        for analyzer in self.analyzers.values():
            r1_idx = unique_regions.index(analyzer.region_pair[0])
            r2_idx = unique_regions.index(analyzer.region_pair[1])

            if analyzer.cca_results:
                pair_results = analyzer.cca_results['pair_results'][analyzer.pair_idx]
                cv_results = pair_results['cv_results']
                mean_cv_R2 = cv_results['mean_cv_R2'].flatten()
                max_r2 = np.max(mean_cv_R2) if len(mean_cv_R2) > 0 else 0

                connectivity_matrix[r1_idx, r2_idx] = max_r2
                connectivity_matrix[r2_idx, r1_idx] = max_r2

        im = ax2.imshow(connectivity_matrix, cmap='viridis', vmin=0, aspect='equal')
        ax2.set_xticks(np.arange(n_regions))
        ax2.set_yticks(np.arange(n_regions))
        ax2.set_xticklabels(unique_regions, rotation=45, ha='right')
        ax2.set_yticklabels(unique_regions)
        ax2.set_title('Maximum Latent Similarity (R²)', fontsize=14)

        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Max R²', fontsize=12)

        for i in range(n_regions):
            for j in range(n_regions):
                if connectivity_matrix[i, j] > 0:
                    ax2.text(j, i, f'{connectivity_matrix[i, j]:.2f}',
                             ha='center', va='center', color='white' if connectivity_matrix[i, j] > 0.5 else 'black',
                             fontsize=9)

        # Plot 3: All components R² heatmap
        # Collect all R² values for each component
        pair_labels = []
        all_r2_values = []

        for pair_key, analyzer in self.analyzers.items():
            if analyzer.cca_results:
                pair_results = analyzer.cca_results['pair_results'][analyzer.pair_idx]
                cv_results = pair_results['cv_results']
                mean_cv_R2 = cv_results['mean_cv_R2'].flatten()

                # Get R² values for n_components_visualize
                n_comp = min(self.config['n_components_visualize'], len(mean_cv_R2))
                r2_values = mean_cv_R2[:n_comp].tolist()

                # Pad with zeros if fewer components than n_components_visualize
                while len(r2_values) < self.config['n_components_visualize']:
                    r2_values.append(0.0)

                all_r2_values.append(r2_values)
                pair_labels.append(f"{analyzer.region_pair[0]}-{analyzer.region_pair[1]}")

        if all_r2_values:
            # Create heatmap data
            r2_matrix = np.array(all_r2_values).T  # Components x Pairs

            im3 = ax3.imshow(r2_matrix, aspect='auto', cmap='plasma', vmin=0)

            # Set labels
            ax3.set_xticks(np.arange(len(pair_labels)))
            ax3.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=10)
            ax3.set_yticks(np.arange(self.config['n_components_visualize']))
            ax3.set_yticklabels([f'Comp {i + 1}' for i in range(self.config['n_components_visualize'])],
                                fontsize=11)
            ax3.set_xlabel('Region Pairs', fontsize=12)
            ax3.set_ylabel('CCA Components', fontsize=12)
            ax3.set_title('Cross-validated R² Values by Component', fontsize=14)

            # Add colorbar
            cbar3 = plt.colorbar(im3, ax=ax3)
            cbar3.set_label('R²', fontsize=12)

            # Add text annotations for R² values
            for i in range(r2_matrix.shape[0]):
                for j in range(r2_matrix.shape[1]):
                    if r2_matrix[i, j] > 0:
                        text_color = 'white' if r2_matrix[i, j] > 0.5 else 'black'
                        ax3.text(j, i, f'{r2_matrix[i, j]:.2f}',
                                 ha='center', va='center', color=text_color, fontsize=8)

        plt.suptitle(f'Neural Population Summary with Component Analysis\nSession: {self.config["session_name"]}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = output_dir / "population_summary_with_components.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Created enhanced population summary: {save_path.name}")

    def generate_integrated_summary(self):
        """Generate comprehensive summary report."""
        print("\nStep 4: Generating Integrated Summary")
        print("-" * 40)

        try:
            summary_dir = Path(self.config['output_base_dir']) / \
                          self.config['session_name'] / "summaries"
            summary_dir.mkdir(parents=True, exist_ok=True)

            integrated_results = {
                'session_info': {
                    'session_name': self.config['session_name'],
                    'analysis_date': str(datetime.now()),
                    'n_region_pairs_analyzed': len(self.analyzers),
                    'n_components_visualized': self.config['n_components_visualize']
                },
                'region_pairs': {}
            }

            for pair_key, analyzer in self.analyzers.items():
                if analyzer.cca_results:
                    pair_results = analyzer.cca_results['pair_results'][analyzer.pair_idx]
                    cv_results = pair_results['cv_results']

                    integrated_results['region_pairs'][pair_key] = {
                        'region1': analyzer.region_pair[0],
                        'region2': analyzer.region_pair[1],
                        'n_neurons_region1': analyzer.neural_data[analyzer.region_pair[0]].shape[0],
                        'n_neurons_region2': analyzer.neural_data[analyzer.region_pair[1]].shape[0],
                        'n_significant_components': len(pair_results['significant_components'].flatten()),
                        'max_R2': float(np.max(cv_results['mean_cv_R2'].flatten())),
                        'mean_R2_top3': float(np.mean(cv_results['mean_cv_R2'].flatten()[:3])),
                        'rastermap_performed': len(analyzer.rastermap_results) > 0
                    }

            json_path = summary_dir / "integrated_results.json"
            with open(json_path, 'w') as f:
                json.dump(integrated_results, f, indent=2)
            print(f"  Saved JSON summary: {json_path.name}")

            self._generate_text_report(integrated_results, summary_dir)

            print("Integrated summary generated")
            return True

        except Exception as e:
            print(f"Summary generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_text_report(self, results, output_dir):
        """Generate human-readable text report."""
        report_path = output_dir / "analysis_report.txt"

        with open(report_path, 'w') as f:
            f.write("Oxford Dataset Integrated Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("Session Information:\n")
            f.write("-" * 20 + "\n")
            session_info = results['session_info']
            for key, value in session_info.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")

            if results['region_pairs']:
                all_r2 = [pair['max_R2'] for pair in results['region_pairs'].values()]
                all_sig = [pair['n_significant_components'] for pair in results['region_pairs'].values()]

                f.write("Overall Statistics:\n")
                f.write("-" * 19 + "\n")
                f.write(f"Average maximum R: {np.mean(all_r2):.3f}\n")
                f.write(f"Highest R: {np.max(all_r2):.3f}\n")
                f.write(f"Average significant components: {np.mean(all_sig):.1f}\n\n")

                f.write("Detailed Results by Region Pair:\n")
                f.write("-" * 33 + "\n\n")

                for pair_key, pair_data in results['region_pairs'].items():
                    f.write(f"{pair_data['region1']} vs {pair_data['region2']}:\n")
                    f.write(f"  Neurons: {pair_data['n_neurons_region1']} + {pair_data['n_neurons_region2']}\n")
                    f.write(f"  Significant components: {pair_data['n_significant_components']}\n")
                    f.write(f"  Maximum R: {pair_data['max_R2']:.4f}\n")
                    f.write(f"  Mean R (top 3): {pair_data['mean_R2_top3']:.4f}\n\n")

        print(f"  Generated text report: {report_path.name}")

    def print_results_summary(self):
        """Print concise summary to console."""
        print("\nAnalysis Results Summary:")
        print("-" * 30)

        if self.analyzers:
            print(f"Session: {self.config['session_name']}")
            print(f"Region pairs analyzed: {len(self.analyzers)}")

            best_r2 = 0
            best_pair = None

            for pair_key, analyzer in self.analyzers.items():
                if analyzer.cca_results:
                    pair_results = analyzer.cca_results['pair_results'][analyzer.pair_idx]
                    cv_results = pair_results['cv_results']
                    mean_cv_R2 = cv_results['mean_cv_R2'].flatten()

                    if len(mean_cv_R2) > 0:
                        max_r2 = np.max(mean_cv_R2)
                        if max_r2 > best_r2:
                            best_r2 = max_r2
                            best_pair = pair_key

            if best_pair:
                print(f"\nStrongest canonical correlation:")
                print(f"  {best_pair}: R = {best_r2:.3f}")

        output_dir = Path(self.config['output_base_dir']) / self.config['session_name']
        print(f"\nOutput directory: {output_dir}")


def create_oxford_configuration():
    """Create configuration for Oxford analysis pipeline."""
    config = {
        'base_results_dir': "/Users/shengyuancai/Downloads/Oxford_dataset/sessions_cued_hit_long_results",
        #'base_results_dir': "/Users/shengyuancai/Downloads/Oxford_dataset/sessions_spont_short_results",


        'output_base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/oxford_integrated_analysis_cued',
        #'output_base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/oxford_integrated_analysis_spont',
        'session_name': "yp021_220405",  # yp013_220211  yp014_220212 yp021_220405 yp021_220331

        'region_pairs': [
            ('ORB', 'STR'),
            ('mPFC', 'STR'),
            ('MOp', 'STR'),
            ('MOs', 'STR'),
            ('MOp', 'MOs')
        ],
        # 'region_pairs': [
        #     ('ORB', 'mPFC'),
        #     ('ORB', 'MOp'),
        #     ('ORB', 'MOs'),
        #     ('ORB', 'STR'),
        #     ('ORB', 'OLF'),
        #     ('mPFC', 'MOp'),
        #     ('mPFC', 'MOs'),
        #     ('mPFC', 'OLF'),
        #     ('mPFC', 'STR'),
        #     ('MOp', 'MOs'),
        #     ('MOp', 'OLF'),
        #     ('MOp', 'STR'),
        #     ('MOs', 'OLF'),
        #     ('MOs', 'STR'),
        #     ('OLF', 'STR'),
        #     ('MOs', 'fiber'),
        #     ('MOp', 'fiber'),
        #     ('OLF', 'fiber'),
        #     ('ORB', 'fiber'),
        #     ('STR', 'fiber')
        # ],
        # 'region_pairs': [
        #     ('ORB', 'STR')
        # ],
        # 'region_pairs': [
        #     ('ORB', 'mPFC')
        # ],
        'n_components_visualize': 3,

    }

    return config


def main_oxford_demonstration():
    """Main demonstration function."""
    print("Oxford Dataset Analysis Pipeline - CORRECTED PCA Handling")
    print("=" * 70)

    config = create_oxford_configuration()

    try:
        pipeline = OxfordIntegratedAnalysisPipeline(config)
        success = pipeline.run_complete_analysis()

        if success:
            print("\n" + "=" * 70)
            print("DEMONSTRATION COMPLETE")
            print("=" * 70)
        else:
            print("\nPipeline encountered issues")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")

    return pipeline if 'pipeline' in locals() else None


if __name__ == "__main__":
    pipeline = main_oxford_demonstration()
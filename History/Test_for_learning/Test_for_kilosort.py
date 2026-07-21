# %%
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd
from kilosort import run_kilosort, DEFAULT_SETTINGS
import xml.etree.ElementTree as ET
from matplotlib import gridspec, rcParams
import json
from datetime import datetime
import glob


class OpenEphysNeuropixelProcessor:
    """
    A comprehensive processor for OpenEphys Neuropixel data that handles
    multiple probe types (1.0 and 2.0) with automatic configuration
    """

    def __init__(self, experiment_path, experiment_name, results_dir=None):
        """
        Initialize the processor with the path to the OpenEphys experiment

        Parameters:
        -----------
        experiment_path : str or Path
            Path to the experiment directory containing settings.xml
        results_dir : str or Path, optional
            Custom directory for saving Kilosort results. If None, uses default location.
        """
        self.experiment_path = Path(experiment_path)
        self.session_name = experiment_name
        self.settings_file = (self.experiment_path.parent / "settings.xml")
        self.probe_configs = {}

        # Set results directory - use custom path if provided, otherwise default
        if results_dir is not None:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = self.experiment_path.parent.parent.parent

        # Parse the settings to understand probe configuration
        self._parse_settings()

    def _parse_settings(self):
        """
        Parse the OpenEphys settings.xml file to extract probe configurations
        """
        print(f"Parsing OpenEphys settings.xml file in {self.experiment_path}")

        if not self.settings_file.exists():
            raise FileNotFoundError(f"Settings file not found: {self.settings_file}")

        # Parse XML
        tree = ET.parse(self.settings_file)
        root = tree.getroot()

        # Find the Neuropix-PXI processor
        neuropix_processor = root.find(".//PROCESSOR[@pluginName='Neuropix-PXI']")
        if neuropix_processor is None:
            raise ValueError("No Neuropix-PXI processor found in settings")

        # Extract stream information
        streams = neuropix_processor.findall(".//STREAM")

        for stream in streams:
            stream_name = stream.get("name")
            device_name = stream.get("device_name")
            sample_rate = float(stream.get("sample_rate"))
            channel_count = int(stream.get("channel_count"))

            # Determine probe type and configuration
            if "2.0" in device_name:
                probe_type = "neuropixels_2.0"
                probe_map = "neuropixPhase3B2_kilosortChanMap.mat"
            else:  # Neuropixels 1.0
                probe_type = "neuropixels_1.0"
                probe_map = "neuropixPhase3B1_kilosortChanMap.mat"

            # Store configuration
            probe_config = {
                'device_name': device_name,
                'probe_type': probe_type,
                'sample_rate': sample_rate,
                'channel_count': channel_count,
                'probe_map': probe_map,
                'stream_name': stream_name
            }

            # For Neuropixels 1.0, we have separate AP and LFP streams
            if "AP" in stream_name or "2.0" in device_name:
                # This is an AP stream or a Neuropixels 2.0 stream (which only has AP)
                probe_name = stream_name.split("-")[0]  # Extract "ProbeA", "ProbeB", etc.
                self.probe_configs[probe_name] = probe_config
                print(f"  Found {probe_name}: {probe_type}, AP data at {sample_rate}Hz")
            elif "LFP" in stream_name:
                # Skip LFP streams for spike sorting (Kilosort works on AP band)
                probe_name = stream_name.split("-")[0]
                print(f"  Found {probe_name}: LFP data at {sample_rate}Hz (skipping for spike sorting)")
                continue

        print(f"  Total probes configured for spike sorting: {len(self.probe_configs)}")

    def _find_continuous_file(self, probe_name):
        """
        Find the continuous.dat file for a specific probe

        Parameters:
        -----------
        probe_name : str
            Name of the probe (e.g., "ProbeA", "ProbeB")

        Returns
        --------
        Path to continuous.dat file
        """
        # Look for the continuous data in the typical OpenEphys structure
        # Format: recording1/continuous/Neuropix-PXI-100.ProbeX[-AP]/continuous.dat

        config = self.probe_configs[probe_name]
        stream_name = config['stream_name']

        # Search in all recording directories
        for recording_dir in self.experiment_path.glob("recording*"):
            continuous_base = recording_dir / "continuous"

            # Try different possible directory naming patterns
            possible_dirs = [
                continuous_base / f"Neuropix-PXI-100.{stream_name}",
                continuous_base / f"Neuropix-PXI.{stream_name}",
                continuous_base / stream_name
            ]

            for probe_dir in possible_dirs:
                continuous_file = probe_dir / "continuous.dat"
                if continuous_file.exists():
                    print(f"    Found continuous data for {probe_name}: {continuous_file}")
                    return continuous_file

        raise FileNotFoundError(f"No continuous.dat found for {probe_name}")

    def _get_probe_settings(self, probe_name):
        """
        Get Kilosort settings optimized for the specific probe type with smart channel detection
        """
        config = self.probe_configs[probe_name]
        continuous_file = self._find_continuous_file(probe_name)

        # Auto-detect number of channels from file size
        file_size = continuous_file.stat().st_size
        bytes_per_sample = 2  # int16 data type

        print(f"    Auto-detecting channels for {continuous_file.name}...")
        print(f"    File size: {file_size:,} bytes")

        # Find all possible channel counts that divide the file size evenly
        valid_channel_counts = []

        # The total number of data points in the file
        total_data_points = file_size // bytes_per_sample

        # Check channel counts that make sense for Neuropixels (typically 300-400)
        # We'll check by finding divisors of total_data_points
        for n_chan in range(300, 385):
            if total_data_points % n_chan == 0:
                samples_per_channel = total_data_points // n_chan
                duration_sec = samples_per_channel / config['sample_rate']

                # Only consider reasonable recording durations (> 10 seconds)
                if duration_sec > 10:
                    valid_channel_counts.append(n_chan)
                    print(f"{n_chan} channels  {samples_per_channel:,} samples {duration_sec:.1f}s")

        # Select the best channel count
        n_chan_bin = self._select_best_channel_count(valid_channel_counts)

        # Verify our choice
        samples_per_channel = total_data_points // n_chan_bin
        duration_sec = samples_per_channel / config['sample_rate']
        print(f"Final selection: {n_chan_bin} channels")
        print(f"Recording duration: {duration_sec:.1f} seconds")

        settings = {
            'data_dir': str(continuous_file.parent),
            'n_chan_bin': n_chan_bin,
            'nblocks': 5,
        }

        return settings, continuous_file, config['probe_map']

    def _select_best_channel_count(self, valid_counts):
        """
        Select the most appropriate channel count from valid options
        """
        if not valid_counts:
            print("    Warning: No valid channel count found. Using 350 as default.")
            return 385

        if len(valid_counts) == 1:
            return valid_counts[0]

        # Multiple valid options - use heuristics to choose the best one
        print(f"    Multiple valid options: {valid_counts}")

        # Common Neuropixels configurations in order of preference
        common_configs = [385, 384, 383]

        # Check if any of our preferred configurations are available
        for preferred in common_configs:
            if preferred in valid_counts:
                print(f"    Selected {preferred} (common Neuropixels configuration)")
                return preferred

        # If no common configuration found, choose the one closest to 385
        closest_to_385 = min(valid_counts, key=lambda x: abs(x - 385))
        print(f"    Selected {closest_to_385} (closest to standard 385)")
        return closest_to_385

    def process_probe(self, probe_name):
        """
        Process a single probe with Kilosort

        Parameters:
        -----------
        probe_name : str
            Name of the probe to process

        Returns:
        --------
        Path to results directory for this probe
        """
        print(f"\n{'=' * 50}")
        print(f"Processing {probe_name}")
        print(f"{'=' * 50}")

        # Get settings for this probe
        settings, continuous_file, probe_map = self._get_probe_settings(probe_name)
        probe_name_dir = f'{self.session_name}_{probe_name}'
        # Create probe-specific results directory
        probe_results_dir = self.results_dir / probe_name_dir
        probe_results_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Probe type: {self.probe_configs[probe_name]['probe_type']}")
        print(f"  Data file: {continuous_file}")
        print(f"  Probe map: {probe_map}")
        print(f"  Results will be saved to: {probe_results_dir}")

        # Check if already processed
        if (probe_results_dir / 'spike_times.npy').exists():
            print(f"  Kilosort results already exist for {probe_name}, skipping...")
            return probe_results_dir

        # Run Kilosort
        print(f"  Running Kilosort 4 on {probe_name}...")
        try:
            ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
                run_kilosort(
                    settings=settings,
                    filename=str(continuous_file),
                    probe_name=probe_map,
                    results_dir=str(probe_results_dir)
                )
            print(f"  Kilosort completed successfully for {probe_name}")
        except Exception as e:
            print(f"  Error running Kilosort on {probe_name}: {str(e)}")
            raise

        return probe_results_dir

    def analyze_probe_results(self, probe_name, n_units_to_plot=5):
        """
        Generate analysis plots for a processed probe

        Parameters:
        -----------
        probe_name : str
            Name of the probe
        n_units_to_plot : int
            Number of units to create detailed plots for
        """

        probe_name = f'{self.session_name}_{probe_name}'
        probe_results_dir = self.results_dir / probe_name

        if not (probe_results_dir / 'spike_times.npy').exists():
            print(f"  No Kilosort results found for {probe_name}")
            return

        print(f"  Analyzing results for {probe_name}...")

        # Load results
        ops = np.load(probe_results_dir / 'ops.npy', allow_pickle=True).item()
        contam_pct = pd.read_csv(probe_results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
        templates = np.load(probe_results_dir / 'templates.npy')
        chan_map = np.load(probe_results_dir / 'channel_map.npy')
        spike_times = np.load(probe_results_dir / 'spike_times.npy')
        spike_clusters = np.load(probe_results_dir / 'spike_clusters.npy')

        # Get sampling rate and convert spike times to seconds
        fs = ops.get('fs', 30000)
        spike_times_sec = spike_times / fs

        # Find best channel for each unit
        chan_best = (templates ** 2).sum(axis=1).argmax(axis=-1)
        chan_best = chan_map[chan_best]

        # Identify good units
        good_units = np.nonzero(contam_pct <= 10)[0]
        print(f"    Found {len(good_units)} good units out of {len(contam_pct)} total units")

        # Create plots directory
        probe_plot_results_dir = probe_results_dir.parent.parent.parent.parent
        probe_plot_results_dir = probe_plot_results_dir / 'unit_quality_analysis'
        probe_plot_results_dir.mkdir(exist_ok=True)

        # Generate summary plot
        self._create_summary_plot(probe_name, ops, spike_times, spike_clusters,
                                  chan_best, chan_map, contam_pct, probe_results_dir, probe_plot_results_dir)

    def _create_summary_plot(self, probe_name, ops, spike_times, spike_clusters,
                             chan_best, chan_map, contam_pct, probe_results_dir, probe_plot_results_dir):
        """Create a summary plot showing drift and spike distribution"""

        # Create figure and gridspec layout
        fig = plt.figure(figsize=(15, 12))
        grid = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.5)
        try:
            camps = pd.read_csv(probe_results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
        except FileNotFoundError:
            print(f"    Warning: cluster_Amplitude.tsv not found for {probe_name}")
            camps = np.array([])  # Create empty array as fallback

        # Plot 1: Drift over time (top-left)
        ax1 = fig.add_subplot(grid[0, 0])
        if 'dshift' in ops:
            dshift = ops['dshift']
            time_points = np.arange(len(dshift)) * 2  # Assuming 4-second intervals
            ax1.plot(time_points, dshift)
            ax1.set_xlabel('Time (sec)')
            ax1.set_ylabel('Drift (m)')
            ax1.set_title('Estimated Drift')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

        # # Plot 2: Spike raster - spans 2 columns (top-middle and top-right)
        # ax2 = fig.add_subplot(grid[0, 1:])
        #
        # # Select a time window for visualization (middle 100 seconds of recording)
        # mid_time =  spike_times.max() /  2  # Middle of recording
        # t1 = np.nonzero(spike_times >  (mid_time - ops['fs']*50))[0][0]
        # t2 = np.nonzero(spike_times >   (mid_time + ops['fs']* 150))[0][0]
        #
        # if len(t1) > 0 and len(t2) > 0:
        #     t1, t2 = t1[0], t2[0]
        #     t2 = min(len(spike_times), t2)
        #
        #     # Get spike times and their corresponding channels for the selected window
        #     window_spike_times = spike_times[t1:t2] / ops['fs']
        #     window_spike_clusters = spike_clusters[t1:t2]
        #     window_spike_channels = chan_best[window_spike_clusters]
        #
        #     # Create raster plot
        #     unique_channels = np.unique(window_spike_channels)
        #
        #     for channel in unique_channels:
        #         channel_mask = window_spike_channels == channel
        #         channel_spike_times = window_spike_times[channel_mask]
        #
        #         if len(channel_spike_times) > 0:
        #             ax2.vlines(channel_spike_times,
        #                        channel - 0.4,
        #                        channel + 0.4,
        #                        colors='black',
        #                        linewidth=0.5,
        #                        alpha=0.7)
        # else:
        #     # Handle the case when no spikes are found in the time window
        #     ax2.text(0.5, 0.5, 'No spikes in selected time window',
        #              horizontalalignment='center', verticalalignment='center',
        #              transform=ax2.transAxes, fontsize=12)
        #
        # ax2.set_xlabel('Time (sec)')
        # ax2.set_ylabel('Channel')
        # ax2.set_ylim([chan_map.max() + 1, -1])
        # ax2.set_title('Spike Distribution (Raster)')

        # Plot 3: Contamination histogram (bottom-left)
        ax3 = fig.add_subplot(grid[0, 1])
        contam_pct = contam_pct[contam_pct < 200]
        new_bins = np.arange(0, 200, 2)
        ax3.hist(contam_pct, bins=new_bins, edgecolor='black', color='gray')
        ax3.axvline(x=10, color='red', linestyle='--', label='10% threshold')
        ax3.set_xlabel('Contamination %')
        ax3.set_ylabel('Number of units')
        ax3.set_title('Unit Contamination Distribution')
        ax3.legend()
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # Plot 4: Firing rates (bottom-middle)
        ax4 = fig.add_subplot(grid[1, 0])
        unique_clusters, cluster_counts = np.unique(spike_clusters, return_counts=True)
        firing_rates = cluster_counts * ops['fs'] / spike_times.max()
        ax4.hist(firing_rates, bins=30, color='gray', edgecolor='black')
        ax4.set_xlabel('Firing Rate (Hz)')
        ax4.set_ylabel('Number of units')
        ax4.set_title('Firing Rate Distribution')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        # Plot 5: Template amplitude distribution (bottom-right)
        ax5 = fig.add_subplot(grid[1, 1])
        if len(camps) > 0:
            ax5.hist(camps, bins=30, color='gray', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Amplitude (V)')
            ax5.set_ylabel('Number of Units')
            ax5.set_title('Template Amplitude Distribution')

            median_amp = np.median(camps)
            ax5.axvline(median_amp, color='red', linestyle='--',
                        label=f'Median: {median_amp:.1f} V')
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'Amplitude data\nnot available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Template Amplitude Distribution')

        plt.suptitle(f'{probe_name} - Summary Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(probe_plot_results_dir / f'{probe_name}_summary.png', dpi=300, bbox_inches='tight')
        # plt.savefig(plots_dir / f'summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    def process_all_probes(self, analyze_results=True):
        """
        Process all probes found in the experiment

        Parameters:
        -----------
        analyze_results : bool
            Whether to generate analysis plots after processing
        """
        print(f"Starting processing of {len(self.probe_configs)} probes...")

        results = {}
        for probe_name in self.probe_configs.keys():
            try:
                results_dir = self.process_probe(probe_name)
                results[probe_name] = results_dir

                if analyze_results:
                    self.analyze_probe_results(probe_name)

            except Exception as e:
                print("Failed to process figure part")
                print(f"  Failed to process {probe_name}: {str(e)}")
                results[probe_name] = None

        # Generate overall summary
        self._generate_experiment_summary(results)

        return results

    def _generate_experiment_summary(self, results):
        """Generate a summary report for the entire experiment"""

        summary_file = self.results_dir / "experiment_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("NEUROPIXEL EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Experiment path: {self.experiment_path}\n")
            f.write(f"Total probes: {len(self.probe_configs)}\n\n")

            for probe_name, config in self.probe_configs.items():
                f.write(f"{probe_name}:\n")
                f.write(f"  Type: {config['probe_type']}\n")
                f.write(f"  Sample rate: {config['sample_rate']} Hz\n")
                f.write(f"  Status: {'Processed' if results.get(probe_name) else 'Failed'}\n")

                if results.get(probe_name):
                    try:
                        results_dir = results[probe_name]
                        contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
                        good_units = np.sum(contam_pct <= 10)
                        total_units = len(contam_pct)

                        f.write(f"  Good units: {good_units}/{total_units}\n")
                        f.write(f"  Results: {results_dir}\n")
                    except:
                        f.write("  Error reading results\n")
                f.write("\n")

            f.write("\nPHY VISUALIZATION INSTRUCTIONS:\n")
            f.write("-" * 30 + "\n")
            for probe_name in self.probe_configs.keys():
                if results.get(probe_name):
                    results_dir = results[probe_name]
                    f.write(f"{probe_name}:\n")
                    f.write(f"  cd {results_dir}\n")
                    f.write(f"  phy template-gui params.py\n\n")

        print(f"Experiment summary saved to: {summary_file}")


class NeuropixelSessionManager:
    """
    Manages multiple Neuropixel recording sessions and automatically processes unprocessed ones
    """

    def __init__(self, base_path):
        """
        Initialize the session manager

        Parameters:
        -----------
        base_path : str or Path
            Base path containing all recording sessions (raw data directory)
            Example: /data/.../Micr/mc_raw/ or /data/.../Shca/sc_raw/
            The manager will automatically create a parallel _proc directory for processed data
        """
        self.base_path = Path(base_path)

        # Create parallel processed directory structure
        # Convert _raw to _proc (e.g., mc_raw -> mc_proc, sc_raw -> sc_proc)
        base_path_str = str(self.base_path)
        if '_raw' in base_path_str:
            self.proc_base_path = Path(base_path_str.replace('_raw', '_proc'))
        else:
            # Fallback: add _proc suffix if _raw not found
            self.proc_base_path = self.base_path.parent / (self.base_path.name + '_proc')

        print(f"Raw data path: {self.base_path}")
        print(f"Processed data path: {self.proc_base_path}")

        self.sessions = {}
        self.processed_sessions = set()
        self.failed_sessions = set()

        # Scan for all sessions
        self._discover_sessions()

    def _discover_sessions(self):
        """
        Discover all OpenEphys sessions in the base directory
        New structure: base_path/{mouse_id}/{session_name}/{timestamp}/Record Node .../experiment*
        Checks for processed results in parallel _proc directory
        """
        print("Discovering OpenEphys sessions...")

        # Pattern for OpenEphys session directories with new structure
        # Format: {mouse_id}/{session_name}/{timestamp}/Record Node */experiment*
        session_patterns = [
            "*/*/*/Record Node */experiment*"
        ]

        found_sessions = []
        for pattern in session_patterns:
            found_sessions.extend(self.base_path.glob(pattern))

        # Filter for directories that contain settings.xml
        for session_path in found_sessions:
            print(session_path)
            settings_file = session_path.parent / "settings.xml"
            if settings_file.exists():
                # Extract session identifier from path
                session_id, session_name = self._get_session_id(session_path)
                self.sessions[session_id] = {
                    'path': session_path,
                    'raw_path': session_path,  # Store original raw path
                    'name': session_name,
                    'status': 'unprocessed',
                    'results': None,
                    'error': None,
                }

                # Check if already processed in the parallel _proc directory
                # Convert raw path to proc path
                relative_path = session_path.relative_to(self.base_path)
                proc_session_path = self.proc_base_path / relative_path
                kilosort_results = proc_session_path

                if kilosort_results.exists() and self._check_results_complete(kilosort_results):
                    self.sessions[session_id]['status'] = 'processed'
                    self.sessions[session_id]['results'] = kilosort_results
                    self.sessions[session_id]['proc_path'] = proc_session_path
                    self.processed_sessions.add(session_id)

        print(f"Found {len(self.sessions)} sessions")
        print(f"  - Already processed: {len(self.processed_sessions)}")
        print(f"  - Unprocessed: {len(self.sessions) - len(self.processed_sessions)}")

    def _get_session_id(self, session_path):
        """
        Extract a unique session identifier from the path
        New format: {mouse_id}/{session_name}/{timestamp}
        Example: mc004/mc004_250613/2025-06-13_14-44-37 -> "mc004_mc004_250613_2025-06-13_14-44-37"
        """
        path_parts = session_path.parts

        # The structure is: .../mc/{mouse_id}/{session_name}/{timestamp}/Record Node .../experiment*
        # We need to extract the last 3 levels before "Record Node"
        try:
            # Find the index of the part containing "Record Node"
            record_node_idx = None
            for i, part in enumerate(path_parts):
                if "Record Node" in part:
                    record_node_idx = i
                    break

            if record_node_idx and record_node_idx >= 3:
                # Extract mouse_id, session_name, and timestamp
                timestamp = path_parts[record_node_idx - 1]
                session_name = path_parts[record_node_idx - 2]
                mouse_id = path_parts[record_node_idx - 3]

                # Create a combined session ID
                session_id = f"{mouse_id}_{session_name}_{timestamp}"
                return session_id, session_name
        except (IndexError, ValueError):
            pass

        # Fallback: use the timestamp directory name if extraction fails
        for part in reversed(path_parts):
            if any(char.isdigit() for char in part) and "2025" in part:
                return part

        # Final fallback: use the experiment directory name
        return path_parts[-1]

    def _check_results_complete(self, results_dir):
        """
        Check if Kilosort results are complete for a session

        Parameters:
        -----------
        results_dir : Path
            Path to kilosort4_results directory

        Returns:
        --------
        bool: True if results appear complete
        """
        # Look for subdirectories (probe results)
        probe_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

        # Check if at least one probe has complete results
        for probe_dir in probe_dirs:
            required_files = [
                'spike_times.npy',
                'spike_clusters.npy',
                'cluster_ContamPct.tsv',
                'params.py'
                'unit_analysis/summary_plot_1.png'
            ]

            if all((probe_dir / file).exists() for file in required_files):
                return True

        return False

    def _prepare_proc_directory(self, session_id):
        """
        Prepare the processed directory for a session and copy TSV files

        Parameters:
        -----------
        session_id : str
            Session identifier

        Returns:
        --------
        Path: Path to the processed session directory
        """
        info = self.sessions[session_id]
        raw_session_path = info['raw_path']
        raw_session_path_related_part = raw_session_path.parent.parent.parent
        # Create the parallel directory structure in _proc
        relative_path = raw_session_path_related_part.relative_to(self.base_path)
        proc_session_path = self.proc_base_path / relative_path

        # Create the processed directory structure
        proc_session_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created processed directory: {proc_session_path}")

        # Find and copy TSV files from the raw session directory
        # TSV files are typically in the parent directories (timestamp or session level)
        # Search pattern: go up to find TSV files in mouse/session directory
        # Example: /data/.../mc_raw/mc004/mc004_250613/mc004-2025-06-13-144516.tsv

        search_dirs = [
            raw_session_path.parent.parent.parent.parent,
            raw_session_path.parent.parent.parent,  # Mouse/session level (e.g., mc004/mc004_250613/)
            raw_session_path.parent.parent,  # Session level
            raw_session_path.parent,  # Timestamp level
        ]

        tsv_files_copied = []
        for search_dir in search_dirs:
            if search_dir.exists():
                for tsv_file in search_dir.glob("*.tsv"):
                    # Copy to the processed session directory
                    dest_file = proc_session_path / tsv_file.name
                    if dest_file.exists():
                        tsv_files_copied.append(tsv_file.name)
                    elif not dest_file.exists():
                        import shutil
                        shutil.copy2(tsv_file, dest_file)
                        tsv_files_copied.append(tsv_file.name)
                        print(f"  Copied TSV file: {tsv_file.name} -> {proc_session_path}")

        if not tsv_files_copied:
            print(f"  Warning: No TSV files found in search directories for {session_id}")
        else:
            print(f"  Successfully copied {len(tsv_files_copied)} TSV file(s)")

        return proc_session_path

    def list_sessions(self, status_filter=None):
        """
        List all sessions, optionally filtered by status

        Parameters:
        -----------
        status_filter : str or None
            Filter by status: 'processed', 'unprocessed', 'failed', or None for all
        """
        print(f"\nSession Summary (filter: {status_filter or 'all'})")
        print("-" * 60)

        for session_id, info in self.sessions.items():
            if status_filter is None or info['status'] == status_filter:
                print(f"{session_id}")
                print(f"  Status: {info['status']}")
                print(f"  Path: {info['path']}")

                if info['results']:
                    print(f"  Results: {info['results']}")
                if info['error']:
                    print(f"  Error: {info['error']}")
                print()

    def process_unprocessed_sessions(self, analyze_results=True, max_sessions=None):
        """
        Process all unprocessed sessions

        Parameters:
        -----------
        analyze_results : bool
            Whether to generate analysis plots
        max_sessions : int or None
            Maximum number of sessions to process (for testing)
        """
        unprocessed = [
            (sid, info) for sid, info in self.sessions.items()
            if info['status'] == 'unprocessed'
        ]

        if max_sessions:
            unprocessed = unprocessed[:max_sessions]

        print(f"\nProcessing {len(unprocessed)} unprocessed sessions...")

        for i, (session_id, info) in enumerate(unprocessed, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing session {i}/{len(unprocessed)}: {session_id}")
            print(f"{'=' * 60}")

            try:
                # Prepare the processed directory and copy TSV files
                print(f"Preparing processed directory for {session_id}...")
                proc_session_path = self._prepare_proc_directory(session_id)

                # Create processor for this session
                # Pass the raw data path for reading, and proc path for saving results
                processor = OpenEphysNeuropixelProcessor(
                    info['raw_path'],
                    info['name'],
                    results_dir=proc_session_path,
                )

                # Process all probes in this session
                results = processor.process_all_probes(analyze_results=analyze_results)

                # Update session status
                self.sessions[session_id]['status'] = 'processed'
                self.sessions[session_id]['results'] = processor.results_dir
                self.sessions[session_id]['proc_path'] = proc_session_path
                self.processed_sessions.add(session_id)

                print(f"✓ Successfully processed {session_id}")

            except Exception as e:
                print(f"✗ Failed to process {session_id}: {str(e)}")
                self.sessions[session_id]['status'] = 'failed'
                self.sessions[session_id]['error'] = str(e)
                self.failed_sessions.add(session_id)

        # Print final summary
        self._print_processing_summary()

    def _print_processing_summary(self):
        """Print a summary of processing results"""
        total = len(self.sessions)
        processed = len(self.processed_sessions)
        failed = len(self.failed_sessions)
        remaining = total - processed - failed

        print(f"\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total sessions: {total}")
        print(f"Successfully processed: {processed}")
        print(f"Failed: {failed}")
        print(f"Remaining unprocessed: {remaining}")

        if failed > 0:
            print(f"\nFailed sessions:")
            for session_id in self.failed_sessions:
                error = self.sessions[session_id]['error']
                print(f"  - {session_id}: {error}")

        if processed > 0:
            print(f"\nTo visualize results with Phy:")
            for session_id in self.processed_sessions:
                results_dir = self.sessions[session_id]['results']
                if results_dir:
                    # List probe subdirectories
                    probe_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
                    print(f"\n{session_id}:")
                    for probe_dir in probe_dirs:
                        print(f"  cd {probe_dir}")
                        print(f"  phy template-gui params.py")

    def process_specific_session(self, session_id, analyze_results=True):
        """
        Process a specific session by ID

        Parameters:
        -----------
        session_id : str
            Session identifier
        analyze_results : bool
            Whether to generate analysis plots
        """
        if session_id not in self.sessions:
            print(f"Session '{session_id}' not found")
            return False

        info = self.sessions[session_id]
        print(f"Processing session: {session_id}")

        try:
            # Prepare the processed directory and copy TSV files
            print(f"Preparing processed directory for {session_id}...")
            proc_session_path = self._prepare_proc_directory(session_id)

            # Create processor for this session
            processor = OpenEphysNeuropixelProcessor(
                info['raw_path'],
                results_dir=proc_session_path
            )
            results = processor.process_all_probes(analyze_results=analyze_results)

            self.sessions[session_id]['status'] = 'processed'
            self.sessions[session_id]['results'] = processor.results_dir
            self.sessions[session_id]['proc_path'] = proc_session_path
            self.processed_sessions.add(session_id)

            print(f"✓ Successfully processed {session_id}")
            return True

        except Exception as e:
            print(f"✗ Failed to process {session_id}: {str(e)}")
            self.sessions[session_id]['status'] = 'failed'
            self.sessions[session_id]['error'] = str(e)
            self.failed_sessions.add(session_id)
            return False


# Example usage functions
def calculate_firing_rates_with_gaussian(spike_times, bin_size_ms=1, sigma=1, t_start=None, t_end=None):
    """
    Calculate firing rates using Gaussian smoothing

    Parameters:
    -----------
    spike_times : array
        Array of spike times in seconds
    bin_size_ms : float
        Size of each bin in milliseconds
    sigma : float
        Standard deviation of the Gaussian kernel in bins
    t_start, t_end : float
        Start and end times in seconds
    """
    if t_start is None:
        t_start = 0 if len(spike_times) == 0 else np.min(spike_times)
    if t_end is None:
        t_end = 1 if len(spike_times) == 0 else np.max(spike_times) + 0.1

    # Create time bins (convert bin_size to seconds)
    bin_size_sec = bin_size_ms / 1000
    bins = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
    time_points = bins[:-1] + bin_size_sec / 2  # Centers of bins

    # Convert spike times to histogram
    hist, _ = np.histogram(spike_times, bins=bins)

    # Convert to firing rate (spikes/second)
    firing_rates = hist / bin_size_sec

    # Apply Gaussian smoothing
    kernel_sigma_bins = sigma
    smoothed_rates = gaussian_filter1d(firing_rates, kernel_sigma_bins)

    return time_points, smoothed_rates


def parse_arguments():
    """
    Parse command-line arguments for configurable script execution

    This allows the script to be run with different base paths without
    editing the code, making it suitable for automated processing pipelines.
    """
    parser = argparse.ArgumentParser(
        description='Kilosort4 Processing Pipeline for Neuropixels Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          # Process data from mc_raw directory (default)
          python kilosort_processing.py --base-path /data/cephfs-2/unmirrored/groups/peng/Micr/mc_raw

          # Process data from sc_raw directory
          python kilosort_processing.py --base-path /data/cephfs-2/unmirrored/groups/peng/Shca/sc_raw

          # Process data from a different project
          python kilosort_processing.py --base-path /data/cephfs-2/unmirrored/groups/peng/OtherProject/data_raw

          # Process only first session for testing
          python kilosort_processing.py --base-path /path/to/data_raw --max-sessions 1
                """
    )

    # Required argument: base path for raw data
    parser.add_argument(
        '--base-path',
        type=str,
        required=True,
        help='Base directory path containing raw recording sessions (e.g., /path/to/mc_raw/)'
    )

    # Optional argument: maximum number of sessions to process
    parser.add_argument(
        '--max-sessions',
        type=int,
        default=None,
        help='Maximum number of sessions to process (default: process all unprocessed sessions)'
    )

    # Optional argument: whether to generate analysis plots
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help='Skip generation of analysis plots (faster processing)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Display configuration
    print("=" * 80)
    print("KILOSORT4 PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Base path: {args.base_path}")
    print(f"Max sessions: {args.max_sessions if args.max_sessions else 'All'}")
    print(f"Generate analysis: {not args.no_analysis}")
    print("=" * 80)
    print()

    # Initialize the session manager with the specified base path
    print("Initializing Neuropixel Session Manager...")
    print("=" * 60)
    manager = NeuropixelSessionManager(args.base_path)
    print("=" * 60)

    # List all sessions to see what was found
    manager.list_sessions()

    # Process all unprocessed sessions
    # Set max_sessions=1 to test with just one session first
    print("\nStarting automatic processing of unprocessed sessions...")
    manager.process_unprocessed_sessions(
        analyze_results=True,  # Generate analysis plots
        max_sessions=None  # Set to 1 for testing, None to process all
    )

    print("\nProcessing complete!")

    # Optionally process a specific session
    # manager.process_specific_session("single_column2025-03-23_17-17-55", analyze_results=True)
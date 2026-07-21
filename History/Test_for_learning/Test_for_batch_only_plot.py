#!/usr/bin/env python3
"""
Batch quality control script for Neuropixels recordings across multiple sessions.

This script implements BombCell-style quality control metrics for spike sorting validation
and generates Rastermap visualizations for entire recording sessions. It automatically
discovers sessions and processes each probe independently, generating both session-level
detailed outputs and a summary CSV across all sessions.

Key differences from trial-aligned processing:
- No behavioral event alignment required
- Focus on unit quality metrics (contamination, firing rate, ISI violations, etc.)
- Visualization of full time series rather than trial snippets
- Session-level summary statistics

Usage:
    python batch_quality_control.py /path/to/processed/data/folder --bin-size 0.1
"""

try:
    from rastermap import Rastermap

    RASTERMAP_AVAILABLE = True
except ImportError:
    RASTERMAP_AVAILABLE = False
    print("Warning: Rastermap not installed. Install with: pip install rastermap")

import numpy as np
import pandas as pd
#import bombcell as bc
from pathlib import Path
import scipy.io as sio
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import gridspec
import warnings
import sys
import traceback


warnings.filterwarnings('ignore')


class SessionQualityControl:
    """
    Perform quality control analysis on a single Neuropixels probe recording.

    This class implements BombCell-inspired quality metrics to assess spike sorting
    quality without requiring behavioral event data. Metrics include:
    - Contamination percentage (false positive rate)
    - Firing rate statistics
    - ISI violation rate
    - Amplitude stability
    - Presence ratio (fraction of recording with activity)
    """

    def __init__(self, probe_path: Path, probe_name: str, session_name: str):
        """
        Initialize quality control processor for a single probe.

        Parameters
        ----------
        probe_path : Path
            Path to Kilosort output directory (e.g., hl001_251113_ProbeA/)
        probe_name : str
            Probe identifier (e.g., 'ProbeA', 'ProbeB')
        session_name : str
            Session identifier (e.g., 'hl001_251113')
        """
        self.probe_path = Path(probe_path)
        self.probe_name = probe_name
        self.session_name = session_name

        # Sampling rate for Neuropixels
        self.fs = 30000

        # Data containers
        self.spike_times = None
        self.spike_clusters = None
        self.all_clusters = None
        self.good_clusters = None
        self.cluster_info = None
        self.quality_metrics = None

    def load_kilosort_data(self) -> None:
        """
        Load essential Kilosort output files.

        Loads spike times, cluster assignments, and quality metrics from
        the Kilosort output directory. Automatically detects sampling rate
        from params.py if available.
        """
        print(f"\n[{self.probe_name}] Loading Kilosort outputs from {self.probe_path.name}...")

        # Load spike times and convert to seconds
        self.spike_times = np.load(self.probe_path / 'spike_times.npy').flatten()
        self.spike_clusters = np.load(self.probe_path / 'spike_clusters.npy').flatten()

        # Check for params.py to get correct sampling rate
        params_path = self.probe_path / 'params.py'
        if params_path.exists():
            with open(params_path, 'r') as f:
                for line in f:
                    if 'sample_rate' in line:
                        self.fs = float(line.split('=')[1].strip())
                        print(f"[{self.probe_name}] Found sampling rate: {self.fs} Hz")
                        break

        # Convert spike times from samples to seconds
        self.spike_times = self.spike_times / self.fs

        # Get all unique clusters
        self.all_clusters = np.unique(self.spike_clusters)

        # Load contamination information if available
        cluster_contam_path = self.probe_path / 'cluster_ContamPct.tsv'
        if cluster_contam_path.exists():
            self.cluster_info = pd.read_csv(cluster_contam_path, sep='\t')
            # Define good units as those with contamination < 10%
            self.good_clusters = self.cluster_info[
                self.cluster_info['ContamPct'] < 10.0
                ]['cluster_id'].values
            print(f"[{self.probe_name}] Found {len(self.good_clusters)} good units "
                  f"(contamination < 10%) out of {len(self.all_clusters)} total")
        else:
            print(f"[{self.probe_name}] Warning: cluster_ContamPct.tsv not found. "
                  f"All {len(self.all_clusters)} clusters will be considered.")
            self.good_clusters = self.all_clusters
            # Create minimal cluster_info DataFrame
            self.cluster_info = pd.DataFrame({
                'cluster_id': self.all_clusters,
                'ContamPct': np.nan
            })

    def compute_quality_metrics(self) -> pd.DataFrame:
        """
        Compute BombCell-style quality metrics for all clusters.

        Quality metrics computed:
        - firing_rate: Mean firing rate across entire recording (Hz)
        - isi_violation_rate: Percentage of ISI violations (< 2ms refractory period)
        - presence_ratio: Fraction of recording with activity (in 1s bins)
        - amplitude_cutoff: Estimated fraction of missing spikes based on amplitude distribution
        - contamination_pct: From Kilosort's cluster_ContamPct.tsv
        - is_good: Boolean indicating if unit passes quality thresholds

        Returns
        -------
        pd.DataFrame
            Quality metrics for each cluster
        """
        print(f"[{self.probe_name}] Computing quality metrics for {len(self.all_clusters)} clusters...")

        # Get total recording duration
        recording_duration = self.spike_times.max()

        metrics_list = []

        for cluster_id in self.all_clusters:
            # Get spike times for this cluster
            cluster_mask = self.spike_clusters == cluster_id
            cluster_spike_times = self.spike_times[cluster_mask]
            n_spikes = len(cluster_spike_times)

            # 1. Firing rate (Hz)
            firing_rate = n_spikes / recording_duration

            # 2. ISI violation rate (refractory period violations)
            # Count spikes with ISI < 2ms (0.002 seconds)
            if n_spikes > 1:
                isis = np.diff(cluster_spike_times)
                isi_violations = np.sum(isis < 0.002)
                isi_violation_rate = 100 * isi_violations / n_spikes
            else:
                isi_violation_rate = np.nan

            # 3. Presence ratio
            # Divide recording into 1-second bins and check which bins have spikes
            if n_spikes > 0:
                n_bins = int(np.ceil(recording_duration))
                spike_bins = np.floor(cluster_spike_times).astype(int)
                spike_bins = spike_bins[spike_bins < n_bins]  # Handle edge case
                occupied_bins = len(np.unique(spike_bins))
                presence_ratio = occupied_bins / n_bins
            else:
                presence_ratio = 0.0

            # 4. Amplitude cutoff estimation
            # This is a simplified version - full BombCell uses more sophisticated methods
            # We estimate based on whether firing rate is stable across recording
            if n_spikes > 50:
                # Divide into 10 epochs and check firing rate stability
                n_epochs = 10
                epoch_duration = recording_duration / n_epochs
                epoch_rates = []
                for i in range(n_epochs):
                    epoch_start = i * epoch_duration
                    epoch_end = (i + 1) * epoch_duration
                    epoch_mask = (cluster_spike_times >= epoch_start) & (cluster_spike_times < epoch_end)
                    epoch_rate = np.sum(epoch_mask) / epoch_duration
                    epoch_rates.append(epoch_rate)

                # Amplitude cutoff proxy: coefficient of variation of firing rate
                mean_rate = np.mean(epoch_rates)
                std_rate = np.std(epoch_rates)
                amplitude_cutoff = std_rate / (mean_rate + 1e-10)  # Add small value to avoid division by zero
            else:
                amplitude_cutoff = np.nan

            # Get contamination from cluster_info if available
            if 'ContamPct' in self.cluster_info.columns:
                contamination = self.cluster_info[
                    self.cluster_info['cluster_id'] == cluster_id
                    ]['ContamPct'].values[0] if cluster_id in self.cluster_info['cluster_id'].values else np.nan
            else:
                contamination = np.nan

            # Determine if unit is "good" based on multiple criteria
            is_good = (
                    (cluster_id in self.good_clusters) and  # Passes contamination threshold
                    (firing_rate > 0.1) and  # Minimum firing rate
                    (presence_ratio > 0.2) and  # Present throughout recording
                    (isi_violation_rate < 0.5 if not np.isnan(isi_violation_rate) else False)  # Low ISI violations
            )

            metrics_list.append({
                'cluster_id': cluster_id,
                'probe': self.probe_name,
                'session': self.session_name,
                'n_spikes': n_spikes,
                'firing_rate': firing_rate,
                'isi_violation_rate': isi_violation_rate,
                'presence_ratio': presence_ratio,
                'amplitude_cutoff': amplitude_cutoff,
                'contamination_pct': contamination,
                'is_good': is_good
            })

        self.quality_metrics = pd.DataFrame(metrics_list)

        n_good = self.quality_metrics['is_good'].sum()
        print(f"[{self.probe_name}] Quality metrics computed: {n_good} good units identified")

        return self.quality_metrics

    def generate_quality_plots(self, output_dir: Path) -> None:
        """
        Generate quality control visualization plots.

        Creates a multi-panel figure showing:
        - Firing rate distribution
        - ISI violation rates
        - Presence ratio distribution
        - Scatter plots of quality metrics

        Parameters
        ----------
        output_dir : Path
            Directory to save the figure
        """
        print(f"[{self.probe_name}] Generating quality control plots...")

        output_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

        good_mask = self.quality_metrics['is_good']

        # 1. Firing rate distribution
        ax = fig.add_subplot(gs[0, 0])
        bins1 = np.arange(0, 20, 0.5)
        ax.hist(self.quality_metrics['firing_rate'], bins=bins1, alpha=0.4, label='All units', color='gray')
        bins2 = np.arange(0, 20, 0.01)
        ax.hist(self.quality_metrics[good_mask]['firing_rate'], bins=bins2, alpha=0.8, label='Good units', color='green')
        ax.set_xlabel('Firing Rate (Hz)')
        ax.set_ylabel('Count')
        ax.set_title(f'{self.probe_name}: Firing Rate Distribution')
        ax.legend()
        ax.set_yscale('log')
        ax.set_xlim([0, 20])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 2. ISI violation rate
        ax = fig.add_subplot(gs[0, 1])
        valid_isi = self.quality_metrics['isi_violation_rate'].notna()
        bins1 = np.arange(0,10,0.1)
        ax.hist(self.quality_metrics[valid_isi]['isi_violation_rate'], bins=bins1, alpha=0.4,
                label='All units', color='gray')
        bins2= np.arange(0, 0.5, 0.1)
        ax.hist(self.quality_metrics[good_mask & valid_isi]['isi_violation_rate'], bins=bins2,
                alpha=0.8, label='Good units', color='green')
        ax.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5%)')
        ax.set_xlabel('ISI Violation Rate (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'{self.probe_name}: ISI Violations')
        ax.legend()
        ax.set_xlim([0, 10])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 3. Presence ratio
        ax = fig.add_subplot(gs[0, 2])
        bins1 = np.arange(0,1,0.01)
        ax.hist(self.quality_metrics['presence_ratio'], bins=bins1, alpha=0.4, label='All units', color='gray')
        ax.hist(self.quality_metrics[good_mask]['presence_ratio'], bins=bins1, alpha=0.8,
                label='Good units', color='green')
        ax.axvline(0.2, color='red', linestyle='--', label='Threshold (0.2)')
        ax.set_xlabel('Presence Ratio')
        ax.set_ylabel('Count')
        ax.set_title(f'{self.probe_name}: Presence Ratio')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 4. Contamination vs Firing Rate
        ax = fig.add_subplot(gs[1, 0])
        valid_contam = self.quality_metrics['contamination_pct'].notna()
        ax.scatter(self.quality_metrics[valid_contam & ~good_mask]['firing_rate'],
                   self.quality_metrics[valid_contam & ~good_mask]['contamination_pct'],
                   alpha=0.5, s=20, color='gray', label='Rejected')
        ax.scatter(self.quality_metrics[valid_contam & good_mask]['firing_rate'],
                   self.quality_metrics[valid_contam & good_mask]['contamination_pct'],
                   alpha=0.7, s=20, color='green', label='Good')
        ax.axhline(10, color='red', linestyle='--', label='Contamination threshold')
        ax.set_xlabel('Firing Rate (Hz)')
        ax.set_ylabel('Contamination (%)')
        ax.set_ylim([0, 200])
        ax.set_title('Contamination vs Firing Rate')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 5. ISI violations vs Firing Rate
        ax = fig.add_subplot(gs[1, 1])
        ax.scatter(self.quality_metrics[valid_isi & ~good_mask]['firing_rate'],
                   self.quality_metrics[valid_isi & ~good_mask]['isi_violation_rate'],
                   alpha=0.5, s=20, color='gray', label='Rejected')
        ax.scatter(self.quality_metrics[valid_isi & good_mask]['firing_rate'],
                   self.quality_metrics[valid_isi & good_mask]['isi_violation_rate'],
                   alpha=0.7, s=20, color='green', label='Good')
        ax.axhline(0.5, color='red', linestyle='--', label='ISI violation threshold')
        ax.set_xlabel('Firing Rate (Hz)')
        ax.set_ylabel('ISI Violation Rate (%)')
        ax.set_title('ISI Violations vs Firing Rate')
        ax.legend()
        ax.set_ylim([0,10])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 6. Presence ratio vs Firing Rate
        ax = fig.add_subplot(gs[1, 2])
        ax.scatter(self.quality_metrics[~good_mask]['firing_rate'],
                   self.quality_metrics[~good_mask]['presence_ratio'],
                   alpha=0.5, s=20, color='gray', label='Rejected')
        ax.scatter(self.quality_metrics[good_mask]['firing_rate'],
                   self.quality_metrics[good_mask]['presence_ratio'],
                   alpha=0.7, s=20, color='green', label='Good')
        ax.axhline(0.2, color='red', linestyle='--', label='Presence threshold')
        ax.set_xlabel('Firing Rate (Hz)')
        ax.set_ylabel('Presence Ratio')
        ax.set_title('Presence Ratio vs Firing Rate')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 7. Summary statistics text
        ax = fig.add_subplot(gs[2, 0:1])
        ax.axis('off')

        summary_text = f"Quality Control Summary: {self.probe_name}\n"
        summary_text += "=" * 60 + "\n\n"
        summary_text += f"Total units: {len(self.quality_metrics)}\n"
        summary_text += f"Good units: {good_mask.sum()} ({100 * good_mask.sum() / len(self.quality_metrics):.1f}%)\n\n"

        summary_text += f"Firing Rate:\n"
        summary_text += f"  Mean: {self.quality_metrics['firing_rate'].mean():.2f} Hz\n"
        summary_text += f"  Median: {self.quality_metrics['firing_rate'].median():.2f} Hz\n"
        summary_text += f"  Range: [{self.quality_metrics['firing_rate'].min():.2f}, "
        summary_text += f"{self.quality_metrics['firing_rate'].max():.2f}] Hz\n\n"

        if valid_isi.any():
            summary_text += f"ISI Violation Rate:\n"
            summary_text += f"  Mean: {self.quality_metrics[valid_isi]['isi_violation_rate'].mean():.3f}%\n"
            summary_text += f"  Units with violations > 0.5%: "
            summary_text += f"{(self.quality_metrics['isi_violation_rate'] > 0.5).sum()}\n\n"

        summary_text += f"Presence Ratio:\n"
        summary_text += f"  Mean: {self.quality_metrics['presence_ratio'].mean():.3f}\n"
        summary_text += f"  Units with presence > 0.2: "
        summary_text += f"{(self.quality_metrics['presence_ratio'] > 0.2).sum()}\n"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.1))

        plt.savefig(output_dir / f'{self.session_name}_{self.probe_name}_quality_control.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[{self.probe_name}] Quality control plots saved")


class MultiProbeSession:
    """
    Process and analyze all probes from a single recording session.

    This class coordinates quality control analysis across multiple probes
    within a session and generates session-level summary statistics and
    visualizations using Rastermap.
    """

    def __init__(self, session_path: Path, session_name: str, bin_size: float = 0.1):
        """
        Initialize multi-probe session processor.

        Parameters
        ----------
        session_path : Path
            Path to session directory containing probe subdirectories
        session_name : str
            Session identifier (e.g., 'hl001_251113')
        bin_size : float
            Bin size in seconds for computing firing rates (default: 0.1s)
        """
        self.session_path = Path(session_path)
        self.session_name = session_name
        self.bin_size = bin_size

        self.probe_processors = {}  # Dictionary of SessionQualityControl objects
        self.all_quality_metrics = []
        self.session_summary = {}

    def find_probe_directories(self) -> List[Tuple[Path, str]]:
        """
        Find all probe subdirectories within the session.

        Expected naming convention: {session_name}_ProbeA, {session_name}_ProbeB, etc.

        Returns
        -------
        List[Tuple[Path, str]]
            List of (probe_path, probe_name) tuples
        """
        probe_dirs = []

        # Look for directories matching the pattern {session_name}_Probe*
        for item in self.session_path.iterdir():
            if item.is_dir() and item.name.startswith(self.session_name):
                # Check if it contains essential Kilosort files
                if (item / 'spike_times.npy').exists() and (item / 'spike_clusters.npy').exists():
                    # Extract probe name (e.g., "ProbeA" from "hl001_251113_ProbeA")
                    probe_name = item.name.replace(f"{self.session_name}_", "")
                    probe_dirs.append((item, probe_name))

        return sorted(probe_dirs)

    def process_all_probes(self) -> None:
        """
        Process quality control for all probes in the session.

        For each probe, loads data, computes quality metrics, and stores
        results for later aggregation and visualization.
        """
        probe_dirs = self.find_probe_directories()

        if not probe_dirs:
            print(f"\nWarning: No probe directories found in {self.session_path}")
            return

        print(f"\n{'=' * 80}")
        print(f"Processing session: {self.session_name}")
        print(f"Found {len(probe_dirs)} probe(s)")
        print(f"{'=' * 80}")

        for probe_path, probe_name in probe_dirs:
            print(f"\nProcessing {probe_name}...")

            # Create quality control processor for this probe
            processor = SessionQualityControl(
                probe_path=probe_path,
                probe_name=probe_name,
                session_name=self.session_name
            )

            # Load data and compute metrics
            processor.load_kilosort_data()
            quality_metrics = processor.compute_quality_metrics()

            # Store results
            self.probe_processors[probe_name] = processor
            self.all_quality_metrics.append(quality_metrics)

    def generate_session_summary(self) -> Dict:
        """
        Generate session-level summary statistics across all probes.

        Creates a dictionary suitable for writing as a single row in the
        master CSV file. Includes good/all unit counts for each probe and
        aggregate statistics.

        Returns
        -------
        Dict
            Session summary statistics with one entry per probe and overall totals
        """
        summary = {
            'session_name': self.session_name,
            'session_path': str(self.session_path),
            'n_probes': len(self.probe_processors),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        total_good = 0
        total_all = 0

        # Add probe-specific columns
        for probe_name, processor in self.probe_processors.items():
            n_good = processor.quality_metrics['is_good'].sum()
            n_all = len(processor.quality_metrics)

            # Create columns like "good_ProbeA", "all_ProbeA", "ratio_ProbeA"
            summary[f'good_{probe_name}'] = n_good
            summary[f'all_{probe_name}'] = n_all
            summary[f'ratio_{probe_name}'] = f"{n_good}/{n_all}"

            # Aggregate statistics
            summary[f'mean_firing_rate_{probe_name}'] = processor.quality_metrics['firing_rate'].mean()
            summary[f'median_contamination_{probe_name}'] = processor.quality_metrics['contamination_pct'].median()

            total_good += n_good
            total_all += n_all

        # Overall session statistics
        summary['total_good_units'] = total_good
        summary['total_all_units'] = total_all
        summary['overall_ratio'] = f"{total_good}/{total_all}"

        self.session_summary = summary
        return summary

    def save_detailed_outputs(self, tsv_dir: Path, plot_dir: Path) -> None:
        """
        Save detailed per-probe TSV files and quality control plots.

        Parameters
        ----------
        tsv_dir : Path
            Directory for saving detailed TSV files
        plot_dir : Path
            Directory for saving quality control plots
        """
        tsv_session_dir = tsv_dir / self.session_name
        plot_session_dir = plot_dir / self.session_name

        tsv_session_dir.mkdir(parents=True, exist_ok=True)
        plot_session_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving detailed outputs for {self.session_name}...")

        for probe_name, processor in self.probe_processors.items():
            # Save quality metrics as TSV
            # tsv_path = tsv_session_dir / f'{self.session_name}_{probe_name}_quality_metrics.tsv'
            # processor.quality_metrics.to_csv(tsv_path, sep='\t', index=False)
            # print(f"  Saved: {tsv_path.name}")

            # Generate and save quality control plots
            processor.generate_quality_plots(plot_session_dir)

    def generate_rastermap_visualization(self, output_dir: Path) -> None:
        """
        Generate Rastermap visualization of full session activity across all good units.

        This creates a population-level visualization showing firing patterns of all
        well-isolated units throughout the entire recording, sorted by Rastermap's
        dimensionality reduction algorithm to reveal functional clustering.

        Parameters
        ----------
        output_dir : Path
            Directory for saving Rastermap figures
        """
        import re
        from matplotlib.gridspec import GridSpec


        if not RASTERMAP_AVAILABLE:
            print(f"\nSkipping Rastermap visualization (rastermap not installed)")
            return

        print(f"\nGenerating Rastermap visualization for {self.session_name}...")

        output_dir.mkdir(parents=True, exist_ok=True)

        sub_sessions: dict[str, dict] = {}  # prefix → {probe_name: processor}
        for probe_name, processor in self.probe_processors.items():
            # Strip trailing _Probe<letter(s)>
            match = re.match(r'^(.*?)_?(Probe[A-Z]+)$', probe_name)
            if match and match.group(1):  # new-style name
                prefix = match.group(1)
            else:  # old-style: just "ProbeA"
                prefix = ""
            sub_sessions.setdefault(prefix, {})[probe_name] = processor
        for prefix, probe_dict in sorted(sub_sessions.items()):

            # Human-readable label for filenames / plot titles
            if prefix:
                sub_session_label = f"{self.session_name}_{prefix}"
            else:
                sub_session_label = self.session_name

            print(f"\nGenerating Rastermap for sub-session: {sub_session_label} "
                  f"({len(probe_dict)} probe(s))")


            print(f'probe_dict {probe_dict}')
            # ---- collect good units ----------------------------------------

            all_spike_times = []
            all_cluster_ids = []
            neuron_labels = []

            for probe_name, processor in probe_dict.items():
                if hasattr(processor, 'unit_type_string') and processor.unit_type_string is not None:
                    good_mask = processor.unit_type_string == 'GOOD'
                    qm_df = pd.DataFrame(processor.quality_metrics)
                    good_clusters = qm_df.loc[good_mask, 'phy_clusterID'].values

                    spike_times_raw = np.load(processor.probe_path / 'spike_times.npy').flatten()
                    spike_clusters = np.load(processor.probe_path / 'spike_clusters.npy').flatten()

                    fs = 30000
                    params_path = processor.probe_path / 'params.py'
                    if params_path.exists():
                        with open(params_path) as f:
                            for line in f:
                                if 'sample_rate' in line:
                                    fs = float(line.split('=')[1].strip())
                                    break
                    spike_times_sec = spike_times_raw / fs

                    for cid in good_clusters:
                        mask = spike_clusters == cid
                        all_spike_times.append(spike_times_sec[mask])
                        neuron_labels.append(f"{probe_name}_c{int(cid)}")

                # --- manual-QC version ---
                else:
                    good_mask = processor.quality_metrics['is_good']
                    good_clusters = processor.quality_metrics[good_mask]['cluster_id'].values
                    for cid in good_clusters:
                        mask = processor.spike_clusters == cid
                        all_spike_times.append(processor.spike_times[mask])
                        neuron_labels.append(f"{probe_name}_c{cid}")

            if not all_spike_times:
                print(f"  No good units found for Rastermap visualization")
                return

            print(f"  Processing {len(all_spike_times)} good units across all probes")

            # Determine recording duration (use maximum spike time across all units)
            recording_duration = max([st.max() for st in all_spike_times if len(st) > 0])

            # Bin spike times into firing rate matrix
            # Matrix shape: (n_neurons, n_time_bins)
            n_bins = int(np.ceil(recording_duration / self.bin_size))
            time_vector = np.arange(0, recording_duration, self.bin_size)

            firing_rate_matrix = np.zeros((len(all_spike_times), n_bins))

            # Create Gaussian kernel for smoothing
            window_size = 0.02  # 50 ms window
            kernel_width = int(window_size / self.bin_size)
            kernel = np.exp(-0.5 * (np.arange(-3 * kernel_width, 3 * kernel_width + 1) / kernel_width) ** 2)
            kernel = kernel / (kernel.sum() * self.bin_size)  # Normalize to preserve firing rate units

            for neuron_idx, spike_times in enumerate(all_spike_times):
                if len(spike_times) > 0:
                    # Step 1: Bin the spike times into discrete counts
                    spike_counts, _ = np.histogram(spike_times, bins=n_bins,
                                                   range=(0, recording_duration))

                    # Step 2: Convert to instantaneous firing rate (spikes per bin / bin duration)
                    #instantaneous_rate = spike_counts / self.bin_size
                    instantaneous_rate = spike_counts
                    # Step 3: Smooth with Gaussian kernel
                    firing_rate_matrix[neuron_idx, :] = np.convolve(instantaneous_rate, kernel, mode='same')

            # Z-score normalize each neuron
            for i in range(firing_rate_matrix.shape[0]):
                if firing_rate_matrix[i, :].std() > 0:
                    firing_rate_matrix[i, :] = (firing_rate_matrix[i, :] - firing_rate_matrix[i,
                                                                           :].mean()) / firing_rate_matrix[i, :].std()

            # Apply Rastermap for sorting
            print(f"  Applying Rastermap dimensionality reduction...")
            try:
                model = Rastermap(
                    n_PCs=firing_rate_matrix.shape[0],  # Use fewer PCs for efficiency
                    locality=0.75,
                    time_lag_window=5,
                    grid_upsample=0,
                    verbose=False
                )
                model.fit(firing_rate_matrix)
                sorted_indices = model.isort

                firing_rate_sorted = firing_rate_matrix[sorted_indices, :]
                neuron_labels_sorted = [neuron_labels[i] for i in sorted_indices]

                print(f"  Rastermap sorting complete")

            except Exception as e:
                print(f"  Rastermap failed: {e}")
                print(f"  Using original order")
                firing_rate_sorted = firing_rate_matrix
                neuron_labels_sorted = neuron_labels

            from matplotlib.gridspec import GridSpec

            # Create figure with GridSpec for precise control
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(2, 2, figure=fig, width_ratios=[20, 1], height_ratios=[3, 1],
                          hspace=0.3, wspace=0.05)

            # Create axes: main heatmap, colorbar space, and population activity
            ax_heatmap = fig.add_subplot(gs[0, 0])
            ax_cbar = fig.add_subplot(gs[0, 1])
            ax_population = fig.add_subplot(gs[1, 0])

            # Main heatmap (your existing code)
            im = ax_heatmap.imshow(firing_rate_sorted, aspect='auto', cmap='RdBu_r',
                                   vmin=-2, vmax=2, interpolation='nearest',
                                   extent=[0, recording_duration / 60, firing_rate_sorted.shape[0], 0])

            ax_heatmap.set_xlabel('Time (minutes)', fontsize=14)
            ax_heatmap.set_ylabel('Neuron (Rastermap sorted)', fontsize=14)
            ax_heatmap.set_title(f'{self.session_name}: Full Session Activity (z-scored firing rates)\n'
                                 f'{len(neuron_labels)} good units across {len(probe_dict)} probe(s)',
                                 fontsize=14)

            # Add colorbar to its dedicated space
            cbar = plt.colorbar(im, cax=ax_cbar)
            cbar.set_label('Z-scored Firing Rate', fontsize=12)

            # Population average activity (your existing code)
            mean_activity = firing_rate_sorted.mean(axis=0)
            time_axis_minutes = time_vector / 60

            ax_population.plot(time_axis_minutes, mean_activity, color='black', linewidth=1)
            ax_population.fill_between(time_axis_minutes, mean_activity, alpha=0.3, color='gray')
            ax_population.set_xlabel('Time (minutes)', fontsize=14)
            ax_population.set_ylabel('Mean Activity\n(z-score)', fontsize=14)
            ax_population.set_title('Average Population Activity', fontsize=14)
            ax_population.set_xlim([0, recording_duration / 60])
            ax_population.spines['top'].set_visible(False)
            ax_population.spines['right'].set_visible(False)

            plt.tight_layout()

            # output_path = output_dir / f'{self.session_name}_rastermap_full_session.png'
            # plt.savefig(output_path, dpi=300, bbox_inches='tight')
            # plt.close()
            #
            # print(f"  Saved: {output_path.name}")
            #
            # # Save sorted firing rate matrix for later analysis
            # np.save(output_dir / f'{self.session_name}_firing_rate_sorted.npy', firing_rate_sorted)
            # with open(output_dir / f'{self.session_name}_neuron_labels.txt', 'w') as f:
            #     for label in neuron_labels_sorted:
            #         f.write(f"{label}\n")

            out_png = output_dir / f'{sub_session_label}_rastermap_full_session.png'
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {out_png.name}")

            # ---- save artefacts -------------------------------------------
            np.save(output_dir / f'{sub_session_label}_firing_rate_sorted.npy',
                    firing_rate_sorted)
            with open(output_dir / f'{sub_session_label}_neuron_labels.txt', 'w') as f:
                for label in neuron_labels_sorted:
                    f.write(f"{label}\n")

class BatchQualityControlProcessor:
    """
    Batch processor for quality control analysis across multiple sessions.

    This is the top-level class that orchestrates the entire pipeline:
    1. Discovers all sessions in a processed data directory
    2. Processes each session independently
    3. Aggregates results into a master summary CSV
    4. Saves detailed outputs organized by session
    """

    def __init__(self, processed_root: Path, experimenter_name: str, bin_size: float = 0.1):
        """
        Initialize batch quality control processor.

        Parameters
        ----------
        processed_root : Path
            Root directory containing processed sessions (e.g., .../hl_proc/)
        experimenter_name : str
            Experimenter identifier for organizing outputs (e.g., 'Holly')
        bin_size : float
            Bin size in seconds for firing rate computation (default: 0.1s)
        """
        self.processed_root = Path(processed_root)
        self.experimenter_name = experimenter_name
        self.bin_size = bin_size

        # Setup output directory structure
        self.output_base = Path('/data/cephfs-2/unmirrored/groups/peng') / experimenter_name
        self.output_base.mkdir(parents=True, exist_ok=True)

        self.tsv_dir = self.output_base / 'session_level_tsv'
        self.plot_dir = self.output_base / 'session_level_plot'
        self.rastermap_dir = self.output_base / 'session_level_rastermap'

        # Storage for results
        self.sessions_found = []
        self.sessions_processed = []
        self.sessions_failed = []
        self.all_session_summaries = []

    def discover_sessions(self) -> List[Dict]:
        """
        Discover all sessions in the processed data directory.

        A session is identified as a directory containing probe subdirectories
        with Kilosort outputs (spike_times.npy, spike_clusters.npy).

        Expected structure:
        processed_root/
            mouse_id/
                session_id/
                    session_id_ProbeA/
                    session_id_ProbeB/

        Returns
        -------
        List[Dict]
            List of discovered session information dictionaries
        """
        print(f"\n{'=' * 80}")
        print(f"Discovering sessions in: {self.processed_root}")
        print(f"{'=' * 80}")

        sessions = []

        # Look for directories that contain probe subdirectories
        for mouse_dir in self.processed_root.iterdir():
            if not mouse_dir.is_dir():
                continue

            for session_dir in mouse_dir.iterdir():
                if not session_dir.is_dir():
                    print(f"Skipping {session_dir}")
                    continue

                # Check if this directory contains probe subdirectories
                has_probes = False
                for item in session_dir.iterdir():
                    if item.is_dir() and item.name.startswith(session_dir.name):
                        # Check for essential Kilosort files
                        if (item / 'spike_times.npy').exists() and (item / 'spike_clusters.npy').exists():
                            has_probes = True
                            print(f"Found {item}")
                            break

                if has_probes:
                    session_info = {
                        'session_path': session_dir,
                        'session_name': session_dir.name,
                        'mouse_id': mouse_dir.name
                    }
                    sessions.append(session_info)
                    print(f"  ✓ Found session: {session_info['session_name']}")

        self.sessions_found = sessions
        print(f"\nTotal sessions discovered: {len(sessions)}")

        return sessions

    def process_session(self, session_info: Dict) -> bool:
        """
        Process a single session through the quality control pipeline.

        Parameters
        ----------
        session_info : Dict
            Session information dictionary from discover_sessions()

        Returns
        -------
        bool
            True if processing succeeded, False otherwise
        """
        session_name = session_info['session_name']

        print(f"\n{'=' * 80}")
        print(f"Processing Session: {session_name}")
        print(f"{'=' * 80}")

        try:
            # Create session processor
            session_processor = MultiProbeSession(
                session_path=session_info['session_path'],
                session_name=session_name,
                bin_size=self.bin_size
            )

            # Process all probes
            session_processor.process_all_probes()

            if not session_processor.probe_processors:
                print(f"  Warning: No probes were successfully processed")
                return False

            # # Generate session summary
            # summary = session_processor.generate_session_summary()
            # self.all_session_summaries.append(summary)

            # Save detailed outputs
            session_processor.save_detailed_outputs(self.tsv_dir, self.plot_dir)

            # # Generate Rastermap visualization
            #session_processor.generate_rastermap_visualization(self.rastermap_dir)

            self.sessions_processed.append(session_info)
            print(f"\n✓ Session {session_name} processed successfully")

            return True

        except Exception as e:
            print(f"\n✗ Error processing session {session_name}:")
            print(f"  {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()

            self.sessions_failed.append({
                'session_info': session_info,
                'error': str(e),
                'traceback': traceback.format_exc()
            })

            return False

    def process_all_sessions(self) -> None:
        """
        Discover and process all sessions in batch mode.

        This is the main entry point for batch processing. It orchestrates
        the entire pipeline and saves all results.
        """
        # Discover sessions
        sessions = self.discover_sessions()

        if not sessions:
            print("\n⚠ No sessions found to process!")
            return

        # Process each session
        print(f"\n{'=' * 80}")
        print(f"Beginning batch processing of {len(sessions)} sessions")
        print(f"{'=' * 80}")

        for i, session_info in enumerate(sessions, 1):
            print(f"\n[Session {i}/{len(sessions)}]")
            self.process_session(session_info)

        # Save master summary CSV
        self.save_master_summary()

        # Print final summary
        self.print_summary()

    def save_master_summary(self) -> None:
        """
        Save the master session summary CSV file.

        This file contains one row per session with quality control statistics
        from all probes aggregated into a single row.
        """
        if not self.all_session_summaries:
            print("\nNo session summaries to save")
            return

        summary_df = pd.DataFrame(self.all_session_summaries)
        summary_path = self.output_base / 'session_summary.csv'

        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Master summary saved: {summary_path}")

    def print_summary(self) -> None:
        """Print and save batch processing summary."""
        print(f"\n{'=' * 80}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'=' * 80}")

        print(f"\nTotal sessions found: {len(self.sessions_found)}")
        print(f"Successfully processed: {len(self.sessions_processed)}")
        print(f"Failed: {len(self.sessions_failed)}")

        if self.sessions_processed:
            print(f"\n✓ Processed sessions:")
            for session in self.sessions_processed:
                print(f"  - {session['session_name']}")

        if self.sessions_failed:
            print(f"\n✗ Failed sessions:")
            for failure in self.sessions_failed:
                print(f"  - {failure['session_info']['session_name']}")
                print(f"    Error: {failure['error']}")

        # Save detailed summary to file
        summary_file = self.output_base / 'batch_processing_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("BATCH QUALITY CONTROL PROCESSING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Processed root: {self.processed_root}\n")
            f.write(f"Experimenter: {self.experimenter_name}\n")
            f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bin size: {self.bin_size}s\n\n")
            f.write(f"Total sessions found: {len(self.sessions_found)}\n")
            f.write(f"Successfully processed: {len(self.sessions_processed)}\n")
            f.write(f"Failed: {len(self.sessions_failed)}\n\n")

            if self.sessions_processed:
                f.write("Processed sessions:\n")
                for session in self.sessions_processed:
                    f.write(f"  ✓ {session['session_name']}\n")
                f.write("\n")

            if self.sessions_failed:
                f.write("Failed sessions:\n")
                for failure in self.sessions_failed:
                    f.write(f"  ✗ {failure['session_info']['session_name']}\n")
                    f.write(f"    Error: {failure['error']}\n\n")

        print(f"\nDetailed summary saved: {summary_file}")
        print(f"\nOutput directories:")
        print(f"  - Master summary: {self.output_base / 'session_summary.csv'}")
        print(f"  - Session TSVs: {self.tsv_dir}")
        print(f"  - Quality plots: {self.plot_dir}")
        print(f"  - Rastermap figures: {self.rastermap_dir}")
        print(f"{'=' * 80}\n")


def main():
    """Main entry point with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch quality control for Neuropixels recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sessions for experimenter 'Holly' with default 100ms bins
  python batch_quality_control.py /data/cephfs-2/unmirrored/groups/peng/Holly/hl_proc --experimenter Holly

  # Use custom bin size for firing rate computation
  python batch_quality_control.py /path/to/proc/data --experimenter Name --bin-size 0.05
        """
    )

    # parser.add_argument('processed_root', type=str,
    #                     default='/Users/shengyuancai/Downloads/Kilosort/kilosort4_results',
    #                     nargs='?',  # This makes it optiona
    #                     help='Path to processed data root directory (e.g., .../hl_proc/)')

    parser.add_argument('processed_root', type=str,
                        help='Path to processed data root directory (e.g., .../hl_proc/)')
    # parser.add_argument('--experimenter', type=str, required=True,
    #                     help='Experimenter name for organizing outputs (e.g., "Holly")')
    parser.add_argument('--bin-size', type=float, default=0.01,
                        help='Bin size in seconds for firing rate computation (default: 0.1)')

    args = parser.parse_args()

    processed_root = Path(args.processed_root)

    if not processed_root.exists():
        print(f"Error: Path does not exist: {processed_root}")
        sys.exit(1)

    print("=" * 80)
    print("Batch Quality Control Pipeline for Neuropixels")
    print("=" * 80)
    print(f"Processed root: {processed_root}")
    print(f"Experimenter: {processed_root.parts[6]}")
    print(f"Bin size: {args.bin_size}s")
    print("=" * 80)

    # Create and run batch processor
    batch_processor = BatchQualityControlProcessor(
        processed_root=processed_root,
        experimenter_name= processed_root.parts[6],
        bin_size=args.bin_size
    )

    batch_processor.process_all_sessions()

    print("\n" + "=" * 80)
    print("✓ Processing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
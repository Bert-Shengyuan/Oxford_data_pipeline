#!/usr/bin/env python3
"""
Batch quality control script for Neuropixels recordings using BombCell.

This script implements the full BombCell quality control pipeline for spike sorting
validation and generates Rastermap visualizations. It automatically discovers sessions
and processes each probe independently, generating:
- Session-level summary CSV with all sessions
- Detailed per-probe TSV files
- BombCell quality control plots
- Rastermap population visualizations

Key features:
- Uses BombCell Python package (pybombcell) directly
- Computes comprehensive quality metrics (contamination, firing rate, ISI violations,
  spatial decay, amplitude stability, waveform properties, etc.)
- Generates BombCell's three summary plot types (waveforms, upset plots, histograms)
- Creates session-level and aggregate summaries

Usage:
    python batch_quality_control_bombcell.py /path/to/processed/data/folder --bin-size 0.1

    Example:
    python batch_quality_control_bombcell.py /data/cephfs-2/unmirrored/groups/peng/Holly/hl_proc
"""

try:
    import bombcell as bc

    BOMBCELL_AVAILABLE = True
except ImportError:
    BOMBCELL_AVAILABLE = False
    print("Warning: pybombcell not installed. Install with: pip install pybombcell")

try:
    from rastermap import Rastermap

    RASTERMAP_AVAILABLE = True
except ImportError:
    RASTERMAP_AVAILABLE = False
    print("Warning: Rastermap not installed. Install with: pip install rastermap")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import gridspec
import warnings
import sys
import traceback
import shutil

warnings.filterwarnings('ignore')

#%%
class SessionQualityControlBombCell:
    """
    Perform BombCell quality control analysis on a single Neuropixels probe recording.

    This class uses the pybombcell package to compute comprehensive quality metrics
    including contamination percentage, firing rate statistics, ISI violations,
    amplitude stability, presence ratio, waveform properties, and spatial decay.
    """

    def __init__(self, probe_path: Path, probe_name: str, session_name: str):
        """
        Initialize BombCell quality control processor for a single probe.

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

        # BombCell output directory
        self.bc_output_dir = self.probe_path / 'bombcell'

        # Data containers
        self.quality_metrics = None
        self.param = None
        self.unit_type = None
        self.unit_type_string = None
        self.figures = None

    @staticmethod
    def _is_bombcell_complete(probe_path: Path) -> bool:
        """Check whether BombCell has already been run for this probe."""
        bc_dir = probe_path / 'bombcell'
        return (
                bc_dir.exists()
                and (bc_dir / 'for_GUI').is_dir()
                and (bc_dir / 'templates._bc_qMetrics.parquet').exists()
        )

    def _load_existing_results(self) -> bool:
        """
        Load BombCell quality metrics from previously saved files instead of
        re-running the full pipeline.

        Reads templates._bc_qMetrics.parquet (or .csv fallback) written by
        pybombcell. Expects a 'bc_unitType' column with values like
        'GOOD', 'MUA', 'NOISE', 'NON-SOMA'.
        """
        try:
            parquet_path = self.bc_output_dir / 'templates._bc_qMetrics.parquet'
            csv_path = self.bc_output_dir / 'templates._bc_qMetrics.csv'

            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                print(f"[{self.probe_name}] No saved metrics file found — will re-run.")
                return False

            # Locate the unit-type column (pybombcell uses 'bc_unitType')
            type_col = next(
                (c for c in ('bc_unitType', 'unitType', 'unit_type') if c in df.columns),
                None
            )
            if type_col is None:
                print(f"[{self.probe_name}] Unit-type column not found in saved file — will re-run.")
                return False

            self.unit_type_string = df[type_col].values
            self.unit_type = np.unique(self.unit_type_string, return_inverse=True)[1]

            # Attach probe / session labels (mirrors what run_bombcell does)
            df = df.copy()
            df['probe'] = self.probe_name
            df['session'] = self.session_name
            self.quality_metrics = df

            n_good = int(np.sum(self.unit_type_string == 'GOOD'))
            n_total = len(self.unit_type_string)
            print(f"[{self.probe_name}] Loaded existing BombCell results: "
                  f"{n_good} good / {n_total} total")
            return True

        except Exception as e:
            print(f"[{self.probe_name}] Failed to load existing results: {e} — will re-run.")
            return False


    def run_bombcell(self) -> bool:
        if not BOMBCELL_AVAILABLE:
            print(f"[{self.probe_name}] Error: bombcell not installed")
            return False

        # ── Skip recomputation if output already exists ────────────────────────
        if self._is_bombcell_complete(self.probe_path):
            print(f"[{self.probe_name}] BombCell output already exists — loading saved results.")
            return self._load_existing_results()
        # ───────────────────────────────────────────────────────────────────────

        print(f"\n[{self.probe_name}] Running BombCell quality metrics...")

        try:
            # Get default BombCell parameters
            self.param = bc.get_default_parameters(str(self.probe_path))

            # Customize parameters for your recording
            # This can adjust these based on your specific needs
            self.param["computeDistanceMetrics"] = 0 # Can be slow for large datasets
            self.param["computeDrift"] = 1  # Can be slow
            self.param["computeTimeChunks"] = 0  # Don't split into time chunks

            # Adjust classification thresholds if needed
            self.param["maxRPVviolations"] = 0.5
            self.param["minNumSpikes"] = 40
            self.param["minPresenceRatio"] = 0.2
            # self.param["lratioMax"] =

            # Run BombCell
            print(f"[{self.probe_name}] Computing quality metrics...")
            (
                self.quality_metrics,
                self.param,
                self.unit_type,
                self.unit_type_string,
                self.figures,
            ) = bc.run_bombcell(
                ks_dir=str(self.probe_path),
                save_path=str(self.bc_output_dir),
                param=self.param,
                return_figures=True
            )

            # Add probe and session identifiers to quality metrics
            self.quality_metrics['probe'] = self.probe_name
            self.quality_metrics['session'] = self.session_name

            # Count good units
            n_good = np.sum(self.unit_type_string == 'GOOD')
            n_total = len(self.unit_type_string)

            print(f"[{self.probe_name}] BombCell complete: {n_good} good units "
                  f"out of {n_total} total")

            return True

        except Exception as e:
            print(f"[{self.probe_name}] Error running BombCell: {e}")
            traceback.print_exc()
            return False

    def save_bombcell_figures(self, output_dir: Path) -> None:
        """
        Save BombCell's summary plots to the session plot directory.

        BombCell generates three types of plots:
        1. Waveform overlay plot
        2. UpSet plots (noise, non-somatic, MUA)
        3. Quality metric histogram distributions

        Parameters
        ----------
        output_dir : Path
            Directory to save the figures
        """
        if self.figures is None:
            print(f"[{self.probe_name}] No figures to save")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{self.probe_name}] Saving BombCell figures...")

        try:
            # Save waveform overlay plot
            if 'waveforms_overlay' in self.figures:
                waveforms_fig = self.figures['waveforms_overlay']
                waveforms_path = output_dir / f'{self.session_name}_{self.probe_name}_waveforms_overlay.png'
                waveforms_fig.savefig(waveforms_path, dpi=300, bbox_inches='tight')
                plt.close(waveforms_fig)
                print(f"  Saved: {waveforms_path.name}")

            # Save upset plots (list of figures)
            if 'upset_plots' in self.figures:
                upset_figs = self.figures['upset_plots']
                for i, upset_fig in enumerate(upset_figs):
                    upset_path = output_dir / f'{self.session_name}_{self.probe_name}_upset_plot_{i + 1}.png'
                    upset_fig.savefig(upset_path, dpi=300, bbox_inches='tight')
                    plt.close(upset_fig)
                    print(f"  Saved: {upset_path.name}")

            # Save histograms plot
            if 'histograms' in self.figures:
                histograms_fig = self.figures['histograms']
                histograms_path = output_dir / f'{self.session_name}_{self.probe_name}_histograms.png'
                histograms_fig.savefig(histograms_path, dpi=300, bbox_inches='tight')
                plt.close(histograms_fig)
                print(f"  Saved: {histograms_path.name}")

        except Exception as e:
            print(f"[{self.probe_name}] Error saving figures: {e}")
            traceback.print_exc()

    def get_quality_summary(self) -> Dict:
        """
        Generate summary statistics for this probe.

        Returns
        -------
        Dict
            Summary statistics including counts of good/MUA/noise units
            and mean values of key quality metrics
        """
        if self.quality_metrics is None:
            return {}

        # Count units by type
        n_good = np.sum(self.unit_type_string == 'GOOD')
        n_mua = np.sum(self.unit_type_string == 'MUA')
        n_noise = np.sum(self.unit_type_string == 'NOISE')
        n_non_somatic = np.sum(self.unit_type_string == 'NON-SOMA')
        n_total = len(self.unit_type_string)

        summary = {
            f'n_good_{self.probe_name}': n_good,
            f'n_mua_{self.probe_name}': n_mua,
            f'n_noise_{self.probe_name}': n_noise,
            f'n_non_somatic_{self.probe_name}': n_non_somatic,
            f'n_total_{self.probe_name}': n_total,
            f'ratio_{self.probe_name}': f"{n_good}/{n_total}",
        }

        # Add mean quality metrics for good units only
        good_mask = self.unit_type_string == 'GOOD'
        if n_good > 0:
            # Get quality metrics dataframe
            qm_df = pd.DataFrame(self.quality_metrics)

            # Select key metrics to summarize
            key_metrics = ['nSpikes', 'presenceRatio', 'fractionRPVs_estimatedTauR',
                           'spatialDecaySlope', 'waveformBaselineFlatness']

            for metric in key_metrics:
                if metric in qm_df.columns:
                    mean_val = qm_df.loc[good_mask, metric].mean()
                    summary[f'mean_{metric}_{self.probe_name}'] = mean_val

        return summary


class MultiProbeSessionBombCell:
    """
    Process and analyze all probes from a single recording session using BombCell.

    This class coordinates BombCell quality control analysis across multiple probes
    within a session and generates session-level summary statistics and
    Rastermap visualizations.
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

        self.probe_processors = {}  # Dictionary of SessionQualityControlBombCell objects
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

    def process_all_probes(self) -> bool:
        """
        Process BombCell quality control for all probes in the session.

        For each probe, runs BombCell analysis and stores results for later
        aggregation and visualization.

        Returns
        -------
        bool
            True if at least one probe was successfully processed
        """
        probe_dirs = self.find_probe_directories()

        if not probe_dirs:
            print(f"\nWarning: No probe directories found in {self.session_path}")
            return False

        print(f"\n{'=' * 80}")
        print(f"Processing session: {self.session_name}")
        print(f"Found {len(probe_dirs)} probe(s)")
        print(f"{'=' * 80}")

        success_count = 0

        for probe_path, probe_name in probe_dirs:
            print(f"\nProcessing {probe_name}...")

            # Create BombCell processor for this probe
            processor = SessionQualityControlBombCell(
                probe_path=probe_path,
                probe_name=probe_name,
                session_name=self.session_name
            )

            # Run BombCell
            if processor.run_bombcell():
                self.probe_processors[probe_name] = processor
                success_count += 1
            else:
                print(f"[{probe_name}] Failed to process")

        return success_count > 0

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
            probe_summary = processor.get_quality_summary()
            summary.update(probe_summary)

            # Extract counts for totals
            n_good = probe_summary.get(f'n_good_{probe_name}', 0)
            n_all = probe_summary.get(f'n_total_{probe_name}', 0)

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
        Save detailed per-probe TSV files and BombCell quality control plots.

        Parameters
        ----------
        tsv_dir : Path
            Base directory for saving detailed TSV files
        plot_dir : Path
            Base directory for saving quality control plots
        """
        tsv_session_dir = tsv_dir / self.session_name
        plot_session_dir = plot_dir / self.session_name

        tsv_session_dir.mkdir(parents=True, exist_ok=True)
        plot_session_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving detailed outputs for {self.session_name}...")

        for probe_name, processor in self.probe_processors.items():
            # Save quality metrics as TSV
            if processor.quality_metrics is not None:
                qm_df = pd.DataFrame(processor.quality_metrics)

                # Add unit type classification
                qm_df.insert(0, 'unit_type', processor.unit_type_string)

                tsv_path = tsv_session_dir / f'{self.session_name}_{probe_name}_quality_metrics.tsv'
                qm_df.to_csv(tsv_path, sep='\t', index=False)
                print(f"  Saved: {tsv_path.name}")

                # Also copy BombCell's individual cluster_*.tsv files to session directory
                bc_output = processor.bc_output_dir
                if bc_output.exists():
                    for cluster_tsv in bc_output.glob('cluster_*.tsv'):
                        dest = tsv_session_dir / f'{self.session_name}_{probe_name}_{cluster_tsv.name}'
                        shutil.copy2(cluster_tsv, dest)

            # Save BombCell's summary plots
            processor.save_bombcell_figures(plot_session_dir)

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
            # ── Skip if Rastermap PNG already exists ──────────────────────────
            out_png = output_dir / f'{sub_session_label}_rastermap_full_session.png'
            if out_png.exists():
                print(f"  Rastermap already exists for {sub_session_label} — skipping.")
                continue
            # ──────────────────────────────────────────────────────────────────
            print(f"\nGenerating Rastermap for sub-session: {sub_session_label} "
                  f"({len(probe_dict)} probe(s))")

            # ---- collect good units ----------------------------------------

            all_spike_times = []
            neuron_labels = []
            print(f'probe_dict {probe_dict}')
            for probe_name, processor in probe_dict.items():
                print(f'probe_name {probe_name}')
                print(f'processor {processor}')
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

            # # Determine recording duration (use maximum spike time across all units)
            # recording_duration = max([st.max() for st in all_spike_times if len(st) > 0])
            #
            # # Bin spike times into firing rate matrix
            # # Matrix shape: (n_neurons, n_time_bins)
            # n_bins = int(np.ceil(recording_duration / self.bin_size))
            # time_vector = np.arange(0, recording_duration, self.bin_size)
            #
            # firing_rate_matrix = np.zeros((len(all_spike_times), n_bins))
            #
            # # Create Gaussian kernel for smoothing
            # window_size = 0.02  # 50 ms window
            # kernel_width = int(window_size / self.bin_size)
            # kernel = np.exp(-0.5 * (np.arange(-3 * kernel_width, 3 * kernel_width + 1) / kernel_width) ** 2)
            # kernel = kernel / (kernel.sum() * self.bin_size)  # Normalize to preserve firing rate units

            # for neuron_idx, spike_times in enumerate(all_spike_times):
            #     if len(spike_times) > 0:
            #         # Step 1: Bin the spike times into discrete counts
            #         spike_counts, _ = np.histogram(spike_times, bins=n_bins,
            #                                        range=(0, recording_duration))
            #
            #         # Step 2: Convert to instantaneous firing rate (spikes per bin / bin duration)
            #         # instantaneous_rate = spike_counts / self.bin_size
            #         instantaneous_rate = spike_counts
            #         # Step 3: Smooth with Gaussian kernel
            #         firing_rate_matrix[neuron_idx, :] = np.convolve(instantaneous_rate, kernel, mode='same')
            #
            # # Z-score normalize each neuron
            # for i in range(firing_rate_matrix.shape[0]):
            #     if firing_rate_matrix[i, :].std() > 0:
            #         firing_rate_matrix[i, :] = (firing_rate_matrix[i, :] - firing_rate_matrix[i,
            #                                                                :].mean()) / firing_rate_matrix[i, :].std()
            # Determine recording duration
            recording_duration = max([st.max() for st in all_spike_times if len(st) > 0])

            # Bin spike times into firing rate matrix
            n_bins = int(np.ceil(recording_duration / self.bin_size))
            time_vector = np.arange(0, recording_duration, self.bin_size)

            firing_rate_matrix = np.zeros((len(all_spike_times), n_bins))

            for neuron_idx, spike_times in enumerate(all_spike_times):
                if len(spike_times) > 0:
                    spike_counts, _ = np.histogram(spike_times, bins=n_bins,
                                                   range=(0, recording_duration))
                    firing_rate_matrix[neuron_idx, :] = spike_counts / self.bin_size

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
                    locality=0,  #locality=0,
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
    Batch processor for BombCell quality control analysis across multiple sessions.

    This is the top-level class that orchestrates the entire pipeline:
    1. Discovers all sessions in a processed data directory
    2. Processes each session independently with BombCell
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

        Returns
        -------
        List[Dict]
            List of discovered session information dictionaries
        """
        print(f"\n{'=' * 80}")
        print(f"Discovering sessions in: {self.processed_root}")
        print(f"{'=' * 80}")

        sessions = []


        for session_dir in self.processed_root.iterdir():  # ← one level only
            if not session_dir.is_dir():
                continue

            has_probes = False
            for item in session_dir.iterdir():
                if item.is_dir() and item.name.startswith(session_dir.name):
                    if (item / 'spike_times.npy').exists() and (item / 'spike_clusters.npy').exists():
                        has_probes = True
                        break

            if has_probes:
                session_info = {
                    'session_path': session_dir,
                    'session_name': session_dir.name,
                    # mouse_id removed — no longer a separate directory level
                }
                sessions.append(session_info)
                print(f"  ✓ Found session: {session_info['session_name']}")

        self.sessions_found = sessions
        print(f"\nTotal sessions discovered: {len(sessions)}")

        return sessions

    def process_session(self, session_info: Dict) -> bool:
        """
        Process a single session through the BombCell quality control pipeline.

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
            session_processor = MultiProbeSessionBombCell(
                session_path=session_info['session_path'],
                session_name=session_name,
                bin_size=self.bin_size
            )

            # Process all probes with BombCell
            if not session_processor.process_all_probes():
                print(f"  Warning: No probes were successfully processed")
                return False

            # Generate session summary
            summary = session_processor.generate_session_summary()
            self.all_session_summaries.append(summary)

            # Save detailed outputs
            session_processor.save_detailed_outputs(self.tsv_dir, self.plot_dir)

            # Generate Rastermap visualization
            session_processor.generate_rastermap_visualization(self.rastermap_dir)

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
        # Check if BombCell is available
        if not BOMBCELL_AVAILABLE:
            print("\nError: pybombcell is not installed!")
            print("Install with: pip install pybombcell")
            return

        # Discover sessions
        sessions = self.discover_sessions()

        if not sessions:
            print("\nNo sessions found to process!")
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
            f.write("BATCH BOMBCELL QUALITY CONTROL PROCESSING SUMMARY\n")
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
        print(f"  - BombCell plots: {self.plot_dir}")
        print(f"  - Rastermap figures: {self.rastermap_dir}")
        print(f"{'=' * 80}\n")


def main():
    """Main entry point with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch BombCell quality control for Neuropixels recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sessions for experimenter 'Holly' with default 100ms bins
  python batch_quality_control_bombcell.py /data/cephfs-2/unmirrored/groups/peng/Holly/hl_proc

  # Use custom bin size for firing rate computation
  python batch_quality_control_bombcell.py /path/to/proc/data --bin-size 0.05
        """
    )

    parser.add_argument(
        'processed_root',
        type=str,
        help='Path to processed data root directory (e.g., .../hl_proc/)'
    )


    parser.add_argument('--bin-size', type=float, default=0.1,
                        help='Bin size in seconds for firing rate computation (default: 0.1)')

    args = parser.parse_args()

    processed_root = Path(args.processed_root)

    if not processed_root.exists():
        print(f"Error: Path does not exist: {processed_root}")
        sys.exit(1)

    # Extract experimenter name from path
    # Assumes structure: .../groups/peng/ExperimenterName/...
    experimenter_name = processed_root.parts[6] if len(processed_root.parts) > 6 else 'Unknown'

    print("=" * 80)
    print("Batch BombCell Quality Control Pipeline for Neuropixels")
    print("=" * 80)
    print(f"Processed root: {processed_root}")
    print(f"Experimenter: {experimenter_name}")
    print(f"Bin size: {args.bin_size}s")
    print("=" * 80)

    # Create and run batch processor
    batch_processor = BatchQualityControlProcessor(
        processed_root=processed_root,
        experimenter_name=experimenter_name,
        bin_size=args.bin_size
    )

    batch_processor.process_all_sessions()

    print("\n" + "=" * 80)
    print("✓ Processing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
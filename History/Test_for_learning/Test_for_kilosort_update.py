# %%
import argparse
import re

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
            Expected structure: .../timestamp/Record Node XXX/experimentN/
        experiment_name : str
            Session name (e.g., 'mc015_260314')
        results_dir : str or Path, optional
            Custom directory for saving Kilosort results. If None, uses default location.
        """
        self.experiment_path = Path(experiment_path)
        self.session_name = experiment_name
        self.settings_file = (self.experiment_path.parent / "settings.xml")
        self.probe_configs = {}

        # Set results directory
        if results_dir is not None:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = self.experiment_path.parent.parent.parent

        # Extract session time + experiment index from the path hierarchy
        # e.g. .../2026-03-15_00-07-59/Record Node 101/experiment1
        #  -> session_time = '000759', experiment_index = 'E1'
        self._extract_path_metadata()

        # Parse the settings to understand probe configuration
        self._parse_settings()

    # ------------------------------------------------------------------
    # NEW: extract session-time and experiment index from path
    # ------------------------------------------------------------------
    def _extract_path_metadata(self):
        """
        Walk the experiment_path parts to pull out:
          - session_time  : HHMMSS string from the timestamp folder
                            e.g. '2026-03-15_00-07-59' -> '000759'
          - experiment_index : 'E1', 'E2', … from the experimentN folder name
        """
        # ---- experiment index ------------------------------------------
        exp_name = self.experiment_path.name          # e.g. 'experiment1'
        exp_num = ''.join(filter(str.isdigit, exp_name))
        self.experiment_index = f"E{exp_num}" if exp_num else "E1"

        # ---- session time from the timestamp directory ------------------
        # Timestamp directories look like: 2026-03-15_00-07-59
        timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_(\d{2})-(\d{2})-(\d{2})$')
        self.session_time = "000000"   # safe default

        for part in self.experiment_path.parts:
            m = timestamp_pattern.match(part)
            if m:
                hh, mm, ss = m.group(1), m.group(2), m.group(3)
                self.session_time = f"{hh}{mm}{ss}"   # e.g. '000759'
                break

        print(f"  Path metadata: session_time={self.session_time}, "
              f"experiment_index={self.experiment_index}")

    def _parse_settings(self):
        """
        Parse the OpenEphys settings.xml file to extract probe configurations
        """
        print(f"Parsing OpenEphys settings.xml file in {self.experiment_path}")

        if not self.settings_file.exists():
            raise FileNotFoundError(f"Settings file not found: {self.settings_file}")

        tree = ET.parse(self.settings_file)
        root = tree.getroot()

        neuropix_processor = root.find(".//PROCESSOR[@pluginName='Neuropix-PXI']")
        if neuropix_processor is None:
            raise ValueError("No Neuropix-PXI processor found in settings")

        streams = neuropix_processor.findall(".//STREAM")

        for stream in streams:
            stream_name   = stream.get("name")
            device_name   = stream.get("device_name")
            sample_rate   = float(stream.get("sample_rate"))
            channel_count = int(stream.get("channel_count"))

            if "2.0" in device_name:
                probe_type = "neuropixels_2.0"
                probe_map  = "neuropixPhase3B2_kilosortChanMap.mat"
            else:
                probe_type = "neuropixels_1.0"
                probe_map  = "neuropixPhase3B1_kilosortChanMap.mat"

            probe_config = {
                'device_name':   device_name,
                'probe_type':    probe_type,
                'sample_rate':   sample_rate,
                'channel_count': channel_count,
                'probe_map':     probe_map,
                'stream_name':   stream_name,
            }

            if "AP" in stream_name or "2.0" in device_name:
                probe_name = stream_name.split("-")[0]
                self.probe_configs[probe_name] = probe_config
                print(f"  Found {probe_name}: {probe_type}, AP data at {sample_rate} Hz")
            elif "LFP" in stream_name:
                probe_name = stream_name.split("-")[0]
                print(f"  Found {probe_name}: LFP data at {sample_rate} Hz "
                      f"(skipping for spike sorting)")

        print(f"  Total probes configured for spike sorting: {len(self.probe_configs)}")

    # ------------------------------------------------------------------
    # CHANGED: find ALL continuous files across every recording* subdir
    # ------------------------------------------------------------------
    def _find_all_continuous_files(self, probe_name):
        """
        Find all continuous.dat files for *probe_name*, one entry per
        recording* directory found inside self.experiment_path.

        Returns
        -------
        list of (recording_index: str, continuous_file: Path)
            e.g. [('R1', PosixPath('.../recording1/continuous/.../continuous.dat')),
                  ('R2', PosixPath('.../recording2/continuous/.../continuous.dat'))]
        """
        config      = self.probe_configs[probe_name]
        stream_name = config['stream_name']

        results = []

        for recording_dir in sorted(self.experiment_path.glob("recording*")):
            # derive recording index: recording1 -> 'R1'
            rec_name = recording_dir.name
            rec_num  = ''.join(filter(str.isdigit, rec_name))
            recording_index = f"R{rec_num}" if rec_num else "R1"

            continuous_base = recording_dir / "continuous"

            # Glob for any Neuropix-PXI-<number>.<stream_name> variant (e.g. -100, -101, ...)
            glob_matches = list(continuous_base.glob(f"Neuropix-PXI-*.{stream_name}"))

            possible_dirs = glob_matches + [
                continuous_base / f"Neuropix-PXI.{stream_name}",  # legacy no-suffix variant
                continuous_base / stream_name,  # bare stream name fallback
            ]

            for probe_dir in possible_dirs:
                continuous_file = probe_dir / "continuous.dat"
                if continuous_file.exists():
                    print(f"    Found continuous data for {probe_name} "
                          f"[{recording_index}]: {continuous_file}")
                    results.append((recording_index, continuous_file))
                    break   # move on to next recording dir

        if not results:
            raise FileNotFoundError(
                f"No continuous.dat found for {probe_name} in any recording* directory"
            )

        return results

    # kept for backward-compatibility (used by _get_probe_settings)
    def _find_continuous_file(self, probe_name):
        """Return the first continuous.dat found (legacy helper)."""
        return self._find_all_continuous_files(probe_name)[0][1]

    # ------------------------------------------------------------------
    # CHANGED: accept continuous_file as an explicit argument
    # ------------------------------------------------------------------
    def _get_probe_settings(self, probe_name, continuous_file):
        """
        Get Kilosort settings optimised for the specific probe type.

        Parameters
        ----------
        probe_name      : str   e.g. 'ProbeA'
        continuous_file : Path  path to continuous.dat for one recording
        """
        config = self.probe_configs[probe_name]

        file_size        = continuous_file.stat().st_size
        bytes_per_sample = 2   # int16

        print(f"    Auto-detecting channels for {continuous_file.name}...")
        print(f"    File size: {file_size:,} bytes")

        total_data_points = file_size // bytes_per_sample
        valid_channel_counts = []

        for n_chan in range(300, 385):
            if total_data_points % n_chan == 0:
                samples_per_channel = total_data_points // n_chan
                duration_sec        = samples_per_channel / config['sample_rate']
                if duration_sec > 10:
                    valid_channel_counts.append(n_chan)
                    print(f"{n_chan} channels  "
                          f"{samples_per_channel:,} samples  "
                          f"{duration_sec:.1f} s")

        n_chan_bin = self._select_best_channel_count(valid_channel_counts)

        samples_per_channel = total_data_points // n_chan_bin
        duration_sec        = samples_per_channel / config['sample_rate']
        print(f"Final selection: {n_chan_bin} channels")
        print(f"Recording duration: {duration_sec:.1f} seconds")

        settings = {
            'data_dir':   str(continuous_file.parent),
            'n_chan_bin': n_chan_bin,
            'nblocks':    5,
        }

        return settings, config['probe_map']

    def _select_best_channel_count(self, valid_counts):
        if not valid_counts:
            print("    Warning: No valid channel count found. Using 385 as default.")
            return 385
        if len(valid_counts) == 1:
            return valid_counts[0]

        print(f"    Multiple valid options: {valid_counts}")
        common_configs = [385, 384, 383]
        for preferred in common_configs:
            if preferred in valid_counts:
                print(f"    Selected {preferred} (common Neuropixels configuration)")
                return preferred

        closest = min(valid_counts, key=lambda x: abs(x - 385))
        print(f"    Selected {closest} (closest to standard 385)")
        return closest

    # ------------------------------------------------------------------
    # CHANGED: loop over all recordings; build full probe_name_dir
    # ------------------------------------------------------------------
    def process_probe(self, probe_name):
        """
        Process a single probe across all its recordings with Kilosort.

        The output directory for each recording is named:
            {session_name}_{session_time}_{experiment_index}_{recording_index}_{probe_name}
        e.g.  mc015_260314_000759_E1_R1_ProbeB

        Returns
        -------
        list of Path
            Result directories (one per recording) for this probe.
        """
        print(f"\n{'=' * 50}")
        print(f"Processing {probe_name}")
        print(f"{'=' * 50}")

        all_continuous = self._find_all_continuous_files(probe_name)
        result_dirs    = []

        for recording_index, continuous_file in all_continuous:
            # ---- build the new naming  -----------------------------------
            # e.g. mc015_260314_000759_E1_R1_ProbeB
            probe_name_dir = (
                f"{self.session_name}"
                f"_{self.session_time}"
                f"_{self.experiment_index}"
                f"_{recording_index}"
                f"_{probe_name}"
            )

            probe_results_dir = self.results_dir / probe_name_dir
            probe_results_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Recording: {recording_index}")
            print(f"  Probe type : {self.probe_configs[probe_name]['probe_type']}")
            print(f"  Data file  : {continuous_file}")
            print(f"  Results dir: {probe_results_dir}")

            if (probe_results_dir / 'spike_times.npy').exists():
                print(f"  Kilosort results already exist – skipping.")
                result_dirs.append(probe_results_dir)
                continue

            # ---- run Kilosort  ------------------------------------------
            settings, probe_map = self._get_probe_settings(
                probe_name, continuous_file
            )

            print(f"  Running Kilosort 4 …")
            try:
                ops, st, clu, tF, Wall, similar_templates, \
                    is_ref, est_contam_rate, kept_spikes = run_kilosort(
                        settings=settings,
                        filename=str(continuous_file),
                        probe_name=probe_map,
                        results_dir=str(probe_results_dir),
                    )
                print(f"  Kilosort completed successfully.")
            except Exception as e:
                print(f"  Error running Kilosort on {probe_name} [{recording_index}]: {e}")
                raise

            result_dirs.append(probe_results_dir)

        return result_dirs   # list (one entry per recording)

    # ------------------------------------------------------------------
    # CHANGED: accept probe_results_dir directly instead of rebuilding it
    # ------------------------------------------------------------------
    def analyze_probe_results(self, probe_results_dir, n_units_to_plot=5):
        """
        Generate analysis plots for a processed probe/recording directory.

        Parameters
        ----------
        probe_results_dir : Path
            Full path to the probe results directory
            (e.g. mc_proc/…/mc015_260314_000759_E1_R1_ProbeB)
        """
        probe_results_dir = Path(probe_results_dir)

        if not (probe_results_dir / 'spike_times.npy').exists():
            print(f"  No Kilosort results found in {probe_results_dir}")
            return

        print(f"  Analyzing results in {probe_results_dir.name} …")

        ops            = np.load(probe_results_dir / 'ops.npy',
                                 allow_pickle=True).item()
        contam_pct     = pd.read_csv(probe_results_dir / 'cluster_ContamPct.tsv',
                                     sep='\t')['ContamPct'].values
        templates      = np.load(probe_results_dir / 'templates.npy')
        chan_map        = np.load(probe_results_dir / 'channel_map.npy')
        spike_times    = np.load(probe_results_dir / 'spike_times.npy')
        spike_clusters = np.load(probe_results_dir / 'spike_clusters.npy')

        fs              = ops.get('fs', 30000)
        chan_best        = (templates ** 2).sum(axis=1).argmax(axis=-1)
        chan_best        = chan_map[chan_best]

        good_units = np.nonzero(contam_pct <= 10)[0]
        print(f"    {len(good_units)} good units out of {len(contam_pct)} total")

        # unit_quality_analysis sits 4 levels up from the probe dir
        probe_plot_results_dir = probe_results_dir.parent.parent.parent.parent
        probe_plot_results_dir = probe_plot_results_dir / 'unit_quality_analysis'
        probe_plot_results_dir.mkdir(exist_ok=True)

        self._create_summary_plot(
            probe_results_dir.name,
            ops, spike_times, spike_clusters,
            chan_best, chan_map, contam_pct,
            probe_results_dir, probe_plot_results_dir,
        )

    def _create_summary_plot(self, probe_label, ops, spike_times, spike_clusters,
                             chan_best, chan_map, contam_pct,
                             probe_results_dir, probe_plot_results_dir):
        """Create a summary plot showing drift and spike distribution."""

        fig  = plt.figure(figsize=(15, 12))
        grid = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.5)

        try:
            camps = pd.read_csv(probe_results_dir / 'cluster_Amplitude.tsv',
                                sep='\t')['Amplitude'].values
        except FileNotFoundError:
            print(f"    Warning: cluster_Amplitude.tsv not found")
            camps = np.array([])

        # -- drift -------------------------------------------------------
        ax1 = fig.add_subplot(grid[0, 0])
        if 'dshift' in ops:
            dshift      = ops['dshift']
            time_points = np.arange(len(dshift)) * 2
            ax1.plot(time_points, dshift)
            ax1.set_xlabel('Time (sec)')
            ax1.set_ylabel('Drift (µm)')
            ax1.set_title('Estimated Drift')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

        # -- contamination histogram -------------------------------------
        ax3 = fig.add_subplot(grid[0, 1])
        contam_pct_clipped = contam_pct[contam_pct < 200]
        new_bins = np.arange(0, 200, 2)
        ax3.hist(contam_pct_clipped, bins=new_bins, edgecolor='black', color='gray')
        ax3.axvline(x=10, color='red', linestyle='--', label='10% threshold')
        ax3.set_xlabel('Contamination %')
        ax3.set_ylabel('Number of units')
        ax3.set_title('Unit Contamination Distribution')
        ax3.legend()
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # -- firing rates ------------------------------------------------
        ax4 = fig.add_subplot(grid[1, 0])
        unique_clusters, cluster_counts = np.unique(spike_clusters, return_counts=True)
        firing_rates = cluster_counts * ops['fs'] / spike_times.max()
        ax4.hist(firing_rates, bins=30, color='gray', edgecolor='black')
        ax4.set_xlabel('Firing Rate (Hz)')
        ax4.set_ylabel('Number of units')
        ax4.set_title('Firing Rate Distribution')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        # -- amplitude distribution --------------------------------------
        ax5 = fig.add_subplot(grid[1, 1])
        if len(camps) > 0:
            ax5.hist(camps, bins=30, color='gray', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Amplitude (µV)')
            ax5.set_ylabel('Number of Units')
            ax5.set_title('Template Amplitude Distribution')
            median_amp = np.median(camps)
            ax5.axvline(median_amp, color='red', linestyle='--',
                        label=f'Median: {median_amp:.1f} µV')
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'Amplitude data\nnot available',
                     ha='center', va='center',
                     transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Template Amplitude Distribution')

        plt.suptitle(f'{probe_label} – Summary Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(probe_plot_results_dir / f'{probe_label}_summary.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # CHANGED: handle list-of-dirs returned by process_probe
    # ------------------------------------------------------------------
    def process_all_probes(self, analyze_results=True):
        """
        Process all probes found in the experiment.

        Returns
        -------
        dict  {probe_name: [result_dir, …]}
        """
        print(f"Starting processing of {len(self.probe_configs)} probes …")

        results = {}
        for probe_name in self.probe_configs.keys():
            try:
                result_dirs = self.process_probe(probe_name)
                results[probe_name] = result_dirs

                if analyze_results:
                    for rd in result_dirs:
                        self.analyze_probe_results(rd)

            except Exception as e:
                print("Failed to process probe")
                print(f"  Failed to process {probe_name}: {e}")
                results[probe_name] = []

        self._generate_experiment_summary(results)
        return results

    def _generate_experiment_summary(self, results):
        """Generate a summary report for the entire experiment."""
        summary_file = self.results_dir / "experiment_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("NEUROPIXEL EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment path  : {self.experiment_path}\n")
            f.write(f"Session time     : {self.session_time}\n")
            f.write(f"Experiment index : {self.experiment_index}\n")
            f.write(f"Total probes     : {len(self.probe_configs)}\n\n")

            for probe_name, config in self.probe_configs.items():
                f.write(f"{probe_name}:\n")
                f.write(f"  Type       : {config['probe_type']}\n")
                f.write(f"  Sample rate: {config['sample_rate']} Hz\n")

                probe_dirs = results.get(probe_name, [])
                f.write(f"  Recordings : {len(probe_dirs)}\n")

                for probe_results_dir in probe_dirs:
                    try:
                        cp = pd.read_csv(probe_results_dir / 'cluster_ContamPct.tsv',
                                         sep='\t')['ContamPct'].values
                        good  = int(np.sum(cp <= 10))
                        total = len(cp)
                        f.write(f"    {probe_results_dir.name}: "
                                f"{good}/{total} good units\n")
                    except Exception:
                        f.write(f"    {probe_results_dir.name}: error reading results\n")
                f.write("\n")

            f.write("\nPHY VISUALISATION INSTRUCTIONS:\n")
            f.write("-" * 30 + "\n")
            for probe_name, probe_dirs in results.items():
                for probe_results_dir in probe_dirs:
                    f.write(f"  cd {probe_results_dir}\n")
                    f.write(f"  phy template-gui params.py\n\n")

        print(f"Experiment summary saved to: {summary_file}")


# ======================================================================
class NeuropixelSessionManager:
    """
    Manages multiple Neuropixel recording sessions and automatically
    processes unprocessed ones.
    """

    def __init__(self, base_path):
        self.base_path = Path(base_path)

        base_path_str = str(self.base_path)
        if '_raw' in base_path_str:
            self.proc_base_path = Path(base_path_str.replace('_raw', '_proc'))
        else:
            self.proc_base_path = self.base_path.parent / (self.base_path.name + '_proc')

        print(f"Raw data path      : {self.base_path}")
        print(f"Processed data path: {self.proc_base_path}")

        self.sessions           = {}
        self.processed_sessions = set()
        self.failed_sessions    = set()

        self._discover_sessions()

    def _discover_sessions(self):
        print("Discovering OpenEphys sessions …")

        session_patterns = ["*/*/*/Record Node */experiment*"]
        found_sessions   = []
        for pattern in session_patterns:
            found_sessions.extend(self.base_path.glob(pattern))

        for session_path in found_sessions:
            settings_file = session_path.parent / "settings.xml"
            if settings_file.exists():
                session_id, session_name = self._get_session_id(session_path)
                self.sessions[session_id] = {
                    'path':     session_path,
                    'raw_path': session_path,
                    'name':     session_name,
                    'status':   'unprocessed',
                    'results':  None,
                    'error':    None,
                }

                # relative_path     = session_path.relative_to(self.base_path)
                # proc_session_path = self.proc_base_path / relative_path
                session_name_dir = session_path.parent.parent.parent  # climbs: experiment1 → Record Node → timestamp → session_name
                relative_path = session_name_dir.relative_to(self.base_path)
                proc_session_path = self.proc_base_path / relative_path
                kilosort_results  = proc_session_path

                if kilosort_results.exists() and \
                        self._check_results_complete(kilosort_results, session_path):
                    self.sessions[session_id]['status']    = 'processed'
                    self.sessions[session_id]['results']   = kilosort_results
                    self.sessions[session_id]['proc_path'] = proc_session_path
                    self.processed_sessions.add(session_id)

        print(f"Found {len(self.sessions)} sessions")
        print(f"  - Already processed : {len(self.processed_sessions)}")
        print(f"  - Unprocessed       : "
              f"{len(self.sessions) - len(self.processed_sessions)}")

    def _get_session_id(self, session_path):
        path_parts = session_path.parts
        try:
            record_node_idx = next(
                i for i, p in enumerate(path_parts) if "Record Node" in p
            )
            if record_node_idx >= 3:
                timestamp = path_parts[record_node_idx - 1]
                session_name = path_parts[record_node_idx - 2]
                mouse_id = path_parts[record_node_idx - 3]
                experiment_name = session_path.name  # ← NEW: e.g. 'experiment1'
                session_id = f"{mouse_id}_{session_name}_{timestamp}_{experiment_name}"
                return session_id, session_name
        except StopIteration:
            pass

        for part in reversed(path_parts):
            if any(c.isdigit() for c in part) and "2025" in part:
                return part, part

        return path_parts[-1], path_parts[-1]

    def _check_results_complete(self, results_dir, raw_session_path):
        """
        Return True only when EVERY raw probe in this session has a complete
        proc folder.

        Parameters
        ----------
        results_dir      : Path  – proc directory for this mouse/session
                                   (e.g. mc_proc/mc015/)
        raw_session_path : Path  – raw experiment dir
                                   (e.g. mc_raw/.../experiment1/)
        """
        required_files = [
            'spike_times.npy',
            'spike_clusters.npy',
            'cluster_ContamPct.tsv',
            'params.py',
        ]

        # ── 1. Collect every raw probe dir across all recordings ──────────────
        raw_probes = []  # list of (recording_dir, probe_dir)
        for recording_dir in sorted(raw_session_path.glob("recording*")):
            continuous_dir = recording_dir / "continuous"
            if not continuous_dir.exists():
                continue
            for probe_dir in sorted(continuous_dir.iterdir()):
                if probe_dir.is_dir() \
                        and "Neuropix-PXI" in probe_dir.name \
                        and "LFP" not in probe_dir.name:  # <── skip LFP bands
                    raw_probes.append((recording_dir, probe_dir))

        if not raw_probes:
            # No probes found at all — treat as incomplete so it gets processed
            return False

        # ── 2. For every raw probe, check its proc counterpart ────────────────
        for recording_dir, probe_dir in raw_probes:
            expected_name = self._derive_proc_folder_name(
                raw_session_path, recording_dir, probe_dir
            )
            proc_probe_dir = results_dir / expected_name

            if not proc_probe_dir.exists():
                print(f"  [incomplete] Missing proc folder: {expected_name}")
                return False

            missing = [f for f in required_files
                       if not (proc_probe_dir / f).exists()]
            if missing:
                print(f"  [incomplete] {expected_name} missing: {missing}")
                return False

        return True  # every probe is accounted for ✓

    def _derive_proc_folder_name(self, session_path, recording_dir, probe_dir):
        """
        Build the proc folder name from raw path components.

        Raw  : .../mc015/2026-03-15_21-13-12/Record Node 101/experiment1/
                    recording1/continuous/Neuropix-PXI-100.ProbeB
        Proc : mc015_260315_211312_E1_R1_ProbeB
        """
        parts = session_path.parts

        # Locate "Record Node …" to anchor all the relative positions
        try:
            rn_idx = next(i for i, p in enumerate(parts) if "Record Node" in p)
        except StopIteration:
            raise ValueError(f"Cannot find 'Record Node' in path: {session_path}")
        print(f'rn_idx: {rn_idx}')
        mouse_id = parts[rn_idx - 3]  # e.g. "mc015"
        print(f'mouse_id: {mouse_id}')
        timestamp1 = parts[rn_idx - 2]  # e.g. "mc015_260315"
        print(f'timestamp: {timestamp1}')
        timestamp2 = parts[rn_idx - 1]  # e.g. "2026-03-15_21-13-12"
        print(f'timestamp: {timestamp2}')
        experiment_name = parts[rn_idx + 1]  # e.g. "experiment1"
        print(f'experiment_name: {experiment_name}')

        # ── Timestamp → YYMMDD_HHMMSS ─────────────────────────────────────────
        mouse_id, date_raw = timestamp1.split("_")
        rest, time_raw = timestamp2.split("_")  # "2026-03-15", "21-13-12"
        #year, month, day = date_raw.split("-")
        #date_str = year[2:] + month + day
        date_str =date_raw  # "260315"
        time_str = time_raw.replace("-", "")  # "211312"

        # ── experiment1 → E1, recording1 → R1 ────────────────────────────────
        exp_str = "E" + experiment_name.replace("experiment", "")
        rec_str = "R" + recording_dir.name.replace("recording", "")

        # ── Neuropix-PXI-100.ProbeB[-AP] → ProbeB ────────────────────────────
        # The proc folder strips the "-AP" suffix if present
        probe_raw = probe_dir.name.split(".")[-1]  # "ProbeB" or "ProbeB-AP"
        probe_name = probe_raw.replace("-AP", "")  # always "ProbeB"

        return f"{mouse_id}_{date_str}_{time_str}_{exp_str}_{rec_str}_{probe_name}"

    def _prepare_proc_directory(self, session_id):
        info                       = self.sessions[session_id]
        raw_session_path           = info['raw_path']
        raw_session_path_rel_part  = raw_session_path.parent.parent.parent
        relative_path              = raw_session_path_rel_part.relative_to(self.base_path)
        proc_session_path          = self.proc_base_path / relative_path
        proc_session_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created processed directory: {proc_session_path}")

        search_dirs = [
            raw_session_path.parent.parent.parent.parent,
            raw_session_path.parent.parent.parent,
            raw_session_path.parent.parent,
            raw_session_path.parent,
        ]

        tsv_files_copied = []
        for search_dir in search_dirs:
            if search_dir.exists():
                for tsv_file in search_dir.glob("*.tsv"):
                    dest_file = proc_session_path / tsv_file.name
                    if not dest_file.exists():
                        import shutil
                        shutil.copy2(tsv_file, dest_file)
                        tsv_files_copied.append(tsv_file.name)
                        print(f"  Copied TSV: {tsv_file.name}")
                    else:
                        tsv_files_copied.append(tsv_file.name)

        if not tsv_files_copied:
            print(f"  Warning: no TSV files found for {session_id}")
        else:
            print(f"  Copied {len(tsv_files_copied)} TSV file(s)")

        return proc_session_path

    def list_sessions(self, status_filter=None):
        print(f"\nSession Summary (filter: {status_filter or 'all'})")
        print("-" * 60)
        for session_id, info in self.sessions.items():
            if status_filter is None or info['status'] == status_filter:
                print(f"{session_id}")
                print(f"  Status : {info['status']}")
                print(f"  Path   : {info['path']}")
                if info['results']:
                    print(f"  Results: {info['results']}")
                if info['error']:
                    print(f"  Error  : {info['error']}")
                print()

    def process_unprocessed_sessions(self, analyze_results=True, max_sessions=None):
        unprocessed = [
            (sid, info) for sid, info in self.sessions.items()
            if info['status'] == 'unprocessed'
        ]
        if max_sessions:
            unprocessed = unprocessed[:max_sessions]

        print(f"\nProcessing {len(unprocessed)} unprocessed sessions …")

        for i, (session_id, info) in enumerate(unprocessed, 1):
            print(f"\n{'=' * 60}")
            print(f"Session {i}/{len(unprocessed)}: {session_id}")
            print(f"{'=' * 60}")
            try:
                proc_session_path = self._prepare_proc_directory(session_id)
                processor = OpenEphysNeuropixelProcessor(
                    info['raw_path'],
                    info['name'],
                    results_dir=proc_session_path,
                )
                results = processor.process_all_probes(analyze_results=analyze_results)
                self.sessions[session_id]['status']    = 'processed'
                self.sessions[session_id]['results']   = processor.results_dir
                self.sessions[session_id]['proc_path'] = proc_session_path
                self.processed_sessions.add(session_id)
                print(f"✓ Successfully processed {session_id}")
            except Exception as e:
                print(f"✗ Failed to process {session_id}: {e}")
                self.sessions[session_id]['status'] = 'failed'
                self.sessions[session_id]['error']  = str(e)
                self.failed_sessions.add(session_id)

        self._print_processing_summary()

    def _print_processing_summary(self):
        total     = len(self.sessions)
        processed = len(self.processed_sessions)
        failed    = len(self.failed_sessions)
        remaining = total - processed - failed

        print(f"\n{'=' * 60}")
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total sessions        : {total}")
        print(f"Successfully processed: {processed}")
        print(f"Failed                : {failed}")
        print(f"Remaining unprocessed : {remaining}")

        if failed:
            print("\nFailed sessions:")
            for sid in self.failed_sessions:
                print(f"  - {sid}: {self.sessions[sid]['error']}")

        if processed:
            print("\nTo visualize results with Phy:")
            for sid in self.processed_sessions:
                results_dir = self.sessions[sid]['results']
                if results_dir:
                    probe_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
                    print(f"\n{sid}:")
                    for pd_ in probe_dirs:
                        print(f"  cd {pd_}")
                        print(f"  phy template-gui params.py")

    def process_specific_session(self, session_id, analyze_results=True):
        if session_id not in self.sessions:
            print(f"Session '{session_id}' not found")
            return False

        info = self.sessions[session_id]
        print(f"Processing session: {session_id}")
        try:
            proc_session_path = self._prepare_proc_directory(session_id)
            processor = OpenEphysNeuropixelProcessor(
                info['raw_path'],
                info['name'],
                results_dir=proc_session_path,
            )
            processor.process_all_probes(analyze_results=analyze_results)
            self.sessions[session_id]['status']    = 'processed'
            self.sessions[session_id]['results']   = processor.results_dir
            self.sessions[session_id]['proc_path'] = proc_session_path
            self.processed_sessions.add(session_id)
            print(f"✓ Successfully processed {session_id}")
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            self.sessions[session_id]['status'] = 'failed'
            self.sessions[session_id]['error']  = str(e)
            self.failed_sessions.add(session_id)
            return False


# ======================================================================
def calculate_firing_rates_with_gaussian(
        spike_times, bin_size_ms=1, sigma=1, t_start=None, t_end=None):
    if t_start is None:
        t_start = 0 if len(spike_times) == 0 else np.min(spike_times)
    if t_end is None:
        t_end = 1 if len(spike_times) == 0 else np.max(spike_times) + 0.1

    bin_size_sec = bin_size_ms / 1000
    bins         = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
    time_points  = bins[:-1] + bin_size_sec / 2

    hist, _        = np.histogram(spike_times, bins=bins)
    firing_rates   = hist / bin_size_sec
    smoothed_rates = gaussian_filter1d(firing_rates, sigma)

    return time_points, smoothed_rates


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Kilosort4 Processing Pipeline for Neuropixels Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_neuropixel_data.py --base-path /data/.../mc_raw
  python process_neuropixel_data.py --base-path /data/.../sc_raw
  python process_neuropixel_data.py --base-path /data/.../mc_raw --max-sessions 1
        """,
    )
    parser.add_argument('--base-path',     type=str, required=True)
    parser.add_argument('--max-sessions',  type=int, default=None)
    parser.add_argument('--no-analysis',   action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    print("=" * 80)
    print("KILOSORT4 PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Base path      : {args.base_path}")
    print(f"Max sessions   : {args.max_sessions or 'All'}")
    print(f"Gen. analysis  : {not args.no_analysis}")
    print("=" * 80)

    manager = NeuropixelSessionManager(args.base_path)
    manager.list_sessions()
    manager.process_unprocessed_sessions(
        analyze_results=not args.no_analysis,
        max_sessions=args.max_sessions,
    )
    print("\nProcessing complete!")
#!/usr/bin/env python3
"""
Oxford Dataset Cross-Trial-Type PCA Projection Analysis
=========================================================

This module implements cross-trial-type principal component analysis (PCA)
projection, where PCA weights trained on one behavioral condition (e.g.,
cued_hit_long) are used to project neural activity from other conditions
(e.g., spont_hit_long, spont_miss_long) into the same latent subspace.

Key Difference from CCA:
------------------------
Unlike CCA (which requires paired recordings to identify shared variance),
PCA operates independently on each region's neural population. Therefore:

1. Each region is aggregated across ALL sessions that recorded it, regardless
   of what other regions were co-recorded in those sessions.

2. When displaying a region pair (A, B) in the upper triangle:
   - Region A projection: mean across all sessions in S_A
   - Region B projection: mean across all sessions in S_B
   where S_A and S_B are computed independently and need not overlap.

Mathematical Framework:
-----------------------
Given PCA coefficients W ∈ R^{n × k} for a region trained on condition c_ref
(cued_hit_long), the principal component projections for any condition c are:

    z_c = X_c @ W

where X_c ∈ R^{T × n} is the trial-averaged neural activity matrix.

Cross-Session Aggregation:
--------------------------
For multi-session analysis, projections are first computed per-session,
then aggregated across sessions:

    z̄_c = (1/N_s) Σ_{s=1}^{N_s} z_{c,s}

Standard error is computed as:
    SEM = σ / √N_s

Author: Oxford Neural Analysis Pipeline
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mat73
import scipy.io as sio
from scipy.stats import zscore, wilcoxon, pearsonr
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Trial type labels and their data directories
# TRIAL_TYPES = {
#     'cued_hit_long': 'sessions_cued_hit_long_results',
#     'spont_hit_long': 'sessions_spont_hit_long_results',
#     'spont_miss_long': 'sessions_spont_miss_long_results'
# }

TRIAL_TYPES = {
    'cued_hit_long': 'sessions_cued_hit_long_results',
    'spont_hit_long': 'sessions_spont_hit_long_results',
}

# Colors for visualization (consistent across figures)
TRIAL_TYPE_COLORS = {
    'cued_hit_long': '#1E88E5',  # Blue
    'spont_hit_long': '#004D40',  # Teal/Green
    'spont_miss_long': '#D81B60'  # Pink/Red
}

# Canonical anatomical ordering for regions
ANATOMICAL_ORDER = [
    'mPFC', 'ORB', 'MOp', 'MOs', 'OLF',  # Cortical regions
    'STR', 'STRv',  # Striatal
    'MD', 'LP', 'VALVM', 'VPMPO', 'ILM',  # Thalamic nuclei
    'HY'  # Hypothalamic
]

# Minimum sessions threshold for cross-session analysis
MIN_SESSIONS_THRESHOLD = 3

# Hierarchical anatomical grouping: maps individual regions to broader categories
# Regions that map to themselves are kept separate
HIERARCHICAL_GROUPING = {
    'mPFC': 'PFC',
    'ORB': 'PFC',
    'ILM': 'PFC',
    'OLF': 'OLF',
    'MOp': 'MOp',
    'MOs': 'MOs',
    'STR': 'Striatum',
    'STRv': 'Striatum',
    'MD': 'Thalamus',
    'VALVM': 'Thalamus',
    'LP': 'Thalamus',
    'VPMPO': 'Thalamus',
    'HY': 'Hypothalamus',
}

# Ordering for hierarchical (aggregated) region display
HIERARCHICAL_ORDER = [
    'PFC','OLF', 'MOp', 'MOs',
    'Striatum', 'Thalamus', 'Hypothalamus'
]


# =============================================================================
# HELPER FUNCTIONS FOR DATA LOADING
# =============================================================================

def load_mat_file(file_path: Path) -> Dict:
    """
    Load MATLAB .mat file with automatic version detection.

    Parameters:
        file_path: Path to the .mat file

    Returns:
        Dictionary containing the loaded data
    """
    try:
        data = mat73.loadmat(str(file_path))
        return data
    except Exception:
        # Fall back to scipy.io for older MATLAB versions
        return sio.loadmat(str(file_path), squeeze_me=True, struct_as_record=False)


def get_anatomical_index(region: str) -> int:
    """Get anatomical ordering index for a region."""
    try:
        return ANATOMICAL_ORDER.index(region)
    except ValueError:
        return len(ANATOMICAL_ORDER)


def sort_pair_by_anatomy(region_i: str, region_j: str) -> Tuple[str, str]:
    """Sort region pair by anatomical order."""
    idx_i = get_anatomical_index(region_i)
    idx_j = get_anatomical_index(region_j)
    if idx_i <= idx_j:
        return (region_i, region_j)
    else:
        return (region_j, region_i)


def get_hierarchical_region(region: str) -> str:
    """Map an individual region to its hierarchical group name."""
    return HIERARCHICAL_GROUPING.get(region, region)


def get_hierarchical_index(region: str) -> int:
    """Get hierarchical ordering index for a region."""
    try:
        return HIERARCHICAL_ORDER.index(region)
    except ValueError:
        return len(HIERARCHICAL_ORDER)


# =============================================================================
# MAIN CLASS: CrossTrialTypePCAAnalyzer
# =============================================================================

class CrossTrialTypePCAAnalyzer:
    """
    Analyzer for cross-trial-type PCA projection for a single session.

    This class implements PCA projection where weights trained on a reference
    condition (typically cued_hit_long) are used to project neural activity
    from other conditions into the same latent subspace.

    Unlike CCA, PCA operates independently on each region. This class
    extracts PCA weights per region and computes projections for all trial types.

    Attributes:
        base_dir: Root directory containing all session results
        session_name: Identifier for the session (e.g., 'yp010_220209')
        reference_type: Trial type used for PCA weight extraction
        trial_type_data: Dictionary holding loaded data per trial type
        pca_weights: Dictionary holding PCA weights per region
        projections: Dictionary holding computed projections per region
    """

    def __init__(
            self,
            base_dir: str,
            session_name: str,
            reference_type: str = 'cued_hit_long',
            n_components: int = 5
    ):
        """
        Initialize the cross-trial-type PCA analyzer.

        Parameters:
            base_dir: Root directory (e.g., '/path/to/Oxford_dataset')
            session_name: Session identifier (e.g., 'yp010_220209')
            reference_type: Trial type for PCA weight extraction (default: cued_hit_long)
            n_components: Number of PCA components to analyze
        """
        self.base_dir = Path(base_dir)
        self.session_name = session_name
        self.reference_type = reference_type
        self.n_components = n_components

        # Data containers
        self.trial_type_data: Dict[str, Dict] = {}
        self.pca_weights: Dict[str, Dict] = {}  # Per-region PCA weights
        self.neural_data: Dict[str, Dict[str, np.ndarray]] = {}  # Per trial type, per region
        self.projections: Dict[str, Dict[str, Dict]] = {}  # Per trial type, per region
        self.statistical_results: Dict[str, Dict] = {}  # Per region

        # Available trial types for this session
        self.available_trial_types: List[str] = []

        # Region information
        self.available_regions: List[str] = []

        # Time axis
        self.time_bins: Optional[np.ndarray] = None

        print("=" * 70)
        print("Cross-Trial-Type PCA Analyzer")
        print("=" * 70)
        print(f"Session: {session_name}")
        print(f"Reference condition: {reference_type}")
        print(f"Components to analyze: {n_components}")

    # =========================================================================
    # DATA LOADING METHODS
    # =========================================================================

    def load_all_trial_types(self) -> bool:
        """
        Load data from all available trial types for this session.

        Returns:
            True if at least reference type and one other type loaded successfully
        """
        print("\n" + "-" * 50)
        print("Loading data from all trial types...")
        print("-" * 50)

        for trial_type, subdir in TRIAL_TYPES.items():
            results_dir = self.base_dir / subdir
            results_file = results_dir / f"{self.session_name}_analysis_results.mat"

            if not results_file.exists():
                print(f"  {trial_type}: NOT FOUND")
                continue

            try:
                data = load_mat_file(results_file)
                self.trial_type_data[trial_type] = data
                self.available_trial_types.append(trial_type)
                print(f"  {trial_type}: LOADED")

                # Extract time bins from first loaded file
                if self.time_bins is None:
                    self._extract_time_bins(data)

            except Exception as e:
                print(f"  {trial_type}: ERROR - {e}")

        # Verify reference type is available
        if self.reference_type not in self.available_trial_types:
            print(f"\nERROR: Reference type '{self.reference_type}' not available")
            return False

        # Need at least reference + one other type
        if len(self.available_trial_types) < 2:
            print("\nERROR: Need at least 2 trial types for comparison")
            return False

        print(f"\nAvailable trial types: {', '.join(self.available_trial_types)}")

        # Extract region information from reference type
        self._extract_region_info()

        return True

    def _extract_time_bins(self, data: Dict) -> None:
        """Extract time axis from loaded data structure."""
        if 'region_data' in data:
            region_data = data['region_data']
            if 'timepoints' in region_data:
                n_timepoints = int(region_data['timepoints'].flatten()[0])
                self.time_bins = np.linspace(-1.5, 3.0, n_timepoints)

        if self.time_bins is None:
            # Default time axis: -1.5s to 3.0s at ~20ms bins
            self.time_bins = np.linspace(-1.5, 3.0, 226)

        print(f"  Time axis: {len(self.time_bins)} bins, {self.time_bins[0]:.2f}s to {self.time_bins[-1]:.2f}s")

    def _extract_region_info(self) -> None:
        """Extract available regions from reference type PCA results."""
        ref_data = self.trial_type_data[self.reference_type]

        # Get regions from PCA results
        if 'pca_results' in ref_data:
            pca_results = ref_data['pca_results']
            # PCA results are organized as: pca_results.REGION_NAME
            for key in pca_results.keys():
                if isinstance(pca_results[key], dict) and 'coefficients' in pca_results[key]:
                    self.available_regions.append(key)

        print(f"Available regions with PCA results: {', '.join(self.available_regions)}")

    # =========================================================================
    # NEURAL DATA EXTRACTION
    # =========================================================================

    def extract_neural_data_for_region(self, region: str) -> bool:
        """
        Extract neural activity matrices for a single region across all trial types.

        Parameters:
            region: Region name (e.g., 'mPFC')

        Returns:
            True if extraction successful for all available trial types
        """
        print(f"\n" + "-" * 50)
        print(f"Extracting neural data for: {region}")
        print("-" * 50)

        for trial_type in self.available_trial_types:
            data = self.trial_type_data[trial_type]

            if 'region_data' not in data or 'regions' not in data['region_data']:
                print(f"  {trial_type}: No region_data found")
                continue

            regions_data = data['region_data']['regions']

            if region not in regions_data:
                print(f"  {trial_type}: Region {region} not found")
                continue

            region_info = regions_data[region]

            if 'spike_data' not in region_info:
                print(f"  {trial_type}: No spike_data for {region}")
                continue

            # Extract spike data and apply neuron selection
            spike_data = region_info['spike_data']

            if 'selected_neurons' in region_info:
                selected_neurons = region_info['selected_neurons'].reshape(-1).astype(int) - 1
                spike_data = spike_data[:, selected_neurons, :]

            if trial_type not in self.neural_data:
                self.neural_data[trial_type] = {}

            self.neural_data[trial_type][region] = spike_data
            n_trials, n_neurons, n_time = spike_data.shape
            print(f"  {trial_type} - {region}: {n_trials} trials × {n_neurons} neurons × {n_time} time")

        # Check if we have data for at least 2 trial types
        count = sum(1 for tt in self.neural_data if region in self.neural_data.get(tt, {}))
        return count >= 2

    # =========================================================================
    # PCA WEIGHT EXTRACTION
    # =========================================================================

    def extract_pca_weights(self, region: str) -> bool:
        """
        Extract PCA weights (coefficients) from reference condition for a region.

        PCA coefficients transform neural activity into principal components:
        z = X @ W

        Parameters:
            region: Region name

        Returns:
            True if PCA weights successfully extracted
        """
        print(f"\n" + "-" * 50)
        print(f"Extracting PCA weights from {self.reference_type} for {region}")
        print("-" * 50)

        ref_data = self.trial_type_data[self.reference_type]

        if 'pca_results' not in ref_data:
            print("ERROR: No PCA results in reference data")
            return False

        pca_results = ref_data['pca_results']

        if region not in pca_results:
            print(f"ERROR: Region {region} not found in PCA results")
            return False

        region_pca = pca_results[region]

        if 'coefficients' not in region_pca:
            print("ERROR: PCA coefficients not found")
            return False

        coefficients = np.array(region_pca['coefficients'])

        # Store weights
        self.pca_weights[region] = {
            'coefficients': coefficients,
            'n_components': coefficients.shape[1] if coefficients.ndim > 1 else 1,
            'n_neurons': coefficients.shape[0]
        }

        # Also extract explained variance if available
        if 'explained_variance' in region_pca:
            self.pca_weights[region]['explained_variance'] = np.array(
                region_pca['explained_variance']
            ).flatten()

        print(f"  {region}: Coefficients shape = {coefficients.shape}")
        print(f"  Available components: {self.pca_weights[region]['n_components']}")

        return True

    # =========================================================================
    # PROJECTION COMPUTATION
    # =========================================================================

    def compute_projections_for_region(self, region: str) -> bool:
        """
        Project neural activity from all trial types onto PCA subspace for a region.

        For each trial type c, computes principal component scores:
            z_c = X_c @ W

        where X_c is the trial-averaged (or per-trial) neural activity.

        Parameters:
            region: Region name

        Returns:
            True if projections computed successfully
        """
        print(f"\n" + "-" * 50)
        print(f"Computing PCA projections for {region}")
        print("-" * 50)

        if region not in self.pca_weights:
            print(f"ERROR: PCA weights not extracted for {region}")
            return False

        W = self.pca_weights[region]['coefficients']
        n_comp_use = min(self.n_components, W.shape[1])

        for trial_type in self.available_trial_types:
            # First try to get pre-computed projections from reference
            if trial_type == self.reference_type:
                ref_data = self.trial_type_data[self.reference_type]
                if 'pca_results' in ref_data and region in ref_data['pca_results']:
                    region_pca = ref_data['pca_results'][region]
                    if 'projections' in region_pca:
                        proj_data = region_pca['projections']
                        if 'mean' in proj_data:
                            # Extract pre-computed mean projections
                            means = proj_data['mean']
                            n_timepoints = len(self.time_bins)

                            z_mean = np.zeros((n_timepoints, n_comp_use))
                            z_std = np.zeros((n_timepoints, n_comp_use))

                            for comp_idx in range(min(len(means), n_comp_use)):
                                mean_data = means[comp_idx]
                                if isinstance(mean_data, np.ndarray):
                                    z_mean[:, comp_idx] = mean_data.flatten()[:n_timepoints]
                                elif isinstance(mean_data, (list, tuple)):
                                    z_mean[:, comp_idx] = np.array(mean_data).flatten()[:n_timepoints]

                                # Get std if available
                                if 'std' in proj_data and comp_idx < len(proj_data['std']):
                                    std_data = proj_data['std'][comp_idx]
                                    if isinstance(std_data, np.ndarray):
                                        z_std[:, comp_idx] = std_data.flatten()[:n_timepoints]
                                    elif isinstance(std_data, (list, tuple)):
                                        z_std[:, comp_idx] = np.array(std_data).flatten()[:n_timepoints]

                            if trial_type not in self.projections:
                                self.projections[trial_type] = {}

                            self.projections[trial_type][region] = {
                                'z_mean': z_mean,
                                'z_std': z_std,
                                'z_sem': None,
                                'z_trials': None,
                                'n_trials': None
                            }

                            print(f"  {trial_type}: Using pre-computed projections")
                            continue

            # Compute projections from neural data
            if trial_type not in self.neural_data or region not in self.neural_data[trial_type]:
                print(f"  {trial_type}: No neural data available for {region}")
                continue

            spike_data = self.neural_data[trial_type][region]
            n_trials, n_neurons, n_timepoints = spike_data.shape

            # Reshape: (n_trials, n_neurons, n_time) -> (n_trials*n_time, n_neurons)
            spike_reshaped = np.transpose(spike_data, (1, 2, 0))  # (n_neurons, n_time, n_trials)
            X = spike_reshaped.reshape(n_neurons, n_trials * n_timepoints).T  # (n_trials*n_time, n_neurons)

            # Z-score normalize
            X_norm = zscore(X, axis=0, nan_policy='omit')
            X_norm = np.nan_to_num(X_norm, nan=0.0)

            # Project onto PCA subspace
            z = X_norm @ W[:, :n_comp_use]  # (n_trials*n_time, n_components)

            # Reshape back: (n_trials*n_time, n_comp) -> (n_time, n_trials, n_comp) -> mean over trials
            z_reshaped = z.reshape(n_timepoints, n_trials, n_comp_use)

            z_mean = np.mean(z_reshaped, axis=1)  # (n_time, n_comp)
            z_std = np.std(z_reshaped, axis=1)
            z_sem = z_std / np.sqrt(n_trials)

            # Store per-trial projections
            z_trials = np.transpose(z_reshaped, (1, 0, 2))  # (n_trials, n_time, n_comp)

            if trial_type not in self.projections:
                self.projections[trial_type] = {}

            self.projections[trial_type][region] = {
                'z_mean': z_mean,
                'z_std': z_std,
                'z_sem': z_sem,
                'z_trials': z_trials,
                'n_trials': n_trials
            }

            print(f"  {trial_type}: {n_trials} trials projected")

        # Check if we have projections for at least 2 trial types
        count = sum(1 for tt in self.projections if region in self.projections.get(tt, {}))
        return count >= 2

    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================

    def compute_statistics_for_region(self, region: str) -> Dict:
        """
        Compute statistical comparisons between trial types for a region.

        Metrics computed:
        1. Peak amplitude: max(|projection|) in post-stimulus window
        2. Temporal correlation: R² between projection time courses

        Parameters:
            region: Region name

        Returns:
            Dictionary containing statistical results for this region
        """
        print(f"\n" + "-" * 50)
        print(f"Computing statistics for {region}")
        print("-" * 50)

        results = {
            'peak_amplitudes': {},
            'temporal_correlations': {}
        }

        # Define post-stimulus window (0 to 1.5s)
        post_stim_mask = (self.time_bins >= 0) & (self.time_bins <= 1.5)

        # Compute peak amplitudes for each trial type
        for trial_type in self.available_trial_types:
            if trial_type not in self.projections or region not in self.projections[trial_type]:
                continue

            proj = self.projections[trial_type][region]
            peaks = []

            n_comp_use = min(self.n_components, proj['z_mean'].shape[1])
            for comp_idx in range(n_comp_use):
                z_post = np.abs(proj['z_mean'][post_stim_mask, comp_idx])
                peaks.append(np.max(z_post))

            results['peak_amplitudes'][trial_type] = peaks

            print(f"  {trial_type} peak amplitudes: {[f'{p:.3f}' for p in peaks[:3]]}")

        # Compute temporal correlations (reference vs other types)
        if self.reference_type in self.projections and region in self.projections[self.reference_type]:
            ref_proj = self.projections[self.reference_type][region]

            for trial_type in self.available_trial_types:
                if trial_type == self.reference_type:
                    continue

                if trial_type not in self.projections or region not in self.projections[trial_type]:
                    continue

                proj = self.projections[trial_type][region]
                correlations = []

                n_comp_use = min(self.n_components, proj['z_mean'].shape[1], ref_proj['z_mean'].shape[1])
                for comp_idx in range(n_comp_use):
                    r, p = pearsonr(ref_proj['z_mean'][:, comp_idx], proj['z_mean'][:, comp_idx])
                    correlations.append({'r': r, 'r2': r ** 2, 'p': p})

                results['temporal_correlations'][f"{self.reference_type}_vs_{trial_type}"] = correlations

                print(f"\n  Temporal correlation ({self.reference_type} vs {trial_type}):")
                for idx, corr in enumerate(correlations[:3]):
                    print(f"    PC{idx+1}: R²={corr['r2']:.3f}")

        self.statistical_results[region] = results
        return results


# =============================================================================
# CROSS-SESSION PCA AGGREGATION CLASS
# =============================================================================

class CrossSessionPCAAnalyzer:
    """
    Aggregator for cross-trial-type PCA analysis across multiple sessions.

    CRITICAL DISTINCTION FROM CCA:
    -----------------------------
    For PCA, each region's data is aggregated across ALL sessions that recorded
    that region, regardless of what other regions were co-recorded. This means:

    - Region A: aggregated from sessions {s1, s2, s5, s7, ...} where A was recorded
    - Region B: aggregated from sessions {s2, s3, s5, s8, ...} where B was recorded

    The session sets for A and B may have limited or no overlap, unlike CCA
    which requires both regions to be recorded in the same session.

    Attributes:
        base_dir: Root directory containing all session results
        region: The region being analyzed
        session_projections: Dictionary mapping session_name → projection data
        aggregated_projections: Cross-session aggregated projections
    """

    def __init__(
            self,
            base_dir: str,
            region: str,
            reference_type: str = 'cued_hit_long',
            n_components: int = 5,
            min_sessions: int = MIN_SESSIONS_THRESHOLD
    ):
        """
        Initialize the cross-session analyzer for a single region.

        Parameters:
            base_dir: Root directory for Oxford dataset
            region: Region name (e.g., 'mPFC')
            reference_type: Trial type for PCA weight training
            n_components: Number of PCA components to analyze
            min_sessions: Minimum sessions required for aggregation
        """
        self.base_dir = Path(base_dir)
        self.region = region
        self.reference_type = reference_type
        self.n_components = n_components
        self.min_sessions = min_sessions

        # Session-level data containers
        self.session_projections: Dict[str, Dict] = {}  # session_name → {trial_type → projection}
        self.session_statistics: Dict[str, Dict] = {}

        # Cross-session aggregated results
        self.aggregated_projections: Dict[str, Dict] = {}  # trial_type → aggregated projection
        self.aggregated_statistics: Dict = {}

        # Time axis (from first valid session)
        self.time_bins: Optional[np.ndarray] = None

        # Available trial types (union across sessions)
        self.available_trial_types: List[str] = []

        print(f"  CrossSessionPCAAnalyzer initialized for region: {region}")

    def add_session_result(
            self,
            session_name: str,
            projections: Dict[str, Dict],
            statistics: Dict,
            time_bins: np.ndarray
    ) -> bool:
        """
        Add results from a single session analyzer for this region.

        Parameters:
            session_name: Session identifier
            projections: Dictionary mapping trial_type → projection data for this region
            statistics: Statistical results for this region
            time_bins: Time axis array

        Returns:
            True if session was successfully added
        """
        if not projections:
            return False

        self.session_projections[session_name] = projections
        self.session_statistics[session_name] = statistics

        # Update time bins from first session
        if self.time_bins is None:
            self.time_bins = time_bins

        return True

    def aggregate_projections(self) -> bool:
        """
        Aggregate projections across all sessions for this region.

        Sign alignment is performed by:
        - For reference trial (cued_hit_long): Identifying baseline latent from first session with positive peak,
          computing correlation of each session with baseline, and flipping sessions with negative correlation
        - For other trials: Using the SAME flip decisions as the reference trial

        Returns:
            True if aggregation successful
        """
        n_sessions = len(self.session_projections)
        if n_sessions < self.min_sessions:
            print(f"    {self.region}: Insufficient sessions ({n_sessions} < {self.min_sessions})")
            return False

        print(f"    Aggregating {self.region} across {n_sessions} sessions")

        # Collect all trial types across all sessions
        all_trial_types = set()
        for session_proj in self.session_projections.values():
            all_trial_types |= set(session_proj.keys())

        if not all_trial_types:
            return False

        self.available_trial_types = list(all_trial_types)

        # Dictionary to store flip decisions from reference trial
        # Format: {session_name: {comp_idx: should_flip}}
        reference_flip_decisions = {}

        # Process reference trial first to determine flip decisions
        for trial_type in self.available_trial_types:
            z_means_all = []
            session_names_ordered = []

            for session_name, session_proj in self.session_projections.items():
                if trial_type not in session_proj:
                    continue

                proj = session_proj[trial_type]
                z_means_all.append(proj['z_mean'])
                session_names_ordered.append(session_name)

            if len(z_means_all) < self.min_sessions:
                continue

            # Stack into array: (n_sessions, n_time, n_components)
            z_stack_raw = np.stack(z_means_all, axis=0)
            n_sess, n_time, n_comp = z_stack_raw.shape

            # Align signs based on correlation with baseline for each component
            z_stack_aligned = np.zeros_like(z_stack_raw)

            for comp_idx in range(n_comp):
                # For reference trial: find baseline and compute flip decisions
                if trial_type == self.reference_type:
                    # Find baseline session (first with positive peak)
                    baseline_idx = None
                    for sess_idx in range(n_sess):
                        z_proj = z_stack_raw[sess_idx, :, comp_idx]
                        peak_val = z_proj[np.argmax(np.abs(z_proj)[74:150])+ 74]
                        if peak_val > 0:
                            baseline_idx = sess_idx
                            break

                    if baseline_idx is None:
                        baseline_idx = 0

                    baseline = z_stack_raw[baseline_idx, :, comp_idx]

                    # Align all sessions based on correlation with baseline and store decisions
                    for sess_idx in range(n_sess):
                        session_name = session_names_ordered[sess_idx]
                        z_proj = z_stack_raw[sess_idx, :, comp_idx]
                        correlation = np.corrcoef(baseline, z_proj)[0, 1]
                        should_flip = correlation < 0

                        # Store flip decision for this session and component
                        if session_name not in reference_flip_decisions:
                            reference_flip_decisions[session_name] = {}
                        reference_flip_decisions[session_name][comp_idx] = should_flip

                        z_stack_aligned[sess_idx, :, comp_idx] = -z_proj if should_flip else z_proj
                else:
                    # For other trials: reuse flip decisions from reference trial
                    for sess_idx in range(n_sess):
                        session_name = session_names_ordered[sess_idx]
                        z_proj = z_stack_raw[sess_idx, :, comp_idx]

                        # Apply flip decision from reference trial if available
                        if session_name in reference_flip_decisions and comp_idx in reference_flip_decisions[session_name]:
                            should_flip = reference_flip_decisions[session_name][comp_idx]
                            z_stack_aligned[sess_idx, :, comp_idx] = -z_proj if should_flip else z_proj
                        else:
                            # Fallback: don't flip if no reference decision available
                            z_stack_aligned[sess_idx, :, comp_idx] = z_proj

            self.aggregated_projections[trial_type] = {
                'z_mean': np.mean(z_stack_aligned, axis=0),
                'z_std': np.std(z_stack_aligned, axis=0),
                'z_sem': np.std(z_stack_aligned, axis=0) / np.sqrt(n_sess),
                'z_sessions': z_stack_aligned,
                'n_sessions': n_sess
            }

            print(f"      {trial_type}: {n_sess} sessions aggregated")

        return len(self.aggregated_projections) >= 2

    def aggregate_temporal_correlations(self) -> Dict:
        """
        Aggregate temporal R² correlations across sessions.

        Returns:
            Dictionary with aggregated correlation data
        """
        aggregated_corr = {}

        for session_name, session_stats in self.session_statistics.items():
            if 'temporal_correlations' not in session_stats:
                continue

            for comparison_key, corr_data in session_stats['temporal_correlations'].items():
                if comparison_key not in aggregated_corr:
                    aggregated_corr[comparison_key] = {
                        f'comp_{k+1}': [] for k in range(self.n_components)
                    }

                for comp_idx, corr in enumerate(corr_data[:self.n_components]):
                    aggregated_corr[comparison_key][f'comp_{comp_idx+1}'].append(corr['r2'])

        self.aggregated_statistics['temporal_correlations'] = aggregated_corr
        return aggregated_corr


# =============================================================================
# HIERARCHICAL AGGREGATION
# =============================================================================

class HierarchicalRegionResult:
    """
    Lightweight container for hierarchical region aggregated data.

    Holds pooled aggregated data from multiple individual regions that have
    been combined according to HIERARCHICAL_GROUPING. Compatible with
    CrossTrialTypePCASummaryVisualizer for visualization.

    Attributes:
        region: Hierarchical region name (e.g., 'Striatum', 'Thalamus')
        time_bins: Time axis array
        aggregated_projections: Pooled projection data per trial type
        aggregated_statistics: Pooled statistical results
        available_trial_types: List of trial types with data
        session_projections: Dict for compatibility (session count via len())
    """

    def __init__(self, region, time_bins, n_components, reference_type='cued_hit_long'):
        self.region = region
        self.time_bins = time_bins
        self.n_components = n_components
        self.reference_type = reference_type
        self.aggregated_projections = {}
        self.aggregated_statistics = {}
        self.available_trial_types = []
        self.session_projections = {}  # For compatibility (session count)


def build_hierarchical_region_analyzers(
    cross_session_analyzers: Dict[str, 'CrossSessionPCAAnalyzer'],
    reference_type: str = 'cued_hit_long',
    n_components: int = 5,
    min_sessions: int = MIN_SESSIONS_THRESHOLD
) -> Dict[str, 'HierarchicalRegionResult']:
    """
    Build hierarchical region analyzers by aggregating individual region data.

    Aggregation procedure:
    1. Map each individual region to its hierarchical group
    2. For single-region groups (e.g., mPFC -> mPFC): wrap existing data
    3. For multi-region groups (e.g., STR+STRv -> Striatum): pool session-level
       data from all contributing regions, then compute summary statistics
    4. Compute temporal correlation statistics for pooled data

    Parameters:
        cross_session_analyzers: Dict of region-level CrossSessionPCAAnalyzer objects
        reference_type: Trial type for PCA weight extraction
        n_components: Number of PCA components
        min_sessions: Minimum pooled sessions for inclusion

    Returns:
        Dict mapping hierarchical region names to HierarchicalRegionResult objects
    """
    print("\n" + "=" * 70)
    print("BUILDING HIERARCHICAL REGION ANALYZERS (PCA)")
    print("=" * 70)

    # Step 1: Group individual regions by hierarchical category
    # {hierarchical_name: [region_name, ...]}
    hierarchical_groups = {}

    for region, analyzer in cross_session_analyzers.items():
        if not analyzer.aggregated_projections:
            continue

        h_name = get_hierarchical_region(region)

        if h_name not in hierarchical_groups:
            hierarchical_groups[h_name] = []
        hierarchical_groups[h_name].append((region, analyzer))
        print(f"  {region} -> {h_name}")

    # Step 2: Build hierarchical region results
    hierarchical_analyzers = {}

    for h_name, contributing_regions in hierarchical_groups.items():
        print(f"\n  Building hierarchical region: {h_name}")
        print(f"    Contributing regions: {[r for r, _ in contributing_regions]}")

        # Get time_bins from first contributing region
        time_bins = contributing_regions[0][1].time_bins

        h_result = HierarchicalRegionResult(
            region=h_name,
            time_bins=time_bins,
            n_components=n_components,
            reference_type=reference_type
        )

        # Pool aggregated projections from all contributing regions
        # {trial_type: {'z_list': [arrays...]}}
        pooled_by_trial = {}

        for region, analyzer in contributing_regions:
            for trial_type, agg in analyzer.aggregated_projections.items():
                if trial_type not in pooled_by_trial:
                    pooled_by_trial[trial_type] = {'z_list': []}

                z_sessions = agg.get('z_sessions')
                if z_sessions is None:
                    continue

                pooled_by_trial[trial_type]['z_list'].append(z_sessions)

        # Compute statistics for pooled data
        for trial_type, pooled in pooled_by_trial.items():
            if not pooled['z_list']:
                continue

            z_all = np.concatenate(pooled['z_list'], axis=0)  # (N_total, T, C)
            n_total = z_all.shape[0]

            if n_total < min_sessions:
                print(f"    {trial_type}: {n_total} pooled sessions (< {min_sessions}, skipping)")
                continue

            h_result.aggregated_projections[trial_type] = {
                'z_mean': np.mean(z_all, axis=0),
                'z_std': np.std(z_all, axis=0),
                'z_sem': np.std(z_all, axis=0) / np.sqrt(n_total),
                'z_sessions': z_all,
                'n_sessions': n_total
            }

            print(f"    {trial_type}: {n_total} pooled sessions")

        h_result.available_trial_types = list(h_result.aggregated_projections.keys())

        # Create a dummy session_projections dict for session count compatibility
        # Use the total count from the reference type
        ref_agg = h_result.aggregated_projections.get(reference_type, {})
        n_total_sessions = ref_agg.get('n_sessions', 0)
        h_result.session_projections = {f'pooled_{i}': {} for i in range(n_total_sessions)}

        # Pool temporal correlations from contributing regions
        pooled_corr = {}

        for region, analyzer in contributing_regions:
            if 'temporal_correlations' not in analyzer.aggregated_statistics:
                continue

            for comp_key, corr_data in analyzer.aggregated_statistics['temporal_correlations'].items():
                if comp_key not in pooled_corr:
                    pooled_corr[comp_key] = {
                        f'comp_{k+1}': [] for k in range(n_components)
                    }

                for comp_idx_key, r2_values in corr_data.items():
                    if comp_idx_key in pooled_corr[comp_key]:
                        pooled_corr[comp_key][comp_idx_key].extend(r2_values)

        h_result.aggregated_statistics['temporal_correlations'] = pooled_corr

        # Only add if we have enough data
        if len(h_result.aggregated_projections) >= 2:
            hierarchical_analyzers[h_name] = h_result
            print(f"    -> Valid hierarchical region: {h_name}")
        else:
            print(f"    -> Insufficient data for: {h_name}")

    print(f"\nTotal hierarchical regions: {len(hierarchical_analyzers)}")
    return hierarchical_analyzers


# =============================================================================
# UPPER TRIANGLE SUMMARY FIGURE CLASS
# =============================================================================

class CrossTrialTypePCASummaryVisualizer:
    """
    Visualizer for creating upper-triangle summary figures for PCA analysis.

    CRITICAL DISTINCTION FROM CCA:
    -----------------------------
    In the upper triangle layout, each cell (i, j) shows:
    - Row region (i): aggregated from ALL sessions recording region i
    - Column region (j): aggregated from ALL sessions recording region j

    The session sets for regions i and j need NOT overlap, unlike CCA.
    """

    def __init__(
            self,
            base_dir: str,
            reference_type: str = 'cued_hit_long',
            n_components: int = 5,
            min_sessions: int = MIN_SESSIONS_THRESHOLD,
            use_hierarchical: bool = False
    ):
        """
        Initialize the summary visualizer.

        Parameters:
            base_dir: Root directory for Oxford dataset
            reference_type: Trial type for PCA weight extraction
            n_components: Number of PCA components to analyze
            min_sessions: Minimum sessions required for a region to be included
            use_hierarchical: If True, use hierarchical region ordering
        """
        self.base_dir = Path(base_dir)
        self.reference_type = reference_type
        self.n_components = n_components
        self.min_sessions = min_sessions
        self.use_hierarchical = use_hierarchical

        # Cross-session analyzers for each region (independent!)
        self.region_analyzers: Dict[str, CrossSessionPCAAnalyzer] = {}

        # Available regions and time axis
        self.available_regions: List[str] = []
        self.time_bins: Optional[np.ndarray] = None

        # Output directory
        if use_hierarchical:
            self.output_dir = self.base_dir / 'Paper_output' / 'cross_trial_type_pca' / 'summary_figures_hierarchical'
        else:
            self.output_dir = self.base_dir / 'Paper_output' / 'cross_trial_type_pca' / 'summary_figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        mode_str = " (HIERARCHICAL)" if use_hierarchical else ""
        print("=" * 70)
        print(f"Cross-Trial-Type PCA Summary Visualizer{mode_str}")
        print("=" * 70)
        print(f"Reference condition: {reference_type}")
        print(f"Minimum sessions: {min_sessions}")
        print(f"Output directory: {self.output_dir}")

    def add_region_analyzer(self, region_analyzer: CrossSessionPCAAnalyzer) -> None:
        """
        Add a cross-session analyzer for a region.

        Parameters:
            region_analyzer: Completed CrossSessionPCAAnalyzer with aggregated results
        """
        region = region_analyzer.region
        self.region_analyzers[region] = region_analyzer

        if region not in self.available_regions:
            self.available_regions.append(region)

        if self.time_bins is None and region_analyzer.time_bins is not None:
            self.time_bins = region_analyzer.time_bins

        n_sess = len(region_analyzer.session_projections)
        print(f"  Added region: {region} ({n_sess} sessions)")

    def _get_ordered_regions(self) -> List[str]:
        """Return available regions in the appropriate order."""
        if self.use_hierarchical:
            return [r for r in HIERARCHICAL_ORDER if r in self.available_regions]
        return [r for r in ANATOMICAL_ORDER if r in self.available_regions]

    def create_projection_matrix_figure(
            self,
            figsize: Tuple[float, float] = (60, 60),
            component_idx: int = 0,
            save_fig: bool = True
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create two upper-triangle figures showing PCA projections across trial types.

        Unlike CCA, each region's projection is computed independently from ALL
        sessions recording that region, regardless of co-recorded regions.

        Creates separate figures for:
        - Figure 1 (row): Projections for the row region
        - Figure 2 (column): Projections for the column region

        Parameters:
            figsize: Figure dimensions
            component_idx: Which PCA component to display (0-indexed)
            save_fig: Whether to save the figure

        Returns:
            Tuple of (row_figure, column_figure)
        """
        print(f"\nCreating PCA projection matrix figures (Component {component_idx + 1})...")
        print("  NOTE: Each region aggregated INDEPENDENTLY across all its sessions")

        ordered_regions = self._get_ordered_regions()
        n_regions = len(ordered_regions)

        if n_regions == 0:
            print("No regions available for plotting")
            return None, None

        figures = []

        for fig_type in ['row', 'column']:
            fig, axes = plt.subplots(n_regions, n_regions, figsize=figsize)

            for i, region_i in enumerate(ordered_regions):
                for j, region_j in enumerate(ordered_regions):
                    ax = axes[i, j] if n_regions > 1 else axes

                    if i == j:
                        # Diagonal: region name with session count
                        analyzer = self.region_analyzers.get(region_i)
                        n_sess = len(analyzer.session_projections) if analyzer else 0
                        ax.text(0.5, 0.5, f'{region_i}\n(n={n_sess})',
                                ha='center', va='center',
                                fontsize=32, fontweight='bold')
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.axis('off')
                    elif i > j:
                        # Lower triangle: hide
                        ax.axis('off')
                    else:
                        # Upper triangle: plot cross-trial-type projections
                        display_region = region_i if fig_type == 'row' else region_j
                        self._plot_region_projections(
                            ax, display_region, component_idx
                        )

            region_label = "Row Region" if fig_type == 'row' else "Column Region"
            fig.suptitle(
                f'PCA Component {component_idx + 1} | {region_label} | '
                f'Reference: {self.reference_type} | n ≥ {self.min_sessions} sessions\n'
                f'(Each region aggregated independently across all its sessions)',
                fontsize=48, fontweight='bold', y=0.995
            )

            plt.tight_layout(rect=[0, 0.01, 1, 0.99])

            if save_fig:
                save_path = self.output_dir / f"pca_projection_matrix_comp{component_idx + 1}_{fig_type}_region.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")

            figures.append(fig)
            plt.close(fig)

        return tuple(figures)

    def _plot_region_projections(
            self,
            ax: plt.Axes,
            region: str,
            component_idx: int
    ) -> None:
        """
        Plot cross-trial-type projections for a single region.

        Parameters:
            ax: Matplotlib axes
            region: Region name
            component_idx: Component to plot (0-indexed)
        """
        analyzer = self.region_analyzers.get(region)

        if analyzer is None or len(analyzer.aggregated_projections) < 2:
            ax.set_visible(False)
            return

        n_sessions = len(analyzer.session_projections)
        if n_sessions < self.min_sessions:
            ax.set_visible(False)
            return

        # Define short labels for trial types
        trial_type_short = {
            'cued_hit_long': 'ch',
            'spont_hit_long': 'sh',
            'spont_miss_long': 'sm'
        }

        time_vec = analyzer.time_bins
        session_count_labels = []

        for trial_type in analyzer.available_trial_types:
            if trial_type not in analyzer.aggregated_projections:
                continue

            agg = analyzer.aggregated_projections[trial_type]

            if component_idx >= agg['z_mean'].shape[1]:
                continue

            mean_proj = agg['z_mean'][:, component_idx]
            sem_proj = agg['z_sem'][:, component_idx]

            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')
            linewidth = 2 if trial_type == self.reference_type else 1.0
            alpha = 0.8 if trial_type == self.reference_type else 0.4
            short_label = trial_type_short.get(trial_type, trial_type.replace('_', ' '))
            n_sess = agg['n_sessions']

            # Plot individual session lines (thin, low alpha)
            if 'z_sessions' in agg:
                session_data = agg['z_sessions']  # Shape: (n_sessions, n_time, n_components)
                for sess_idx in range(session_data.shape[0]):
                    sess_proj = session_data[sess_idx, :, component_idx]
                    ax.plot(time_vec, sess_proj, color=color, linewidth=0.5,
                            alpha=0.2, zorder=1)

            # Plot mean projection on top
            ax.plot(time_vec, mean_proj, color=color, linewidth=linewidth,
                    label=f'{short_label} (n={n_sess})', alpha=alpha, zorder=3)
            ax.fill_between(time_vec, mean_proj - sem_proj, mean_proj + sem_proj,
                            alpha=0.15, color=color, zorder=2)

            session_count_labels.append(f'n_{short_label[:2]}={n_sess}')

        # Reference line at t=0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=3)

        # Formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.8, linestyle=':', linewidth=2)
        ax.set_yticks(np.arange(0, 10, 2))
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)
        ax.set_ylim([-1.5, 3])
        ax.set_xticks([-1.5, 0, 2, 3])
        ax.set_xticklabels(['-1.5', '0', '2', '3'], fontsize=20)
        ax.tick_params(axis='both', which='major', width=2, length=8)

        for spine in ax.spines.values():
            spine.set_linewidth(3)

        # Add region and session counts annotation
        session_count_text = ', '.join(session_count_labels)
        ax.text(0.02, 0.98, f'{region}\n{session_count_text}', transform=ax.transAxes,
                fontsize=12, va='top', ha='left')

    def create_r2_boxplot_matrix_figure(
            self,
            figsize: Tuple[float, float] = (60, 60),
            save_fig: bool = True
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create two upper-triangle figures showing R² boxplots for top 3 components.

        Parameters:
            figsize: Figure dimensions
            save_fig: Whether to save the figure

        Returns:
            Tuple of (row_figure, column_figure)
        """
        print("\nCreating R² boxplot matrix figures...")

        ordered_regions = self._get_ordered_regions()
        n_regions = len(ordered_regions)

        if n_regions == 0:
            print("No regions available for plotting")
            return None, None

        figures = []

        for fig_type in ['row', 'column']:
            fig, axes = plt.subplots(n_regions, n_regions, figsize=figsize)

            for i, region_i in enumerate(ordered_regions):
                for j, region_j in enumerate(ordered_regions):
                    ax = axes[i, j] if n_regions > 1 else axes

                    if i == j:
                        # Diagonal: region name
                        analyzer = self.region_analyzers.get(region_i)
                        n_sess = len(analyzer.session_projections) if analyzer else 0
                        ax.text(0.5, 0.5, f'{region_i}\n(n={n_sess})',
                                ha='center', va='center',
                                fontsize=32, fontweight='bold')
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.axis('off')
                    elif i > j:
                        # Lower triangle: hide
                        ax.axis('off')
                    else:
                        # Upper triangle: plot R² boxplots
                        display_region = region_i if fig_type == 'row' else region_j
                        self._plot_region_r2_boxplots(ax, display_region)

            region_label = "Row Region" if fig_type == 'row' else "Column Region"
            fig.suptitle(
                f'Cross-Session Temporal R² Distribution (Top 3 Components)\n'
                f'{region_label} | Reference: {self.reference_type} | n ≥ {self.min_sessions} sessions',
                fontsize=48, fontweight='bold', y=0.995
            )

            plt.tight_layout(rect=[0, 0.01, 1, 0.99])

            if save_fig:
                save_path = self.output_dir / f"pca_r2_boxplot_matrix_{fig_type}_region.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")

            figures.append(fig)
            plt.close(fig)

        return tuple(figures)

    def _plot_region_r2_boxplots(
            self,
            ax: plt.Axes,
            region: str
    ) -> None:
        """
        Plot R² boxplots for a single region.

        Parameters:
            ax: Matplotlib axes
            region: Region name
        """
        analyzer = self.region_analyzers.get(region)

        if analyzer is None:
            ax.set_visible(False)
            return

        if 'temporal_correlations' not in analyzer.aggregated_statistics:
            ax.set_visible(False)
            return

        correlations = analyzer.aggregated_statistics['temporal_correlations']
        n_comp_show = min(3, self.n_components)

        comparisons = list(correlations.keys())
        n_comparisons = len(comparisons)

        if n_comparisons == 0:
            ax.set_visible(False)
            return

        positions = []
        data_to_plot = []
        colors = []

        for comp_idx in range(n_comp_show):
            comp_key = f'comp_{comp_idx + 1}'

            for idx, comparison in enumerate(comparisons):
                r2_values = correlations[comparison].get(comp_key, [])

                if r2_values:
                    pos = comp_idx * (n_comparisons + 0.5) + idx
                    positions.append(pos)
                    data_to_plot.append(r2_values)

                    other_type = comparison.replace(f"{self.reference_type}_vs_", "")
                    colors.append(TRIAL_TYPE_COLORS.get(other_type, 'gray'))

        if not data_to_plot:
            ax.set_visible(False)
            return

        # Create boxplots
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.4, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Set x-axis labels
        comp_positions = [(i * (n_comparisons + 0.5) + (n_comparisons - 1) / 2)
                          for i in range(n_comp_show)]
        ax.set_xticks(comp_positions)
        ax.set_xticklabels([f'PC{i+1}' for i in range(n_comp_show)], fontsize=18)

        ax.set_ylim([0, 1.05])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0', '0.5', '1'], fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Add region annotation
        ax.text(0.02, 0.98, region, transform=ax.transAxes,
                fontsize=14, va='top', ha='left')


# =============================================================================
# PIPELINE ORCHESTRATION CLASS
# =============================================================================

class CrossTrialTypePCAPipeline:
    """
    Complete pipeline for cross-trial-type PCA analysis.

    This class orchestrates the entire analysis workflow for multiple
    sessions, with cross-session aggregation performed INDEPENDENTLY
    for each region (unlike CCA which requires paired regions).
    """

    def __init__(self, config: Dict):
        """
        Initialize the pipeline with configuration.

        Parameters:
            config: Dictionary containing:
                - base_dir: Root directory for Oxford dataset
                - sessions: List of session names to analyze
                - reference_type: Trial type for PCA weight extraction
                - n_components: Number of PCA components
                - output_base_dir: Base output directory
                - enable_cross_session: Whether to perform cross-session aggregation
                - min_sessions: Minimum sessions for cross-session analysis
        """
        self.config = config
        self.analyzers: Dict[str, CrossTrialTypePCAAnalyzer] = {}
        self.results: Dict = {}

        # Cross-session analyzers for each REGION (not pair!)
        self.cross_session_analyzers: Dict[str, CrossSessionPCAAnalyzer] = {}

        # Summary visualizer
        self.summary_visualizer: Optional[CrossTrialTypePCASummaryVisualizer] = None

        # Hierarchical aggregation results
        self.hierarchical_analyzers: Dict = {}
        self.hierarchical_summary_visualizer: Optional[CrossTrialTypePCASummaryVisualizer] = None

        # Validate config
        required_keys = ['base_dir', 'sessions']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        # Set defaults
        config.setdefault('reference_type', 'cued_hit_long')
        config.setdefault('n_components', 5)
        config.setdefault('enable_cross_session', True)
        config.setdefault('min_sessions', MIN_SESSIONS_THRESHOLD)

        print("=" * 70)
        print("Cross-Trial-Type PCA Pipeline Initialized")
        print("=" * 70)
        print("NOTE: PCA aggregation is INDEPENDENT per region")
        print("      (sessions need not include both regions)")
        self._print_config()

    def _print_config(self) -> None:
        """Print configuration summary."""
        print("\nConfiguration:")
        for k, v in self.config.items():
            if isinstance(v, list) and len(v) > 5:
                print(f"  {k}: [{len(v)} items]")
            else:
                print(f"  {k}: {v}")

    def run_session_analysis(self, session_name: str) -> bool:
        """
        Run complete analysis for a single session.

        Parameters:
            session_name: Session identifier

        Returns:
            True if analysis completed successfully
        """
        print("\n" + "=" * 70)
        print(f"ANALYZING SESSION: {session_name}")
        print("=" * 70)

        try:
            # Initialize analyzer
            analyzer = CrossTrialTypePCAAnalyzer(
                base_dir=self.config['base_dir'],
                session_name=session_name,
                reference_type=self.config['reference_type'],
                n_components=self.config['n_components']
            )

            # Load all trial types
            if not analyzer.load_all_trial_types():
                print(f"Failed to load trial types for {session_name}")
                return False

            # Analyze each region independently
            for region in analyzer.available_regions:
                print(f"\n--- Analyzing region: {region} ---")

                # Extract neural data
                if not analyzer.extract_neural_data_for_region(region):
                    print(f"  Skipping - insufficient data")
                    continue

                # Extract PCA weights from reference
                if not analyzer.extract_pca_weights(region):
                    print(f"  Skipping - PCA weights not available")
                    continue

                # Compute projections
                if not analyzer.compute_projections_for_region(region):
                    print(f"  Skipping - projection failed")
                    continue

                # Compute statistics
                analyzer.compute_statistics_for_region(region)

                # Add to cross-session analyzer if enabled
                if self.config.get('enable_cross_session', True):
                    if region not in self.cross_session_analyzers:
                        self.cross_session_analyzers[region] = CrossSessionPCAAnalyzer(
                            base_dir=self.config['base_dir'],
                            region=region,
                            reference_type=self.config['reference_type'],
                            n_components=self.config['n_components'],
                            min_sessions=self.config.get('min_sessions', MIN_SESSIONS_THRESHOLD)
                        )

                    # Collect projections for this region across trial types
                    region_projections = {}
                    for tt in analyzer.projections:
                        if region in analyzer.projections[tt]:
                            region_projections[tt] = analyzer.projections[tt][region]

                    region_stats = analyzer.statistical_results.get(region, {})

                    self.cross_session_analyzers[region].add_session_result(
                        session_name,
                        region_projections,
                        region_stats,
                        analyzer.time_bins
                    )

            # Store analyzer
            self.analyzers[session_name] = analyzer

            print(f"\n✓ Session {session_name} analysis complete")
            return True

        except Exception as e:
            print(f"ERROR in session {session_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_sessions(self) -> Dict:
        """
        Run analysis for all configured sessions.

        Returns:
            Dictionary with success status per session
        """
        results = {}

        for session_name in self.config['sessions']:
            success = self.run_session_analysis(session_name)
            results[session_name] = success

        # Print summary
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        successful = sum(results.values())
        total = len(results)
        print(f"Sessions analyzed: {successful}/{total}")

        for session, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {session}")

        self.results = results
        return results

    def run_cross_session_aggregation(self) -> None:
        """
        Perform cross-session aggregation and create summary figures.

        For PCA, aggregation is performed INDEPENDENTLY for each region
        across all sessions that recorded that region.
        """
        print("\n" + "=" * 70)
        print("CROSS-SESSION PCA AGGREGATION")
        print("=" * 70)
        print("NOTE: Each region aggregated independently across its sessions")

        min_sessions = self.config.get('min_sessions', MIN_SESSIONS_THRESHOLD)

        # Initialize summary visualizer
        self.summary_visualizer = CrossTrialTypePCASummaryVisualizer(
            base_dir=self.config['base_dir'],
            reference_type=self.config['reference_type'],
            n_components=self.config['n_components'],
            min_sessions=min_sessions
        )

        # Process each region independently
        valid_regions = 0
        for region, cross_session_analyzer in self.cross_session_analyzers.items():
            n_sessions = len(cross_session_analyzer.session_projections)

            if n_sessions < min_sessions:
                print(f"  {region}: {n_sessions} sessions (skipping, < {min_sessions})")
                continue

            print(f"\n  Aggregating {region} ({n_sessions} sessions)...")

            # Perform aggregation
            if cross_session_analyzer.aggregate_projections():
                cross_session_analyzer.aggregate_temporal_correlations()

                # Add to summary visualizer
                self.summary_visualizer.add_region_analyzer(cross_session_analyzer)
                valid_regions += 1

        print(f"\nValid regions for summary: {valid_regions}")

        # Create summary figures if we have enough regions
        if valid_regions >= 2:
            print("\nCreating summary figures...")

            # Create figures for first 3 components
            for comp_idx in range(min(3, self.config['n_components'])):
                self.summary_visualizer.create_projection_matrix_figure(
                    component_idx=comp_idx,
                    save_fig=True
                )

            # R² boxplots
            self.summary_visualizer.create_r2_boxplot_matrix_figure(
                save_fig=True
            )
        else:
            print("Not enough valid regions for summary figures")

        # Run hierarchical aggregation if enabled
        if self.config.get('use_hierarchical', False):
            self._run_hierarchical_aggregation()

    def _run_hierarchical_aggregation(self) -> None:
        """
        Perform hierarchical aggregation of region-level cross-session results.

        After region-level cross-session aggregation is complete, this method:
        1. Groups individual regions by hierarchical categories
        2. Pools session-level data from contributing regions
        3. Creates a separate summary visualizer with hierarchical ordering
        4. Generates hierarchical summary figures
        """
        print("\n" + "=" * 70)
        print("HIERARCHICAL PCA AGGREGATION")
        print("=" * 70)

        min_sessions = self.config.get('min_sessions', MIN_SESSIONS_THRESHOLD)

        # Build hierarchical region analyzers from region-level cross-session data
        hierarchical_analyzers = build_hierarchical_region_analyzers(
            self.cross_session_analyzers,
            reference_type=self.config['reference_type'],
            n_components=self.config['n_components'],
            min_sessions=min_sessions
        )

        if not hierarchical_analyzers:
            print("No valid hierarchical regions found")
            return

        # Store for external access
        self.hierarchical_analyzers = hierarchical_analyzers

        # Create hierarchical summary visualizer
        self.hierarchical_summary_visualizer = CrossTrialTypePCASummaryVisualizer(
            base_dir=self.config['base_dir'],
            reference_type=self.config['reference_type'],
            n_components=self.config['n_components'],
            min_sessions=min_sessions,
            use_hierarchical=True
        )

        # Add hierarchical region analyzers
        for h_name, h_result in hierarchical_analyzers.items():
            self.hierarchical_summary_visualizer.add_region_analyzer(h_result)

        valid_regions = len(hierarchical_analyzers)
        print(f"\nValid hierarchical regions for summary: {valid_regions}")

        # Create hierarchical summary figures
        if valid_regions >= 2:
            print("\nCreating hierarchical summary figures...")

            # Create figures for first 3 components
            for comp_idx in range(min(3, self.config['n_components'])):
                self.hierarchical_summary_visualizer.create_projection_matrix_figure(
                    component_idx=comp_idx,
                    save_fig=True
                )

            # R² boxplots
            self.hierarchical_summary_visualizer.create_r2_boxplot_matrix_figure(
                save_fig=True
            )
        else:
            print("Not enough valid hierarchical regions for summary figures")


# =============================================================================
# MAIN DEMONSTRATION FUNCTION
# =============================================================================

def main():
    """Demonstration of cross-trial-type PCA analysis."""

    # Configuration
    config = {
        'base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset',
        'sessions': ['yp010_220209'],  # Example session
        'reference_type': 'cued_hit_long',
        'n_components': 5,
        'enable_cross_session': True,
        'min_sessions': MIN_SESSIONS_THRESHOLD
    }

    print("=" * 70)
    print("Cross-Trial-Type PCA Analysis Pipeline")
    print("=" * 70)

    # Initialize and run pipeline
    pipeline = CrossTrialTypePCAPipeline(config)
    results = pipeline.run_all_sessions()

    # Run cross-session aggregation if enabled
    if config.get('enable_cross_session', True):
        pipeline.run_cross_session_aggregation()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return pipeline


if __name__ == "__main__":
    pipeline = main()

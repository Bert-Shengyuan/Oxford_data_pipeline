#!/usr/bin/env python3
"""
Oxford Dataset Cross-Trial-Type CCA Projection Analysis
=========================================================

This module implements cross-trial-type canonical correlation analysis (CCA)
projection, where CCA weights trained on one behavioral condition (e.g.,
cued_hit_long) are used to project neural activity from other conditions
(e.g., spont_hit_long, spont_miss_long) into the same latent subspace.

Mathematical Framework:
-----------------------
Given CCA weights $\mathbf{A} \in \mathbb{R}^{n_i \times k}$ for region $i$
and $\mathbf{B} \in \mathbb{R}^{n_j \times k}$ for region $j$ trained on
condition $c_{\text{ref}}$ (cued_hit_long), the canonical variates for
any condition $c$ are:

$$\mathbf{u}_c = \mathbf{X}_c^{(i)} \mathbf{A}, \quad \mathbf{v}_c = \mathbf{X}_c^{(j)} \mathbf{B}$$

where $\mathbf{X}_c^{(i)} \in \mathbb{R}^{T \times n_i}$ is the trial-averaged
neural activity matrix for region $i$ in condition $c$.

The cross-condition projection similarity is quantified by:
$$R^2_{c_1, c_2} = \text{corr}^2(\mathbf{u}_{c_1}, \mathbf{u}_{c_2})$$

Cross-Session Aggregation:
--------------------------
For multi-session analysis, projections are first computed per-session,
then aggregated across sessions:
$$\bar{\mathbf{u}}_c = \frac{1}{N_s} \sum_{s=1}^{N_s} \mathbf{u}_{c,s}$$

Standard error is computed as:
$$\text{SEM} = \frac{\sigma}{\sqrt{N_s}}$$

Author: Oxford Neural Analysis Pipeline
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mat73
import scipy.io as sio
from scipy.stats import zscore, wilcoxon, ttest_rel, pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Optional: Rastermap for neural population sorting
try:
    from rastermap import Rastermap

    RASTERMAP_AVAILABLE = True
except ImportError:
    RASTERMAP_AVAILABLE = False
    print("Warning: Rastermap not installed. Install with: pip install rastermap")

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Trial type labels and their data directories
TRIAL_TYPES = {
    'cued_hit_long': 'sessions_cued_hit_long_results',
    'spont_hit_long': 'sessions_spont_hit_long_results',
}

# TRIAL_TYPES = {
#     'cued_hit_long': 'sessions_cued_hit_long_results',
#     # 'spont_hit_long': 'sessions_spont_hit_long_results',
#     # 'spont_miss_long': 'sessions_spont_miss_long_results'
# }

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


def extract_string_field(data_dict: Dict, field_name: str) -> Optional[str]:
    """
    Extract string from various MATLAB data structures.

    Handles the polymorphism of MATLAB string representations in Python.
    """
    try:
        if field_name not in data_dict:
            return None
        value = data_dict[field_name]
        if isinstance(value, str):
            return value
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return str(value.item())
            elif value.dtype.char == 'U':
                return str(value[0])
        elif isinstance(value, list) and len(value) > 0:
            return str(value[0])
        return None
    except Exception:
        return None


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


# =============================================================================
# MAIN CLASS: CrossTrialTypeCCAAnalyzer
# =============================================================================

class CrossTrialTypeCCAAnalyzer:
    """
    Analyzer for cross-trial-type CCA projection.

    This class implements the methodology where CCA weights trained on a
    reference condition (typically cued_hit_long) are used to project
    neural activity from other conditions into the same latent subspace.

    The key insight is that if two conditions evoke similar neural
    coordination patterns, their projections onto the reference CCA
    subspace should be highly correlated.

    Attributes:
        base_dir: Root directory containing all session results
        session_name: Identifier for the session (e.g., 'yp010_220209')
        reference_type: Trial type used for CCA weight training
        trial_type_data: Dictionary holding loaded data per trial type
        cca_weights: Dictionary holding CCA weights (A, B matrices)
        projections: Dictionary holding computed projections
        rastermap_order: Global rastermap sorting indices from reference
    """

    def __init__(
            self,
            base_dir: str,
            session_name: str,
            reference_type: str = 'cued_hit_long',
            n_components: int = 5
    ):
        """
        Initialize the cross-trial-type CCA analyzer.

        Parameters:
            base_dir: Root directory (e.g., '/path/to/Oxford_dataset')
            session_name: Session identifier (e.g., 'yp010_220209')
            reference_type: Trial type for CCA weight training (default: cued_hit_long)
            n_components: Number of CCA components to analyze
        """
        self.base_dir = Path(base_dir)
        self.session_name = session_name
        self.reference_type = reference_type
        self.n_components = n_components

        # Data containers
        self.trial_type_data: Dict[str, Dict] = {}
        self.cca_weights: Dict[str, Dict] = {}
        self.neural_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.projections: Dict[str, Dict] = {}
        self.statistical_results: Dict = {}

        # Rastermap results (computed once from reference)
        self.global_rastermap_results: Optional[Dict] = None
        self.region_rastermap_results: Dict[str, Dict] = {}

        # Available trial types for this session
        self.available_trial_types: List[str] = []

        # Region information
        self.available_regions: List[str] = []
        self.available_pairs: List[Tuple[str, str, int]] = []

        # Current region pair being analyzed
        self.current_region_pair: Optional[Tuple[str, str]] = None

        # Time axis
        self.time_bins: Optional[np.ndarray] = None

        print("=" * 70)
        print("Cross-Trial-Type CCA Analyzer")
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

        This method scans for available trial type data files and loads
        neural activity, CCA results, and PCA results for each.

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
            # elif 'time_axis' in region_data:
            #     self.time_bins = np.array(region_data['time_axis']).flatten()

        if self.time_bins is None:
            # Default time axis: -1.5s to 3.0s at ~20ms bins
            self.time_bins = np.linspace(-1.5, 3.0, 226)

        print(f"  Time axis: {len(self.time_bins)} bins, {self.time_bins[0]:.2f}s to {self.time_bins[-1]:.2f}s")

    def _extract_region_info(self) -> None:
        """Extract available regions and pairs from reference type CCA results."""
        ref_data = self.trial_type_data[self.reference_type]

        # Extract available regions
        if 'region_data' in ref_data and 'regions' in ref_data['region_data']:
            regions_data = ref_data['region_data']['regions']
            self.available_regions = list(regions_data.keys())
            print(f"Available regions: {', '.join(self.available_regions)}")

        # Extract CCA pairs from reference type
        if 'cca_results' in ref_data and 'pair_results' in ref_data['cca_results']:
            pair_results = ref_data['cca_results']['pair_results']

            for pair_idx, pair_result in enumerate(pair_results):
                if isinstance(pair_result, dict):
                    region_i = extract_string_field(pair_result, 'region_i')
                    region_j = extract_string_field(pair_result, 'region_j')

                    if region_i and region_j:
                        self.available_pairs.append((region_i, region_j, pair_idx))

            print(f"Available region pairs ({len(self.available_pairs)}):")
            for r1, r2, idx in self.available_pairs:
                print(f"  [{idx}] {r1} vs {r2}")

    # =========================================================================
    # NEURAL DATA EXTRACTION
    # =========================================================================

    def extract_neural_data(self, region_pair: Tuple[str, str]) -> bool:
        """
        Extract neural activity matrices for a region pair across all trial types.

        Parameters:
            region_pair: Tuple of (region_i, region_j) names

        Returns:
            True if extraction successful for all available trial types
        """
        region_i, region_j = region_pair
        #self.current_region_pair = region_pair
        print(f"\n" + "-" * 50)
        print(f"Extracting neural data for: {region_i} vs {region_j}")
        print("-" * 50)

        self.neural_data = {}

        for trial_type in self.available_trial_types:
            data = self.trial_type_data[trial_type]

            if 'region_data' not in data or 'regions' not in data['region_data']:
                print(f"  {trial_type}: No region_data found")
                continue

            regions_data = data['region_data']['regions']
            trial_neural_data = {}

            for region_name in [region_i, region_j]:
                if region_name not in regions_data:
                    print(f"  {trial_type}: Region {region_name} not found")
                    continue

                region_info = regions_data[region_name]

                if 'spike_data' not in region_info:
                    print(f"  {trial_type}: No spike_data for {region_name}")
                    continue

                # Extract spike data and apply neuron selection
                spike_data = region_info['spike_data']

                if 'selected_neurons' in region_info:
                    selected_neurons = region_info['selected_neurons'].reshape(-1).astype(int) - 1
                    spike_data = spike_data[:, selected_neurons, :]

                trial_neural_data[region_name] = spike_data
                n_trials, n_neurons, n_time = spike_data.shape
                print(f"  {trial_type} - {region_name}: {n_trials} trials × {n_neurons} neurons × {n_time} time")

            if len(trial_neural_data) == 2:
                self.neural_data[trial_type] = trial_neural_data

        return len(self.neural_data) >= 2

    # =========================================================================
    # CCA WEIGHT EXTRACTION
    # =========================================================================

    def extract_cca_weights(self, region_pair: Tuple[str, str]) -> bool:
        """
        Extract CCA weights (A, B matrices) from reference condition.

        The CCA weights transform neural activity into canonical variates:
        u = X_i @ A, v = X_j @ B

        These weights are computed on the reference condition and will be
        used to project neural activity from all conditions.

        Parameters:
            region_pair: Tuple of (region_i, region_j) names

        Returns:
            True if CCA weights successfully extracted
        """
        region_i, region_j = region_pair
        print(f"\n" + "-" * 50)
        print(f"Extracting CCA weights from {self.reference_type}")
        print("-" * 50)

        ref_data = self.trial_type_data[self.reference_type]

        if 'cca_results' not in ref_data:
            print("ERROR: No CCA results in reference data")
            return False

        cca_results = ref_data['cca_results']

        if 'pair_results' not in cca_results:
            print("ERROR: No pair_results in CCA results")
            return False

        # Find the correct pair index and track if order is reversed
        pair_idx = None
        pair_is_reversed = False
        for r1, r2, idx in self.available_pairs:
            if r1 == region_i and r2 == region_j:
                pair_idx = idx
                pair_is_reversed = False
                break
            elif r1 == region_j and r2 == region_i:
                pair_idx = idx
                pair_is_reversed = True
                break

        if pair_idx is None:
            print(f"ERROR: Region pair {region_i}-{region_j} not found in CCA results")
            return False

        pair_result = cca_results['pair_results'][pair_idx]

        # Extract weight matrices
        if 'mean_A_matrix' not in pair_result or 'mean_B_matrix' not in pair_result:
            print("ERROR: CCA weight matrices not found")
            return False

        A_matrix = pair_result['mean_A_matrix']
        B_matrix = pair_result['mean_B_matrix']

        # Extract mean vectors for centering (if available)
        mu_A = pair_result.get('mean_A', None)
        mu_B = pair_result.get('mean_B', None)

        # If pair was found in reversed order, swap A↔B to match input region order
        # A_matrix in file corresponds to stored region_i, B_matrix to stored region_j
        # If reversed: stored region_i = input region_j, so we need to swap
        if pair_is_reversed:
            A_matrix, B_matrix = B_matrix, A_matrix
            mu_A, mu_B = mu_B, mu_A
            print(f"  Note: Pair found in reversed order, swapping A↔B matrices")

        # Store weights
        self.cca_weights = {
            'region_i': region_i,
            'region_j': region_j,
            'A': A_matrix,
            'B': B_matrix,
            'mu_A': mu_A,
            'mu_B': mu_B,
            'n_components': A_matrix.shape[1] if A_matrix.ndim > 1 else 1,
            'pair_is_reversed': pair_is_reversed  # Track for projection extraction
        }

        print(f"  Region i ({region_i}): A matrix shape = {A_matrix.shape}")
        print(f"  Region j ({region_j}): B matrix shape = {B_matrix.shape}")
        print(f"  Available components: {self.cca_weights['n_components']}")

        return True

    # =========================================================================
    # PROJECTION COMPUTATION
    # =========================================================================

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

    def compute_projections(self) -> bool:
        """
        Project neural activity from all trial types onto CCA subspace.

        For each trial type $c$, computes canonical variates:
        $$\mathbf{u}_c = \bar{\mathbf{X}}_c^{(i)} \mathbf{A}$$
        $$\mathbf{v}_c = \bar{\mathbf{X}}_c^{(j)} \mathbf{B}$$

        where $\bar{\mathbf{X}}_c$ is the trial-averaged neural activity.

        Returns:
            True if projections computed successfully
        """
        print(f"\n" + "-" * 50)
        print("Computing CCA projections across trial types")
        print("-" * 50)

        if not self.cca_weights:
            print("ERROR: CCA weights not extracted")
            return False

        region_i = self.cca_weights['region_i']
        region_j = self.cca_weights['region_j']

        A = self.cca_weights['A']
        B = self.cca_weights['B']

        self.projections = {}

        for trial_type in self.available_trial_types:
            # For cued_hit_long, use pre-computed projections if available
            if trial_type == 'cued_hit_long':
                trial_data = self.trial_type_data[trial_type]
                if 'cca_results' in trial_data and 'pair_results' in trial_data['cca_results']:
                    # Find the correct pair index
                    pair_idx = None
                    for r1, r2, idx in self.available_pairs:
                        if (r1 == region_i and r2 == region_j) or (r1 == region_j and r2 == region_i):
                            pair_idx = idx
                            break

                    if pair_idx is not None:
                        pair_result = trial_data['cca_results']['pair_results'][pair_idx]
                        if 'projections' in pair_result and 'components' in pair_result['projections']:
                            # Extract pre-computed projections
                            extracted_proj = self._extract_projections(pair_result['projections'])
                            if extracted_proj is not None:
                                # Reshape extracted projections to match expected format
                                # extracted_proj[comp_idx] = {'region_i_mean': array, 'region_j_mean': array}
                                n_components = len(extracted_proj)
                                n_timepoints = len(extracted_proj[0]['region_i_mean']) if 0 in extracted_proj else len(self.time_bins)

                                # Initialize arrays
                                u_mean = np.zeros((n_timepoints, n_components))
                                v_mean = np.zeros((n_timepoints, n_components))

                                # Check if pair was reversed - if so, swap region_i_mean ↔ region_j_mean
                                # to maintain correct mapping: u→region_i, v→region_j
                                pair_is_reversed = self.cca_weights.get('pair_is_reversed', False)

                                # Fill in projection data
                                for comp_idx in range(n_components):
                                    if comp_idx in extracted_proj:
                                        if pair_is_reversed:
                                            # Swap: stored region_i = input region_j, so assign region_j_mean to u
                                            u_mean[:, comp_idx] = extracted_proj[comp_idx]['region_j_mean'][:n_timepoints]
                                            v_mean[:, comp_idx] = extracted_proj[comp_idx]['region_i_mean'][:n_timepoints]
                                        else:
                                            u_mean[:, comp_idx] = extracted_proj[comp_idx]['region_i_mean'][:n_timepoints]
                                            v_mean[:, comp_idx] = extracted_proj[comp_idx]['region_j_mean'][:n_timepoints]

                                # Store projections (without trial-level data since we only have averages)
                                self.projections[trial_type] = {
                                    'u_mean': u_mean,
                                    'v_mean': v_mean,
                                    'u_trials': None,  # Not available in extracted projections
                                    'v_trials': None,  # Not available in extracted projections
                                    'u_std': None,
                                    'v_std': None,
                                    'u_sem': None,
                                    'v_sem': None,
                                    'n_trials': None
                                }

                                if pair_is_reversed:
                                    print(f"  {trial_type}: Using pre-computed projections (swapped for reversed pair order)")
                                else:
                                    print(f"  {trial_type}: Using pre-computed projections from CCA results")
                                continue
            elif trial_type not in self.neural_data:
                continue


            else:
                neural = self.neural_data[trial_type]

                # Get neural activity for each region
                X_i = neural[region_i]  # shape: (n_trials, n_neurons, n_time)
                X_j = neural[region_j]

                n_trials, n_neurons, n_timepoints = X_i.shape

                region_i_sampled_p = np.transpose(X_i, (1, 2, 0))
                region_j_sampled_p = np.transpose(X_j, (1, 2, 0))

                # Reshape and transpose: (neurons, timepoints, trials) → (trials×timepoints, neurons)
                X = region_i_sampled_p.reshape(n_neurons, n_trials * n_timepoints).T
                Y = region_j_sampled_p.reshape(region_j_sampled_p.shape[0], n_trials * n_timepoints).T

                X_i_norm = zscore(X, axis=0, nan_policy='omit')
                X_j_norm = zscore(Y, axis=0, nan_policy='omit')


                X_i_norm = np.nan_to_num(X_i_norm, nan=0.0)
                X_j_norm = np.nan_to_num(X_j_norm, nan=0.0)

                # Project onto CCA subspace: (n_time, n_components)
                u = (X_i_norm @ A[:, :self.n_components]).T
                v = (X_j_norm @ B[:, :self.n_components]).T


                u_trials_p = u.reshape(self.n_components, n_timepoints, n_trials)
                v_trials_p = v.reshape(self.n_components, n_timepoints, n_trials)

                # Compute trial-averaged activity
                u_final = np.mean(u_trials_p, axis=2).T
                v_final = np.mean(v_trials_p, axis=2).T

                u_trials = np.transpose(u_trials_p, (2, 1, 0))
                v_trials = np.transpose(v_trials_p, (2, 1, 0))

                # Store projections
                self.projections[trial_type] = {
                    'u_mean': u_final,  # Trial-averaged projection region i
                    'v_mean': v_final,  # Trial-averaged projection region j
                    'u_trials': u_trials,  # Per-trial projections region i
                    'v_trials': v_trials,  # Per-trial projections region j
                    'u_std': np.std(u_trials, axis=0),
                    'v_std': np.std(v_trials, axis=0),
                    'u_sem': np.std(u_trials, axis=0) / np.sqrt(n_trials),
                    'v_sem': np.std(v_trials, axis=0) / np.sqrt(n_trials),
                    'n_trials': n_trials
                }

                print(f"  {trial_type}: {n_trials} trials projected")

        return len(self.projections) >= 2

    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================

    def compute_statistics(self) -> Dict:
        """
        Compute statistical comparisons between trial types.

        Metrics computed:
        1. Peak amplitude: max(|projection|) in post-stimulus window
        2. Temporal correlation: R² between projection time courses
        3. Wilcoxon signed-rank test: paired comparison of peak values
        4. Bootstrap confidence intervals: uncertainty on means

        Returns:
            Dictionary containing all statistical results
        """
        print(f"\n" + "-" * 50)
        print("Computing statistical comparisons")
        print("-" * 50)

        results = {
            'peak_amplitudes': {},
            'temporal_correlations': {},
            'pairwise_tests': {}
        }

        # Define post-stimulus window (0 to 1.5s)
        post_stim_mask = (self.time_bins >= 0) & (self.time_bins <= 1.5)

        # Compute peak amplitudes for each trial type and component
        for trial_type, proj in self.projections.items():
            peaks_u = []
            peaks_v = []

            for comp_idx in range(self.n_components):
                # Trial-averaged peaks
                u_post = np.abs(proj['u_mean'][post_stim_mask, comp_idx])
                v_post = np.abs(proj['v_mean'][post_stim_mask, comp_idx])

                peaks_u.append(np.max(u_post))
                peaks_v.append(np.max(v_post))

            results['peak_amplitudes'][trial_type] = {
                'region_i': peaks_u,
                'region_j': peaks_v
            }

            print(f"  {trial_type} peak amplitudes:")
            print(f"    Region i: {[f'{p:.3f}' for p in peaks_u[:3]]}")
            print(f"    Region j: {[f'{p:.3f}' for p in peaks_v[:3]]}")

        # Compute temporal correlations (reference vs other types)
        ref_proj = self.projections[self.reference_type]

        for trial_type, proj in self.projections.items():
            if trial_type == self.reference_type:
                continue

            corr_u = []
            corr_v = []

            for comp_idx in range(self.n_components):
                # Correlation of projection time courses
                r_u, p_u = pearsonr(ref_proj['u_mean'][:, comp_idx],
                                    proj['u_mean'][:, comp_idx])
                r_v, p_v = pearsonr(ref_proj['v_mean'][:, comp_idx],
                                    proj['v_mean'][:, comp_idx])

                corr_u.append({'r': r_u, 'r2': r_u ** 2, 'p': p_u})
                corr_v.append({'r': r_v, 'r2': r_v ** 2, 'p': p_v})

            results['temporal_correlations'][f"{self.reference_type}_vs_{trial_type}"] = {
                'region_i': corr_u,
                'region_j': corr_v
            }

            print(f"\n  Temporal correlation ({self.reference_type} vs {trial_type}):")

        # Store statistical results (significance tests are performed at cross-session level)
        self.statistical_results = results
        return results

# =============================================================================
# CROSS-SESSION AGGREGATION CLASS
# =============================================================================

class CrossSessionCCAAnalyzer:
    """
    Aggregator for cross-trial-type CCA analysis across multiple sessions.
    
    This class collects results from multiple sessions for a given region pair
    and computes cross-session statistics including mean ± SEM projections
    and temporal correlation distributions.
    
    Mathematical Framework:
    -----------------------
    For cross-session aggregation of projections:
    
    $$\bar{\mathbf{u}}_{c,\text{pop}} = \frac{1}{N_s} \sum_{s=1}^{N_s} \bar{\mathbf{u}}_{c,s}$$
    
    where $\bar{\mathbf{u}}_{c,s}$ is the session-mean projection.
    
    The cross-session SEM is:
    $$\text{SEM}_{\text{pop}} = \frac{\sigma_{\text{sessions}}}{\sqrt{N_s}}$$
    
    Attributes:
        base_dir: Root directory containing all session results
        region_pair: Tuple of (region_i, region_j) being analyzed
        session_results: Dictionary mapping session_name → analyzer results
        aggregated_projections: Cross-session aggregated projections
        aggregated_statistics: Cross-session statistical summaries
    """
    
    def __init__(
            self,
            base_dir: str,
            region_pair: Tuple[str, str],
            reference_type: str = 'cued_hit_long',
            n_components: int = 5,
            min_sessions: int = MIN_SESSIONS_THRESHOLD
    ):
        """
        Initialize the cross-session analyzer.
        
        Parameters:
            base_dir: Root directory for Oxford dataset
            region_pair: Tuple of (region_i, region_j) to analyze
            reference_type: Trial type for CCA weight training
            n_components: Number of CCA components to analyze
            min_sessions: Minimum sessions required for aggregation
        """
        self.base_dir = Path(base_dir)


        self.region_pair = sort_pair_by_anatomy(*region_pair)



        self.reference_type = reference_type
        self.n_components = n_components
        self.min_sessions = min_sessions
        
        # Session-level data containers
        self.session_analyzers: Dict[str, CrossTrialTypeCCAAnalyzer] = {}
        self.session_projections: Dict[str, Dict] = {}
        self.session_statistics: Dict[str, Dict] = {}
        
        # Cross-session aggregated results
        self.aggregated_projections: Dict[str, Dict] = {}
        self.aggregated_statistics: Dict = {}
        
        # Time axis (from first valid session)
        self.time_bins: Optional[np.ndarray] = None
        
        # Available trial types (intersection across sessions)
        self.available_trial_types: List[str] = []
        
        # Output directory
        region_i, region_j = self.region_pair
        # self.output_dir = self.base_dir / 'Paper_output' / 'cross_trial_type_cca' / 'cross_session' / f'{region_i}_{region_j}'
        # self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("Cross-Session CCA Analyzer")
        print("=" * 70)
        print(f"Region pair: {region_i} vs {region_j}")
        print(f"Reference condition: {reference_type}")
        print(f"Minimum sessions: {min_sessions}")
        # print(f"Output directory: {self.output_dir}")
    
    def add_session_result(
            self,
            session_name: str,
            analyzer: CrossTrialTypeCCAAnalyzer,
            swap_uv: bool = False
    ) -> bool:
        """
        Add results from a single session analyzer.

        Parameters:
            session_name: Session identifier
            analyzer: Completed CrossTrialTypeCCAAnalyzer with projections
            swap_uv: If True, swap u and v data to match the canonical pair ordering.
                     This is needed when sort_pair_by_anatomy flipped the original
                     region order. After swapping:
                     - u data corresponds to self.region_pair[0] (row region)
                     - v data corresponds to self.region_pair[1] (column region)

        Returns:
            True if session was successfully added
        """
        if not analyzer.projections:
            print(f"  {session_name}: No projections available")
            return False

        # Store analyzer
        self.session_analyzers[session_name] = analyzer

        # If pair was flipped, swap u↔v to maintain correct region-data mapping
        if swap_uv:
            swapped_projections = {}
            for trial_type, proj in analyzer.projections.items():
                swapped_projections[trial_type] = {
                    'u_mean': proj['v_mean'],  # swap: u ← v
                    'v_mean': proj['u_mean'],  # swap: v ← u
                    'u_trials': proj['v_trials'],
                    'v_trials': proj['u_trials'],
                    'u_std': proj['v_std'],
                    'v_std': proj['u_std'],
                    'u_sem': proj['v_sem'],
                    'v_sem': proj['u_sem'],
                    'n_trials': proj['n_trials']
                }
            self.session_projections[session_name] = swapped_projections

            # Also swap statistical results for region_i ↔ region_j
            swapped_stats = self._swap_statistics(analyzer.statistical_results)
            self.session_statistics[session_name] = swapped_stats
            print(f"  Added session: {session_name} (u↔v swapped for canonical ordering)")
        else:
            self.session_projections[session_name] = analyzer.projections
            self.session_statistics[session_name] = analyzer.statistical_results
            print(f"  Added session: {session_name}")

        # Update time bins from first session
        if self.time_bins is None:
            self.time_bins = analyzer.time_bins

        return True

    def _swap_statistics(self, stats: Dict) -> Dict:
        """
        Swap region_i and region_j in statistical results.

        Parameters:
            stats: Original statistical results dictionary

        Returns:
            New dictionary with region_i and region_j swapped
        """
        swapped = {}

        # Swap peak amplitudes
        if 'peak_amplitudes' in stats:
            swapped['peak_amplitudes'] = {}
            for trial_type, peaks in stats['peak_amplitudes'].items():
                swapped['peak_amplitudes'][trial_type] = {
                    'region_i': peaks['region_j'],
                    'region_j': peaks['region_i']
                }

        # Swap temporal correlations
        if 'temporal_correlations' in stats:
            swapped['temporal_correlations'] = {}
            for comp_key, corr_data in stats['temporal_correlations'].items():
                swapped['temporal_correlations'][comp_key] = {
                    'region_i': corr_data['region_j'],
                    'region_j': corr_data['region_i']
                }

        # Swap pairwise tests
        if 'pairwise_tests' in stats:
            swapped['pairwise_tests'] = {}
            for comp_key, test_data in stats['pairwise_tests'].items():
                swapped['pairwise_tests'][comp_key] = {
                    'region_i': test_data['region_j'],
                    'region_j': test_data['region_i']
                }

        return swapped
    
    def aggregate_projections(self) -> bool:
        """
        Aggregate projections across all sessions.

        Computes mean ± SEM by:
        1. First averaging within each session (already done in per-session analysis)
        2. Then averaging across sessions independently for each trial type
        3. Aligning signs based on correlation with baseline latent

        Sign alignment is performed by:
        - Identifying baseline latent from first session with positive peak
        - Computing correlation of each session with baseline
        - Flipping sessions with negative correlation

        Note: Sessions do NOT need to contain all trial types simultaneously.
        Each trial type is aggregated independently using only sessions that
        contain that trial type. For example:
        - cued_hit_long: averaged across 15 sessions
        - spont_hit_long: averaged across 8 sessions
        - spont_miss_long: averaged across 6 sessions

        Returns:
            True if aggregation successful
        """
        n_sessions = len(self.session_projections)
        if n_sessions < self.min_sessions:
            print(f"Insufficient sessions: {n_sessions} < {self.min_sessions}")
            return False

        print(f"\n" + "-" * 50)
        print(f"Aggregating projections across {n_sessions} sessions")
        print("-" * 50)

        # Collect ALL trial types that appear in ANY session (union, not intersection)
        # Each trial type will be aggregated independently
        all_trial_types = set()
        for session_proj in self.session_projections.values():
            all_trial_types |= set(session_proj.keys())

        if not all_trial_types:
            print("No trial types found across sessions")
            return False

        self.available_trial_types = list(all_trial_types)
        print(f"Available trial types (aggregated independently): {', '.join(self.available_trial_types)}")

        # Aggregate for each trial type
        for trial_type in self.available_trial_types:
            # Collect session-level means
            u_means_all = []
            v_means_all = []

            for session_name, session_proj in self.session_projections.items():
                if trial_type not in session_proj:
                    continue

                proj = session_proj[trial_type]
                u_means_all.append(proj['u_mean'][:,:])
                v_means_all.append(proj['v_mean'][:,:])

            if len(u_means_all) < self.min_sessions:
                continue

            # Stack into arrays: (n_sessions, n_time, n_components)
            u_stack_raw = np.stack(u_means_all, axis=0)
            v_stack_raw = np.stack(v_means_all, axis=0)

            # Align signs based on correlation with baseline for each component
            n_sess, n_time, n_comp = u_stack_raw.shape
            u_stack_aligned = np.zeros_like(u_stack_raw)
            v_stack_aligned = np.zeros_like(v_stack_raw)

            for comp_idx in range(n_comp):
                # Find baseline session (first with positive peak) for u
                u_baseline_idx = None
                for sess_idx in range(n_sess):
                    u_proj = u_stack_raw[sess_idx, :, comp_idx]
                    peak_val = u_proj[np.argmax(np.abs(u_proj))]
                    if peak_val > 0:
                        u_baseline_idx = sess_idx
                        break

                if u_baseline_idx is None:
                    u_baseline_idx = 0

                u_baseline = u_stack_raw[u_baseline_idx, :, comp_idx]

                # Find baseline session for v
                v_baseline_idx = None
                for sess_idx in range(n_sess):
                    v_proj = v_stack_raw[sess_idx, :, comp_idx]
                    peak_val = v_proj[np.argmax(np.abs(v_proj)[74:150])+ 74]
                    if peak_val > 0:
                        v_baseline_idx = sess_idx
                        break

                if v_baseline_idx is None:
                    v_baseline_idx = 0

                v_baseline = v_stack_raw[v_baseline_idx, :, comp_idx]

                # Align all sessions based on correlation with baseline
                for sess_idx in range(n_sess):
                    u_proj = u_stack_raw[sess_idx, :, comp_idx]
                    u_corr = np.corrcoef(u_baseline, u_proj)[0, 1]
                    u_stack_aligned[sess_idx, :, comp_idx] = u_proj if u_corr >= 0 else -u_proj

                    v_proj = v_stack_raw[sess_idx, :, comp_idx]
                    v_corr = np.corrcoef(v_baseline, v_proj)[0, 1]
                    v_stack_aligned[sess_idx, :, comp_idx] = v_proj if v_corr >= 0 else -v_proj

            u_mean_final = np.mean(u_stack_aligned, axis=0)
            v_mean_final = np.mean(v_stack_aligned, axis=0)

            max_idx = np.argmax(np.abs(u_mean_final)[74:150,0]) + 74
            u_mean_final = u_mean_final if u_mean_final[max_idx,0] > 0 else -u_mean_final

            max_idx = np.argmax(np.abs(v_mean_final)[74:150,0]) + 74
            v_mean_final = u_mean_final if v_mean_final[max_idx,0] > 0 else -v_mean_final

            #v_mean_final = v_mean_final if v_mean_final[np.argmax(np.abs(v_mean_final)[74:150,0])+ 74,0] >0 else -v_mean_final
            # Compute cross-session statistics

            self.aggregated_projections[trial_type] = {
                'u_mean': u_mean_final,  # (n_time, n_components)
                'v_mean': v_mean_final,
                'u_std': np.std(u_stack_aligned, axis=0),
                'v_std': np.std(v_stack_aligned, axis=0),
                'u_sem': np.std(u_stack_aligned, axis=0) / np.sqrt(n_sess),
                'v_sem': np.std(v_stack_aligned, axis=0) / np.sqrt(n_sess),
                'u_sessions': u_stack_aligned,  # Keep individual session data for plotting
                'v_sessions': v_stack_aligned,
                'n_sessions': n_sess
            }

            print(f"  {trial_type}: aggregated {n_sess} sessions")

        return len(self.aggregated_projections) >= 2
    
    def aggregate_temporal_correlations(self) -> Dict:
        """
        Aggregate temporal R² correlations across sessions.
        
        Collects per-session R² values for boxplot visualization.
        
        Returns:
            Dictionary with aggregated correlation data
        """
        print(f"\n" + "-" * 50)
        print("Aggregating temporal correlations")
        print("-" * 50)
        
        aggregated_corr = {}
        
        for session_name, session_stats in self.session_statistics.items():
            if 'temporal_correlations' not in session_stats:
                continue
            
            for comparison_key, corr_data in session_stats['temporal_correlations'].items():
                if comparison_key not in aggregated_corr:
                    aggregated_corr[comparison_key] = {
                        'region_i': {f'comp_{k+1}': [] for k in range(self.n_components)},
                        'region_j': {f'comp_{k+1}': [] for k in range(self.n_components)}
                    }
                
                # Collect R² values per component
                for comp_idx, corr in enumerate(corr_data['region_i'][:self.n_components]):
                    aggregated_corr[comparison_key]['region_i'][f'comp_{comp_idx+1}'].append(corr['r2'])
                
                for comp_idx, corr in enumerate(corr_data['region_j'][:self.n_components]):
                    aggregated_corr[comparison_key]['region_j'][f'comp_{comp_idx+1}'].append(corr['r2'])
        
        self.aggregated_statistics['temporal_correlations'] = aggregated_corr

        # Print summary
        for comparison_key, data in aggregated_corr.items():
            n_vals = len(data['region_i']['comp_1'])
            print(f"  {comparison_key}: {n_vals} sessions")

        return aggregated_corr

    def aggregate_pairwise_tests(self) -> Dict:
        """
        Aggregate pairwise statistical test results across sessions.

        Collects p-values from Wilcoxon tests for cross-session analysis.
        Creates data structure for p-value heatmap visualization.

        Returns:
            Dictionary with aggregated p-values organized by comparison and component
        """
        print(f"\n" + "-" * 50)
        print("Aggregating pairwise test p-values")
        print("-" * 50)

        aggregated_pvals = {}

        for session_name, session_stats in self.session_statistics.items():
            if 'pairwise_tests' not in session_stats:
                continue

            for comparison_key, test_data in session_stats['pairwise_tests'].items():
                if comparison_key not in aggregated_pvals:
                    aggregated_pvals[comparison_key] = {
                        'region_i': {f'comp_{k+1}': [] for k in range(self.n_components)},
                        'region_j': {f'comp_{k+1}': [] for k in range(self.n_components)}
                    }

                # Collect Wilcoxon p-values per component
                for test_result in test_data.get('region_i', []):
                    comp_idx = test_result.get('component', 1) - 1
                    if comp_idx < self.n_components:
                        aggregated_pvals[comparison_key]['region_i'][f'comp_{comp_idx+1}'].append(
                            test_result.get('wilcoxon_p', 1.0)
                        )

                for test_result in test_data.get('region_j', []):
                    comp_idx = test_result.get('component', 1) - 1
                    if comp_idx < self.n_components:
                        aggregated_pvals[comparison_key]['region_j'][f'comp_{comp_idx+1}'].append(
                            test_result.get('wilcoxon_p', 1.0)
                        )

        self.aggregated_statistics['pairwise_tests'] = aggregated_pvals

        # Print summary
        for comparison_key, data in aggregated_pvals.items():
            n_vals = len(data['region_i']['comp_1']) if data['region_i']['comp_1'] else 0
            print(f"  {comparison_key}: {n_vals} sessions")

        return aggregated_pvals

    def compute_r2_based_significance(self) -> Dict:
        """
        Compute significance tests based on R² values across all sessions.

        Uses Wilcoxon signed-rank test to compare R² distributions between
        different trial type comparisons across sessions. For each pairwise
        comparison (e.g., cued_hit vs spont_hit), collects R² values from all
        sessions and tests if they are significantly different.

        Statistical approach:
        - For each component and region, collect R² values across sessions
        - Use Wilcoxon signed-rank test between R² distributions of different comparisons
        - Also test if R² is significantly above 0.5 (one-sample Wilcoxon)

        Returns:
            Dictionary containing:
            - 'pairwise_r2_tests': p-values from comparing R² between trial type pairs
            - 'r2_above_threshold': p-values testing if R² > 0.5
        """
        print(f"\n" + "-" * 50)
        print("Computing R²-based significance tests across sessions")
        print("-" * 50)

        if 'temporal_correlations' not in self.aggregated_statistics:
            print("  Warning: No temporal correlations available. Run aggregate_temporal_correlations() first.")
            return {}

        correlations = self.aggregated_statistics['temporal_correlations']
        n_comp_show = min(3, self.n_components)

        r2_significance = {
            'pairwise_r2_tests': {},  # Wilcoxon between different comparison R² values
            'r2_above_threshold': {}   # One-sample test if R² > 0.5
        }

        # Get comparison keys (e.g., 'cued_hit_long_vs_spont_hit_long')
        comparison_keys = list(correlations.keys())

        # Test 1: Compare R² values between different trial type comparisons
        # e.g., is R² for cued_vs_spont_hit different from R² for cued_vs_spont_miss?
        for i, comp1 in enumerate(comparison_keys):
            for comp2 in comparison_keys[i + 1:]:
                test_key = f"{comp1}_VS_{comp2}"
                r2_significance['pairwise_r2_tests'][test_key] = {
                    'region_i': {},
                    'region_j': {}
                }

                for comp_idx in range(n_comp_show):
                    comp_key = f'comp_{comp_idx + 1}'

                    # Get R² values for both comparisons
                    r2_comp1_i = np.array(correlations[comp1]['region_i'].get(comp_key, []))
                    r2_comp2_i = np.array(correlations[comp2]['region_i'].get(comp_key, []))
                    r2_comp1_j = np.array(correlations[comp1]['region_j'].get(comp_key, []))
                    r2_comp2_j = np.array(correlations[comp2]['region_j'].get(comp_key, []))

                    # Use minimum length for paired comparison
                    n_min_i = min(len(r2_comp1_i), len(r2_comp2_i))
                    n_min_j = min(len(r2_comp1_j), len(r2_comp2_j))

                    # Wilcoxon signed-rank test for region_i
                    if n_min_i >= 5:
                        try:
                            stat_i, p_i = wilcoxon(r2_comp1_i[:n_min_i], r2_comp2_i[:n_min_i])
                            mean_diff_i = np.mean(r2_comp1_i[:n_min_i] - r2_comp2_i[:n_min_i])
                        except ValueError:
                            p_i, mean_diff_i = 1.0, 0.0
                    else:
                        p_i, mean_diff_i = 1.0, 0.0

                    # Wilcoxon signed-rank test for region_j
                    if n_min_j >= 5:
                        try:
                            stat_j, p_j = wilcoxon(r2_comp1_j[:n_min_j], r2_comp2_j[:n_min_j])
                            mean_diff_j = np.mean(r2_comp1_j[:n_min_j] - r2_comp2_j[:n_min_j])
                        except ValueError:
                            p_j, mean_diff_j = 1.0, 0.0
                    else:
                        p_j, mean_diff_j = 1.0, 0.0

                    r2_significance['pairwise_r2_tests'][test_key]['region_i'][comp_key] = {
                        'wilcoxon_p': p_i,
                        'mean_diff': mean_diff_i,
                        'n_sessions': n_min_i,
                        'mean_r2_comp1': np.mean(r2_comp1_i) if len(r2_comp1_i) > 0 else 0,
                        'mean_r2_comp2': np.mean(r2_comp2_i) if len(r2_comp2_i) > 0 else 0
                    }

                    r2_significance['pairwise_r2_tests'][test_key]['region_j'][comp_key] = {
                        'wilcoxon_p': p_j,
                        'mean_diff': mean_diff_j,
                        'n_sessions': n_min_j,
                        'mean_r2_comp1': np.mean(r2_comp1_j) if len(r2_comp1_j) > 0 else 0,
                        'mean_r2_comp2': np.mean(r2_comp2_j) if len(r2_comp2_j) > 0 else 0
                    }

        # Test 2: Test if R² is significantly above 0.5 (one-sample test)
        threshold = 0.5
        for comparison in comparison_keys:
            r2_significance['r2_above_threshold'][comparison] = {
                'region_i': {},
                'region_j': {}
            }

            for comp_idx in range(n_comp_show):
                comp_key = f'comp_{comp_idx + 1}'

                r2_values_i = np.array(correlations[comparison]['region_i'].get(comp_key, []))
                r2_values_j = np.array(correlations[comparison]['region_j'].get(comp_key, []))

                # One-sample Wilcoxon test against threshold
                if len(r2_values_i) >= 5:
                    try:
                        # Test if values are significantly > threshold
                        shifted_i = r2_values_i - threshold
                        stat_i, p_i = wilcoxon(shifted_i, alternative='greater')
                    except ValueError:
                        p_i = 1.0
                else:
                    p_i = 1.0

                if len(r2_values_j) >= 5:
                    try:
                        shifted_j = r2_values_j - threshold
                        stat_j, p_j = wilcoxon(shifted_j, alternative='greater')
                    except ValueError:
                        p_j = 1.0
                else:
                    p_j = 1.0

                r2_significance['r2_above_threshold'][comparison]['region_i'][comp_key] = {
                    'wilcoxon_p': p_i,
                    'mean_r2': np.mean(r2_values_i) if len(r2_values_i) > 0 else 0,
                    'n_sessions': len(r2_values_i)
                }

                r2_significance['r2_above_threshold'][comparison]['region_j'][comp_key] = {
                    'wilcoxon_p': p_j,
                    'mean_r2': np.mean(r2_values_j) if len(r2_values_j) > 0 else 0,
                    'n_sessions': len(r2_values_j)
                }

        self.aggregated_statistics['r2_significance'] = r2_significance

        # Print summary
        print("\n  R² significance test results:")
        for comparison, data in r2_significance['r2_above_threshold'].items():
            for comp_idx in range(n_comp_show):
                comp_key = f'comp_{comp_idx + 1}'
                p_i = data['region_i'][comp_key]['wilcoxon_p']
                p_j = data['region_j'][comp_key]['wilcoxon_p']
                mean_i = data['region_i'][comp_key]['mean_r2']
                mean_j = data['region_j'][comp_key]['mean_r2']
                n_i = data['region_i'][comp_key]['n_sessions']
                n_j = data['region_j'][comp_key]['n_sessions']

                sig_i = '*' if p_i < 0.05 else ''
                sig_j = '*' if p_j < 0.05 else ''

                print(f"    {comparison} - {comp_key}:")
                print(f"      Region I: R²={mean_i:.3f}, p={p_i:.4f}{sig_i} (n={n_i})")
                print(f"      Region J: R²={mean_j:.3f}, p={p_j:.4f}{sig_j} (n={n_j})")

        return r2_significance

    def create_cross_session_projection_figure(
            self,
            figsize: Tuple[float, float] = (16, 12),
            save_fig: bool = True
    ) -> plt.Figure:
        """
        Create figure showing cross-session aggregated projections.
        
        Shows mean ± SEM across sessions for each trial type.
        
        Parameters:
            figsize: Figure dimensions
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        region_i, region_j = self.region_pair
        n_comp_show = min(3, self.n_components)
        n_sessions = len(self.session_projections)
        
        fig = plt.figure(figsize=figsize)
        fontsize = 12
        
        # Create grid: 2 rows × n_comp_show columns
        gs = fig.add_gridspec(2, n_comp_show, height_ratios=[1, 1],
                              hspace=0.35, wspace=0.3)
        
        # Row 1: Region i projections
        for comp_idx in range(n_comp_show):
            ax = fig.add_subplot(gs[0, comp_idx])
            self._plot_cross_session_projections(
                ax, 'u_mean', 'u_sem', comp_idx, region_i, fontsize
            )
            if comp_idx == 0:
                ax.set_ylabel(f'{region_i}\nProjection', fontsize=fontsize)
            ax.set_title(f'CCA Component {comp_idx + 1}', fontsize=fontsize + 1)
        
        # Row 2: Region j projections
        for comp_idx in range(n_comp_show):
            ax = fig.add_subplot(gs[1, comp_idx])
            self._plot_cross_session_projections(
                ax, 'v_mean', 'v_sem', comp_idx, region_j, fontsize
            )
            if comp_idx == 0:
                ax.set_ylabel(f'{region_j}\nProjection', fontsize=fontsize)
            ax.set_xlabel('Time (s)', fontsize=fontsize)
        
        # Add overall title
        plt.suptitle(
            f'Cross-Session Aggregated CCA Projections (n={n_sessions} sessions)\n'
            f'{region_i} vs {region_j} | CCA trained on {self.reference_type}',
            fontsize=fontsize + 2, fontweight='bold', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_fig:
            save_path = self.output_dir / f"cross_session_projection_{region_i}_{region_j}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def _plot_cross_session_projections(
            self,
            ax: plt.Axes,
            mean_key: str,
            sem_key: str,
            comp_idx: int,
            region_name: str,
            fontsize: int
    ) -> None:
        """Plot cross-session aggregated projections for a single component."""
        # Build session count annotation (n_a, n_b, n_c format)
        session_counts = []

        for trial_type in self.available_trial_types:
            if trial_type not in self.aggregated_projections:
                continue

            agg = self.aggregated_projections[trial_type]
            mean_proj = np.abs(agg[mean_key][:, comp_idx])
            sem_proj = agg[sem_key][:, comp_idx]

            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')
            linewidth = 2 if trial_type == self.reference_type else 1.5

            # Shorter label for legend
            short_label = trial_type.replace('_long', '').replace('_', ' ')
            ax.plot(self.time_bins, mean_proj, color=color, linewidth=linewidth,
                    label=f'{short_label} (n={agg["n_sessions"]})',
                    alpha=0.9)
            ax.fill_between(self.time_bins, mean_proj - sem_proj, mean_proj + sem_proj,
                            alpha=0.2, color=color)

            # Collect session counts for annotation
            session_counts.append(f'{agg["n_sessions"]}')

        # Add stimulus onset marker
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvspan(0, 1.5, alpha=0.05, color='gray')
        ax.set_ylim([0, 6])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=fontsize - 2)
        ax.legend(fontsize=fontsize - 3, loc='upper right')
    
    def create_temporal_correlation_boxplot_figure(
            self,
            figsize: Tuple[float, float] = (14, 10),
            save_fig: bool = True
    ) -> plt.Figure:
        """
        Create figure with boxplots of temporal R² across sessions.
        
        Parameters:
            figsize: Figure dimensions
            save_fig: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        region_i, region_j = self.region_pair
        
        if 'temporal_correlations' not in self.aggregated_statistics:
            print("No temporal correlations to plot")
            return None
        
        correlations = self.aggregated_statistics['temporal_correlations']
        n_comp_show = min(3, self.n_components)
        
        fig = plt.figure(figsize=figsize)
        fontsize = 12
        
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        
        # Plot for region i
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_correlation_boxplots(ax1, correlations, 'region_i', region_i, 
                                        n_comp_show, fontsize)
        
        # Plot for region j
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_correlation_boxplots(ax2, correlations, 'region_j', region_j,
                                        n_comp_show, fontsize)
        
        plt.suptitle(
            f'Cross-Session Temporal R² Distribution\n'
            f'{region_i} vs {region_j} | n={len(self.session_projections)} sessions',
            fontsize=fontsize + 2, fontweight='bold', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_fig:
            save_path = self.output_dir / f"temporal_correlation_boxplot_{region_i}_{region_j}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def _plot_correlation_boxplots(
            self,
            ax: plt.Axes,
            correlations: Dict,
            region_key: str,
            region_name: str,
            n_comp_show: int,
            fontsize: int
    ) -> None:
        """Plot boxplots of temporal R² for a single region."""
        comparisons = list(correlations.keys())
        n_comparisons = len(comparisons)
        
        positions = []
        data_to_plot = []
        colors = []
        labels = []
        
        for comp_idx in range(n_comp_show):
            comp_key = f'comp_{comp_idx + 1}'
            
            for idx, comparison in enumerate(comparisons):
                r2_values = correlations[comparison][region_key][comp_key]
                
                if r2_values:
                    # Position: grouped by component, offset by comparison
                    pos = comp_idx * (n_comparisons + 1) + idx
                    positions.append(pos)
                    data_to_plot.append(r2_values)
                    
                    # Color based on comparison trial type
                    other_type = comparison.replace(f"{self.reference_type}_vs_", "")
                    colors.append(TRIAL_TYPE_COLORS.get(other_type, 'gray'))
                    
                    if comp_idx == 0:
                        labels.append(other_type.replace('_', ' '))
        
        if not data_to_plot:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return
        
        # Create boxplots
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set x-axis labels
        comp_positions = [(i * (n_comparisons + 1) + (n_comparisons - 1) / 2) 
                          for i in range(n_comp_show)]
        ax.set_xticks(comp_positions)
        ax.set_xticklabels([f'Comp {i + 1}' for i in range(n_comp_show)], fontsize=fontsize)
        
        ax.set_ylabel('Temporal R²', fontsize=fontsize)
        ax.set_title(f'{region_name}', fontsize=fontsize + 1)
        ax.set_ylim([0, 1.0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend
        if labels:
            from matplotlib.patches import Patch
            unique_labels = list(dict.fromkeys(labels))
            unique_colors = [TRIAL_TYPE_COLORS.get(l.replace(' ', '_'), 'gray') for l in unique_labels]
            legend_elements = [Patch(facecolor=c, alpha=0.7, label=l) 
                              for c, l in zip(unique_colors, unique_labels)]
            ax.legend(handles=legend_elements, fontsize=fontsize - 2, loc='lower right')


# =============================================================================
# UPPER TRIANGLE SUMMARY FIGURE CLASSES
# =============================================================================

class CrossTrialTypeSummaryVisualizer:
    """
    Visualizer for creating upper-triangle summary figures across all region pairs.
    
    Following the layout from CCA_test_all.py, this class creates:
    1. Figure 1: First component projections for all region pairs
    2. Figure 2: R² boxplots of top 3 components comparing trial types
    
    The upper triangle format allows visualization of all unique region pairs
    in a single figure, with anatomical ordering ensuring consistent layout.
    """
    
    def __init__(
            self,
            base_dir: str,
            reference_type: str = 'cued_hit_long',
            n_components: int = 5,
            min_sessions: int = MIN_SESSIONS_THRESHOLD
    ):
        """
        Initialize the summary visualizer.
        
        Parameters:
            base_dir: Root directory for Oxford dataset
            reference_type: Trial type for CCA weight training
            n_components: Number of CCA components to analyze
            min_sessions: Minimum sessions required for a pair to be included
        """
        self.base_dir = Path(base_dir)
        self.reference_type = reference_type
        self.n_components = n_components
        self.min_sessions = min_sessions
        
        # Cross-session analyzers for each region pair
        self.pair_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer] = {}
        
        # Available regions and time axis
        self.available_regions: List[str] = []
        self.time_bins: Optional[np.ndarray] = None
        
        # Output directory
        self.output_dir = self.base_dir / 'Paper_output' / 'cross_trial_type_cca' / 'summary_figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("Cross-Trial-Type Summary Visualizer")
        print("=" * 70)
        print(f"Reference condition: {reference_type}")
        print(f"Minimum sessions: {min_sessions}")
    
    def add_pair_analyzer(
            self,
            pair_analyzer: CrossSessionCCAAnalyzer
    ) -> None:
        """
        Add a cross-session analyzer for a region pair.
        
        Parameters:
            pair_analyzer: Completed CrossSessionCCAAnalyzer with aggregated results
        """
        pair_key = pair_analyzer.region_pair
        self.pair_analyzers[pair_key] = pair_analyzer
        
        # Update available regions
        for region in pair_key:
            if region not in self.available_regions:
                self.available_regions.append(region)
        
        # Update time bins
        if self.time_bins is None and pair_analyzer.time_bins is not None:
            self.time_bins = pair_analyzer.time_bins
        
        print(f"  Added pair: {pair_key[0]} vs {pair_key[1]}")
    
    def _get_ordered_regions(self) -> List[str]:
        """Return available regions in anatomical order."""
        return [r for r in ANATOMICAL_ORDER if r in self.available_regions]
    
    def create_projection_matrix_figure(
            self,
            figsize: Tuple[float, float] = (40, 40),
            component_idx: int = 0,
            save_fig: bool = True
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create two upper-triangle figures showing component projections.

        Creates separate figures for:
        - Figure 1 (row): Projections for the region in the row position (u data)
        - Figure 2 (column): Projections for the region in the column position (v data)

        Parameters:
            figsize: Figure dimensions
            component_idx: Which CCA component to display (0-indexed)
            save_fig: Whether to save the figure

        Returns:
            Tuple of (row_figure, column_figure)
        """
        print(f"\nCreating projection matrix figures (Component {component_idx + 1})...")

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
                        ax.text(0.5, 0.5, region_i, ha='center', va='center',
                                fontsize=32, fontweight='bold')
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.axis('off')
                    elif i > j:
                        # Lower triangle: hide
                        ax.axis('off')
                    else:
                        # Upper triangle: plot cross-trial-type projections
                        # For row figure: plot data from row region (region_i → u)
                        # For column figure: plot data from column region (region_j → v)
                        self._plot_pair_projections_single_region(
                            ax, region_i, region_j, component_idx, fig_type
                        )

            region_label = "Row Region" if fig_type == 'row' else "Column Region"
            fig.suptitle(
                # f'Cross-Trial-Type CCA Projections - Component {component_idx + 1}\n'
                f'Component {component_idx + 1}| {region_label} | Reference: {self.reference_type} | n ≥ {self.min_sessions} sessions',
                fontsize=48, fontweight='bold', y=0.995
            )

            plt.tight_layout(rect=[0, 0.01, 1, 0.99])

            if save_fig:
                save_path = self.output_dir / f"projection_matrix_comp{component_idx + 1}_{fig_type}_region.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")

            figures.append(fig)
            plt.close(fig)

        return tuple(figures)

    def _plot_pair_projections_single_region(
            self,
            ax: plt.Axes,
            region_i: str,
            region_j: str,
            component_idx: int,
            which_region: str
    ) -> None:
        """
        Plot cross-trial-type projections for a single region from a pair.

        Parameters:
            ax: Matplotlib axes
            region_i: Row region name (in upper triangle layout)
            region_j: Column region name (in upper triangle layout)
            component_idx: Component to plot (0-indexed)
            which_region: 'row' for region_i (u data), 'column' for region_j (v data)
        """
        # Find pair analyzer (check both orderings)
        pair_key = sort_pair_by_anatomy(region_i, region_j)
        pair_analyzer = self.pair_analyzers.get(pair_key)

        if pair_analyzer is None:
            # Try alternate key
            alt_key = (pair_key[1], pair_key[0])
            pair_analyzer = self.pair_analyzers.get(alt_key)

        if pair_analyzer is None or len(pair_analyzer.aggregated_projections) < 2:
            ax.set_visible(False)
            return

        n_sessions = pair_analyzer.aggregated_projections.get(
            self.reference_type, {}).get('n_sessions', 0)

        if n_sessions < self.min_sessions:
            ax.set_visible(False)
            return

        # Determine which data to use based on region position
        # pair_key is in canonical (sorted) order
        # pair_key[0] = row region in canonical order → u data
        # pair_key[1] = column region in canonical order → v data
        if which_region == 'row':
            # We want data for region_i (the row in the matrix)
            # Check if region_i is pair_key[0] (u) or pair_key[1] (v)
            if region_i == pair_key[0]:
                mean_key, sem_key = 'u_mean', 'u_sem'
            else:
                mean_key, sem_key = 'v_mean', 'v_sem'
            display_region = region_i
        else:  # 'column'
            # We want data for region_j (the column in the matrix)
            if region_j == pair_key[1]:
                mean_key, sem_key = 'v_mean', 'v_sem'
            else:
                mean_key, sem_key = 'u_mean', 'u_sem'
            display_region = region_j

        # Plot projections for each trial type and collect session counts
        time_vec = pair_analyzer.time_bins
        session_count_labels = []

        # Define short labels for trial types (a, b, c style)
        trial_type_short = {
            'cued_hit_long': 'ch',
            'spont_hit_long': 'sh',
            'spont_miss_long': 'sm'
        }

        # Determine session data key based on which region we're plotting
        if which_region == 'row':
            if region_i == pair_key[0]:
                session_key = 'u_sessions'
            else:
                session_key = 'v_sessions'
        else:  # 'column'
            if region_j == pair_key[1]:
                session_key = 'v_sessions'
            else:
                session_key = 'u_sessions'

        for trial_type in pair_analyzer.available_trial_types:
            if trial_type not in pair_analyzer.aggregated_projections:
                continue

            agg = pair_analyzer.aggregated_projections[trial_type]

            mean_proj = agg[mean_key][:, component_idx]
            sem_proj = agg[sem_key][:, component_idx]

            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')
            linestyle = '-' if trial_type == self.reference_type else '-'
            linewidth = 2 if trial_type == self.reference_type else 1.0
            short_label = trial_type_short.get(trial_type, trial_type.replace('_', ' '))
            n_sess = agg['n_sessions']

            # Plot individual session lines (thin, low alpha)
            if session_key in agg:
                session_data = agg[session_key]  # Shape: (n_sessions, n_time, n_components)
                for sess_idx in range(session_data.shape[0]):
                    sess_proj = session_data[sess_idx, :, component_idx]
                    ax.plot(self.time_bins, sess_proj, color=color, linewidth=0.5,
                            alpha=0.2, zorder=1)

            # Plot mean projection on top
            if trial_type == self.reference_type:
                ax.plot(self.time_bins, mean_proj, color=color, linestyle=linestyle,
                        linewidth=linewidth, label=f'{short_label} (n={n_sess})',
                        alpha=0.8, zorder=3)
                ax.fill_between(self.time_bins, mean_proj - sem_proj, mean_proj + sem_proj,
                                alpha=0.15, color=color, zorder=2)
            else:
                ax.plot(self.time_bins, mean_proj, color=color, linestyle=linestyle,
                        linewidth=linewidth, label=f'{short_label} (n={n_sess})',
                        alpha=0.4, zorder=3)
                ax.fill_between(self.time_bins, mean_proj - sem_proj, mean_proj + sem_proj,
                                alpha=0.15, color=color, zorder=2)

            # Collect session count for annotation
            session_count_labels.append(f'n_{short_label[:2]}={n_sess}')

        # Reference line at t=0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=3)

        # Formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.8, linestyle=':', linewidth=2)
        ax.set_yticks(np.arange(0, 10, 2))
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)
        ax.set_ylim([-1.5, 5])
        ax.set_xticks([-1.5, 0, 2, 3])
        ax.set_xticklabels(['-1.5', '0', '2', '3'], fontsize=20)
        ax.tick_params(axis='both', which='major', width=2, length=8)

        for spine in ax.spines.values():
            spine.set_linewidth(3)

        # Add session counts annotation for each trial type (n_a, n_b, n_c format)
        session_count_text = ', '.join(session_count_labels)
        ax.text(0.02, 0.98, f'{display_region}\n{session_count_text}', transform=ax.transAxes,
                fontsize=12, va='top', ha='left')
    
    def create_r2_boxplot_matrix_figure(
            self,
            figsize: Tuple[float, float] = (40, 40),
            save_fig: bool = True
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create two upper-triangle figures showing R² boxplots for top 3 components.

        Creates separate figures for:
        - Figure 1 (row): R² for the region in the row position
        - Figure 2 (column): R² for the region in the column position

        Each cell shows boxplots comparing cued_hit_long vs spont_hit_long
        and spont_miss_long across sessions.

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
                        ax.text(0.5, 0.5, region_i, ha='center', va='center',
                                fontsize=32, fontweight='bold')
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.axis('off')
                    elif i > j:
                        # Lower triangle: hide
                        ax.axis('off')
                    else:
                        # Upper triangle: plot R² boxplots for single region
                        self._plot_pair_r2_boxplots_single_region(
                            ax, region_i, region_j, fig_type
                        )

            region_label = "Row Region" if fig_type == 'row' else "Column Region"
            fig.suptitle(
                f'Cross-Session Temporal R² Distribution (Top 3 Components)\n'
                f'{region_label} | Reference: {self.reference_type} | n ≥ {self.min_sessions} sessions',
                fontsize=48, fontweight='bold', y=0.995
            )

            plt.tight_layout(rect=[0, 0.01, 1, 0.99])

            if save_fig:
                save_path = self.output_dir / f"r2_boxplot_matrix_{fig_type}_region.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")

            figures.append(fig)
            plt.close(fig)

        return tuple(figures)

    def _plot_pair_r2_boxplots_single_region(
            self,
            ax: plt.Axes,
            region_i: str,
            region_j: str,
            which_region: str
    ) -> None:
        """
        Plot R² boxplots for a single region from a pair.

        Parameters:
            ax: Matplotlib axes
            region_i: Row region name (in upper triangle layout)
            region_j: Column region name (in upper triangle layout)
            which_region: 'row' for region_i, 'column' for region_j
        """
        # Find pair analyzer
        pair_key = sort_pair_by_anatomy(region_i, region_j)
        pair_analyzer = self.pair_analyzers.get(pair_key)

        if pair_analyzer is None:
            alt_key = (pair_key[1], pair_key[0])
            pair_analyzer = self.pair_analyzers.get(alt_key)

        if pair_analyzer is None:
            ax.set_visible(False)
            return

        if 'temporal_correlations' not in pair_analyzer.aggregated_statistics:
            ax.set_visible(False)
            return

        correlations = pair_analyzer.aggregated_statistics['temporal_correlations']
        n_comp_show = min(3, self.n_components)

        # Determine which region's data to use
        # pair_key[0] → region_i in canonical order → 'region_i' key in stats
        # pair_key[1] → region_j in canonical order → 'region_j' key in stats
        if which_region == 'row':
            # We want data for region_i (the row in the matrix)
            if region_i == pair_key[0]:
                region_key = 'region_i'
            else:
                region_key = 'region_j'
            display_region = region_i
        else:  # 'column'
            # We want data for region_j (the column in the matrix)
            if region_j == pair_key[0]:
                region_key = 'region_i'
            else:
                region_key = 'region_j'
            display_region = region_j

        # Prepare data for boxplots
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
                # Get R² for the specific region only
                r2_values = correlations[comparison][region_key].get(comp_key, [])

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
        ax.set_xticklabels([f'C{i+1}' for i in range(n_comp_show)], fontsize=18)

        ax.set_ylim([0, 1.05])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0', '0.5', '1'], fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Add region annotation
        ax.text(0.02, 0.98, display_region, transform=ax.transAxes,
                fontsize=14, va='top', ha='left')

    def create_pvalue_heatmap_matrix_figure(
            self,
            figsize: Tuple[float, float] = (40, 40),
            save_fig: bool = True
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create two upper-triangle figures showing p-value heatmaps.

        Each subplot contains a 3×3 grid showing p-values for components 1-3
        across the three trial type comparisons.

        Creates separate figures for:
        - Figure 1 (row): P-values for the region in the row position
        - Figure 2 (column): P-values for the region in the column position

        Parameters:
            figsize: Figure dimensions
            save_fig: Whether to save the figure

        Returns:
            Tuple of (row_figure, column_figure)
        """
        print("\nCreating cross-session p-value heatmap matrix figures...")

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
                        ax.text(0.5, 0.5, region_i, ha='center', va='center',
                                fontsize=32, fontweight='bold')
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.axis('off')
                    elif i > j:
                        # Lower triangle: hide
                        ax.axis('off')
                    else:
                        # Upper triangle: plot p-value heatmap for single region
                        self._plot_pair_pvalue_heatmap(
                            ax, region_i, region_j, fig_type
                        )

            region_label = "Row Region" if fig_type == 'row' else "Column Region"
            fig.suptitle(
                f'Cross-Session P-Value Heatmap (Components 1-3)\n'
                f'{region_label} | Reference: {self.reference_type} | n ≥ {self.min_sessions} sessions',
                fontsize=48, fontweight='bold', y=0.995
            )

            plt.tight_layout(rect=[0, 0.01, 1, 0.99])

            if save_fig:
                save_path = self.output_dir / f"pvalue_heatmap_matrix_{fig_type}_region.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")

            figures.append(fig)
            plt.close(fig)

        return tuple(figures)

    def _plot_pair_pvalue_heatmap(
            self,
            ax: plt.Axes,
            region_i: str,
            region_j: str,
            which_region: str
    ) -> None:
        """
        Plot cross-session p-value heatmap for a single region from a pair.

        Creates a grid where:
        - Rows: Trial type comparisons (R² tests for cued vs spont_hit, cued vs spont_miss)
        - Columns: CCA Components 1-3

        The heatmap shows -log10(p-value) from Wilcoxon tests on R² values across sessions.

        Parameters:
            ax: Matplotlib axes
            region_i: Row region name (in upper triangle layout)
            region_j: Column region name (in upper triangle layout)
            which_region: 'row' for region_i, 'column' for region_j
        """
        # Find pair analyzer
        pair_key = sort_pair_by_anatomy(region_i, region_j)
        pair_analyzer = self.pair_analyzers.get(pair_key)

        if pair_analyzer is None:
            ax.set_visible(False)
            return

        # Use R²-based significance tests (Wilcoxon on R² across sessions)
        if 'r2_significance' not in pair_analyzer.aggregated_statistics:
            ax.set_visible(False)
            return

        r2_significance = pair_analyzer.aggregated_statistics['r2_significance']
        n_comp_show = min(3, self.n_components)

        # Determine which region's data to use
        if which_region == 'row':
            if region_i == pair_key[0]:
                region_key = 'region_i'
            else:
                region_key = 'region_j'
            display_region = region_i
        else:  # 'column'
            if region_j == pair_key[0]:
                region_key = 'region_i'
            else:
                region_key = 'region_j'
            display_region = region_j

        # Get comparisons from R² above threshold tests
        r2_threshold_tests = r2_significance.get('r2_above_threshold', {})
        comparisons = list(r2_threshold_tests.keys())
        n_comparisons = len(comparisons)

        if n_comparisons == 0:
            ax.set_visible(False)
            return

        # Build p-value matrix: (n_comparisons, n_components)
        # Use Wilcoxon p-values from R² significance tests
        p_matrix = np.ones((n_comparisons, n_comp_show))

        for i, comparison in enumerate(comparisons):
            for j in range(n_comp_show):
                comp_key = f'comp_{j + 1}'
                test_result = r2_threshold_tests[comparison][region_key].get(comp_key, {})

                if test_result:
                    p_matrix[i, j] = test_result.get('wilcoxon_p', 1.0)

        # Convert to -log10 for visualization (higher = more significant)
        log_p_matrix = -np.log10(p_matrix + 1e-10)

        # Plot heatmap
        im = ax.imshow(log_p_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=3.0)

        # Add text annotations
        for i in range(n_comparisons):
            for j in range(n_comp_show):
                p_val = log_p_matrix[i, j]
                # Show significance markers
                text = f'{p_val:.2f}'
                if p_val > 1.3:  # p < 0.05
                    text += '*'
                if p_val > 2.0:  # p < 0.01
                    text += '*'
                if p_val > 3.0:  # p < 0.001
                    text += '*'
                text_color = 'white' if p_val > 1.5 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=14, color=text_color, fontweight='bold')

        # Set axis labels
        ax.set_xticks(np.arange(n_comp_show))
        ax.set_yticks(np.arange(n_comparisons))
        ax.set_xticklabels([f'C{i+1}' for i in range(n_comp_show)], fontsize=16)

        # Format comparison labels for y-axis (simplified for R² tests)
        y_labels = []
        for comp in comparisons:
            # Simplify trial type names
            short_name = comp.replace('_long', '').replace('cued_hit', 'cued').replace('spont_', 's_')
            parts = short_name.split('_vs_')
            if len(parts) == 2:
                label = f"R²>{0.5}\n{parts[0]} vs {parts[1]}"
            else:
                label = f"R²>{0.5}\n{short_name}"
            y_labels.append(label)
        ax.set_yticklabels(y_labels, fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Add region annotation
        ax.set_title(f'{display_region}', fontsize=16, fontweight='bold')


# =============================================================================
# PIPELINE ORCHESTRATION CLASS
# =============================================================================

class CrossTrialTypeCCAPipeline:
    """
    Complete pipeline for cross-trial-type CCA analysis.

    This class orchestrates the entire analysis workflow for multiple
    sessions and region pairs, with optional cross-session aggregation.
    """

    def __init__(self, config: Dict):
        """
        Initialize the pipeline with configuration.

        Parameters:
            config: Dictionary containing:
                - base_dir: Root directory for Oxford dataset
                - sessions: List of session names to analyze
                - region_pairs: List of (region_i, region_j) tuples
                - reference_type: Trial type for CCA training
                - n_components: Number of CCA components
                - output_dir: Base output directory
                - enable_cross_session: Whether to perform cross-session aggregation
        """
        self.config = config
        self.analyzers: Dict[str, CrossTrialTypeCCAAnalyzer] = {}
        self.results: Dict = {}
        
        # Cross-session analyzers (organized by region pair)
        self.cross_session_analyzers: Dict[Tuple[str, str], CrossSessionCCAAnalyzer] = {}
        
        # Summary visualizer
        self.summary_visualizer: Optional[CrossTrialTypeSummaryVisualizer] = None

        # Validate config
        required_keys = ['base_dir', 'sessions']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        # Set defaults
        config.setdefault('reference_type', 'cued_hit_long')
        config.setdefault('n_components', 5)
        config.setdefault('region_pairs', None)  # Auto-detect if None
        config.setdefault('enable_cross_session', True)
        config.setdefault('min_sessions', MIN_SESSIONS_THRESHOLD)

        print("=" * 70)
        print("Cross-Trial-Type CCA Pipeline Initialized")
        print("=" * 70)
        self._print_config()

    def _print_config(self) -> None:
        """Print configuration summary."""
        print("Configuration:")
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
            analyzer = CrossTrialTypeCCAAnalyzer(
                base_dir=self.config['base_dir'],
                session_name=session_name,
                reference_type=self.config['reference_type'],
                n_components=self.config['n_components']
            )

            # Load all trial types
            if not analyzer.load_all_trial_types():
                print(f"Failed to load trial types for {session_name}")
                return False

            # Determine region pairs to analyze
            region_pairs = self.config.get('region_pairs')
            if region_pairs is None:
                region_pairs = [(r1, r2) for r1, r2, _ in analyzer.available_pairs]

            # Analyze each region pair
            for region_pair in region_pairs:
                region_i, region_j = region_pair
                print(f"\n--- Analyzing {region_i} vs {region_j} ---")

                # Extract neural data
                if not analyzer.extract_neural_data(region_pair):
                    print(f"  Skipping - insufficient data")
                    continue

                # Extract CCA weights from reference
                if not analyzer.extract_cca_weights(region_pair):
                    print(f"  Skipping - CCA weights not available")
                    continue

                # Compute projections
                if not analyzer.compute_projections():
                    print(f"  Skipping - projection failed")
                    continue

                # Compute statistics
                analyzer.compute_statistics()

                # Add to cross-session analyzer if enabled
                if self.config.get('enable_cross_session', True):
                    pair_key = sort_pair_by_anatomy(region_i, region_j)

                    # Check if pair was flipped by sort_pair_by_anatomy
                    # If flipped, u/v data need to be swapped to maintain correct region mapping
                    is_flipped = (pair_key[0] != region_i)

                    if pair_key not in self.cross_session_analyzers:
                        self.cross_session_analyzers[pair_key] = CrossSessionCCAAnalyzer(
                            base_dir=self.config['base_dir'],
                            region_pair=pair_key,
                            reference_type=self.config['reference_type'],
                            n_components=self.config['n_components'],
                            min_sessions=self.config.get('min_sessions', MIN_SESSIONS_THRESHOLD)
                        )

                    self.cross_session_analyzers[pair_key].add_session_result(
                        session_name, analyzer, swap_uv=is_flipped
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
        
        This should be called after run_all_sessions() to aggregate
        results across sessions and create the upper-triangle summary figures.
        """
        print("\n" + "=" * 70)
        print("CROSS-SESSION AGGREGATION")
        print("=" * 70)
        
        min_sessions = self.config.get('min_sessions', MIN_SESSIONS_THRESHOLD)
        
        # Initialize summary visualizer
        self.summary_visualizer = CrossTrialTypeSummaryVisualizer(
            base_dir=self.config['base_dir'],
            reference_type=self.config['reference_type'],
            n_components=self.config['n_components'],
            min_sessions=min_sessions
        )
        
        # Process each region pair
        valid_pairs = 0
        for pair_key, cross_session_analyzer in self.cross_session_analyzers.items():
            n_sessions = len(cross_session_analyzer.session_projections)
            
            if n_sessions < min_sessions:
                print(f"  {pair_key[0]} vs {pair_key[1]}: {n_sessions} sessions (skipping, < {min_sessions})")
                continue
            
            print(f"\n  Aggregating {pair_key[0]} vs {pair_key[1]} ({n_sessions} sessions)...")
            
            # Perform aggregation
            if cross_session_analyzer.aggregate_projections():
                cross_session_analyzer.aggregate_temporal_correlations()
                cross_session_analyzer.aggregate_pairwise_tests()
                # Compute R²-based significance using Wilcoxon test across sessions
                cross_session_analyzer.compute_r2_based_significance()

                # Add to summary visualizer
                self.summary_visualizer.add_pair_analyzer(cross_session_analyzer)
                valid_pairs += 1

        print(f"\nValid pairs for summary: {valid_pairs}")

        # Create summary figures if we have enough pairs
        if valid_pairs >= 1:
            print("\nCreating summary figures...")

            # Figures 1a & 1b: First component projections (row and column regions)
            self.summary_visualizer.create_projection_matrix_figure(
                component_idx=0,
                save_fig=True
            )

            # Figures 2a & 2b: R² boxplots (row and column regions)
            self.summary_visualizer.create_r2_boxplot_matrix_figure(
                save_fig=True
            )

            # # Figures 3a & 3b: P-value heatmaps (row and column regions)
            # self.summary_visualizer.create_pvalue_heatmap_matrix_figure(
            #     save_fig=True
            # )
        else:
            print("Not enough valid pairs for summary figures")


# =============================================================================
# MAIN DEMONSTRATION FUNCTION
# =============================================================================

def main():
    """Demonstration of cross-trial-type CCA analysis."""

    # Configuration
    config = {
        'base_dir': '/Users/shengyuancai/Downloads/Oxford_dataset',
        'sessions': ['yp010_220209'],  # Example session
        'reference_type': 'cued_hit_long',
        'n_components': 5,
        'region_pairs': [
            ('mPFC', 'STR'),
            ('MOp', 'STR'),
            ('MOs', 'STR'),
            ('ORB', 'STR')
        ],
        'enable_cross_session': True,
        'min_sessions': MIN_SESSIONS_THRESHOLD
    }

    print("=" * 70)
    print("Cross-Trial-Type CCA Analysis Pipeline")
    print("=" * 70)

    # Initialize and run pipeline
    pipeline = CrossTrialTypeCCAPipeline(config)
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

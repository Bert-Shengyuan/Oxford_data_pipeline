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

Statistical Comparison:
-----------------------
1. Peak amplitude comparison: max projection within post-stimulus window
2. Temporal correlation: correlation between projection time courses
3. Wilcoxon signed-rank test: non-parametric paired comparison
4. Bootstrap confidence intervals: robust uncertainty estimation

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
    'spont_miss_long': 'sessions_spont_miss_long_results'
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

        # Time axis
        self.time_bins: Optional[np.ndarray] = None

        # Output directory
        self.output_dir = Path(base_dir) / 'Paper_output' / 'cross_trial_type_cca' / session_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Cross-Trial-Type CCA Analyzer")
        print("=" * 70)
        print(f"Session: {session_name}")
        print(f"Reference condition: {reference_type}")
        print(f"Components to analyze: {n_components}")
        print(f"Output directory: {self.output_dir}")

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
            elif 'time_axis' in region_data:
                self.time_bins = np.array(region_data['time_axis']).flatten()

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

        # Find the correct pair index
        pair_idx = None
        for r1, r2, idx in self.available_pairs:
            if (r1 == region_i and r2 == region_j) or (r1 == region_j and r2 == region_i):
                pair_idx = idx
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

        # Store CCA weights
        self.cca_weights = {
            'region_i': region_i,
            'region_j': region_j,
            'A': A_matrix,
            'B': B_matrix,
            'mu_A': mu_A,
            'mu_B': mu_B,
            'pair_idx': pair_idx,
            'pair_result': pair_result
        }

        print(f"  Region {region_i}: A matrix shape {A_matrix.shape}")
        print(f"  Region {region_j}: B matrix shape {B_matrix.shape}")

        # Extract cross-validated R² values for reference
        if 'cv_results' in pair_result:
            cv_results = pair_result['cv_results']
            mean_cv_R2 = cv_results.get('mean_cv_R2', [])
            if len(mean_cv_R2) > 0:
                print(f"  Reference CV-R²: {mean_cv_R2[:self.n_components]}")

        return True

    # =========================================================================
    # PROJECTION COMPUTATION
    # =========================================================================

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
            if trial_type not in self.neural_data:
                continue

            neural = self.neural_data[trial_type]

            # Get neural activity for each region
            X_i = neural[region_i]  # shape: (n_trials, n_neurons, n_time)
            X_j = neural[region_j]


            X_i_mean_test = np.mean(X_i, axis=0).T # (n_time, n_neurons)
            X_j_mean_test = np.mean(X_j, axis=0).T


            n_trials, n_neurons, n_timepoints = X_i.shape

            region_i_sampled_p = np.transpose(X_i, (1, 2, 0))
            region_j_sampled_p = np.transpose(X_j, (1, 2, 0))

            # Reshape and transpose: (neurons, timepoints, trials) → (trials×timepoints, neurons)
            X = region_i_sampled_p.reshape(n_neurons, n_trials * n_timepoints).T
            Y = region_j_sampled_p.reshape(n_neurons, n_trials * n_timepoints).T

            X_i_norm = zscore(X, axis=0, nan_policy='omit')
            X_j_norm = zscore(Y, axis=0, nan_policy='omit')
            X_i_norm = np.nan_to_num(X_i_norm, nan=0.0)
            X_j_norm = np.nan_to_num(X_j_norm, nan=0.0)



            # Z-score normalize across time
            # X_i_norm = zscore(X_i_mean, axis=0, nan_policy='omit')
            # X_j_norm = zscore(X_j_mean, axis=0, nan_policy='omit')
            # X_i_norm = np.nan_to_num(X_i_norm, nan=0.0)
            # X_j_norm = np.nan_to_num(X_j_norm, nan=0.0)


            # Project onto CCA subspace: (n_time, n_components)
            u = (X_i_norm @ A[:, :self.n_components]).T
            v = (X_j_norm @ B[:, :self.n_components]).T
            u_trials_p = u.reshape(self.n_components, n_timepoints, n_trials)
            v_trials_p = v.reshape(self.n_components, n_timepoints, n_trials)

            # # Compute trial-averaged activity: (n_time, n_neurons)
            u_final = np.mean(u_trials_p,axis=2).T
            v_final = np.mean(v_trials_p,axis=2).T

            u_trials = np.transpose(u_trials_p, (2, 1, 0))
            v_trials = np.transpose(v_trials_p, (2, 1, 0))

            # Also compute trial-level projections for statistics
            # n_trials = X_i.shape[0]
            # u_trials = np.zeros((n_trials, X_i.shape[2], self.n_components))
            # v_trials = np.zeros((n_trials, X_j.shape[2], self.n_components))
            # for trial_idx in range(n_trials):
            #     X_i_trial = X_i[trial_idx, :, :].T  # (n_time, n_neurons)
            #     X_j_trial = X_j[trial_idx, :, :].T
            #
            #     X_i_trial_norm = zscore(X_i_trial, axis=0, nan_policy='omit')
            #     X_j_trial_norm = zscore(X_j_trial, axis=0, nan_policy='omit')
            #     X_i_trial_norm = np.nan_to_num(X_i_trial_norm, nan=0.0)
            #     X_j_trial_norm = np.nan_to_num(X_j_trial_norm, nan=0.0)
            #
            #     u_trials[trial_idx] = X_i_trial_norm @ A[:, :self.n_components]
            #     v_trials[trial_idx] = X_j_trial_norm @ B[:, :self.n_components]

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
            # print(f"    Region i R²: {[f'{c['r2']:.3f}' for c in corr_u[:3]]}")
            # print(f"    Region j R²: {[f'{c['r2']:.3f}' for c in corr_v[:3]]}")

        # Pairwise statistical tests on peak values
        trial_types = list(self.projections.keys())

        for i, type1 in enumerate(trial_types):
            for type2 in trial_types[i + 1:]:
                comparison_key = f"{type1}_vs_{type2}"

                test_results = {
                    'region_i': [],
                    'region_j': []
                }

                proj1 = self.projections[type1]
                proj2 = self.projections[type2]

                for comp_idx in range(self.n_components):
                    # Compute per-trial peak values
                    peaks1_u = np.max(np.abs(proj1['u_trials'][:, post_stim_mask, comp_idx]), axis=1)
                    peaks2_u = np.max(np.abs(proj2['u_trials'][:, post_stim_mask, comp_idx]), axis=1)
                    peaks1_v = np.max(np.abs(proj1['v_trials'][:, post_stim_mask, comp_idx]), axis=1)
                    peaks2_v = np.max(np.abs(proj2['v_trials'][:, post_stim_mask, comp_idx]), axis=1)

                    # Use minimum number of trials for paired comparison
                    n_min = min(len(peaks1_u), len(peaks2_u))

                    if n_min >= 5:
                        # Wilcoxon signed-rank test
                        stat_u, p_wilcox_u = wilcoxon(peaks1_u[:n_min], peaks2_u[:n_min])
                        stat_v, p_wilcox_v = wilcoxon(peaks1_v[:n_min], peaks2_v[:n_min])

                        # Paired t-test
                        t_u, p_ttest_u = ttest_rel(peaks1_u[:n_min], peaks2_u[:n_min])
                        t_v, p_ttest_v = ttest_rel(peaks1_v[:n_min], peaks2_v[:n_min])

                        # Effect size (Cohen's d)
                        diff_u = peaks1_u[:n_min] - peaks2_u[:n_min]
                        diff_v = peaks1_v[:n_min] - peaks2_v[:n_min]
                        d_u = np.mean(diff_u) / np.std(diff_u) if np.std(diff_u) > 0 else 0
                        d_v = np.mean(diff_v) / np.std(diff_v) if np.std(diff_v) > 0 else 0

                        test_results['region_i'].append({
                            'component': comp_idx + 1,
                            'wilcoxon_p': p_wilcox_u,
                            'ttest_p': p_ttest_u,
                            'cohens_d': d_u,
                            'mean_diff': np.mean(diff_u),
                            'n_trials': n_min
                        })

                        test_results['region_j'].append({
                            'component': comp_idx + 1,
                            'wilcoxon_p': p_wilcox_v,
                            'ttest_p': p_ttest_v,
                            'cohens_d': d_v,
                            'mean_diff': np.mean(diff_v),
                            'n_trials': n_min
                        })

                results['pairwise_tests'][comparison_key] = test_results

                print(f"\n  Statistical tests ({comparison_key}):")
                for r in test_results['region_i'][:3]:
                    sig = '*' if r['wilcoxon_p'] < 0.05 else ''
                    print(f"    Comp {r['component']}: Wilcoxon p={r['wilcoxon_p']:.4f}{sig}, d={r['cohens_d']:.2f}")

        self.statistical_results = results
        return results

    # =========================================================================
    # RASTERMAP ORDERING (FROM REFERENCE ONLY)
    # =========================================================================

    def compute_global_rastermap(self) -> Optional[Dict]:
        """
        Compute global Rastermap ordering using reference trial type only.

        This ordering will be applied to visualizations of all trial types
        to ensure consistent neuron ordering across comparisons.

        Returns:
            Dictionary with Rastermap results or None if unavailable
        """
        if not RASTERMAP_AVAILABLE:
            print("Rastermap not available - skipping neuron ordering")
            return None

        print(f"\n" + "-" * 50)
        print(f"Computing Rastermap ordering from {self.reference_type}")
        print("-" * 50)

        ref_neural = self.neural_data.get(self.reference_type, {})

        if not ref_neural:
            print("No neural data for reference type")
            return None

        all_neural_data = []
        region_indices = {}
        current_idx = 0

        for region_name, spike_data in ref_neural.items():
            # Shape: (n_trials, n_neurons, n_time)
            n_trials, n_neurons, n_time = spike_data.shape

            # Reshape to (n_neurons, n_time * n_trials) for Rastermap
            reshaped = spike_data.transpose(1, 0, 2).reshape(n_neurons, -1)
            all_neural_data.append(reshaped)

            region_indices[region_name] = {
                'start': current_idx,
                'end': current_idx + n_neurons,
                'n_neurons': n_neurons
            }
            current_idx += n_neurons

            print(f"  {region_name}: {n_neurons} neurons")

        # Pool all data
        pooled_data = np.vstack(all_neural_data)
        print(f"  Pooled data: {pooled_data.shape}")

        # Normalize
        pooled_norm = zscore(pooled_data, axis=1, nan_policy='omit')
        pooled_norm = np.nan_to_num(pooled_norm, nan=0.0)

        # Fit Rastermap
        model = Rastermap(
            n_PCs=min(pooled_data.shape[0], 200),
            locality=0,
            grid_upsample=5
        )
        model.fit(pooled_norm)

        self.global_rastermap_results = {
            'sorting_idx': model.isort,
            'embedding': model.embedding,
            'region_indices': region_indices,
            'n_total_neurons': pooled_data.shape[0],
            'n_time': n_time,
            'n_trials': n_trials
        }

        print(f"  Rastermap complete: {len(model.isort)} neurons sorted")

        return self.global_rastermap_results

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================

    def create_projection_comparison_figure(
            self,
            region_pair: Tuple[str, str],
            figsize: Tuple[float, float] = (16, 12),
            save_fig: bool = True
    ) -> plt.Figure:
        """
        Create comprehensive figure comparing CCA projections across trial types.

        Layout:
        - Row 1: Region i projections (all components, all trial types)
        - Row 2: Region j projections (all components, all trial types)
        - Row 3: Peak amplitude comparison bar plots

        Parameters:
            region_pair: Tuple of (region_i, region_j)
            figsize: Figure dimensions
            save_fig: Whether to save the figure

        Returns:
            Matplotlib figure object
        """
        region_i, region_j = region_pair
        n_comp_show = min(3, self.n_components)

        fig = plt.figure(figsize=figsize)
        fontsize = 12

        # Create grid: 3 rows × n_comp_show columns
        gs = fig.add_gridspec(2, n_comp_show, height_ratios=[1, 1],
                              hspace=0.35, wspace=0.3)

        # Row 1: Region i projections
        for comp_idx in range(n_comp_show):
            ax = fig.add_subplot(gs[0, comp_idx])
            self._plot_projections_single_component(
                ax, 'u_mean', 'u_sem', comp_idx, region_i, fontsize
            )
            if comp_idx == 0:
                ax.set_ylabel(f'{region_i}\nProjection', fontsize=fontsize)
            ax.set_title(f'CCA Component {comp_idx + 1}', fontsize=fontsize + 1)

        # Row 2: Region j projections
        for comp_idx in range(n_comp_show):
            ax = fig.add_subplot(gs[1, comp_idx])
            self._plot_projections_single_component(
                ax, 'v_mean', 'v_sem', comp_idx, region_j, fontsize
            )
            if comp_idx == 0:
                ax.set_ylabel(f'{region_j}\nProjection', fontsize=fontsize)
            ax.set_xlabel('Time (s)', fontsize=fontsize)

        # # Row 3: Peak amplitude comparison
        # ax_peak = fig.add_subplot(gs[2, :])
        # self._plot_peak_amplitude_comparison(ax_peak, fontsize)

        # Add overall title
        plt.suptitle(
            f'Cross-Trial-Type CCA Projections\n'
            f'{region_i} vs {region_j} | CCA trained on {self.reference_type}',
            fontsize=fontsize + 2, fontweight='bold', y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_fig:
            save_path = self.output_dir / f"projection_comparison_{region_i}_{region_j}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def _plot_projections_single_component(
            self,
            ax: plt.Axes,
            mean_key: str,
            sem_key: str,
            comp_idx: int,
            region_name: str,
            fontsize: int
    ) -> None:
        """Plot projections for a single component across all trial types."""
        for trial_type in self.available_trial_types:
            if trial_type not in self.projections:
                continue

            proj = self.projections[trial_type]
            mean_proj = np.abs(proj[mean_key][:, comp_idx])
            sem_proj = proj[sem_key][:, comp_idx]

            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')
            linestyle = '-' if trial_type == self.reference_type else '-'
            linewidth = 2.5 if trial_type == self.reference_type else 1.0
            if trial_type == self.reference_type:
                ax.plot(self.time_bins, mean_proj, color=color, linestyle=linestyle,
                        linewidth=linewidth, label=trial_type.replace('_', ' '),
                        alpha=0.8)
                ax.fill_between(self.time_bins, mean_proj - sem_proj, mean_proj + sem_proj,
                                alpha=0.15, color=color)
            else:
                ax.plot(self.time_bins, mean_proj, color=color, linestyle=linestyle,
                        linewidth=linewidth, label=trial_type.replace('_', ' '),
                        alpha=0.4)
                ax.fill_between(self.time_bins, mean_proj - sem_proj, mean_proj + sem_proj,
                                alpha=0.15, color=color)
        # Add stimulus onset marker
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvspan(0, 1.5, alpha=0.05, color='gray')
        ax.set_ylim([0, 6])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=fontsize - 2)
        ax.legend(fontsize=fontsize - 3, loc='upper right')

    def _plot_peak_amplitude_comparison(self, ax: plt.Axes, fontsize: int) -> None:
        """Plot peak amplitude comparison as grouped bar chart."""
        trial_types = list(self.projections.keys())
        n_types = len(trial_types)
        n_comp_show = min(3, self.n_components)

        # Prepare data
        bar_width = 0.25
        x_base = np.arange(n_comp_show)

        for idx, trial_type in enumerate(trial_types):
            peaks = self.statistical_results['peak_amplitudes'][trial_type]
            region_i_peaks = peaks['region_i'][:n_comp_show]
            region_j_peaks = peaks['region_j'][:n_comp_show]
            avg_peaks = [(a + b) / 2 for a, b in zip(region_i_peaks, region_j_peaks)]

            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')
            offset = (idx - n_types / 2 + 0.5) * bar_width

            ax.bar(x_base + offset, avg_peaks, bar_width, color=color,
                   label=trial_type.replace('_', ' '), alpha=0.8, edgecolor='black')

        ax.set_xticks(x_base)
        ax.set_xticklabels([f'Comp {i + 1}' for i in range(n_comp_show)], fontsize=fontsize)
        ax.set_ylabel('Peak Amplitude\n(avg. both regions)', fontsize=fontsize)
        ax.set_xlabel('CCA Component', fontsize=fontsize)
        ax.legend(fontsize=fontsize - 2, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)

    def create_statistical_summary_figure(
            self,
            region_pair: Tuple[str, str],
            figsize: Tuple[float, float] = (14, 10),
            save_fig: bool = True
    ) -> plt.Figure:
        """
        Create figure showing statistical comparison results.

        Layout:
        - Top row: Temporal correlation heatmap
        - Bottom row: Pairwise test p-values and effect sizes

        Parameters:
            region_pair: Tuple of (region_i, region_j)
            figsize: Figure dimensions
            save_fig: Whether to save the figure

        Returns:
            Matplotlib figure object
        """
        region_i, region_j = region_pair

        fig = plt.figure(figsize=figsize)
        fontsize = 12

        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        # Plot 1: Temporal correlation R² values
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_temporal_correlation_bars(ax1, 'region_i', region_i, fontsize)

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_temporal_correlation_bars(ax2, 'region_j', region_j, fontsize)

        # Plot 2: P-value heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_pvalue_heatmap(ax3, fontsize)

        # # Plot 3: Effect size comparison
        # ax4 = fig.add_subplot(gs[1, 1])
        # self._plot_effect_sizes(ax4, fontsize)

        plt.suptitle(
            f'Statistical Comparison: Cross-Trial-Type CCA Projections\n'
            f'{region_i} vs {region_j}',
            fontsize=fontsize + 2, fontweight='bold', y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_fig:
            save_path = self.output_dir / f"statistical_summary_{region_i}_{region_j}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def _plot_temporal_correlation_bars(
            self,
            ax: plt.Axes,
            region_key: str,
            region_name: str,
            fontsize: int
    ) -> None:
        """Plot temporal correlation R² as bar chart."""
        correlations = self.statistical_results['temporal_correlations']
        n_comp_show = min(3, self.n_components)

        bar_width = 0.35
        x_base = np.arange(n_comp_show)

        for idx, (comparison, corr_data) in enumerate(correlations.items()):
            r2_values = [c['r2'] for c in corr_data[region_key][:n_comp_show]]
            offset = (idx - len(correlations) / 2 + 0.5) * bar_width

            # Extract comparison trial type for color
            other_type = comparison.replace(f"{self.reference_type}_vs_", "")
            color = TRIAL_TYPE_COLORS.get(other_type, 'gray')

            ax.bar(x_base + offset, r2_values, bar_width, color=color,
                   label=comparison.replace('_', ' '), alpha=0.8, edgecolor='black')

        ax.set_xticks(x_base)
        ax.set_xticklabels([f'Comp {i + 1}' for i in range(n_comp_show)], fontsize=fontsize - 2)
        ax.set_ylabel('Temporal R²', fontsize=fontsize)
        ax.set_xlabel('CCA Component', fontsize=fontsize)
        ax.set_title(f'{region_name}', fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 3)
        ax.set_ylim([0, 1.0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)

    def _plot_pvalue_heatmap(self, ax: plt.Axes, fontsize: int) -> None:
        """Plot p-value heatmap for pairwise comparisons."""
        pairwise = self.statistical_results['pairwise_tests']
        n_comp_show = min(3, self.n_components)

        comparisons = list(pairwise.keys())
        if not comparisons:
            ax.text(0.5, 0.5, 'No pairwise tests available',
                    ha='center', va='center', fontsize=fontsize)
            ax.axis('off')
            return

        # Create p-value matrix
        p_matrix = np.zeros((len(comparisons), n_comp_show))

        for i, comp in enumerate(comparisons):
            for j in range(n_comp_show):
                if j < len(pairwise[comp]['region_i']):
                    # Average p-value across both regions
                    p_i = pairwise[comp]['region_i'][j]['wilcoxon_p']
                    p_j = pairwise[comp]['region_j'][j]['wilcoxon_p']
                    p_matrix[i, j] = (p_i + p_j) / 2

        # Plot heatmap
        p_matrix = -np.log10(p_matrix + 1e-10)
        im = ax.imshow(p_matrix, cmap='Reds', aspect='auto',vmax=2.5,vmin=0)

        ax.set_xticks(np.arange(n_comp_show))
        ax.set_yticks(np.arange(len(comparisons)))
        ax.set_xticklabels([f'Comp {i + 1}' for i in range(n_comp_show)], fontsize=fontsize - 2)
        ax.set_yticklabels([c.replace('_', '\n') for c in comparisons], fontsize=fontsize - 3)
        ax.set_xlabel('CCA Component', fontsize=fontsize)
        ax.set_title('Significance (-log₁₀ p)', fontsize=fontsize + 1)

        # Add text annotations
        for i in range(len(comparisons)):
            for j in range(n_comp_show):
                p_val = p_matrix[i, j]
                text = f'{p_val:.3f}'
                if p_val > 1.3:
                    text += '*'
                if p_val > 2:
                    text += '*'
                ax.text(j, i, text, ha='center', va='center', fontsize=fontsize - 3,
                        color='white' if (p_val + 1e-10) > 1 else 'black')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_effect_sizes(self, ax: plt.Axes, fontsize: int) -> None:
        """Plot Cohen's d effect sizes."""
        pairwise = self.statistical_results['pairwise_tests']
        n_comp_show = min(3, self.n_components)

        comparisons = list(pairwise.keys())
        if not comparisons:
            ax.text(0.5, 0.5, 'No effect sizes available',
                    ha='center', va='center', fontsize=fontsize)
            ax.axis('off')
            return

        bar_width = 0.25
        x_base = np.arange(n_comp_show)

        for idx, comp in enumerate(comparisons):
            d_values = []
            for j in range(n_comp_show):
                if j < len(pairwise[comp]['region_i']):
                    d_i = pairwise[comp]['region_i'][j]['cohens_d']
                    d_j = pairwise[comp]['region_j'][j]['cohens_d']
                    d_values.append((d_i + d_j) / 2)
                else:
                    d_values.append(0)

            offset = (idx - len(comparisons) / 2 + 0.5) * bar_width

            # Get color from comparison name
            other_type = comp.split('_vs_')[-1]
            color = TRIAL_TYPE_COLORS.get(other_type, 'gray')

            ax.bar(x_base + offset, d_values, bar_width, color=color,
                   label=comp.replace('_', ' '), alpha=0.8, edgecolor='black')

        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
        ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)

        ax.set_xticks(x_base)
        ax.set_xticklabels([f'Comp {i + 1}' for i in range(n_comp_show)], fontsize=fontsize - 2)
        ax.set_ylabel("Cohen's d", fontsize=fontsize)
        ax.set_xlabel('CCA Component', fontsize=fontsize)
        ax.set_title('Effect Size (avg. both regions)', fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 3, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)

    def create_detailed_temporal_figure(
            self,
            region_pair: Tuple[str, str],
            component_idx: int = 0,
            figsize: Tuple[float, float] = (18, 10),
            save_fig: bool = True
    ) -> plt.Figure:
        """
        Create detailed temporal comparison figure for a single component.

        Shows both regions' projections with individual trial variability.

        Parameters:
            region_pair: Tuple of (region_i, region_j)
            component_idx: CCA component to display (0-indexed)
            figsize: Figure dimensions
            save_fig: Whether to save the figure

        Returns:
            Matplotlib figure object
        """
        region_i, region_j = region_pair

        fig = plt.figure(figsize=figsize)
        fontsize = 14

        gs = fig.add_gridspec(2, len(self.available_trial_types),
                              hspace=0.3, wspace=0.25)

        for col_idx, trial_type in enumerate(self.available_trial_types):
            if trial_type not in self.projections:
                continue

            proj = self.projections[trial_type]
            color = TRIAL_TYPE_COLORS.get(trial_type, 'gray')

            # Row 1: Region i
            ax1 = fig.add_subplot(gs[0, col_idx])
            self._plot_detailed_projection(
                ax1, proj['u_mean'][:, component_idx],
                proj['u_trials'][:, :, component_idx],
                color, trial_type, fontsize
            )
            ax1.set_ylim([0, 5])
            ax1.set_title(f'{trial_type.replace("_", " ")}\n(n={proj["n_trials"]} trials)',
                          fontsize=fontsize)
            if col_idx == 0:
                ax1.set_ylabel(f'{region_i}\nProjection', fontsize=fontsize)

            # Row 2: Region j
            ax2 = fig.add_subplot(gs[1, col_idx])
            self._plot_detailed_projection(
                ax2, proj['v_mean'][:, component_idx],
                proj['v_trials'][:, :, component_idx],
                color, trial_type, fontsize
            )
            if col_idx == 0:
                ax2.set_ylabel(f'{region_j}\nProjection', fontsize=fontsize)
            ax2.set_xlabel('Time (s)', fontsize=fontsize)
            ax2.set_ylim([0, 5])

        plt.suptitle(
            f'CCA Component {component_idx + 1} Projections\n'
            f'{region_i} vs {region_j} | CCA trained on {self.reference_type}',
            fontsize=fontsize + 2, fontweight='bold', y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if save_fig:
            save_path = self.output_dir / f"detailed_comp{component_idx + 1}_{region_i}_{region_j}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def _plot_detailed_projection(
            self,
            ax: plt.Axes,
            mean_proj: np.ndarray,
            trial_projs: np.ndarray,
            color: str,
            trial_type: str,
            fontsize: int
    ) -> None:
        """Plot detailed projection with individual trial traces."""
        # Take absolute value
        mean_proj = np.abs(mean_proj)
        trial_projs = np.abs(trial_projs)

        # Plot individual trials (light)
        for trial_idx in range(min(20, trial_projs.shape[0])):
            ax.plot(self.time_bins, trial_projs[trial_idx], color=color,
                    alpha=0.1, linewidth=0.5)

        # Plot mean with Std
        sem = np.std(trial_projs, axis=0) / np.sqrt(trial_projs.shape[0])
        std = np.std(trial_projs, axis=0)
        ax.plot(self.time_bins, mean_proj, color=color, linewidth=2.5, alpha=0.9)
        # ax.fill_between(self.time_bins, mean_proj - sem, mean_proj + sem,
        #                 alpha=0.2, color=color)
        ax.fill_between(self.time_bins, mean_proj - std, mean_proj + std,
                        alpha=0.2, color=color)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvspan(0, 1.5, alpha=0.05, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=fontsize - 2)

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_summary_report(self, region_pair: Tuple[str, str]) -> Path:
        """
        Generate comprehensive text and JSON summary reports.

        Parameters:
            region_pair: Tuple of (region_i, region_j)

        Returns:
            Path to the generated report directory
        """
        region_i, region_j = region_pair
        report_dir = self.output_dir / 'reports'
        report_dir.mkdir(exist_ok=True)

        # JSON report
        json_report = {
            'session': self.session_name,
            'region_pair': [region_i, region_j],
            'reference_type': self.reference_type,
            'available_trial_types': self.available_trial_types,
            'n_components': self.n_components,
            'analysis_date': str(datetime.now()),
            'statistical_results': self._serialize_stats_for_json()
        }

        json_path = report_dir / f'analysis_summary_{region_i}_{region_j}.json'
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)

        # Text report
        text_path = report_dir / f'analysis_report_{region_i}_{region_j}.txt'
        with open(text_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Cross-Trial-Type CCA Projection Analysis Report\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Session: {self.session_name}\n")
            f.write(f"Region Pair: {region_i} vs {region_j}\n")
            f.write(f"Reference Type: {self.reference_type}\n")
            f.write(f"Trial Types Analyzed: {', '.join(self.available_trial_types)}\n")
            f.write(f"Analysis Date: {datetime.now()}\n\n")

            f.write("-" * 50 + "\n")
            f.write("Trial Counts:\n")
            f.write("-" * 50 + "\n")
            for tt, proj in self.projections.items():
                f.write(f"  {tt}: {proj['n_trials']} trials\n")

            f.write("\n" + "-" * 50 + "\n")
            f.write("Peak Amplitudes (Component 1):\n")
            f.write("-" * 50 + "\n")
            for tt, peaks in self.statistical_results['peak_amplitudes'].items():
                f.write(f"  {tt}:\n")
                f.write(f"    {region_i}: {peaks['region_i'][0]:.4f}\n")
                f.write(f"    {region_j}: {peaks['region_j'][0]:.4f}\n")

            f.write("\n" + "-" * 50 + "\n")
            f.write("Statistical Tests (Component 1):\n")
            f.write("-" * 50 + "\n")
            for comp, tests in self.statistical_results['pairwise_tests'].items():
                f.write(f"\n  {comp}:\n")
                if tests['region_i']:
                    t = tests['region_i'][0]
                    f.write(f"    {region_i}: Wilcoxon p={t['wilcoxon_p']:.4f}, ")
                    f.write(f"Cohen's d={t['cohens_d']:.3f}\n")
                if tests['region_j']:
                    t = tests['region_j'][0]
                    f.write(f"    {region_j}: Wilcoxon p={t['wilcoxon_p']:.4f}, ")
                    f.write(f"Cohen's d={t['cohens_d']:.3f}\n")

        print(f"Reports saved to: {report_dir}")
        return report_dir

    def _serialize_stats_for_json(self) -> Dict:
        """Convert statistical results to JSON-serializable format."""

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        return convert(self.statistical_results)


# =============================================================================
# PIPELINE ORCHESTRATION CLASS
# =============================================================================

class CrossTrialTypeCCAPipeline:
    """
    Complete pipeline for cross-trial-type CCA analysis.

    This class orchestrates the entire analysis workflow for multiple
    sessions and region pairs.
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
        """
        self.config = config
        self.analyzers: Dict[str, CrossTrialTypeCCAAnalyzer] = {}
        self.results: Dict = {}

        # Validate config
        required_keys = ['base_dir', 'sessions']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        # Set defaults
        config.setdefault('reference_type', 'cued_hit_long')
        config.setdefault('n_components', 5)
        config.setdefault('region_pairs', None)  # Auto-detect if None

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

                # Create visualizations
                analyzer.create_projection_comparison_figure(region_pair)
                analyzer.create_statistical_summary_figure(region_pair)

                # Create detailed figures for first 3 components
                for comp_idx in range(min(3, analyzer.n_components)):
                    analyzer.create_detailed_temporal_figure(region_pair, comp_idx)

                # Generate reports
                analyzer.generate_summary_report(region_pair)

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
        ]
    }

    print("=" * 70)
    print("Cross-Trial-Type CCA Analysis Pipeline")
    print("=" * 70)

    # Initialize and run pipeline
    pipeline = CrossTrialTypeCCAPipeline(config)
    results = pipeline.run_all_sessions()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return pipeline


if __name__ == "__main__":
    pipeline = main()
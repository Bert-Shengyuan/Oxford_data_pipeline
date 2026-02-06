#!/usr/bin/env python3
"""
Dimensionality-Communication Relationship Analysis
==================================================

This script investigates the relationship between:
1. Intrinsic signal dimensionality (from PCA cumulative variance)
2. Cross-regional communication capacity (from CCA R² values)
3. Shared subspace dimensionality (number of significant CCA components)

Scientific Questions:
--------------------
Q1: Do high-dimensional regions (broad PCA spectrum) have greater
    communication capacity (higher CCA R²)?

Q2: Is communication dimensionality limited by the lower-dimensional
    partner (bottleneck hypothesis)?

Q3: Are communication strength and dimensionality dissociable
    (single strong channel vs. distributed multiplexing)?

Mathematical Framework:
----------------------
Effective dimensionality (participation ratio):
    d_eff = (Σλᵢ)² / Σλᵢ²

Communication capacity:
    I_shared ≈ Σ ρₖ²

Shared subspace dimensionality:
    n_sig = number of significant CCA components

Author: Shengyuan Cai
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
import mat73

# Configure visualization
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)


class DimensionalityAnalyzer:
    """
    Analyzer for investigating relationships between intrinsic dimensionality
    and cross-regional communication capacity.
    """

    def __init__(self, results_base_dir: str, results_subdir: str, output_dir: str):
        """
        Initialize analyzer with paths to session results.

        Parameters:
        -----------
        results_base_dir : str
            Base directory containing all session results
        results_subdir : str
            Subdirectory with analysis results (e.g., 'sessions_cued_hit_long_results')
        output_dir : str
            Directory for saving analysis outputs
        """
        self.results_base_dir = Path(results_base_dir)
        self.results_dir = self.results_base_dir / results_subdir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for analysis results
        self.region_dimensionality = {}  # region -> effective dimensionality
        self.pair_communication = {}  # (region1, region2) -> communication metrics
        self.all_data = []  # List of dicts for correlation analysis

        print("=" * 70)
        print("DIMENSIONALITY-COMMUNICATION RELATIONSHIP ANALYZER")
        print("=" * 70)
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")

    def compute_effective_dimensionality(
            self,
            explained_variance: np.ndarray,
            method: str = 'participation_ratio'
    ) -> float:
        """
        Compute effective dimensionality from PCA explained variance.

        Multiple measures of dimensionality capture different aspects of
        the eigenspectrum distribution:

        1. Participation Ratio (Gao et al., 2017):
           d_eff = (Σλᵢ)² / Σλᵢ²

           Interpretation: Number of equally-weighted components that would
           produce the observed variance. High values indicate distributed
           variance across many components.

        2. Shannon Entropy-based (Stringer et al., 2019):
           d_eff = exp(-Σ pᵢ log pᵢ)
           where pᵢ = λᵢ / Σλⱼ

           Interpretation: Exponential of Shannon entropy of the variance
           distribution. More sensitive to the full spectrum shape.

        3. 90% Cumulative Variance:
           d_90 = argmin_k {Σᵢ₌₁ᵏ λᵢ / Σⱼ λⱼ ≥ 0.90}

           Interpretation: Number of components needed to capture 90% of
           variance. Simple threshold-based measure.

        Parameters:
        -----------
        explained_variance : np.ndarray
            PCA explained variance values (can be percentages or raw eigenvalues)
        method : str
            Method to use: 'participation_ratio', 'entropy', or 'cumulative_90'

        Returns:
        --------
        d_eff : float
            Effective dimensionality measure
        """
        # Normalize to ensure we're working with a probability distribution
        total_var = np.sum(explained_variance)
        p = explained_variance / total_var

        if method == 'participation_ratio':
            # Participation ratio: (Σλᵢ)² / Σλᵢ²
            d_eff = (np.sum(explained_variance) ** 2) / np.sum(explained_variance ** 2)

        elif method == 'entropy':
            # Shannon entropy-based: exp(H) where H = -Σ pᵢ log pᵢ
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            entropy = -np.sum(p * np.log(p + epsilon))
            d_eff = np.exp(entropy)

        elif method == 'cumulative_90':
            # Number of components to reach 90% cumulative variance
            cumulative = np.cumsum(p)
            d_eff = np.argmax(cumulative >= 0.90) + 1

        else:
            raise ValueError(f"Unknown method: {method}")

        return d_eff

    def load_and_process_all_sessions(self) -> None:
        """
        Load all session files and extract dimensionality metrics.

        This method:
        1. Loads each session's PCA and CCA results
        2. Computes effective dimensionality for each region
        3. Extracts communication metrics for each region pair
        4. Aggregates data for cross-session correlation analysis
        """
        print("\nLoading session data...")
        print("-" * 50)

        # Find all analysis result files
        result_files = list(self.results_dir.glob("*_analysis_results.mat"))
        print(f"Found {len(result_files)} session files")

        for result_file in result_files:
            session_name = result_file.stem.replace("_analysis_results", "")
            self._process_session(result_file, session_name)

        print("\n" + "-" * 50)
        print(f"Processed {len(result_files)} sessions")
        print(f"Total region-pair observations: {len(self.all_data)}")

    def _process_session(self, file_path: Path, session_name: str) -> None:
        """Process a single session file."""
        try:
            # Load session data
            data = mat73.loadmat(str(file_path))

            if 'pca_results' not in data or 'cca_results' not in data:
                print(f"  {session_name}: Missing PCA or CCA results, skipping")
                return

            pca_results = data['pca_results']
            cca_results = data['cca_results']

            # Extract PCA dimensionality for each region
            session_region_dim = {}
            METADATA_KEYS = {'config', 'session_name', 'analysis_timestamp', '__len__'}

            for region_name, region_pca in pca_results.items():
                if not isinstance(region_pca, dict):
                    continue
                if region_name in METADATA_KEYS:
                    continue
                if 'explained_variance' in region_pca:
                    explained_var = np.array(region_pca['explained_variance']).flatten()

                    # Compute multiple dimensionality measures
                    d_pr = self.compute_effective_dimensionality(
                        explained_var, method='participation_ratio'
                    )
                    d_ent = self.compute_effective_dimensionality(
                        explained_var, method='entropy'
                    )
                    d_90 = self.compute_effective_dimensionality(
                        explained_var, method='cumulative_90'
                    )

                    # Also store cumulative variance at specific points
                    cumulative_var = np.cumsum(explained_var)
                    total_var = np.sum(explained_var)
                    cumulative_pct = cumulative_var / total_var * 100

                    session_region_dim[region_name] = {
                        'participation_ratio': d_pr,
                        'entropy_based': d_ent,
                        'cumulative_90': d_90,
                        'pca_1': cumulative_pct[0] if len(cumulative_pct) > 0 else 0,
                        'pca_3': cumulative_pct[2] if len(cumulative_pct) > 2 else 0,
                        'pca_5': cumulative_pct[4] if len(cumulative_pct) > 4 else 0,
                        'pca_10': cumulative_pct[9] if len(cumulative_pct) > 9 else 0,
                        'session': session_name
                    }

            # Extract CCA communication metrics for each pair
            if 'pair_results' not in cca_results:
                return

            pair_results = cca_results['pair_results']

            for pair_idx, pair_result in enumerate(pair_results):
                if not isinstance(pair_result, dict):
                    continue

                # Extract region names
                region_i = self._extract_string(pair_result, 'region_i')
                region_j = self._extract_string(pair_result, 'region_j')

                if not region_i or not region_j:
                    continue

                # Check if we have dimensionality info for both regions
                if region_i not in session_region_dim or region_j not in session_region_dim:
                    continue

                # Extract CCA metrics
                if 'cv_results' not in pair_result:
                    continue

                cv_results = pair_result['cv_results']
                mean_cv_R2 = np.array(cv_results['mean_cv_R2']).flatten()

                # Number of significant components
                if 'significant_components' in pair_result:
                    n_sig = len(np.array(pair_result['significant_components']).flatten())
                else:
                    # Fallback: count components with R² > threshold
                    n_sig = np.sum(mean_cv_R2 > 0.01)

                # Communication capacity metrics
                max_R2 = np.max(mean_cv_R2) if len(mean_cv_R2) > 0 else 0
                mean_R2_top3 = np.mean(mean_cv_R2[:3]) if len(mean_cv_R2) >= 3 else np.mean(mean_cv_R2)
                sum_R2 = np.sum(mean_cv_R2)

                # Create data entry for correlation analysis
                data_entry = {
                    'session': session_name,
                    'region_i': region_i,
                    'region_j': region_j,
                    'dim_i_pr': session_region_dim[region_i]['participation_ratio'],
                    'dim_j_pr': session_region_dim[region_j]['participation_ratio'],
                    'dim_i_ent': session_region_dim[region_i]['entropy_based'],
                    'dim_j_ent': session_region_dim[region_j]['entropy_based'],
                    'dim_i_90': session_region_dim[region_i]['cumulative_90'],
                    'dim_j_90': session_region_dim[region_j]['cumulative_90'],
                    'dim_min_pr': min(session_region_dim[region_i]['participation_ratio'],
                                      session_region_dim[region_j]['participation_ratio']),
                    'dim_max_pr': max(session_region_dim[region_i]['participation_ratio'],
                                      session_region_dim[region_j]['participation_ratio']),
                    'dim_mean_pr': (session_region_dim[region_i]['participation_ratio'] +
                                    session_region_dim[region_j]['participation_ratio']) / 2,
                    'n_sig_components': n_sig,
                    'max_R2': max_R2,
                    'mean_R2_top3': mean_R2_top3,
                    'sum_R2': sum_R2,
                    'pca1_i': session_region_dim[region_i]['pca_1'],
                    'pca1_j': session_region_dim[region_j]['pca_1'],
                    'pca10_i': session_region_dim[region_i]['pca_10'],
                    'pca10_j': session_region_dim[region_j]['pca_10']
                }

                self.all_data.append(data_entry)

            print(f"  {session_name}: {len(session_region_dim)} regions, "
                  f"{len([d for d in self.all_data if d['session'] == session_name])} pairs")

        except Exception as e:
            print(f"  {session_name}: Error processing - {str(e)}")

    def _extract_string(self, data: dict, field: str) -> Optional[str]:
        """Helper to extract string from MATLAB structures."""
        try:
            if field not in data:
                return None
            value = data[field]
            if isinstance(value, str):
                return value
            elif isinstance(value, np.ndarray):
                if value.dtype.kind in ['U', 'S']:
                    return str(value.item())
            elif isinstance(value, list) and len(value) > 0:
                return str(value[0])
            return None
        except:
            return None

    def test_hypothesis_1(self) -> Dict:
        """
        Test Hypothesis 1: High-dimensional regions have greater communication capacity.

        Statistical Approach:
        --------------------
        For each region pair, we have:
        - Mean dimensionality: (d_i + d_j) / 2
        - Communication capacity: Σ ρₖ²

        We test for positive correlation using:
        1. Pearson correlation (linear relationship)
        2. Spearman correlation (monotonic relationship)

        Expected Result if H1 is True:
        Positive correlation with p < 0.05

        Returns:
        --------
        results : dict
            Statistical test results and scatter plot data
        """
        print("\n" + "=" * 70)
        print("HYPOTHESIS 1: Dimensionality → Communication Capacity")
        print("=" * 70)

        if len(self.all_data) == 0:
            print("No data loaded. Run load_and_process_all_sessions() first.")
            return {}

        # Extract relevant variables
        dim_mean = np.array([d['dim_mean_pr'] for d in self.all_data])
        sum_R2 = np.array([d['sum_R2'] for d in self.all_data])
        max_R2 = np.array([d['max_R2'] for d in self.all_data])

        # Compute correlations
        r_pearson_sum, p_pearson_sum = pearsonr(dim_mean, sum_R2)
        r_spearman_sum, p_spearman_sum = spearmanr(dim_mean, sum_R2)

        r_pearson_max, p_pearson_max = pearsonr(dim_mean, max_R2)
        r_spearman_max, p_spearman_max = spearmanr(dim_mean, max_R2)

        print("\nCorrelation: Mean Dimensionality vs Sum R²")
        print(f"  Pearson r = {r_pearson_sum:.3f}, p = {p_pearson_sum:.4f}")
        print(f"  Spearman ρ = {r_spearman_sum:.3f}, p = {p_spearman_sum:.4f}")

        print("\nCorrelation: Mean Dimensionality vs Max R²")
        print(f"  Pearson r = {r_pearson_max:.3f}, p = {p_pearson_max:.4f}")
        print(f"  Spearman ρ = {r_spearman_max:.3f}, p = {p_spearman_max:.4f}")

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Dimensionality vs Sum R²
        ax1 = axes[0]
        ax1.scatter(dim_mean, sum_R2, alpha=0.6, s=80, edgecolor='black', linewidth=0.5)

        # Add regression line
        z = np.polyfit(dim_mean, sum_R2, 1)
        p = np.poly1d(z)
        x_line = np.linspace(dim_mean.min(), dim_mean.max(), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
                 label=f'Linear fit: r={r_pearson_sum:.3f}')

        ax1.set_xlabel('Mean Effective Dimensionality\n(Participation Ratio)', fontsize=12)
        ax1.set_ylabel('Total Communication Capacity (Σ R²)', fontsize=12)
        ax1.set_title('Hypothesis 1: Dimensionality → Communication\n(Sum R²)',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Dimensionality vs Max R²
        ax2 = axes[1]
        ax2.scatter(dim_mean, max_R2, alpha=0.6, s=80, edgecolor='black',
                    linewidth=0.5, color='darkgreen')

        z2 = np.polyfit(dim_mean, max_R2, 1)
        p2 = np.poly1d(z2)
        ax2.plot(x_line, p2(x_line), "r--", alpha=0.8, linewidth=2,
                 label=f'Linear fit: r={r_pearson_max:.3f}')

        ax2.set_xlabel('Mean Effective Dimensionality\n(Participation Ratio)', fontsize=12)
        ax2.set_ylabel('Maximum Communication Strength (Max R²)', fontsize=12)
        ax2.set_title('Hypothesis 1: Dimensionality → Communication\n(Max R²)',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / "hypothesis1_dimensionality_capacity.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Visualization saved: {save_path}")

        results = {
            'correlation_sum_R2': {
                'pearson_r': r_pearson_sum,
                'pearson_p': p_pearson_sum,
                'spearman_rho': r_spearman_sum,
                'spearman_p': p_spearman_sum
            },
            'correlation_max_R2': {
                'pearson_r': r_pearson_max,
                'pearson_p': p_pearson_max,
                'spearman_rho': r_spearman_max,
                'spearman_p': p_spearman_max
            },
            'n_observations': len(self.all_data)
        }

        return results

    def test_hypothesis_2(self) -> Dict:
        """
        Test Hypothesis 2: Communication limited by lower-dimensional partner.

        Bottleneck Hypothesis:
        ---------------------
        If communication requires coordination across multiple dimensions,
        the capacity should be constrained by whichever region has fewer
        independent dimensions. Formally:

        n_sig ∝ min(d_i, d_j) rather than max or mean

        We test this by comparing correlations between n_sig and:
        1. min(d_i, d_j)  - bottleneck prediction
        2. max(d_i, d_j)  - "rich get richer" alternative
        3. mean(d_i, d_j) - symmetric alternative

        Statistical Test:
        Use Steiger's Z-test to compare dependent correlations

        Returns:
        --------
        results : dict
            Correlation coefficients and statistical comparison
        """
        print("\n" + "=" * 70)
        print("HYPOTHESIS 2: Bottleneck Architecture")
        print("=" * 70)

        if len(self.all_data) == 0:
            return {}

        # Extract variables
        n_sig = np.array([d['n_sig_components'] for d in self.all_data])
        dim_min = np.array([d['dim_min_pr'] for d in self.all_data])
        dim_max = np.array([d['dim_max_pr'] for d in self.all_data])
        dim_mean = np.array([d['dim_mean_pr'] for d in self.all_data])

        # Compute correlations
        r_min, p_min = pearsonr(dim_min, n_sig)
        r_max, p_max = pearsonr(dim_max, n_sig)
        r_mean, p_mean = pearsonr(dim_mean, n_sig)

        print("\nCorrelation: Dimensionality → Number of Significant Components")
        print(f"  min(d_i, d_j):  r = {r_min:.3f}, p = {p_min:.4f}")
        print(f"  max(d_i, d_j):  r = {r_max:.3f}, p = {p_max:.4f}")
        print(f"  mean(d_i, d_j): r = {r_mean:.3f}, p = {p_mean:.4f}")

        if abs(r_min) > abs(r_max):
            print("\n→ Strongest correlation with MIN → Supports bottleneck hypothesis")
        elif abs(r_max) > abs(r_min):
            print("\n→ Strongest correlation with MAX → Contradicts bottleneck")
        else:
            print("\n→ Similar correlations → Inconclusive")

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (dim_vals, dim_name, r_val) in enumerate([
            (dim_min, 'Min(d₁, d₂)', r_min),
            (dim_max, 'Max(d₁, d₂)', r_max),
            (dim_mean, 'Mean(d₁, d₂)', r_mean)
        ]):
            ax = axes[idx]
            ax.scatter(dim_vals, n_sig, alpha=0.6, s=80, edgecolor='black', linewidth=0.5)

            z = np.polyfit(dim_vals, n_sig, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(dim_vals.min(), dim_vals.max(), 100)
            ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2,
                    label=f'r = {r_val:.3f}')

            ax.set_xlabel(f'{dim_name}\n(Participation Ratio)', fontsize=12)
            ax.set_ylabel('Number of Significant\nCCA Components', fontsize=12)
            ax.set_title(f'n_sig vs {dim_name}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Hypothesis 2: Bottleneck Architecture Test',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.output_dir / "hypothesis2_bottleneck.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Visualization saved: {save_path}")

        results = {
            'corr_min': {'r': r_min, 'p': p_min},
            'corr_max': {'r': r_max, 'p': p_max},
            'corr_mean': {'r': r_mean, 'p': p_mean},
            'bottleneck_supported': abs(r_min) > abs(r_max)
        }

        return results

    def test_hypothesis_3(self) -> Dict:
        """
        Test Hypothesis 3: Communication strength and dimensionality are dissociable.

        Theoretical Predictions:
        -----------------------
        Two communication architectures are possible:

        Architecture A: "Dedicated Line"
        - High max R² (strong first component)
        - Low n_sig (few significant components)
        - Single dominant communication channel

        Architecture B: "Distributed Multiplex"
        - Moderate max R² (no single dominant component)
        - High n_sig (many significant components)
        - Information distributed across channels

        We test for dissociation by examining the correlation structure:
        If dissociation exists, max_R² and n_sig should have low correlation

        Returns:
        --------
        results : dict
            Clustering analysis and correlation structure
        """
        print("\n" + "=" * 70)
        print("HYPOTHESIS 3: Dissociation of Strength and Dimensionality")
        print("=" * 70)

        if len(self.all_data) == 0:
            return {}

        # Extract variables
        max_R2 = np.array([d['max_R2'] for d in self.all_data])
        n_sig = np.array([d['n_sig_components'] for d in self.all_data])

        # Test for dissociation
        r_dissoc, p_dissoc = pearsonr(max_R2, n_sig)

        print(f"\nCorrelation: Max R² vs n_sig")
        print(f"  r = {r_dissoc:.3f}, p = {p_dissoc:.4f}")

        if abs(r_dissoc) < 0.3 and p_dissoc > 0.05:
            print("\n→ Weak/non-significant correlation → Supports dissociation")
        else:
            print("\n→ Strong correlation → No evidence for dissociation")

        # Classify region pairs into architectural types
        # Use median split for binary classification
        median_max_R2 = np.median(max_R2)
        median_n_sig = np.median(n_sig)

        architecture_types = []
        for r2, ns in zip(max_R2, n_sig):
            if r2 > median_max_R2 and ns < median_n_sig:
                architecture_types.append('Dedicated Line')
            elif r2 < median_max_R2 and ns > median_n_sig:
                architecture_types.append('Distributed Multiplex')
            elif r2 > median_max_R2 and ns > median_n_sig:
                architecture_types.append('Strong & Distributed')
            else:
                architecture_types.append('Weak & Sparse')

        # Count architectural types
        from collections import Counter
        type_counts = Counter(architecture_types)

        print("\nArchitectural Type Distribution:")
        for arch_type, count in type_counts.items():
            pct = count / len(architecture_types) * 100
            print(f"  {arch_type}: {count} ({pct:.1f}%)")

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Scatter with architectural regions
        colors_map = {
            'Dedicated Line': 'red',
            'Distributed Multiplex': 'blue',
            'Strong & Distributed': 'green',
            'Weak & Sparse': 'gray'
        }

        for arch_type in set(architecture_types):
            mask = np.array([t == arch_type for t in architecture_types])
            ax1.scatter(max_R2[mask], n_sig[mask],
                        label=arch_type,
                        color=colors_map[arch_type],
                        alpha=0.7, s=100, edgecolor='black', linewidth=0.5)

        # Add quadrant lines
        ax1.axvline(x=median_max_R2, color='black', linestyle='--', alpha=0.3, linewidth=2)
        ax1.axhline(y=median_n_sig, color='black', linestyle='--', alpha=0.3, linewidth=2)

        ax1.set_xlabel('Maximum Communication Strength (Max R²)', fontsize=12)
        ax1.set_ylabel('Communication Dimensionality (n_sig)', fontsize=12)
        ax1.set_title('Communication Architecture Types', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Bar chart of architectural type frequencies
        arch_types_list = list(type_counts.keys())
        counts_list = [type_counts[t] for t in arch_types_list]
        colors_list = [colors_map[t] for t in arch_types_list]

        ax2.bar(range(len(arch_types_list)), counts_list, color=colors_list,
                alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(arch_types_list)))
        ax2.set_xticklabels(arch_types_list, rotation=15, ha='right')
        ax2.set_ylabel('Number of Region Pairs', fontsize=12)
        ax2.set_title('Architecture Type Distribution', fontsize=13, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)

        # Add count labels on bars
        for i, count in enumerate(counts_list):
            ax2.text(i, count + 0.5, str(count), ha='center', fontsize=11, fontweight='bold')

        plt.suptitle('Hypothesis 3: Dissociation Analysis',
                     fontsize=15, fontweight='bold', y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / "hypothesis3_dissociation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Visualization saved: {save_path}")

        results = {
            'correlation': {'r': r_dissoc, 'p': p_dissoc},
            'dissociation_supported': abs(r_dissoc) < 0.3,
            'architecture_distribution': dict(type_counts)
        }

        return results

    def create_comprehensive_summary(self) -> None:
        """
        Create comprehensive summary figure integrating all three hypotheses.
        """
        print("\n" + "=" * 70)
        print("Creating Comprehensive Summary")
        print("=" * 70)

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

        # Extract key variables
        dim_mean = np.array([d['dim_mean_pr'] for d in self.all_data])
        dim_min = np.array([d['dim_min_pr'] for d in self.all_data])
        sum_R2 = np.array([d['sum_R2'] for d in self.all_data])
        max_R2 = np.array([d['max_R2'] for d in self.all_data])
        n_sig = np.array([d['n_sig_components'] for d in self.all_data])

        # Row 1: Hypothesis 1 - Dimensionality vs Communication
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.scatter(dim_mean, sum_R2, alpha=0.6, s=100, c=n_sig, cmap='viridis',
                    edgecolor='black', linewidth=0.5)
        z = np.polyfit(dim_mean, sum_R2, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(dim_mean.min(), dim_mean.max(), 100)
        ax1.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=3)
        ax1.set_xlabel('Mean Effective Dimensionality', fontsize=13)
        ax1.set_ylabel('Total Communication (Σ R²)', fontsize=13)
        ax1.set_title('H1: Dimensionality → Communication Capacity',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label('n_sig', fontsize=11)

        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.scatter(dim_mean, max_R2, alpha=0.6, s=100, c=n_sig, cmap='plasma',
                    edgecolor='black', linewidth=0.5)
        z2 = np.polyfit(dim_mean, max_R2, 1)
        p_fit2 = np.poly1d(z2)
        ax2.plot(x_line, p_fit2(x_line), "r--", alpha=0.8, linewidth=3)
        ax2.set_xlabel('Mean Effective Dimensionality', fontsize=13)
        ax2.set_ylabel('Max Communication (Max R²)', fontsize=13)
        ax2.set_title('H1: Dimensionality → Max Strength',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('n_sig', fontsize=11)

        # Row 2: Hypothesis 2 - Bottleneck
        for idx, (vals, name) in enumerate([
            (dim_min, 'Min'),
            (np.array([d['dim_max_pr'] for d in self.all_data]), 'Max'),
            (dim_mean, 'Mean')
        ]):
            ax = fig.add_subplot(gs[1, idx])
            ax.scatter(vals, n_sig, alpha=0.6, s=80, edgecolor='black', linewidth=0.5)
            z = np.polyfit(vals, n_sig, 1)
            p_fit = np.poly1d(z)
            x_line_local = np.linspace(vals.min(), vals.max(), 100)
            r, _ = pearsonr(vals, n_sig)
            ax.plot(x_line_local, p_fit(x_line_local), "r--", alpha=0.8, linewidth=2,
                    label=f'r={r:.3f}')
            ax.set_xlabel(f'{name}(d₁, d₂)', fontsize=12)
            ax.set_ylabel('n_sig' if idx == 0 else '', fontsize=12)
            ax.set_title(f'H2: {name} → n_sig', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        # Row 3: Hypothesis 3 - Dissociation
        ax7 = fig.add_subplot(gs[2, :2])

        median_max_R2 = np.median(max_R2)
        median_n_sig = np.median(n_sig)

        architecture_types = []
        colors_map = {
            'Dedicated': 'red',
            'Distributed': 'blue',
            'Strong+Dist': 'green',
            'Weak+Sparse': 'gray'
        }

        for r2, ns in zip(max_R2, n_sig):
            if r2 > median_max_R2 and ns < median_n_sig:
                architecture_types.append('Dedicated')
            elif r2 < median_max_R2 and ns > median_n_sig:
                architecture_types.append('Distributed')
            elif r2 > median_max_R2 and ns > median_n_sig:
                architecture_types.append('Strong+Dist')
            else:
                architecture_types.append('Weak+Sparse')

        for arch_type in set(architecture_types):
            mask = np.array([t == arch_type for t in architecture_types])
            ax7.scatter(max_R2[mask], n_sig[mask],
                        label=arch_type,
                        color=colors_map[arch_type],
                        alpha=0.7, s=120, edgecolor='black', linewidth=0.5)

        ax7.axvline(x=median_max_R2, color='black', linestyle='--', alpha=0.3, linewidth=2)
        ax7.axhline(y=median_n_sig, color='black', linestyle='--', alpha=0.3, linewidth=2)
        ax7.set_xlabel('Max R²', fontsize=13)
        ax7.set_ylabel('n_sig', fontsize=13)
        ax7.set_title('H3: Architecture Types', fontsize=14, fontweight='bold')
        ax7.legend(fontsize=11)
        ax7.grid(True, alpha=0.3)

        # Architecture distribution
        ax8 = fig.add_subplot(gs[2, 2:])
        from collections import Counter
        type_counts = Counter(architecture_types)
        types_list = list(type_counts.keys())
        counts_list = [type_counts[t] for t in types_list]
        colors_list = [colors_map[t] for t in types_list]

        ax8.bar(range(len(types_list)), counts_list, color=colors_list,
                alpha=0.8, edgecolor='black', linewidth=1.5)
        ax8.set_xticks(range(len(types_list)))
        ax8.set_xticklabels(types_list, fontsize=11)
        ax8.set_ylabel('Count', fontsize=13)
        ax8.set_title('H3: Distribution', fontsize=14, fontweight='bold')
        ax8.grid(True, axis='y', alpha=0.3)

        for i, count in enumerate(counts_list):
            ax8.text(i, count + 0.5, str(count), ha='center', fontsize=12, fontweight='bold')

        plt.suptitle('Comprehensive Dimensionality-Communication Analysis',
                     fontsize=17, fontweight='bold', y=0.995)

        save_path = self.output_dir / "comprehensive_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Comprehensive summary saved: {save_path}")


def main():
    """Main analysis pipeline."""

    # Configuration
    BASE_DIR = "/Users/shengyuancai/Downloads/Oxford_dataset"
    RESULTS_SUBDIR = "sessions_cued_hit_long_results"
    OUTPUT_DIR = "/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/dimensionality_analysis"

    print("=" * 70)
    print("DIMENSIONALITY-COMMUNICATION RELATIONSHIP ANALYSIS")
    print("=" * 70)

    # Initialize analyzer
    analyzer = DimensionalityAnalyzer(
        results_base_dir=BASE_DIR,
        results_subdir=RESULTS_SUBDIR,
        output_dir=OUTPUT_DIR
    )

    # Load all session data
    analyzer.load_and_process_all_sessions()

    # Test three hypotheses
    h1_results = analyzer.test_hypothesis_1()
    h2_results = analyzer.test_hypothesis_2()
    h3_results = analyzer.test_hypothesis_3()

    # Create comprehensive summary
    analyzer.create_comprehensive_summary()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
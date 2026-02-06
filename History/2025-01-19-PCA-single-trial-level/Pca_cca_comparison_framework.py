#!/usr/bin/env python3
"""
PCA-CCA Comparison Framework for Multi-Region Neural Analysis
==============================================================

This script implements a comprehensive geometric and statistical framework
for comparing PCA (unsupervised) and CCA (supervised) dimensionality reduction
results from simultaneously recorded brain regions.

Mathematical Framework:
-----------------------
1. Principal Angles: Quantify similarity between PCA subspaces across regions
2. CCA Participation Ratios: Decompose CCA weights onto PCA bases
3. Temporal Similarity: Cross-correlation between PCA and CCA projections
4. Variance Decomposition: Partition variance into shared vs. private components

Author: Shengyuan Cai
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.linalg import svd, orth
from scipy.stats import pearsonr
import mat73

# Configure visualization
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)


class PCA_CCA_Comparator:
    """
    Comprehensive comparison framework for PCA and CCA results.

    This class implements geometric analyses to understand the relationship
    between variance structure (PCA) and cross-regional coordination (CCA).

    Key Analyses:
    ------------
    1. Subspace geometry: Principal angles between regional PCA subspaces
    2. Weight decomposition: How CCA weights project onto PCA bases
    3. Temporal dynamics: Correlation between PCA and CCA projections
    4. Variance accounting: How much of each region's variance is shared
    """

    def __init__(self, session_data_path: str, output_dir: str):
        """
        Initialize comparator with session data.

        Parameters:
        -----------
        session_data_path : str
            Path to *_analysis_results.mat file containing both PCA and CCA results
        output_dir : str
            Directory for saving analysis outputs
        """
        self.session_path = Path(session_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load session data
        print(f"Loading session data from: {self.session_path}")
        self.session_data = mat73.loadmat(str(self.session_path))

        # Extract PCA and CCA results
        self.pca_results = self.session_data.get('pca_results', None)
        self.cca_results = self.session_data.get('cca_results', None)

        if self.pca_results is None or self.cca_results is None:
            raise ValueError("Session data must contain both pca_results and cca_results")

        # Storage for analysis results
        self.comparison_results = {}

        print("PCA-CCA Comparator initialized successfully")

    def compute_principal_angles(
            self,
            region1: str,
            region2: str,
            n_components: int = 10
    ) -> np.ndarray:
        """
        Compute principal angles between PCA subspaces of two regions.

        Mathematical Background:
        -----------------------
        Given two subspaces S₁ ⊂ ℝⁿ and S₂ ⊂ ℝⁿ with orthonormal bases
        U₁ ∈ ℝⁿˣᵈ and U₂ ∈ ℝⁿˣᵈ, the principal angles θₖ ∈ [0, π/2]
        are defined by:

        cos(θₖ) = max_{u ∈ S₁, v ∈ S₂} u^T v

        subject to u^T u = v^T v = 1 and orthogonality with previous solutions.

        Implementation uses SVD of U₁^T U₂, where singular values σₖ = cos(θₖ).

        Small angles (θ → 0) indicate high subspace similarity, meaning the
        regions have aligned variance structure even before considering their
        temporal coordination (which CCA captures).

        Parameters:
        -----------
        region1, region2 : str
            Names of regions to compare
        n_components : int
            Number of PCA components defining each subspace

        Returns:
        --------
        angles : np.ndarray
            Principal angles in radians, shape (n_components,)
        """
        # Extract PCA bases for each region
        if region1 not in self.pca_results or region2 not in self.pca_results:
            raise ValueError(f"PCA results not available for {region1} or {region2}")

        # Get PCA coefficient matrices (eigenvectors)
        # These should be stored as (n_neurons, n_components)
        pca1 = self.pca_results[region1]
        pca2 = self.pca_results[region2]

        if 'coefficients' not in pca1 or 'coefficients' not in pca2:
            raise ValueError("PCA coefficients not found")

        U1 = pca1['coefficients'][:, :n_components]  # (n_neurons_1, n_components)
        U2 = pca2['coefficients'][:, :n_components]  # (n_neurons_2, n_components)

        # For regions with different neuron counts, we need to work in
        # the joint ambient space or compute angles via projections
        # Here we assume both regions were recorded, so we use the method
        # of computing angles via SVD of the cross-product

        if U1.shape[0] != U2.shape[0]:
            print(f"Warning: {region1} and {region2} have different neuron counts")
            print(f"Cannot compute principal angles directly")
            return None

        # Ensure orthonormality (PCA coefficients should already be orthonormal)
        U1 = orth(U1)
        U2 = orth(U2)

        # Compute cross-product and perform SVD
        M = U1.T @ U2  # (n_components, n_components)
        _, sigma, _ = svd(M)

        # Singular values are cosines of principal angles
        # Clip to [0, 1] to handle numerical errors
        sigma = np.clip(sigma, 0, 1)

        # Convert to angles
        angles = np.arccos(sigma)

        return angles

    def compute_cca_participation_ratios(
            self,
            region1: str,
            region2: str,
            pair_idx: int,
            n_pca_components: int = 10,
            n_cca_components: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Decompose CCA weights onto PCA bases to reveal weight structure.

        Mathematical Framework:
        ----------------------
        For CCA component k with canonical weights a_k (region1) and b_k (region2),
        and PCA bases {w_i} (region1) and {v_j} (region2), compute:

        α_ki = ⟨a_k, w_i⟩² = (a_k^T w_i)²
        β_kj = ⟨b_k, v_j⟩² = (b_k^T v_j)²

        These squared projections reveal how each CCA component is constructed
        from the PCA basis vectors. Key insights:

        - If α_k1 ≈ 1, CCA component k aligns with the first PCA mode (high variance)
        - If α_k is distributed across many i, CCA emerges from multiple PCA modes
        - Comparing α and β patterns reveals whether regions use similar or
          different variance structures to achieve correlation

        Parameters:
        -----------
        region1, region2 : str
            Region names
        pair_idx : int
            Index of region pair in CCA results
        n_pca_components : int
            Number of PCA components to consider
        n_cca_components : int
            Number of CCA components to analyze

        Returns:
        --------
        participation : dict
            Dictionary containing:
            - 'region1_participation': (n_cca, n_pca) matrix of α values
            - 'region2_participation': (n_cca, n_pca) matrix of β values
            - 'region1_effective_dim': effective dimensionality for each CCA component
            - 'region2_effective_dim': effective dimensionality for each CCA component
        """
        # Extract CCA canonical weights
        pair_results = self.cca_results['pair_results'][pair_idx]

        # CCA weights should be stored in mean_A_matrix and mean_B_matrix
        # Shape: (n_neurons, n_components)
        A_matrix = pair_results['mean_A_matrix']  # Region1 weights
        B_matrix = pair_results['mean_B_matrix']  # Region2 weights

        # Extract PCA bases
        pca1 = self.pca_results[region1]
        pca2 = self.pca_results[region2]

        W = pca1['coefficients'][:, :n_pca_components]  # PCA basis for region1
        V = pca2['coefficients'][:, :n_pca_components]  # PCA basis for region2

        # Initialize participation matrices
        n_cca_actual = min(n_cca_components, A_matrix.shape[1])
        participation_1 = np.zeros((n_cca_actual, n_pca_components))
        participation_2 = np.zeros((n_cca_actual, n_pca_components))

        # Compute participation ratios
        for k in range(n_cca_actual):
            a_k = A_matrix[:, k]  # CCA weight vector for region1, component k
            b_k = B_matrix[:, k]  # CCA weight vector for region2, component k

            # Project onto PCA bases and square
            for i in range(n_pca_components):
                participation_1[k, i] = (a_k.T @ W[:, i]) ** 2
                participation_2[k, i] = (b_k.T @ V[:, i]) ** 2

        # Normalize each row to sum to 1 (convert to probability distribution)
        participation_1_norm = participation_1 / (participation_1.sum(axis=1, keepdims=True) + 1e-10)
        participation_2_norm = participation_2 / (participation_2.sum(axis=1, keepdims=True) + 1e-10)

        # Compute effective dimensionality using Shannon entropy
        # High entropy → CCA weight distributed across many PCA modes
        # Low entropy → CCA weight dominated by few PCA modes
        def effective_dim(probs):
            """Compute effective dimensionality from probability distribution."""
            # H = -Σ p_i log(p_i)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            # Effective dimension = exp(H)
            return np.exp(entropy)

        eff_dim_1 = effective_dim(participation_1_norm)
        eff_dim_2 = effective_dim(participation_2_norm)

        results = {
            'region1_participation': participation_1,
            'region2_participation': participation_2,
            'region1_participation_norm': participation_1_norm,
            'region2_participation_norm': participation_2_norm,
            'region1_effective_dim': eff_dim_1,
            'region2_effective_dim': eff_dim_2,
            'region1_name': region1,
            'region2_name': region2
        }

        return results

    def compute_temporal_similarity(
            self,
            region1: str,
            region2: str,
            pair_idx: int,
            n_components: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Compute cross-correlation between PCA and CCA temporal projections.

        Conceptual Framework:
        --------------------
        Even if CCA and PCA weights differ substantially in neuron space,
        their temporal projections might exhibit similar dynamics. This analysis
        reveals whether the two methods capture similar functional patterns
        despite different spatial weightings.

        For each CCA component k and each PCA component i:

        ρ(k,i) = Corr[p_CCA^k(t), p_PCA^i(t)]

        where p_CCA^k(t) and p_PCA^i(t) are the temporal projections.

        High correlations indicate that PCA and CCA identify similar temporal
        structure, just with different emphasis on which neurons contribute.

        Parameters:
        -----------
        region1, region2 : str
            Region names
        pair_idx : int
            Index of region pair in CCA results
        n_components : int
            Number of components to analyze for each method

        Returns:
        --------
        similarity : dict
            Contains correlation matrices and significance values
        """
        # Extract CCA temporal projections
        pair_results = self.cca_results['pair_results'][pair_idx]
        cca_projections = pair_results['projections']

        # Extract PCA temporal projections
        pca1_proj = self.pca_results[region1]['projections']
        pca2_proj = self.pca_results[region2]['projections']

        # Initialize correlation matrices
        # Shape: (n_cca_comp, n_pca_comp) for each region
        corr_region1 = np.zeros((n_components, n_components))
        corr_region2 = np.zeros((n_components, n_components))
        pval_region1 = np.ones((n_components, n_components))
        pval_region2 = np.ones((n_components, n_components))

        # Compute correlations
        for k_cca in range(n_components):
            # Get CCA temporal projection for component k
            cca_comp = cca_projections['components'][k_cca][0]
            cca_proj_1 = cca_comp['region_i_mean'].flatten()
            cca_proj_2 = cca_comp['region_j_mean'].flatten()

            for k_pca in range(n_components):
                # Get PCA temporal projection
                pca_proj_1 = pca1_proj['mean'][k_pca].flatten()
                pca_proj_2 = pca2_proj['mean'][k_pca].flatten()

                # Ensure same length (truncate to minimum)
                min_len = min(len(cca_proj_1), len(pca_proj_1))

                # Compute correlation for region1
                r1, p1 = pearsonr(cca_proj_1[:min_len], pca_proj_1[:min_len])
                corr_region1[k_cca, k_pca] = r1
                pval_region1[k_cca, k_pca] = p1

                # Compute correlation for region2
                min_len = min(len(cca_proj_2), len(pca_proj_2))
                r2, p2 = pearsonr(cca_proj_2[:min_len], pca_proj_2[:min_len])
                corr_region2[k_cca, k_pca] = r2
                pval_region2[k_cca, k_pca] = p2

        results = {
            'region1_correlations': corr_region1,
            'region2_correlations': corr_region2,
            'region1_pvalues': pval_region1,
            'region2_pvalues': pval_region2,
            'region1_name': region1,
            'region2_name': region2
        }

        return results

    def compute_variance_decomposition(
            self,
            region1: str,
            region2: str,
            pair_idx: int,
            n_components: int = 10
    ) -> Dict[str, float]:
        """
        Decompose each region's variance into shared (CCA) and private components.

        Mathematical Framework:
        ----------------------
        For region X with covariance matrix C_X, we decompose the variance as:

        Total Variance = Tr(C_X)

        Shared Variance (via CCA) = Σ_k ρ_k² · Var(X a_k)

        where ρ_k is the canonical correlation and a_k is the canonical weight.

        Private Variance = Variance orthogonal to CCA subspace

        This decomposition reveals what fraction of each region's "signal"
        participates in cross-regional coordination versus remains private.

        Parameters:
        -----------
        region1, region2 : str
            Region names
        pair_idx : int
            Index of region pair
        n_components : int
            Number of variance components to consider

        Returns:
        --------
        decomposition : dict
            Variance fractions for each region and component type
        """
        # Extract CCA results
        pair_results = self.cca_results['pair_results'][pair_idx]
        cv_results = pair_results['cv_results']

        # Get canonical correlations (R² values)
        mean_cv_R2 = cv_results['mean_cv_R2'].flatten()
        n_sig = len(pair_results['significant_components'].flatten())

        # Get CCA weights
        A_matrix = pair_results['mean_A_matrix']
        B_matrix = pair_results['mean_B_matrix']

        # Get PCA explained variance
        pca1 = self.pca_results[region1]
        pca2 = self.pca_results[region2]

        explained_var1 = pca1['explained'][:n_components]
        explained_var2 = pca2['explained'][:n_components]

        # Total variance (from PCA)
        total_var1 = np.sum(explained_var1)
        total_var2 = np.sum(explained_var2)

        # Shared variance (from CCA)
        # For each CCA component k, variance explained = ρ_k² · Var(X a_k)
        # We approximate Var(X a_k) using the PCA variance structure

        shared_var1 = 0
        shared_var2 = 0

        for k in range(min(n_sig, len(mean_cv_R2))):
            r_squared = mean_cv_R2[k]

            # Project CCA weight onto PCA basis to estimate variance
            # This is an approximation: exact computation requires trial data
            a_k = A_matrix[:, k]
            b_k = B_matrix[:, k]

            W1 = pca1['coefficients'][:, :n_components]
            W2 = pca2['coefficients'][:, :n_components]

            # Participation of CCA weight in PCA space
            participation1 = np.array([(a_k.T @ W1[:, i]) ** 2 for i in range(n_components)])
            participation2 = np.array([(b_k.T @ W2[:, i]) ** 2 for i in range(n_components)])

            # Weighted variance contribution
            var_contribution1 = np.sum(participation1 * explained_var1)
            var_contribution2 = np.sum(participation2 * explained_var2)

            shared_var1 += r_squared * var_contribution1
            shared_var2 += r_squared * var_contribution2

        # Private variance
        private_var1 = total_var1 - shared_var1
        private_var2 = total_var2 - shared_var2

        results = {
            'region1_total_variance': total_var1,
            'region2_total_variance': total_var2,
            'region1_shared_variance': shared_var1,
            'region2_shared_variance': shared_var2,
            'region1_private_variance': private_var1,
            'region2_private_variance': private_var2,
            'region1_shared_fraction': shared_var1 / total_var1,
            'region2_shared_fraction': shared_var2 / total_var2,
            'n_significant_components': n_sig,
            'max_canonical_correlation': np.sqrt(mean_cv_R2[0]) if len(mean_cv_R2) > 0 else 0
        }

        return results

    def analyze_region_pair(
            self,
            region1: str,
            region2: str,
            pair_idx: int,
            create_visualizations: bool = True
    ) -> Dict:
        """
        Perform complete PCA-CCA comparison for a region pair.

        This is the main analysis function that orchestrates all comparisons
        and generates comprehensive visualizations.

        Parameters:
        -----------
        region1, region2 : str
            Names of regions to analyze
        pair_idx : int
            Index of this pair in CCA results
        create_visualizations : bool
            Whether to generate and save plots

        Returns:
        --------
        results : dict
            Complete analysis results for this region pair
        """
        print(f"\n{'=' * 70}")
        print(f"Analyzing {region1} vs {region2}")
        print(f"{'=' * 70}")

        results = {
            'region1': region1,
            'region2': region2,
            'pair_idx': pair_idx
        }

        # 1. Principal angles between PCA subspaces
        print("\n1. Computing principal angles between PCA subspaces...")
        try:
            angles = self.compute_principal_angles(region1, region2, n_components=10)
            if angles is not None:
                results['principal_angles'] = angles
                print(f"   Mean angle: {np.mean(np.degrees(angles)):.2f}°")
                print(f"   Min angle: {np.min(np.degrees(angles)):.2f}°")
                print(f"   Max angle: {np.max(np.degrees(angles)):.2f}°")
        except Exception as e:
            print(f"   Could not compute principal angles: {e}")
            results['principal_angles'] = None

        # 2. CCA participation ratios
        print("\n2. Computing CCA participation ratios on PCA bases...")
        try:
            participation = self.compute_cca_participation_ratios(
                region1, region2, pair_idx, n_pca_components=10, n_cca_components=5
            )
            results['participation'] = participation
            print(f"   Region1 effective dimensions: {participation['region1_effective_dim']}")
            print(f"   Region2 effective dimensions: {participation['region2_effective_dim']}")
        except Exception as e:
            print(f"   Could not compute participation ratios: {e}")
            results['participation'] = None

        # 3. Temporal similarity analysis
        print("\n3. Computing temporal correlations between PCA and CCA...")
        try:
            temporal_sim = self.compute_temporal_similarity(
                region1, region2, pair_idx, n_components=5
            )
            results['temporal_similarity'] = temporal_sim
            print(f"   Mean correlation (region1): {np.mean(np.abs(temporal_sim['region1_correlations'])):.3f}")
            print(f"   Mean correlation (region2): {np.mean(np.abs(temporal_sim['region2_correlations'])):.3f}")
        except Exception as e:
            print(f"   Could not compute temporal similarity: {e}")
            results['temporal_similarity'] = None

        # 4. Variance decomposition
        print("\n4. Decomposing variance into shared and private components...")
        try:
            var_decomp = self.compute_variance_decomposition(
                region1, region2, pair_idx, n_components=10
            )
            results['variance_decomposition'] = var_decomp
            print(f"   Region1 shared fraction: {var_decomp['region1_shared_fraction'] * 100:.1f}%")
            print(f"   Region2 shared fraction: {var_decomp['region2_shared_fraction'] * 100:.1f}%")
        except Exception as e:
            print(f"   Could not compute variance decomposition: {e}")
            results['variance_decomposition'] = None

        # Generate visualizations
        if create_visualizations:
            self._create_comparison_visualizations(results)

        # Store in class attribute
        pair_key = f"{region1}_vs_{region2}"
        self.comparison_results[pair_key] = results

        return results

    def _create_comparison_visualizations(self, results: Dict):
        """Create comprehensive visualization panel for PCA-CCA comparison."""
        region1 = results['region1']
        region2 = results['region2']

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Principal angles (if available)
        if results['principal_angles'] is not None:
            ax1 = fig.add_subplot(gs[0, 0])
            angles_deg = np.degrees(results['principal_angles'])
            ax1.plot(range(1, len(angles_deg) + 1), angles_deg, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Principal Angle Index', fontsize=12)
            ax1.set_ylabel('Angle (degrees)', fontsize=12)
            ax1.set_title(f'PCA Subspace Similarity\n{region1} vs {region2}', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 90])

        # Plot 2 & 3: CCA participation ratios
        if results['participation'] is not None:
            part = results['participation']

            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(part['region1_participation_norm'], aspect='auto', cmap='YlOrRd')
            ax2.set_xlabel('PCA Component', fontsize=12)
            ax2.set_ylabel('CCA Component', fontsize=12)
            ax2.set_title(f'{region1}\nCCA Participation on PCA Basis', fontsize=13, fontweight='bold')
            plt.colorbar(im2, ax=ax2, label='Squared Projection')

            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(part['region2_participation_norm'], aspect='auto', cmap='YlOrRd')
            ax3.set_xlabel('PCA Component', fontsize=12)
            ax3.set_ylabel('CCA Component', fontsize=12)
            ax3.set_title(f'{region2}\nCCA Participation on PCA Basis', fontsize=13, fontweight='bold')
            plt.colorbar(im3, ax=ax3, label='Squared Projection')

        # Plot 4 & 5: Temporal correlations
        if results['temporal_similarity'] is not None:
            temp_sim = results['temporal_similarity']

            ax4 = fig.add_subplot(gs[1, 0])
            im4 = ax4.imshow(temp_sim['region1_correlations'], aspect='auto',
                             cmap='RdBu_r', vmin=-1, vmax=1)
            ax4.set_xlabel('PCA Component', fontsize=12)
            ax4.set_ylabel('CCA Component', fontsize=12)
            ax4.set_title(f'{region1}\nTemporal Correlation (PCA vs CCA)', fontsize=13, fontweight='bold')
            plt.colorbar(im4, ax=ax4, label='Correlation')

            ax5 = fig.add_subplot(gs[1, 1])
            im5 = ax5.imshow(temp_sim['region2_correlations'], aspect='auto',
                             cmap='RdBu_r', vmin=-1, vmax=1)
            ax5.set_xlabel('PCA Component', fontsize=12)
            ax5.set_ylabel('CCA Component', fontsize=12)
            ax5.set_title(f'{region2}\nTemporal Correlation (PCA vs CCA)', fontsize=13, fontweight='bold')
            plt.colorbar(im5, ax=ax5, label='Correlation')

        # Plot 6: Variance decomposition
        if results['variance_decomposition'] is not None:
            var_decomp = results['variance_decomposition']

            ax6 = fig.add_subplot(gs[1, 2])

            categories = [region1, region2]
            shared_fracs = [
                var_decomp['region1_shared_fraction'],
                var_decomp['region2_shared_fraction']
            ]
            private_fracs = [
                var_decomp['region1_private_variance'] / var_decomp['region1_total_variance'],
                var_decomp['region2_private_variance'] / var_decomp['region2_total_variance']
            ]

            x = np.arange(len(categories))
            width = 0.35

            ax6.bar(x, shared_fracs, width, label='Shared (CCA)', color='steelblue')
            ax6.bar(x, private_fracs, width, bottom=shared_fracs,
                    label='Private', color='lightcoral')

            ax6.set_ylabel('Variance Fraction', fontsize=12)
            ax6.set_title('Variance Decomposition:\nShared vs Private', fontsize=13, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(categories)
            ax6.legend()
            ax6.set_ylim([0, 1])
            ax6.grid(True, axis='y', alpha=0.3)

        # Plot 7: Effective dimensionality comparison
        if results['participation'] is not None:
            ax7 = fig.add_subplot(gs[2, :])
            part = results['participation']

            n_cca = len(part['region1_effective_dim'])
            x = np.arange(n_cca)
            width = 0.35

            ax7.bar(x - width / 2, part['region1_effective_dim'], width,
                    label=f'{region1} Effective Dim', color='steelblue', alpha=0.8)
            ax7.bar(x + width / 2, part['region2_effective_dim'], width,
                    label=f'{region2} Effective Dim', color='darkgreen', alpha=0.8)

            ax7.set_xlabel('CCA Component', fontsize=12)
            ax7.set_ylabel('Effective Dimensionality', fontsize=12)
            ax7.set_title('CCA Weight Structure: How Many PCA Modes Contribute?',
                          fontsize=13, fontweight='bold')
            ax7.set_xticks(x)
            ax7.set_xticklabels([f'CC{i + 1}' for i in range(n_cca)])
            ax7.legend()
            ax7.grid(True, axis='y', alpha=0.3)

        plt.suptitle(f'PCA-CCA Comprehensive Comparison: {region1} vs {region2}',
                     fontsize=16, fontweight='bold', y=0.98)

        # Save figure
        save_path = self.output_dir / f"pca_cca_comparison_{region1}_vs_{region2}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Visualization saved: {save_path}")


def main_analysis_example():
    """
    Demonstration of the PCA-CCA comparison framework.
    """
    # Configuration - adjust paths for your system
    SESSION_PATH = ("/Users/shengyuancai/Downloads/Oxford_dataset/sessions_cued_hit_long_results/"
                    "yp013_220211_analysis_results.mat")#yp013_220211
    OUTPUT_DIR = "/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/pca_cca_comparison"

    print("=" * 70)
    print("PCA-CCA COMPARISON FRAMEWORK")
    print("=" * 70)

    # Initialize comparator
    comparator = PCA_CCA_Comparator(
        session_data_path=SESSION_PATH,
        output_dir=OUTPUT_DIR
    )

    # Define region pairs to analyze
    # These should correspond to pairs that have CCA results
    region_pairs = [
        ('MOp', 'MOs', 0),  # (region1, region2, pair_idx)
        ('MOp', 'STR', 1),
        ('MOs', 'STR', 2),
        # Add more pairs as needed
    ]

    # Analyze each pair
    for region1, region2, pair_idx in region_pairs:
        try:
            results = comparator.analyze_region_pair(
                region1=region1,
                region2=region2,
                pair_idx=pair_idx,
                create_visualizations=True
            )
        except Exception as e:
            print(f"\nError analyzing {region1} vs {region2}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return comparator


if __name__ == "__main__":
    comparator = main_analysis_example()
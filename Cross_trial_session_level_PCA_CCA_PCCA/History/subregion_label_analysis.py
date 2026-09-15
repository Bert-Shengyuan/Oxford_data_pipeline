#!/usr/bin/env python3
"""
Subregion Label Analysis for the Oxford cued-hit-long-cue-onset dataset
=========================================================================

Standalone script that iterates over every session's
`{session}_analysis_results.mat` file produced by the MATLAB pipeline in
`Matlab_part/` (specifically `perform_region_analysis.m`, which attaches a
`subregion_labels` cell array to every valid brain region and pre-selects a
fixed `target_neurons` = 50 neuron subsample per region via `selected_neurons`).

For both the full neuron population and the 50-neuron subsample, this script:

  1. Extracts all unique subregion labels and exports them to a CSV file.
  2. Computes the cross-session average proportion of each subregion label
     within each brain region, and visualizes the result.
  3. Counts, per session, the number of neurons belonging to each subregion
     label within each brain region, and exports the result to a CSV file.

Data source:
    /Users/shengyuancai/Downloads/Oxford_dataset/cued_hit_long_cue_onset_results
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path('/Users/shengyuancai/Downloads/Oxford_dataset/cued_hit_long_cue_onset_results')
FILE_GLOB = '*_analysis_results.mat'
OUTPUT_DIR = Path('/Users/shengyuancai/Downloads/Oxford_dataset/Paper_output/subregion_label_outputs')

POPULATIONS = {
    'full': 'Full neuron dataset',
    'subset': 'Selected 50-neuron subset',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def decode_h5_string(f: h5py.File, ref) -> str:
    """Decode a MATLAB char-array pointed to by an HDF5 object reference."""
    arr = np.asarray(f[ref][()]).flatten()
    return ''.join(chr(int(c)) for c in arr)


def load_session_subregions(file_path: Path) -> Tuple[str, Dict[str, Dict[str, List[str]]]]:
    """
    Load per-region subregion labels for one session's analysis_results.mat.

    Reads directly with h5py (the files are MATLAB v7.3 / HDF5) and only
    touches the small metadata fields (neuron indices, subregion labels),
    never the large `spike_data` arrays, so this is fast even across many
    sessions.

    Returns:
        (session_name, {region_name: {'full': [labels...], 'subset': [labels...]}})
        'full' holds one label per neuron recorded in that region/session.
        'subset' holds the labels for the 50 neurons in
        region_data.regions.<region>.selected_neurons (the fixed-size,
        cross-region-comparable subsample used by the CCA/PCA pipeline).
    """
    session_name = file_path.name.replace('_analysis_results.mat', '')
    region_labels: Dict[str, Dict[str, List[str]]] = {}

    with h5py.File(file_path, 'r') as f:
        regions_group = f['region_data']['regions']

        for region_name in regions_group.keys():
            g = regions_group[region_name]

            target_neurons = int(np.asarray(g['target_neurons'][()]).flatten()[0])
            if target_neurons <= 0:
                # Region did not meet the minimum-neuron threshold for subsampling.
                continue

            label_refs = g['subregion_labels'][:, 0]
            labels_full = [decode_h5_string(f, ref) for ref in label_refs]

            # MATLAB indices are 1-based.
            selected_idx = np.asarray(g['selected_neurons'][()]).flatten().astype(int) - 1
            labels_subset = [labels_full[i] for i in selected_idx]

            region_labels[region_name] = {'full': labels_full, 'subset': labels_subset}

    return session_name, region_labels


def load_all_sessions(data_dir: Path) -> List[Tuple[str, Dict[str, Dict[str, List[str]]]]]:
    files = sorted(data_dir.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f'No files matching {FILE_GLOB!r} found in {data_dir}')

    sessions = []
    for fp in files:
        sessions.append(load_session_subregions(fp))
    return sessions


# =============================================================================
# TABLE CONSTRUCTION
# =============================================================================

def build_counts_table(
    sessions: List[Tuple[str, Dict[str, Dict[str, List[str]]]]],
    population: str,
) -> pd.DataFrame:
    """
    Long-format table with one row per (session, region, subregion_label):
    the number of neurons carrying that label, the total neuron count for
    that session/region, and the resulting within-session proportion.
    """
    records = []
    for session_name, region_labels in sessions:
        for region_name, populations in region_labels.items():
            labels = populations[population]
            total = len(labels)
            if total == 0:
                continue
            for label, count in Counter(labels).items():
                records.append({
                    'session': session_name,
                    'region': region_name,
                    'subregion_label': label,
                    'n_neurons': count,
                    'total_neurons_in_region': total,
                    'proportion': count / total,
                })

    df = pd.DataFrame.from_records(records)
    return df.sort_values(['region', 'session', 'n_neurons'], ascending=[True, True, False]).reset_index(drop=True)


def build_unique_labels_table(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Unique subregion labels, with the brain region(s) each was observed under."""
    grouped = counts_df.groupby('subregion_label').agg(
        brain_regions=('region', lambda s: ';'.join(sorted(s.unique()))),
        n_sessions_observed=('session', 'nunique'),
        total_neurons_across_dataset=('n_neurons', 'sum'),
    ).reset_index()
    return grouped.sort_values('subregion_label').reset_index(drop=True)


def build_cross_session_proportion_table(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-session average proportion of each subregion label within each
    brain region. Averaging is over the sessions in which that region was
    present; sessions where the region occurs but a given label is absent
    contribute a proportion of 0 for that label.
    """
    region_sessions = counts_df.groupby('region')['session'].unique()
    region_labels = counts_df.groupby('region')['subregion_label'].unique()

    # Build the full (region, session, label) grid for sessions where the
    # region is present, then left-merge observed proportions onto it.
    grid_records = []
    for region_name, sessions_with_region in region_sessions.items():
        for label in region_labels[region_name]:
            for session_name in sessions_with_region:
                grid_records.append((region_name, session_name, label))
    grid = pd.DataFrame(grid_records, columns=['region', 'session', 'subregion_label'])

    merged = grid.merge(
        counts_df[['region', 'session', 'subregion_label', 'proportion']],
        on=['region', 'session', 'subregion_label'],
        how='left',
    )
    merged['proportion'] = merged['proportion'].fillna(0.0)

    summary = merged.groupby(['region', 'subregion_label']).agg(
        mean_proportion=('proportion', 'mean'),
        std_proportion=('proportion', 'std'),
        n_sessions_with_region=('proportion', 'size'),
    ).reset_index()
    summary['std_proportion'] = summary['std_proportion'].fillna(0.0)

    return summary.sort_values(['region', 'mean_proportion'], ascending=[True, False]).reset_index(drop=True)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_region_proportions(prop_df: pd.DataFrame, tag: str, output_path: Path) -> None:
    """Grid of bar charts (one per brain region) of mean subregion proportion."""
    regions = sorted(prop_df['region'].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(regions) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, region_name in enumerate(regions):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = prop_df[prop_df['region'] == region_name]
        n_sessions = sub['n_sessions_with_region'].iloc[0]

        ax.bar(
            sub['subregion_label'], sub['mean_proportion'],
            yerr=sub['std_proportion'], capsize=3,
            color='#4C72B0', edgecolor='black', linewidth=0.5,
        )
        ax.set_title(f'{region_name} (n={n_sessions} sessions)')
        ax.set_ylabel('Mean proportion')
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.set_ylim(0, 1.0)

    for idx in range(len(regions), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    fig.suptitle(
        f'Cross-session average subregion proportion within each brain region\n({POPULATIONS[tag]})',
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150)
    print(f'  Saved figure: {output_path}')
    plt.show()


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    sessions: List[Tuple[str, Dict[str, Dict[str, List[str]]]]],
    population: str,
) -> None:
    print(f'\n--- {POPULATIONS[population]} ---')

    counts_df = build_counts_table(sessions, population)

    # Step 1: unique subregion labels.
    unique_df = build_unique_labels_table(counts_df)
    unique_path = OUTPUT_DIR / f'unique_subregion_labels_{population}.csv'
    unique_df.to_csv(unique_path, index=False)
    print(f'  [1] {len(unique_df)} unique subregion labels -> {unique_path}')
    print(unique_df.to_string(index=False))

    # Step 2: cross-session average proportion per region, + visualization.
    prop_df = build_cross_session_proportion_table(counts_df)
    prop_path = OUTPUT_DIR / f'subregion_proportion_by_region_{population}.csv'
    prop_df.to_csv(prop_path, index=False)
    print(f'\n  [2] Cross-session average proportions -> {prop_path}')
    plot_path = OUTPUT_DIR / f'subregion_proportion_by_region_{population}.png'
    plot_region_proportions(prop_df, population, plot_path)

    # Step 3: per-session neuron counts per subregion label within each region.
    counts_path = OUTPUT_DIR / f'subregion_counts_per_session_{population}.csv'
    counts_df.to_csv(counts_path, index=False)
    print(f'\n  [3] Per-session subregion neuron counts -> {counts_path}')
    print(counts_df.head(20).to_string(index=False))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f'Loading sessions from {DATA_DIR} ...')
    sessions = load_all_sessions(DATA_DIR)
    print(f'Loaded {len(sessions)} sessions.')

    for population in POPULATIONS:
        run_pipeline(sessions, population)


if __name__ == '__main__':
    main()

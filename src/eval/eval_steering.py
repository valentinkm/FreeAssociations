#!/usr/bin/env python3
"""
evaluate_yoked_generation.py - Compares the alignment of yoked-steered vs.
non-steered models against specific human demographic subgroups.

(v6 - Improves plot aesthetics and correctness by setting y-axis limit
 and filtering noisy categories.)
"""

import json
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import re

# --- Setup Project Path ---
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Import from existing project ---
try:
    from src.extract_profiles import age_bin, norm_gender, native_bin, edu_bin, slugify
except ImportError as e:
    print(f"❌ Error: Could not import from 'src/extract_profiles.py'. Ensure file exists.")
    sys.exit(1)

# --- Configuration & Paths ---
STEERED_DATA_PATH = PROJECT_ROOT / "runs" / "yoked_generation" / "yoked_steered_responses.jsonl"
NONSTEERED_DATA_PATH = PROJECT_ROOT / "runs" / "yoked_generation" / "yoked_nonsteered_responses.jsonl"
HUMAN_DATA_PATH = PROJECT_ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"
PLOT_OUTPUT_PATH = PROJECT_ROOT / "plots" / "yoked_steering_subgroup_validation_v4.png"

# --- Helper Function ---
def wjacc(h_ctr, m_ctr):
    """Calculates the weighted Jaccard similarity between two Counters."""
    if not isinstance(h_ctr, Counter) or not isinstance(m_ctr, Counter) or not h_ctr or not m_ctr:
        return 0.0
    all_keys = h_ctr.keys() | m_ctr.keys()
    inter = sum(min(h_ctr.get(k, 0), m_ctr.get(k, 0)) for k in all_keys)
    union = sum(max(h_ctr.get(k, 0), m_ctr.get(k, 0)) for k in all_keys)
    return inter / union if union else 0.0

# --- Main Functions ---

def load_and_process_data():
    """Loads and processes all human and LLM data, returning structured dataframes."""
    print("🔄 Loading and processing all data sources...")
    
    # Load Human Data and create profile IDs
    df_human = pd.read_csv(HUMAN_DATA_PATH)
    df_human.dropna(subset=['cue', 'age', 'gender', 'nativeLanguage', 'country', 'education', 'R1', 'R2', 'R3'], how='any', inplace=True)
    df_human['age_bin'] = df_human['age'].apply(age_bin)
    df_human['gender_bin'] = df_human['gender'].apply(norm_gender)
    df_human['country_bin'] = df_human['country'].str.strip().apply(slugify).replace('', 'unspecified')
    df_human['education_bin'] = df_human['education'].apply(edu_bin)
    df_human['native_bin'] = df_human['nativeLanguage'].apply(native_bin)
    df_human_melted = pd.melt(df_human, id_vars=['cue', 'age_bin', 'gender_bin', 'country_bin', 'education_bin'], value_vars=['R1', 'R2', 'R3'], value_name='response').dropna()
    df_human_melted['response'] = df_human_melted['response'].str.lower()
    
    # Load Steered LLM Data
    steered_records = []
    with open(STEERED_DATA_PATH, 'r') as f:
        for line in f:
            record = json.loads(line)
            for resp in record.get('responses', []):
                profile_parts = re.match(
                    r"profile_age_(?P<age_bin>.+?)_gender_(?P<gender_bin>.+?)_native_(?P<native_bin>.+?)_country_(?P<country_bin>.+?)_edu_(?P<education_bin>.+)",
                    resp['profile_id']
                )
                if profile_parts:
                    parts_dict = profile_parts.groupdict()
                    for word in resp.get('set', []):
                        steered_records.append({
                            'cue': record['cue'],
                            'age_bin': parts_dict['age_bin'],
                            'gender_bin': parts_dict['gender_bin'],
                            'country_bin': parts_dict['country_bin'],
                            'education_bin': parts_dict['education_bin'],
                            'response': word.lower()
                        })
    df_steered = pd.DataFrame(steered_records)

    # Load and Aggregate Non-Steered LLM Data
    nonsteered_data = {}
    with open(NONSTEERED_DATA_PATH, 'r') as f:
        for line in f:
            record = json.loads(line)
            cue = record['cue']
            all_words = [word.lower() for resp in record.get('responses', []) for word in resp.get('set', [])]
            if all_words: nonsteered_data[cue] = Counter(all_words)

    print("✅ All data loaded and processed.")
    return df_human_melted, df_steered, nonsteered_data

def run_evaluation():
    """Main function to run the full validation analysis and generate plots."""
    
    df_human, df_steered, nonsteered_data_agg = load_and_process_data()
    
    print("\n--- Calculating Alignment Scores by Subgroup ---")
    
    demographic_axes = ['gender_bin', 'age_bin', 'education_bin', 'country_bin']
    all_results = []

    for axis in tqdm(demographic_axes, desc="Processing Demographic Axes"):
        unique_subgroups = df_human[axis].unique()
        for subgroup in unique_subgroups:
            human_subgroup_df = df_human[df_human[axis] == subgroup]
            steered_subgroup_df = df_steered[df_steered[axis] == subgroup]

            human_norms = {cue: Counter(grp['response']) for cue, grp in human_subgroup_df.groupby('cue')}
            steered_norms = {cue: Counter(grp['response']) for cue, grp in steered_subgroup_df.groupby('cue')}
            
            for cue, h_ctr in human_norms.items():
                s_ctr = steered_norms.get(cue, Counter())
                ns_ctr = nonsteered_data_agg.get(cue, Counter())
                
                if s_ctr and ns_ctr:
                    steered_alignment_score = wjacc(h_ctr, s_ctr)
                    
                    if steered_alignment_score == 0:
                        print("\n--- DEBUG: Zero Alignment Detected ---")
                        print(f"Cue: '{cue}', Subgroup: {axis}='{subgroup}'")
                        print(f"  - Top 5 Human Norms:   {h_ctr.most_common(5)}")
                        print(f"  - Top 5 Steered Norms: {s_ctr.most_common(5)}")
                        print("--------------------------------------")

                    all_results.append({
                        'axis': axis.replace('_bin', ''),
                        'subgroup': subgroup,
                        'cue': cue,
                        'steered_alignment': steered_alignment_score,
                        'nonsteered_alignment': wjacc(h_ctr, ns_ctr)
                    })
    
    results_df = pd.DataFrame(all_results)
    
    # --- Generating Improved Plots ---
    print("\n--- Generating Subgroup Plots ---")
    PLOT_OUTPUT_PATH.parent.mkdir(exist_ok=True)
    
    plot_df = results_df.melt(
        id_vars=['axis', 'subgroup', 'cue'],
        value_vars=['steered_alignment', 'nonsteered_alignment'],
        var_name='Model Type', value_name='Alignment (WJacc)'
    )
    plot_df['Model Type'] = plot_df['Model Type'].str.replace('_alignment', '').str.title()
    
    # <<< FIX: Filter out 'other' gender for a cleaner plot >>>
    plot_df = plot_df[plot_df['subgroup'] != 'other']
    
    top_countries = results_df[results_df['axis'] == 'country']['subgroup'].value_counts().nlargest(4).index
    plot_df['subgroup_display'] = plot_df.apply(
        lambda row: row['subgroup'] if row['axis'] != 'country' or row['subgroup'] in top_countries else 'other',
        axis=1
    )

    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()
    
    axis_map = {'gender': 0, 'age': 1, 'education': 2, 'country': 3}
    
    for axis_name, ax_idx in axis_map.items():
        ax = axes[ax_idx]
        axis_data = plot_df[plot_df['axis'] == axis_name]
        
        # <<< FIX: Sort order numerically for education for better presentation >>>
        if axis_name == 'education':
            order = sorted(axis_data['subgroup_display'].unique(), key=lambda x: int(x))
        else:
            order = sorted(axis_data['subgroup_display'].unique())
        
        sns.violinplot(
            data=axis_data, x='subgroup_display', y='Alignment (WJacc)',
            hue='Model Type', ax=ax, order=order, split=True, inner='quartile',
            palette={'Steered': 'lightblue', 'Nonsteered': 'lightcoral'}
        )
        ax.set_title(axis_name.title(), fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('Alignment (WJacc)' if ax_idx % 2 == 0 else '')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # <<< FIX: Set y-axis lower limit to 0 >>>
        ax.set_ylim(bottom=0)
        
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add a single, shared legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Model Type', loc='upper right', bbox_to_anchor=(0.98, 0.95))

    fig.suptitle('Subgroup Alignment: Steered vs. Non-Steered Model', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    
    print(f"\n✅ Plot saved to '{PLOT_OUTPUT_PATH}'")

if __name__ == "__main__":
    run_evaluation()

#!/usr/bin/env python3
"""
analyze_spp_human_baseline.py - A script to test if human free-association
data can predict human semantic priming effects using the "Holistic Similarity"
method.

This script serves as a "gold standard" baseline to compare against the LLM's
performance on the same task. It makes no API calls.
"""

import sys
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import cosine

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# --- Setup Project Path ---
PROJECT_ROOT = Path.cwd()

# --- Configuration ---
SPP_DATA_PATH = PROJECT_ROOT / "generalize_spp" / "spp_naming_data_raw.xlsx"
SWOW_DATA_PATH = PROJECT_ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"
PLOT_OUTPUT_PATH = PROJECT_ROOT / "plots" / "spp_human_baseline_analysis.png"

SPP_RELATED_COND = 1
SPP_UNRELATED_COND = 2

# --- Main Functions ---

def load_human_lexicon() -> dict:
    """
    Loads SWOW data and creates a 'human lexicon' where each word maps to a
    Counter of its free associations.
    """
    print("--- Step 1: Loading Human Free-Association Lexicon ---")
    try:
        df = pd.read_csv(SWOW_DATA_PATH, usecols=['cue', 'R1', 'R2', 'R3'])
    except FileNotFoundError:
        print(f"❌ ERROR: SWOW file not found at '{SWOW_DATA_PATH}'.")
        return {}

    # Create a frequency counter for each cue word's associations
    human_lexicon = {}
    for cue, grp in tqdm(df.groupby("cue"), desc="Building Human Lexicon"):
        # Combine all three response columns, drop NaNs, and count frequencies
        all_responses = pd.concat([grp['R1'], grp['R2'], grp['R3']]).dropna().str.lower()
        human_lexicon[cue.lower()] = Counter(all_responses)
        
    print(f"✅ Loaded {len(human_lexicon)} word vectors from SWOW data.")
    return human_lexicon

def prepare_priming_data() -> pd.DataFrame:
    """Loads and processes SPP data to return a clean DataFrame of related pairs."""
    print("\n--- Step 2: Preparing SPP Priming Data ---")
    try:
        df = pd.read_excel(SPP_DATA_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: SPP data file not found at '{SPP_DATA_PATH}'.")
        return pd.DataFrame()

    df = df[df['target.ACC'] == 1.0].copy()
    df['target.RT'] = pd.to_numeric(df['target.RT'], errors='coerce')
    df.dropna(subset=['target.RT', 'prime', 'target'], inplace=True)
    df['log_RT'] = np.log(df['target.RT'])
    
    df_filtered = df[df['primecond'].isin([SPP_RELATED_COND, SPP_UNRELATED_COND])]
    rt_means = df_filtered.groupby(['target', 'primecond'])['log_RT'].mean().reset_index()
    rt_pivot = rt_means.pivot_table(index='target', columns='primecond', values='log_RT').reset_index()
    rt_pivot.rename(columns={SPP_RELATED_COND: 'mean_log_RT_related', SPP_UNRELATED_COND: 'mean_log_RT_unrelated'}, inplace=True)
    rt_pivot.dropna(subset=['mean_log_RT_related', 'mean_log_RT_unrelated'], inplace=True)
    rt_pivot['priming_effect'] = rt_pivot['mean_log_RT_unrelated'] - rt_pivot['mean_log_RT_related']
    
    priming_effects = rt_pivot[['target', 'priming_effect']]
    related_pairs = df[df['primecond'] == SPP_RELATED_COND][['prime', 'target']].drop_duplicates()
    
    spp_data = pd.merge(related_pairs, priming_effects, on='target')
    print(f"✅ Prepared {len(spp_data)} unique related (prime, target) pairs.")
    return spp_data


def analyze_human_baseline(spp_data: pd.DataFrame, human_lexicon: dict):
    """Performs the main analysis using the human lexicon."""
    print("\n--- Step 3: Analyzing Human Data Baseline ---")
    
    # Filter SPP pairs to only include those where both words exist in the human lexicon
    available_words = set(human_lexicon.keys())
    spp_data_analyzable = spp_data[
        spp_data['prime'].str.lower().isin(available_words) &
        spp_data['target'].str.lower().isin(available_words)
    ].copy()

    print(f"🔄 Calculating associative cosine similarity for {len(spp_data_analyzable)} pairs...")
    results = []
    for task in tqdm(spp_data_analyzable.itertuples(), total=len(spp_data_analyzable), desc="Calculating Similarity"):
        prime_vec = human_lexicon.get(task.prime.lower())
        target_vec = human_lexicon.get(task.target.lower())

        if not prime_vec or not target_vec:
            continue
        
        vocab = sorted(list(prime_vec.keys() | target_vec.keys()))
        v1 = [prime_vec.get(word, 0) for word in vocab]
        v2 = [target_vec.get(word, 0) for word in vocab]
        similarity = 1 - cosine(v1, v2) if np.any(v1) and np.any(v2) else 0

        results.append({
            "prime": task.prime, "target": task.target,
            "priming_effect": task.priming_effect,
            "human_associative_similarity": similarity,
        })
    
    if not results:
        print("❌ No results generated. This is unexpected.")
        return
        
    analysis_df = pd.DataFrame(results)

    # Run regression and plot
    print("🔄 Running final regression analysis...")
    y = analysis_df['priming_effect']
    X = sm.add_constant(analysis_df['human_associative_similarity'])
    model = sm.OLS(y, X).fit()
    print("\n--- Human Baseline Regression Results ---")
    print(model.summary())

    PLOT_OUTPUT_PATH.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 7))
    ax = sns.regplot(x='human_associative_similarity', y='priming_effect', data=analysis_df,
                     scatter_kws={'alpha': 0.4, 'color': 'blue'}, line_kws={'color': '#00008B'}) # Dark blue
    ax.set_title('Human Baseline: Associative Similarity vs. Semantic Priming Effect', fontsize=16)
    ax.set_xlabel('Human Associative Cosine Similarity (from SWOW)', fontsize=12)
    ax.set_ylabel('Semantic Priming Effect (Log RT Difference)', fontsize=12)
    stats_text = f'$R^2 = {model.rsquared:.3f}$\n$p = {model.pvalues["human_associative_similarity"]:.3g}$'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='#ADD8E6', alpha=0.5))
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analysis complete. Plot saved to '{PLOT_OUTPUT_PATH}'")


if __name__ == "__main__":
    human_lexicon = load_human_lexicon()
    spp_test_data = prepare_priming_data()
    
    if not spp_test_data.empty and human_lexicon:
        analyze_human_baseline(spp_test_data, human_lexicon)

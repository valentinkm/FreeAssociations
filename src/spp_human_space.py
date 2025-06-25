#!/usr/bin/env python3
"""
analyze_spp_association_space.py - A script that tests if the semantic
distance within a human free-association space can predict semantic priming.

This script implements the following logic:
1.  Trains a Word2Vec model on the SWOW free-association data to create a
    "human association space".
2.  Loads and processes the SPP data to get prime-target pairs and their
    priming effects.
3.  For each pair, calculates the cosine similarity between the prime and
    target vectors from the trained association space model.
4.  Runs a regression to test: priming_effect ~ human_semantic_distance.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress warnings from gensim about model updates
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# --- Required: Install gensim for Word2Vec -> pip install gensim ---
try:
    from gensim.models import Word2Vec
except ImportError:
    print("❌ Error: The 'gensim' library is required for this script.")
    print("   Please install it by running: pip install gensim")
    sys.exit(1)

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# --- Setup Project Path ---
PROJECT_ROOT = Path.cwd()

# --- Configuration ---
SPP_DATA_PATH = PROJECT_ROOT / "generalize_spp" / "spp_naming_data_raw.xlsx"
# Using the complete SWOW data is better for training a rich model
SWOW_DATA_PATH = PROJECT_ROOT / "Small World of Words" / "SWOW-EN.complete.20180827.csv"
PLOT_OUTPUT_PATH = PROJECT_ROOT / "plots" / "spp_association_space_analysis.png"

SPP_RELATED_COND = 1
SPP_UNRELATED_COND = 2

# Word2Vec Model Parameters
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 5 # Ignore words with very few associations

# --- Main Functions ---

def train_association_space_model() -> Word2Vec:
    """
    Loads SWOW data, treats it as a corpus, and trains a Word2Vec model.
    """
    print("--- Step 1: Training Human Association Space Model (Word2Vec) ---")
    try:
        df = pd.read_csv(SWOW_DATA_PATH, usecols=['cue', 'R1', 'R2', 'R3'])
        df.dropna(subset=['cue'], inplace=True)
    except FileNotFoundError:
        print(f"❌ ERROR: SWOW data file not found at '{SWOW_DATA_PATH}'.")
        return None

    print("🔄 Preparing corpus from SWOW associations...")
    # Create "sentences" where each sentence is a cue followed by its associations
    corpus = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Creating Corpus"):
        sentence = [str(row.cue).lower()]
        if pd.notna(row.R1): sentence.append(str(row.R1).lower())
        if pd.notna(row.R2): sentence.append(str(row.R2).lower())
        if pd.notna(row.R3): sentence.append(str(row.R3).lower())
        corpus.append(sentence)

    print("🔄 Training Word2Vec model... (This may take a few minutes)")
    model = Word2Vec(
        sentences=corpus,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4 # Use multiple cores if available
    )
    print(f"✅ Model trained. Vocabulary size: {len(model.wv.key_to_index)}")
    return model

def prepare_spp_data() -> pd.DataFrame:
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


def analyze_semantic_distance(spp_data: pd.DataFrame, w2v_model: Word2Vec):
    """Performs the main analysis using the trained Word2Vec model."""
    print("\n--- Step 3: Analyzing Semantic Distance vs. Priming ---")
    
    print("🔄 Calculating semantic distance (cosine similarity) for each pair...")
    results = []
    for task in tqdm(spp_data.itertuples(), total=len(spp_data), desc="Calculating Similarity"):
        prime = str(task.prime).lower()
        target = str(task.target).lower()
        
        # Check if both words exist in the trained model's vocabulary
        if prime in w2v_model.wv and target in w2v_model.wv:
            similarity = w2v_model.wv.similarity(prime, target)
            results.append({
                "prime": prime,
                "target": target,
                "priming_effect": task.priming_effect,
                "human_semantic_distance": similarity,
            })

    if not results:
        print("❌ No results generated. Check for overlap between SPP words and the trained model vocabulary.")
        return
        
    analysis_df = pd.DataFrame(results)
    print(f"✅ Calculated semantic distance for {len(analysis_df)} pairs.")

    print("🔄 Running final regression analysis...")
    y = analysis_df['priming_effect']
    X = sm.add_constant(analysis_df['human_semantic_distance'])
    model = sm.OLS(y, X).fit()
    print("\n--- Human Association Space Regression Results ---")
    print(model.summary())

    PLOT_OUTPUT_PATH.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 7))
    ax = sns.regplot(x='human_semantic_distance', y='priming_effect', data=analysis_df,
                     scatter_kws={'alpha': 0.4, 'color': '#2E86C1'}, line_kws={'color': '#1B4F72'})
    ax.set_title('Human Baseline: Semantic Distance vs. Priming Effect', fontsize=16)
    ax.set_xlabel('Human Semantic Distance (Cosine Similarity from Word2Vec)', fontsize=12)
    ax.set_ylabel('Semantic Priming Effect (Log RT Difference)', fontsize=12)
    stats_text = f'$R^2 = {model.rsquared:.3f}$\n$p = {model.pvalues["human_semantic_distance"]:.3g}$'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='#D6EAF8', alpha=0.7))
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analysis complete. Plot saved to '{PLOT_OUTPUT_PATH}'")


if __name__ == "__main__":
    association_model = train_association_space_model()
    spp_test_data = prepare_spp_data()
    
    if not spp_test_data.empty and association_model:
        analyze_semantic_distance(spp_test_data, association_model)


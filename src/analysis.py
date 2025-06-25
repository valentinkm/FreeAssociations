"""
analysis.py
───────────
Core analysis functions for comparing human vs. LLM data and running regressions.
"""
import pandas as pd
import numpy as np
import json
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import cosine

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Import paths, constants, and data loading functions
from . import settings
from . import data_utils

def _robust_flatten(items):
    """Recursively flattens a list that may contain nested lists of strings."""
    for x in items:
        if isinstance(x, list):
            yield from _robust_flatten(x)
        elif isinstance(x, str):
            yield x.lower()

def analyze_holistic_similarity(spp_data_to_analyze: pd.DataFrame):
    """Performs the LLM holistic similarity analysis using the pre-generated lexicon."""
    print("\n--- Analyzing Holistic Similarity (LLM vs. Human Priming) ---")
    
    try:
        print(f"🔄 Loading LLM Association Lexicon from '{settings.LEXICON_PATH}'...")
        llm_lexicon = {}
        with open(settings.LEXICON_PATH, 'r') as f:
            for line in f:
                record = json.loads(line)
                all_words = list(_robust_flatten(record.get('association_sets', [])))
                llm_lexicon[record['word']] = Counter(all_words)
        print(f"✅ Loaded {len(llm_lexicon)} word vectors from lexicon.")
    except FileNotFoundError:
        print(f"❌ ERROR: Lexicon file not found at '{settings.LEXICON_PATH}'.")
        return

    print("🔄 Filtering SPP pairs to match available lexicon data...")
    available_words_in_lexicon = set(llm_lexicon.keys())
    spp_data_analyzable = spp_data_to_analyze[
        spp_data_to_analyze['prime'].str.lower().isin(available_words_in_lexicon) &
        spp_data_to_analyze['target'].str.lower().isin(available_words_in_lexicon)
    ].copy()

    if spp_data_analyzable.empty:
        print("❌ No SPP pairs found where both prime and target exist in the generated lexicon.")
        return

    print(f"✅ Found {len(spp_data_analyzable)} pairs to analyze.")

    print("🔄 Calculating associative cosine similarity for each pair...")
    results = []
    for task in tqdm(spp_data_analyzable.itertuples(), total=len(spp_data_analyzable), desc="Calculating Similarity"):
        prime_vec = llm_lexicon.get(task.prime.lower())
        target_vec = llm_lexicon.get(task.target.lower())

        if not prime_vec or not target_vec: continue
        
        vocab = sorted(list(prime_vec.keys() | target_vec.keys()))
        v1 = [prime_vec.get(word, 0) for word in vocab]
        v2 = [target_vec.get(word, 0) for word in vocab]
        similarity = 1 - cosine(v1, v2) if np.any(v1) and np.any(v2) else 0

        results.append({
            "prime": task.prime, "target": task.target,
            "priming_effect": task.priming_effect,
            "associative_cosine_similarity": similarity,
        })
    
    if not results:
        print("❌ No results generated. This is unexpected with the new sampling method.")
        return
        
    analysis_df = pd.DataFrame(results)

    if len(analysis_df) < 2:
        print("\n❌ Error: Not enough valid pairs with data in the lexicon to run a regression.")
        return

    print("🔄 Running final regression analysis...")
    y = analysis_df['priming_effect']
    X = sm.add_constant(analysis_df['associative_cosine_similarity'])
    model = sm.OLS(y, X).fit()
    print("\n--- Holistic Similarity Regression Results ---")
    print(model.summary())

    plt.figure(figsize=(10, 7))
    ax = sns.regplot(x='associative_cosine_similarity', y='priming_effect', data=analysis_df,
                      scatter_kws={'alpha': 0.5}, line_kws={'color': '#4C72B0'})
    ax.set_title('Generalizability: LLM Associative Similarity vs. Semantic Priming Effect', fontsize=16)
    ax.set_xlabel('LLM Associative Cosine Similarity', fontsize=12)
    ax.set_ylabel('Semantic Priming Effect (Log RT Difference)', fontsize=12)
    stats_text = f'$R^2 = {model.rsquared:.3f}$\n$p = {model.pvalues["associative_cosine_similarity"]:.3g}$'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='#AED6F1', alpha=0.5))
    plt.savefig(settings.PLOT_HOLISTIC_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analysis complete. Plot saved to '{settings.PLOT_HOLISTIC_PATH}'")

def analyze_human_baseline():
    """Performs the human baseline analysis (SWOW vs. Human Priming)."""
    print("\n--- Analyzing Human Data Baseline ---")
    
    human_lexicon = data_utils.load_human_lexicon()
    spp_data = data_utils.prepare_priming_data()
    
    if not human_lexicon or spp_data.empty:
        print("❌ Cannot run human baseline analysis due to missing data.")
        return

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

        if not prime_vec or not target_vec: continue
        
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

    print("🔄 Running final regression analysis...")
    y = analysis_df['priming_effect']
    X = sm.add_constant(analysis_df['human_associative_similarity'])
    model = sm.OLS(y, X).fit()
    print("\n--- Human Baseline Regression Results ---")
    print(model.summary())

    plt.figure(figsize=(10, 7))
    ax = sns.regplot(x='human_associative_similarity', y='priming_effect', data=analysis_df,
                      scatter_kws={'alpha': 0.4, 'color': 'blue'}, line_kws={'color': '#00008B'})
    ax.set_title('Human Baseline: Associative Similarity vs. Semantic Priming Effect', fontsize=16)
    ax.set_xlabel('Human Associative Cosine Similarity (from SWOW)', fontsize=12)
    ax.set_ylabel('Semantic Priming Effect (Log RT Difference)', fontsize=12)
    stats_text = f'$R^2 = {model.rsquared:.3f}$\n$p = {model.pvalues["human_associative_similarity"]:.3g}$'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='#ADD8E6', alpha=0.5))
    plt.savefig(settings.PLOT_HUMAN_BASELINE_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analysis complete. Plot saved to '{settings.PLOT_HUMAN_BASELINE_PATH}'")

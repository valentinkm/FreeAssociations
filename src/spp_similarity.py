#!/usr/bin/env python3
"""
analyze_spp_holistic_v2.py - A "smarter" script that first samples
(prime, target) pairs and then generates data only for the necessary words.

This script ensures maximum efficiency for a targeted probe by guaranteeing
that all generated data is usable in the final analysis.
"""

import sys
import pandas as pd
import numpy as np
import json
import argparse
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import cosine

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# --- Setup Project Path ---
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Import from existing project ---
try:
    from src.run_lm import call_model
except ImportError as e:
    print(f"❌ Error: Could not import from 'src'. Ensure required files exist.")
    sys.exit(1)

# --- Configuration ---
SPP_DATA_PATH = PROJECT_ROOT / "generalize_spp" / "spp_naming_data_raw.xlsx"
LEXICON_PATH = PROJECT_ROOT / "runs" / "llm_association_lexicon.jsonl"
PLOT_OUTPUT_PATH = PROJECT_ROOT / "plots" / "spp_holistic_similarity_analysis_v2.png"
SWOW_DATA_PATH = PROJECT_ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"


PROMPT_TO_USE = "participant_default_question"
NUM_SETS_PER_WORD = 25

NUM_PAIRS_TO_PROBE = 50
RANDOM_SEED = 42
SPP_RELATED_COND = 1
SPP_UNRELATED_COND = 2

# --- Main Functions ---

def get_and_sample_spp_pairs() -> tuple[pd.DataFrame, set]:
    """
    Loads SPP data, intelligently samples pairs by prioritizing those with
    existing lexicon data, and returns the pairs and required vocabulary.
    """
    print("--- Step 1: Intelligently Sampling (Prime, Target) Pairs ---")
    
    def prepare_priming_data(path):
        df = pd.read_excel(path)
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
        return pd.merge(related_pairs, priming_effects, on='target')

    all_pairs_df = prepare_priming_data(SPP_DATA_PATH)
    
    try:
        swow_cues = set(pd.read_csv(SWOW_DATA_PATH, usecols=['cue'])['cue'].dropna().str.lower().unique())
        all_pairs_df = all_pairs_df[
            all_pairs_df['prime'].str.lower().isin(swow_cues) &
            all_pairs_df['target'].str.lower().isin(swow_cues)
        ].copy()
    except FileNotFoundError:
        print(f"⚠️ Warning: SWOW file not found. Skipping SWOW filtering.")

    print(f"🔄 Checking existing lexicon at '{LEXICON_PATH}'...")
    processed_words = set()
    if LEXICON_PATH.exists():
        with open(LEXICON_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed_words.add(json.loads(line)['word'].lower())
                except (json.JSONDecodeError, KeyError): continue
    
    pairs_ready = all_pairs_df[
        all_pairs_df['prime'].str.lower().isin(processed_words) &
        all_pairs_df['target'].str.lower().isin(processed_words)
    ]
    pairs_needing_generation = all_pairs_df.drop(pairs_ready.index)

    print(f"✅ Found {len(pairs_ready)} pairs ready for immediate analysis.")

    if len(pairs_ready) >= NUM_PAIRS_TO_PROBE:
        print(f"🔄 Sampling {NUM_PAIRS_TO_PROBE} pairs from the 'ready' pool...")
        sampled_pairs_df = pairs_ready.sample(n=NUM_PAIRS_TO_PROBE, random_state=RANDOM_SEED)
    else:
        print(f"Taking all {len(pairs_ready)} ready pairs.")
        num_more_needed = NUM_PAIRS_TO_PROBE - len(pairs_ready)
        if len(pairs_needing_generation) < num_more_needed:
            print(f"⚠️ Warning: Not enough remaining pairs. Taking all {len(pairs_needing_generation)} available.")
            num_more_needed = len(pairs_needing_generation)
        
        print(f"🔄 Sampling an additional {num_more_needed} pairs that require generation...")
        newly_sampled_pairs = pairs_needing_generation.sample(n=num_more_needed, random_state=RANDOM_SEED)
        sampled_pairs_df = pd.concat([pairs_ready, newly_sampled_pairs])

    primes = set(sampled_pairs_df['prime'].dropna().str.lower().unique())
    targets = set(sampled_pairs_df['target'].dropna().str.lower().unique())
    vocabulary_needed = primes.union(targets)
    vocabulary_to_generate = vocabulary_needed - processed_words
    
    print(f"✅ Final analysis will use {len(sampled_pairs_df)} pairs.")
    print(f"   This requires {len(vocabulary_to_generate)} new words to be generated.")
    return sampled_pairs_df, vocabulary_to_generate

def generate_lexicon_data(vocabulary: set):
    """Generates LLM associations for any words not already in the central lexicon."""
    print("\n--- Step 2: Generating LLM Association Lexicon (if needed) ---")
    if not vocabulary:
        print("✅ No new words to generate. All required data is already in the lexicon.")
        return

    LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(LEXICON_PATH, "a", encoding="utf-8") as f:
        for word in tqdm(sorted(list(vocabulary)), desc="Building Lexicon"):
            response_sets = call_model(
                cue=word,
                prompt_key=PROMPT_TO_USE,
                num_sets_to_generate=NUM_SETS_PER_WORD
            )
            if response_sets:
                record = {"word": word, "association_sets": response_sets}
                f.write(json.dumps(record) + "\n")

    print("✅ Lexicon generation complete.")


def analyze_holistic_similarity(spp_data_to_analyze: pd.DataFrame):
    """Performs the main analysis using the pre-generated lexicon."""
    print("\n--- Step 3: Analyzing Holistic Similarity ---")
    
    # <<< FIX: Added a robust, recursive flattening function >>>
    def robust_flatten(items):
        """Recursively flattens a list that may contain nested lists of strings."""
        for x in items:
            if isinstance(x, list):
                yield from robust_flatten(x)
            elif isinstance(x, str):
                yield x.lower()

    try:
        print(f"🔄 Loading LLM Association Lexicon from '{LEXICON_PATH}'...")
        llm_lexicon = {}
        with open(LEXICON_PATH, 'r') as f:
            for line in f:
                record = json.loads(line)
                # Use the new robust flattening function here
                all_words = list(robust_flatten(record.get('association_sets', [])))
                llm_lexicon[record['word']] = Counter(all_words)
        print(f"✅ Loaded {len(llm_lexicon)} word vectors from lexicon.")
    except FileNotFoundError:
        print(f"❌ ERROR: Lexicon file not found at '{LEXICON_PATH}'.")
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

    PLOT_OUTPUT_PATH.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 7))
    ax = sns.regplot(x='associative_cosine_similarity', y='priming_effect', data=analysis_df,
                     scatter_kws={'alpha': 0.5}, line_kws={'color': '#4C72B0'})
    ax.set_title('Generalizability: Associative Similarity vs. Semantic Priming Effect', fontsize=16)
    ax.set_xlabel('LLM Associative Cosine Similarity', fontsize=12)
    ax.set_ylabel('Semantic Priming Effect (Log RT Difference)', fontsize=12)
    stats_text = f'$R^2 = {model.rsquared:.3f}$\n$p = {model.pvalues["associative_cosine_similarity"]:.3g}$'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='#AED6F1', alpha=0.5))
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analysis complete. Plot saved to '{PLOT_OUTPUT_PATH}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SPP Holistic Similarity analysis.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip LLM generation and only run evaluation.")
    args = parser.parse_args()
    
    spp_pairs_to_analyze, vocab_to_generate = get_and_sample_spp_pairs()
    
    if not args.evaluate_only:
        if vocab_to_generate:
            generate_lexicon_data(vocab_to_generate)
    
    analyze_holistic_similarity(spp_pairs_to_analyze)

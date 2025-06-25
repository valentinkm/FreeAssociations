"""
data_utils.py
─────────────
Functions for loading, processing, and sampling data from raw sources like
SWOW and SPP.
"""
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from collections import Counter

# Import paths and constants from the central settings file
from . import settings

def prepare_priming_data(spp_path=settings.SPP_DATA_PATH):
    """Loads and processes SPP data to return a clean DataFrame of related pairs."""
    print(f"\n--- Preparing SPP Priming Data from '{spp_path.name}' ---")
    try:
        df = pd.read_excel(spp_path)
    except FileNotFoundError:
        print(f"❌ ERROR: SPP data file not found at '{spp_path}'.")
        return pd.DataFrame()

    df = df[df['target.ACC'] == 1.0].copy()
    df['target.RT'] = pd.to_numeric(df['target.RT'], errors='coerce')
    df.dropna(subset=['target.RT', 'prime', 'target'], inplace=True)
    df['log_RT'] = np.log(df['target.RT'])
    
    related_cond = settings.SPP_CONSTANTS['RELATED_COND']
    unrelated_cond = settings.SPP_CONSTANTS['UNRELATED_COND']
    
    df_filtered = df[df['primecond'].isin([related_cond, unrelated_cond])]
    rt_means = df_filtered.groupby(['target', 'primecond'])['log_RT'].mean().reset_index()
    rt_pivot = rt_means.pivot_table(index='target', columns='primecond', values='log_RT').reset_index()
    rt_pivot.rename(columns={related_cond: 'mean_log_RT_related', unrelated_cond: 'mean_log_RT_unrelated'}, inplace=True)
    rt_pivot.dropna(subset=['mean_log_RT_related', 'mean_log_RT_unrelated'], inplace=True)
    rt_pivot['priming_effect'] = rt_pivot['mean_log_RT_unrelated'] - rt_pivot['mean_log_RT_related']
    
    priming_effects = rt_pivot[['target', 'priming_effect']]
    related_pairs = df[df['primecond'] == related_cond][['prime', 'target']].drop_duplicates()
    
    spp_data = pd.merge(related_pairs, priming_effects, on='target')
    print(f"✅ Prepared {len(spp_data)} unique related (prime, target) pairs.")
    return spp_data

def get_spp_data_for_analysis():
    """
    Intelligently samples SPP pairs and identifies the vocabulary needed for LLM generation.
    This is the main data getter for the LLM holistic analysis.
    """
    print("--- Step 1: Intelligently Sampling (Prime, Target) Pairs ---")
    
    all_pairs_df = prepare_priming_data()
    
    print(f"🔄 Checking existing lexicon at '{settings.LEXICON_PATH}'...")
    processed_words = set()
    if settings.LEXICON_PATH.exists():
        with open(settings.LEXICON_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_words.add(json.loads(line)['word'].lower())
                except (json.JSONDecodeError, KeyError):
                    continue
    
    pairs_ready = all_pairs_df[
        all_pairs_df['prime'].str.lower().isin(processed_words) &
        all_pairs_df['target'].str.lower().isin(processed_words)
    ]
    pairs_needing_generation = all_pairs_df.drop(pairs_ready.index)

    print(f"✅ Found {len(pairs_ready)} pairs ready for immediate analysis.")

    num_to_probe = settings.SPP_CONSTANTS['NUM_PAIRS_TO_PROBE']
    random_seed = settings.SPP_CONSTANTS['RANDOM_SEED']

    if len(pairs_ready) >= num_to_probe:
        print(f"🔄 Sampling {num_to_probe} pairs from the 'ready' pool...")
        sampled_pairs_df = pairs_ready.sample(n=num_to_probe, random_state=random_seed)
    else:
        print(f"Taking all {len(pairs_ready)} ready pairs.")
        num_more_needed = num_to_probe - len(pairs_ready)
        if len(pairs_needing_generation) < num_more_needed:
            print(f"⚠️ Warning: Not enough remaining pairs. Taking all {len(pairs_needing_generation)} available.")
            num_more_needed = len(pairs_needing_generation)
        
        print(f"🔄 Sampling an additional {num_more_needed} pairs that require generation...")
        newly_sampled_pairs = pairs_needing_generation.sample(n=num_more_needed, random_state=random_seed)
        sampled_pairs_df = pd.concat([pairs_ready, newly_sampled_pairs])

    primes = set(sampled_pairs_df['prime'].dropna().str.lower().unique())
    targets = set(sampled_pairs_df['target'].dropna().str.lower().unique())
    vocabulary_needed = primes.union(targets)
    vocabulary_to_generate = vocabulary_needed - processed_words
    
    print(f"✅ Final analysis will use {len(sampled_pairs_df)} pairs.")
    print(f"   This requires {len(vocabulary_to_generate)} new words to be generated.")
    return sampled_pairs_df, vocabulary_to_generate

def load_human_lexicon():
    """
    Loads SWOW data and creates a 'human lexicon' where each word maps to a
    Counter of its free associations.
    """
    print("--- Loading Human Free-Association Lexicon ---")
    try:
        df = pd.read_csv(settings.SWOW_DATA_PATH, usecols=['cue', 'R1', 'R2', 'R3'])
    except FileNotFoundError:
        print(f"❌ ERROR: SWOW file not found at '{settings.SWOW_DATA_PATH}'.")
        return {}

    human_lexicon = {}
    for cue, grp in tqdm(df.groupby("cue"), desc="Building Human Lexicon"):
        all_responses = pd.concat([grp['R1'], grp['R2'], grp['R3']]).dropna().str.lower()
        human_lexicon[cue.lower()] = Counter(all_responses)
        
    print(f"✅ Loaded {len(human_lexicon)} word vectors from SWOW data.")
    return human_lexicon

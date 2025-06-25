"""
data_utils.py
─────────────
Functions for loading, processing, and sampling data from raw sources like
SWOW, SPP, and 3TT.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

from . import settings

# --- SPP Data Loading ---

def _prepare_spp_data(spp_path):
    """Internal helper to load and process SPP data."""
    df = pd.read_excel(spp_path)
    df = df[df['target.ACC'] == 1.0].copy()
    df['target.RT'] = pd.to_numeric(df['target.RT'], errors='coerce')
    df.dropna(subset=['target.RT', 'prime', 'target'], inplace=True)
    df['log_RT'] = np.log(df['target.RT'])
    df_filtered = df[df['primecond'].isin([1, 2])]
    rt_means = df_filtered.groupby(['target', 'primecond'])['log_RT'].mean().reset_index()
    rt_pivot = rt_means.pivot_table(index='target', columns='primecond', values='log_RT').reset_index()
    rt_pivot.rename(columns={1: 'mean_log_RT_related', 2: 'mean_log_RT_unrelated'}, inplace=True)
    rt_pivot.dropna(subset=['mean_log_RT_related', 'mean_log_RT_unrelated'], inplace=True)
    rt_pivot['priming_effect'] = rt_pivot['mean_log_RT_unrelated'] - rt_pivot['mean_log_RT_related']
    priming_effects = rt_pivot[['target', 'priming_effect']]
    related_pairs = df[df['primecond'] == 1][['prime', 'target']].drop_duplicates()
    return pd.merge(related_pairs, priming_effects, on='target')

def get_spp_data_for_analysis(lexicon_path: Path, num_pairs_to_probe: int):
    """Samples SPP pairs and identifies the vocabulary needed for LLM generation."""
    print(f"\n--- Sampling {num_pairs_to_probe} SPP pairs ---")
    all_pairs_df = _prepare_spp_data(settings.SPP_DATA_PATH)
    
    print(f"Checking existing words in '{lexicon_path.name}'...")
    processed_words = set()
    if lexicon_path.exists():
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed_words.add(json.loads(line)['word'].lower())
                except (json.JSONDecodeError, KeyError): continue
    
    pairs_ready = all_pairs_df[all_pairs_df['prime'].str.lower().isin(processed_words) & all_pairs_df['target'].str.lower().isin(processed_words)]
    pairs_needing_generation = all_pairs_df.drop(pairs_ready.index)

    if len(pairs_ready) >= num_pairs_to_probe:
        sampled_pairs_df = pairs_ready.sample(n=num_pairs_to_probe, random_state=settings.GENERALIZE_DEFAULTS['random_seed'])
    else:
        num_more_needed = num_pairs_to_probe - len(pairs_ready)
        if len(pairs_needing_generation) < num_more_needed:
            num_more_needed = len(pairs_needing_generation)
        newly_sampled_pairs = pairs_needing_generation.sample(n=num_more_needed, random_state=settings.GENERALIZE_DEFAULTS['random_seed'])
        sampled_pairs_df = pd.concat([pairs_ready, newly_sampled_pairs])

    primes = set(sampled_pairs_df['prime'].dropna().str.lower().unique())
    targets = set(sampled_pairs_df['target'].dropna().str.lower().unique())
    vocabulary_needed = primes.union(targets)
    vocabulary_to_generate = vocabulary_needed - processed_words
    
    print(f"✅ Sampled {len(sampled_pairs_df)} pairs. Need to generate {len(vocabulary_to_generate)} new words.")
    return sampled_pairs_df, vocabulary_to_generate

# --- 3TT Data Loading ---

def get_3tt_data_for_analysis(lexicon_path: Path, num_triplets_to_probe: int):
    """
    Loads and samples 3TT data from the Results Summary file and identifies
    the vocabulary needed for LLM generation.
    """
    print(f"\n--- Loading and sampling {num_triplets_to_probe} 3TT triplets ---")
    try:
        # <<< FIX: Removed the old, incorrect loading logic. This is the only file needed. >>>
        all_triplets_df = pd.read_csv(settings.TTT_RESULTS_PATH)
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find 3TT data file at '{settings.TTT_RESULTS_PATH}'. Details: {e}")
        return None, None
    
    # Clean column names and drop rows with missing essential data
    all_triplets_df.rename(columns={'anchor': 'cue', 'target1': 'choiceA', 'target2': 'choiceB'}, inplace=True)
    all_triplets_df.dropna(subset=['cue', 'choiceA', 'choiceB', 'chosen'], inplace=True)
    
    # Determine the word string of the human's choice
    all_triplets_df['human_related_choice'] = np.where(
        all_triplets_df['chosen'] == 1, 
        all_triplets_df['choiceA'], 
        all_triplets_df['choiceB']
    )

    # --- Now apply the same sampling logic as SPP ---
    print(f"Checking existing words in '{lexicon_path.name}'...")
    processed_words = set()
    if lexicon_path.exists():
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed_words.add(json.loads(line)['word'].lower())
                except (json.JSONDecodeError, KeyError): continue

    all_triplets_df['is_ready'] = all_triplets_df.apply(
        lambda row: row['cue'].lower() in processed_words and \
                    row['choiceA'].lower() in processed_words and \
                    row['choiceB'].lower() in processed_words,
        axis=1
    )
    
    triplets_ready = all_triplets_df[all_triplets_df['is_ready'] == True]
    triplets_needing_generation = all_triplets_df[all_triplets_df['is_ready'] == False]

    if len(triplets_ready) >= num_triplets_to_probe:
        sampled_df = triplets_ready.sample(n=num_triplets_to_probe, random_state=settings.GENERALIZE_DEFAULTS['random_seed'])
    else:
        num_more_needed = num_triplets_to_probe - len(triplets_ready)
        if len(triplets_needing_generation) < num_more_needed:
            num_more_needed = len(triplets_needing_generation)
        newly_sampled = triplets_needing_generation.sample(n=num_more_needed, random_state=settings.GENERALIZE_DEFAULTS['random_seed'])
        sampled_df = pd.concat([triplets_ready, newly_sampled]) if not triplets_ready.empty else newly_sampled

    cues = set(sampled_df['cue'].str.lower())
    choices_a = set(sampled_df['choiceA'].str.lower())
    choices_b = set(sampled_df['choiceB'].str.lower())
    vocabulary_needed = cues.union(choices_a).union(choices_b)
    vocabulary_to_generate = vocabulary_needed - processed_words

    print(f"✅ Sampled {len(sampled_df)} triplets. Need to generate {len(vocabulary_to_generate)} new words.")
    return sampled_df, vocabulary_to_generate
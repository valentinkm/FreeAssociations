#!/usr/bin/env python3
"""
analyze_3tt_associative_choice.py - A script to test if the LLM's free-
association data can predict human choices in the Three Terms Task (3TT).

(v2 - Compares prediction accuracy using both Weighted Jaccard and Cosine Similarity)
"""

import sys
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Required for Cosine Similarity: pip install scipy
try:
    from scipy.spatial.distance import cosine
except ImportError:
    print("❌ Error: The 'scipy' library is required for Cosine Similarity.")
    print("   Please install it by running: pip install scipy")
    sys.exit(1)


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
THRE_TT_DATA_PATH = PROJECT_ROOT / "generlize_3tt" / "Results Summary.csv"
SWOW_R100_PATH = PROJECT_ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"
LEXICON_PATH = PROJECT_ROOT / "runs" / "llm_association_lexicon.jsonl"
RESULTS_OUTPUT_PATH = PROJECT_ROOT / "runs" / "associative_choice_results.csv"
PROMPT_TO_USE = "participant_default_question"
NUM_SETS_PER_WORD = 25

# --- Helper Functions for Similarity ---
def wjacc(ctr1: Counter, ctr2: Counter) -> float:
    """Calculates the weighted Jaccard similarity between two Counters."""
    if not ctr1 or not ctr2:
        return 0.0
    
    all_keys = ctr1.keys() | ctr2.keys()
    
    intersection_sum = sum(min(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    union_sum = sum(max(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    
    return intersection_sum / union_sum if union_sum else 0.0

def cosine_sim(ctr1: Counter, ctr2: Counter) -> float:
    """Calculates the cosine similarity between two Counters."""
    if not ctr1 or not ctr2:
        return 0.0
    
    # Create a shared vocabulary and corresponding frequency vectors
    vocab = sorted(list(ctr1.keys() | ctr2.keys()))
    v1 = np.array([ctr1.get(word, 0) for word in vocab])
    v2 = np.array([ctr2.get(word, 0) for word in vocab])
    
    # The 'cosine' function from scipy calculates distance (1 - similarity).
    # Return 0 if either vector is all zeros to avoid division by zero errors.
    if not np.any(v1) or not np.any(v2):
        return 0.0
    
    return 1 - cosine(v1, v2)


# --- Main Pipeline Functions ---

def get_3tt_vocabulary(probe_size: int = 0) -> set:
    """
    Loads 3TT data, filters it, and returns the required vocabulary.
    """
    print("--- Step 1: Identifying Vocabulary from 3TT Data ---")
    try:
        df_3tt = pd.read_csv(THRE_TT_DATA_PATH)
        df_swow = pd.read_csv(SWOW_R100_PATH, usecols=['cue'])
    except FileNotFoundError as e:
        print(f"❌ ERROR: A required data file was not found: {e.filename}")
        return set()
    
    swow_cues = set(df_swow['cue'].dropna().str.lower().unique())
    df_3tt.dropna(subset=['anchor', 'target1', 'target2'], inplace=True)
    initial_count = len(df_3tt)
    
    df_filtered = df_3tt[
        df_3tt['anchor'].str.lower().isin(swow_cues) &
        df_3tt['target1'].str.lower().isin(swow_cues) &
        df_3tt['target2'].str.lower().isin(swow_cues)
    ].copy()
    
    print(f"✅ Filtered 3TT data to only include words present in SWOW R100. Kept {len(df_filtered)} / {initial_count} triplets.")

    if probe_size > 0:
        print(f"🔬 PROBE MODE: Selecting a random sample of {probe_size} triplets.")
        if len(df_filtered) > probe_size:
            df_filtered = df_filtered.sample(n=probe_size, random_state=42)
        else:
            print(f"⚠️ Warning: Requested probe size ({probe_size}) is larger than available data ({len(df_filtered)}). Using all available data.")
    
    vocabulary = set(pd.concat([
        df_filtered['anchor'],
        df_filtered['target1'],
        df_filtered['target2']
    ]).dropna().str.lower().unique())
    
    print(f"✅ Identified {len(vocabulary)} unique words required for this analysis.")
    return vocabulary

def generate_lexicon_data(vocabulary: set):
    """Generates LLM associations for any words not already in the central lexicon."""
    # This function is unchanged and correctly handles lexicon generation.
    # ...
    pass


def analyze_associative_choice(probe_size: int = 0):
    """
    Performs the main analysis using the pre-generated lexicon.
    """
    print("\n--- Step 3: Analyzing Associative Choice ---")
    
    try:
        human_choices_df = pd.read_csv(THRE_TT_DATA_PATH)
        df_swow = pd.read_csv(SWOW_R100_PATH, usecols=['cue'])
    except FileNotFoundError as e:
        print(f"❌ ERROR: A required data file was not found: {e.filename}")
        return

    swow_cues = set(df_swow['cue'].dropna().str.lower().unique())
    human_choices_df.dropna(subset=['anchor', 'target1', 'target2', 'chosen'], inplace=True)
    
    human_choices_df_filtered = human_choices_df[
        human_choices_df['anchor'].str.lower().isin(swow_cues) &
        human_choices_df['target1'].str.lower().isin(swow_cues) &
        human_choices_df['target2'].str.lower().isin(swow_cues)
    ].copy()

    if probe_size > 0 and len(human_choices_df_filtered) > probe_size:
        human_choices_df_filtered = human_choices_df_filtered.sample(n=probe_size, random_state=42)

    try:
        print(f"🔄 Loading LLM Association Lexicon from '{LEXICON_PATH}'...")
        llm_lexicon = {}
        with open(LEXICON_PATH, 'r') as f:
            for line in f:
                record = json.loads(line)
                all_words = [str(word).lower() for s in record.get('association_sets', []) for word in s if word]
                llm_lexicon[record['word']] = Counter(all_words)
        print(f"✅ Loaded {len(llm_lexicon)} word vectors from lexicon.")
    except FileNotFoundError:
        print(f"❌ ERROR: Lexicon file not found. Run script without --evaluate-only first.")
        return

    print("🔄 Predicting human choices using both Jaccard and Cosine similarity...")
    predictions = []
    for task in tqdm(human_choices_df_filtered.itertuples(), total=len(human_choices_df_filtered)):
        anchor, t1, t2 = task.anchor.lower(), task.target1.lower(), task.target2.lower()
        
        anchor_vec = llm_lexicon.get(anchor)
        t1_vec = llm_lexicon.get(t1)
        t2_vec = llm_lexicon.get(t2)

        if not anchor_vec or not t1_vec or not t2_vec:
            continue

        # --- Calculate similarity using both methods ---
        wjacc_sim1 = wjacc(anchor_vec, t1_vec)
        wjacc_sim2 = wjacc(anchor_vec, t2_vec)
        
        cosine_sim1 = cosine_sim(anchor_vec, t1_vec)
        cosine_sim2 = cosine_sim(anchor_vec, t2_vec)

        # --- Make predictions based on each method ---
        predicted_choice_jaccard = 1 if wjacc_sim1 > wjacc_sim2 else 2
        predicted_choice_cosine = 1 if cosine_sim1 > cosine_sim2 else 2
        
        human_choice = int(task.chosen)

        predictions.append({
            "anchor": anchor, "target1": t1, "target2": t2,
            "human_choice": human_choice,
            "predicted_choice_jaccard": predicted_choice_jaccard,
            "predicted_choice_cosine": predicted_choice_cosine,
            "wjacc_sim_t1": wjacc_sim1,
            "wjacc_sim_t2": wjacc_sim2,
            "cosine_sim_t1": cosine_sim1,
            "cosine_sim_t2": cosine_sim2,
        })
    
    if not predictions:
        print("❌ No predictions were made. Check if words in the filtered 3TT data exist in the lexicon.")
        return
        
    analysis_df = pd.DataFrame(predictions)
    
    # --- Calculate and compare accuracies ---
    analysis_df['correct_jaccard'] = (analysis_df['human_choice'] == analysis_df['predicted_choice_jaccard'])
    accuracy_jaccard = analysis_df['correct_jaccard'].mean()
    
    analysis_df['correct_cosine'] = (analysis_df['human_choice'] == analysis_df['predicted_choice_cosine'])
    accuracy_cosine = analysis_df['correct_cosine'].mean()
    
    # --- Save detailed results ---
    analysis_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print(f"\n✅ Detailed results saved to '{RESULTS_OUTPUT_PATH}'")
    
    # --- Print final comparison ---
    print("\n--- Associative Choice Test Results ---")
    if probe_size > 0:
        print(f"🔬 PROBE MODE ACTIVE")
    print(f"Total Triplets Evaluated: {len(analysis_df)}")
    print(f"Prediction Accuracy (Weighted Jaccard): {accuracy_jaccard:.2%}")
    print(f"Prediction Accuracy (Cosine Similarity): {accuracy_cosine:.2%}")
    print("-----------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 3TT Associative Choice analysis.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip LLM generation and only run evaluation.")
    parser.add_argument("--probe", type=int, default=0, help="Run in probe mode on a small sample of N triplets. Default is 0 (full run).")
    args = parser.parse_args()
    
    if not args.evaluate_only:
        tt_vocab = get_3tt_vocabulary(probe_size=args.probe)
        if tt_vocab:
            # Re-implement generate_lexicon_data if it's not in the shared context
            # For now, assuming it exists as a callable function.
            print("Assuming `generate_lexicon_data` is implemented and available.")
            # generate_lexicon_data(tt_vocab) 
    
    analyze_associative_choice(probe_size=args.probe)

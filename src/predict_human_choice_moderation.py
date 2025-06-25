#!/usr/bin/env python3
"""
predict_human_choice_with_moderation.py - A script to test if the LLM's free-
association data can predict human choices in the 3TT, and if this
prediction is moderated by human-LLM alignment.

(v4 - Implements a robust, modular data pipeline)
"""

import sys
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Required for Cosine Similarity and Regression: pip install scipy statsmodels
try:
    from scipy.spatial.distance import cosine
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("❌ Error: The 'scipy', 'statsmodels', 'matplotlib', and 'seaborn' libraries are required.")
    print("   Please install them by running: pip install scipy statsmodels matplotlib seaborn")
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
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_OUTPUT_PATH = PROJECT_ROOT / "runs" / "associative_choice_results_with_moderation.csv"
MODERATION_PLOT_PATH = PLOTS_DIR / "associative_choice_moderated_by_alignment.png"
PROMPT_TO_USE = "participant_default_question"
NUM_SETS_PER_WORD = 25 # Number of association sets to generate per word

# --- Helper Functions ---
def wjacc(h_ctr: Counter, m_words: list) -> float:
    """Calculates weighted Jaccard similarity between a human Counter and LLM word list."""
    m_ctr = Counter(m_words)
    if not h_ctr or not m_ctr: return 0.0
    all_keys = h_ctr.keys() | m_ctr.keys()
    inter = sum(min(h_ctr.get(k, 0), m_ctr.get(k, 0)) for k in all_keys)
    union = sum(max(h_ctr.get(k, 0), m_ctr.get(k, 0)) for k in all_keys)
    return inter / union if union else 0.0

def cosine_sim(ctr1: Counter, ctr2: Counter) -> float:
    """Calculates the cosine similarity between two Counters."""
    if not ctr1 or not ctr2: return 0.0
    vocab = sorted(list(ctr1.keys() | ctr2.keys()))
    v1 = np.array([ctr1.get(word, 0) for word in vocab])
    v2 = np.array([ctr2.get(word, 0) for word in vocab])
    if not np.any(v1) or not np.any(v2): return 0.0
    return 1 - cosine(v1, v2)

# --- Main Pipeline Functions ---

def get_tasks_and_vocabulary(probe_size: int = 0) -> tuple[pd.DataFrame, set]:
    """
    Loads 3TT data, filters for valid tasks, optionally samples for a probe run,
    and returns the tasks and the required vocabulary.
    """
    print("--- Step 1: Identifying Tasks and Required Vocabulary ---")
    try:
        df_3tt = pd.read_csv(THRE_TT_DATA_PATH)
        df_swow = pd.read_csv(SWOW_R100_PATH, usecols=['cue'])
    except FileNotFoundError as e:
        print(f"❌ ERROR: A required data file was not found: {e.filename}.")
        return pd.DataFrame(), set()

    swow_cues = set(df_swow['cue'].dropna().str.lower().unique())
    df_3tt.dropna(subset=['anchor', 'target1', 'target2', 'chosen'], inplace=True)
    
    df_filtered = df_3tt[
        df_3tt['anchor'].str.lower().isin(swow_cues) &
        df_3tt['target1'].str.lower().isin(swow_cues) &
        df_3tt['target2'].str.lower().isin(swow_cues)
    ].copy()
    print(f"✅ Found {len(df_filtered)} valid triplets with all words present in SWOW.")

    tasks_to_run = df_filtered
    if probe_size > 0:
        print(f"🔬 PROBE MODE: Selecting a random sample of {probe_size} triplets.")
        if len(df_filtered) > probe_size:
            tasks_to_run = df_filtered.sample(n=probe_size, random_state=42)
        else:
            print(f"⚠️ Warning: Requested probe size ({probe_size}) is larger than available data. Using all {len(df_filtered)} triplets.")
    
    vocabulary = set(pd.concat([
        tasks_to_run['anchor'],
        tasks_to_run['target1'],
        tasks_to_run['target2']
    ]).dropna().str.lower().unique())
    
    print(f"✅ Identified {len(vocabulary)} unique words required for this run.")
    return tasks_to_run, vocabulary

def generate_lexicon_data(vocabulary: set):
    """Generates LLM associations for any words not already in the central lexicon."""
    print("\n--- Step 2: Generating LLM Association Lexicon (if needed) ---")
    LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    processed_words = set()
    if LEXICON_PATH.exists():
        with open(LEXICON_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed_words.add(json.loads(line)['word'])
                except (json.JSONDecodeError, KeyError): continue

    tasks_to_run = sorted([word for word in vocabulary if isinstance(word, str) and word not in processed_words])
    
    if not tasks_to_run:
        print("✅ All required words already exist in the lexicon.")
        return

    print(f"🔄 Found {len(tasks_to_run)} new words to generate associations for...")
    with open(LEXICON_PATH, "a", encoding="utf-8") as f:
        for word in tqdm(tasks_to_run, desc="Building Lexicon"):
            response_sets = call_model(
                cue=word,
                prompt_key=PROMPT_TO_USE,
                num_sets_to_generate=NUM_SETS_PER_WORD
            )
            if response_sets:
                record = {"word": word, "association_sets": response_sets}
                f.write(json.dumps(record) + "\n")

    print("✅ Lexicon generation complete.")


def analyze_choice_and_moderation(tasks_df: pd.DataFrame):
    """
    Performs the main analysis: predicts choices and tests for moderation by alignment.
    """
    print("\n--- Step 3: Predicting Human Choice and Testing for Moderation ---")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load LLM Lexicon & Human Norms
    print("🔄 Loading lexicons and norms...")
    try:
        df_swow = pd.read_csv(SWOW_R100_PATH)
        llm_lexicon, llm_lexicon_raw_words = {}, {}
        with open(LEXICON_PATH, 'r') as f:
            for line in f:
                record = json.loads(line)
                word = record['word']
                all_words = [str(w).lower() for s in record.get('association_sets', []) for w in s if w]
                llm_lexicon[word] = Counter(all_words)
                llm_lexicon_raw_words[word] = all_words
    except FileNotFoundError:
        print(f"❌ ERROR: A required data file was not found. Cannot run analysis.")
        return

    # 2. Predict Human Choice
    print("🔄 Predicting human choices using associative similarity...")
    predictions = []
    for task in tqdm(tasks_df.itertuples(), total=len(tasks_df), desc="Predicting Choices"):
        anchor, t1, t2 = task.anchor.lower(), task.target1.lower(), task.target2.lower()
        
        anchor_vec, t1_vec, t2_vec = llm_lexicon.get(anchor), llm_lexicon.get(t1), llm_lexicon.get(t2)
        if not anchor_vec or not t1_vec or not t2_vec: continue

        cosine_sim1 = cosine_sim(anchor_vec, t1_vec)
        cosine_sim2 = cosine_sim(anchor_vec, t2_vec)
        predicted_choice_cosine = 1 if cosine_sim1 > cosine_sim2 else 2
        
        predictions.append({
            "anchor": anchor, "target1": t1, "target2": t2,
            "human_choice": int(task.chosen),
            "predicted_choice_cosine": predicted_choice_cosine,
        })
    
    analysis_df = pd.DataFrame(predictions)
    analysis_df['correct_cosine'] = (analysis_df['human_choice'] == analysis_df['predicted_choice_cosine'])
    accuracy_cosine = analysis_df['correct_cosine'].mean()
    print(f"✅ Prediction Accuracy (Cosine Similarity): {accuracy_cosine:.2%}")
    
    # 3. Moderation Analysis
    print("🔄 Calculating item-based alignment scores for moderation analysis...")
    human_norms = {c.lower(): Counter(pd.concat([g['R1'], g['R2'], g['R3']]).dropna().str.lower()) for c, g in df_swow.groupby("cue")}
    
    alignment_scores = {
        word: wjacc(human_norms.get(word, Counter()), llm_words)
        for word, llm_words in llm_lexicon_raw_words.items()
    }
    
    analysis_df['avg_triplet_alignment'] = analysis_df.apply(
        lambda r: np.mean([alignment_scores.get(r['anchor'],0), alignment_scores.get(r['target1'],0), alignment_scores.get(r['target2'],0)]), axis=1
    )
    
    print("🔄 Running moderation analysis (Logistic Regression)...")
    y = analysis_df['correct_cosine']
    X = sm.add_constant(analysis_df['avg_triplet_alignment'])
    logit_model = sm.Logit(y.astype(float), X.astype(float)).fit(disp=0)
    
    print("\n--- Moderation Analysis Summary ---")
    print(logit_model.summary())

    plt.figure(figsize=(10, 6))
    sns.regplot(x='avg_triplet_alignment', y='correct_cosine', data=analysis_df, 
                logistic=True, ci=95,
                scatter_kws={'alpha': 0.2, 'color': 'darkcyan'},
                line_kws={'color': 'teal'})
    plt.title('Prediction Accuracy Moderated by Human-LLM Alignment', fontsize=16)
    plt.xlabel('Average Triplet Alignment with Human Norms', fontsize=12)
    plt.ylabel('Probability of Correct Prediction', fontsize=12)
    plt.savefig(MODERATION_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Moderation plot saved to '{MODERATION_PLOT_PATH}'")

    analysis_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print(f"✅ Detailed results with moderation data saved to '{RESULTS_OUTPUT_PATH}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3TT choice prediction and moderation analysis.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip LLM generation and only run evaluation.")
    parser.add_argument("--probe", type=int, default=0, help="Run in probe mode on a small sample of N triplets. Default is 0 (full run).")
    args = parser.parse_args()
    
    tasks_to_analyze, vocab_to_generate = get_tasks_and_vocabulary(probe_size=args.probe)
    
    if not args.evaluate_only:
        if vocab_to_generate:
            generate_lexicon_data(vocab_to_generate)
    
    if not tasks_to_analyze.empty:
        analyze_choice_and_moderation(tasks_to_analyze)

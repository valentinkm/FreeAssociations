#!/usr/bin/env python3
"""
test_alignment_predicts_performance.py - A script to test if the LLM's alignment
on free associations generalizes to predict its accuracy on a semantic
judgment task (3TT).

(v3 - Separates general and subgroup analyses, uses continuous regression for both)
"""

import sys
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# --- Setup Project Path ---
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Import from existing project ---
try:
    from src.settings import CFG
    from src.run_lm import call_model, client
    from src.extract_profiles import norm_gender # Import specific helper
except ImportError as e:
    print(f"❌ Error: Could not import from 'src'. Ensure required files exist.")
    sys.exit(1)

# --- Configuration ---
# File Paths
THRE_TT_DATA_PATH = PROJECT_ROOT / "generlize_3tt" / "Results Summary.csv"
SWOW_DATA_PATH = PROJECT_ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"
LEXICON_PATH = PROJECT_ROOT / "runs" / "llm_association_lexicon.jsonl"
TTT_CHOICES_PATH = PROJECT_ROOT / "runs" / "3tt_llm_choices.jsonl"
PLOTS_DIR = PROJECT_ROOT / "plots"
GENERAL_PLOT_PATH = PLOTS_DIR / "alignment_accuracy_general_test.png"
SUBGROUP_REGRESSION_PLOT_PATH = PLOTS_DIR / "alignment_accuracy_subgroup_regression.png"
OVERALL_SUBGROUP_COMPARISON_PATH = PLOTS_DIR / "alignment_accuracy_overall_subgroup_comparison.png"

# Generation Parameters
PROMPT_TO_USE = "participant_default_question"

# --- Helper Functions ---
def wjacc(h_ctr, m_words):
    """Calculates the weighted Jaccard similarity."""
    m_ctr = Counter(m_words)
    inter = sum(min(h_ctr.get(w, 0), m_ctr.get(w, 0)) for w in h_ctr.keys() | m_ctr.keys())
    union = sum(max(h_ctr.get(w, 0), m_ctr.get(w, 0)) for w in h_ctr.keys() | m_ctr.keys())
    return inter / union if union else 0.0

# --- Phase 1: Data Preparation (Functions are placeholders assuming data exists) ---
def prepare_and_generate_data():
    """Placeholder for data generation logic."""
    print("This script assumes data has been generated. Use --evaluate-only flag.")
    print("To run data generation, use the appropriate dedicated scripts.")
    # In a full run, this would call prepare_probe_data(), 
    # generate_lexicon_data(), and perform_3tt_task().
    pass

# --- Phase 2: Analysis ---

def analyze_generalizability():
    """Loads all data and runs the final analysis, including subgroup generalizability."""
    print("\n--- Analyzing Generalizability of Alignment vs. Accuracy ---")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load all required data files
    print("🔄 Loading all data sources...")
    try:
        df_human_choices = pd.read_csv(THRE_TT_DATA_PATH)
        df_llm_choices = pd.read_json(TTT_CHOICES_PATH, lines=True)
        df_swow = pd.read_csv(SWOW_DATA_PATH) # Load full SWOW for demographics
        
        lexicon_data = {}
        with open(LEXICON_PATH, 'r') as f:
            for line in f:
                rec = json.loads(line)
                lexicon_data[rec['word']] = rec.get('association_sets', rec.get('sets', []))
    except FileNotFoundError as e:
        print(f"❌ A required data file is missing: {e}. Please run the script without --evaluate-only first.")
        return

    # 2. Calculate alignment scores (General and Subgroup)
    print("🔄 Calculating general and subgroup alignment scores...")
    df_swow.dropna(subset=['cue', 'R1', 'R2', 'R3', 'gender'], inplace=True)
    
    general_human_norms = {c.lower(): Counter(pd.concat([g['R1'], g['R2'], g['R3']]).dropna().str.lower()) for c, g in df_swow.groupby("cue")}
    
    df_swow['gender_bin'] = df_swow['gender'].apply(norm_gender)
    male_norms = {c.lower(): Counter(pd.concat([g['R1'], g['R2'], g['R3']]).dropna().str.lower()) for c, g in df_swow[df_swow['gender_bin'] == 'male'].groupby("cue")}
    female_norms = {c.lower(): Counter(pd.concat([g['R1'], g['R2'], g['R3']]).dropna().str.lower()) for c, g in df_swow[df_swow['gender_bin'] == 'female'].groupby("cue")}

    alignment_scores = {}
    for word, sets in tqdm(lexicon_data.items(), desc="Calculating Alignment Scores"):
        llm_words = [str(w).lower() for s in sets for w in s if w]
        w_lower = word.lower()
        alignment_scores[w_lower] = {
            'general': wjacc(general_human_norms.get(w_lower, Counter()), llm_words),
            'male': wjacc(male_norms.get(w_lower, Counter()), llm_words),
            'female': wjacc(female_norms.get(w_lower, Counter()), llm_words),
        }

    # 3. Prepare the final analysis DataFrame
    print("🔄 Preparing final analysis DataFrame...")
    df_human_choices['human_choice_word'] = np.where(df_human_choices['chosen'] == 1, df_human_choices['target1'], df_human_choices['target2'])
    
    analysis_df = pd.merge(df_human_choices, df_llm_choices, on=['anchor', 'target1', 'target2'])
    analysis_df['correct'] = (analysis_df['llm_choice'].str.lower() == analysis_df['human_choice_word'].str.lower()).astype(int)
    
    for align_type in ['general', 'male', 'female']:
        analysis_df[f'avg_triplet_alignment_{align_type}'] = analysis_df.apply(
            lambda row: np.mean([
                alignment_scores.get(row['anchor'].lower(), {}).get(align_type, 0),
                alignment_scores.get(row['target1'].lower(), {}).get(align_type, 0),
                alignment_scores.get(row['target2'].lower(), {}).get(align_type, 0)
            ]), axis=1
        )
    
    # --- Analysis Part 1: General (Cue-Based) Alignment Generalizability ---
    print("\n--- Running General Alignment Generalizability Test ---")
    y = analysis_df['correct']
    X_general = sm.add_constant(analysis_df['avg_triplet_alignment_general'])
    logit_model_general = sm.Logit(y, X_general).fit(disp=0)
    print("\n--- Logistic Regression Summary (General Alignment) ---")
    print(logit_model_general.summary())
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x='avg_triplet_alignment_general', y='correct', data=analysis_df, logistic=True, ci=95,
                scatter_kws={'alpha': 0.2}, line_kws={'color': 'purple'})
    plt.title('LLM Accuracy vs. General Human Alignment', fontsize=16)
    plt.xlabel('Triplet Alignment with General Human Norms', fontsize=12)
    plt.ylabel('Probability of Correct Performance', fontsize=12)
    plt.savefig(GENERAL_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ General alignment plot saved to '{GENERAL_PLOT_PATH}'")
    plt.close()

    # --- Analysis Part 2: Subgroup Alignment Generalizability ---
    print("\n--- Running Subgroup Alignment Generalizability Test ---")
    
    # Model for Male alignment
    X_male = sm.add_constant(analysis_df['avg_triplet_alignment_male'])
    logit_model_male = sm.Logit(y, X_male).fit(disp=0)
    print("\n--- Logistic Regression Summary (Alignment with Male Norms) ---")
    print(logit_model_male.summary())

    # Model for Female alignment
    X_female = sm.add_constant(analysis_df['avg_triplet_alignment_female'])
    logit_model_female = sm.Logit(y, X_female).fit(disp=0)
    print("\n--- Logistic Regression Summary (Alignment with Female Norms) ---")
    print(logit_model_female.summary())

    # Visualize both subgroup models on the same plot
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.regplot(x='avg_triplet_alignment_male', y='correct', data=analysis_df, ax=ax, logistic=True, ci=None,
                scatter=False, line_kws={'color': 'blue', 'label': 'vs. Male Alignment'})
    sns.regplot(x='avg_triplet_alignment_female', y='correct', data=analysis_df, ax=ax, logistic=True, ci=None,
                scatter=False, line_kws={'color': 'orange', 'label': 'vs. Female Alignment'})
    sns.scatterplot(x='avg_triplet_alignment_general', y='correct', data=analysis_df, ax=ax, alpha=0.1, color='grey', marker='.')

    ax.set_title('LLM Accuracy Predicted by Alignment with Gender Subgroups', fontsize=16)
    ax.set_xlabel('Triplet Alignment with Subgroup Norms', fontsize=12)
    ax.set_ylabel('Probability of Correct Performance', fontsize=12)
    ax.legend(title='Predictor Model')
    plt.savefig(SUBGROUP_REGRESSION_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ Subgroup comparison regression plot saved to '{SUBGROUP_REGRESSION_PLOT_PATH}'")
    plt.close()

    # --- Analysis Part 3: Overall Performance Comparison by Subgroup Alignment ---
    print("\n--- Comparing Overall Performance for High-Alignment Subgroup Items ---")

    # Define "high alignment" as the top 50% of scores for each subgroup
    analysis_df['male_align_bin'] = pd.qcut(analysis_df['avg_triplet_alignment_male'], q=2, labels=['Low', 'High'])
    analysis_df['female_align_bin'] = pd.qcut(analysis_df['avg_triplet_alignment_female'], q=2, labels=['Low', 'High'])

    # Calculate mean accuracy on the "High" alignment trials for each subgroup
    accuracy_on_high_male_align = analysis_df[analysis_df['male_align_bin'] == 'High']['correct'].mean()
    accuracy_on_high_female_align = analysis_df[analysis_df['female_align_bin'] == 'High']['correct'].mean()
    
    print(f"Accuracy on trials with High Male-Alignment: {accuracy_on_high_male_align:.2%}")
    print(f"Accuracy on trials with High Female-Alignment: {accuracy_on_high_female_align:.2%}")

    # Create a simple bar plot for direct comparison
    plot_data = pd.DataFrame([
        {'Subgroup': 'Male-Aligned Items', 'Accuracy': accuracy_on_high_male_align},
        {'Subgroup': 'Female-Aligned Items', 'Accuracy': accuracy_on_high_female_align}
    ])

    plt.figure(figsize=(8, 6))
    sns.barplot(data=plot_data, x='Subgroup', y='Accuracy', palette=['lightblue', 'lightcoral'])
    plt.title('Overall LLM Performance on High-Alignment Items by Subgroup', fontsize=16)
    plt.xlabel('Item Set based on High Alignment with Subgroup', fontsize=12)
    plt.ylabel('Average LLM Accuracy', fontsize=12)
    plt.ylim(0, 1.0)
    plt.savefig(OVERALL_SUBGROUP_COMPARISON_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ Overall subgroup performance plot saved to '{OVERALL_SUBGROUP_COMPARISON_PATH}'")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 3TT Alignment Accuracy Test with Subgroup Analysis.")
    parser.add_argument("--evaluate-only", action="store_true", default=True, help="Skip all data generation and only run the final analysis.")
    args = parser.parse_args()
    
    if not args.evaluate_only:
        prepare_and_generate_data()
    
    analyze_generalizability()

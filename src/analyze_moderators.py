#!/usr/bin/env python3
"""
analyze_moderators.py - A script to analyze how alignment moderates the
accuracy of predicting human choices in the 3TT task.

This script will:
1.  Load the detailed prediction results.
2.  Load human and LLM association data to calculate alignment scores.
3.  Part 1: Analyze if prediction accuracy is higher for triplets with
    higher overall human-LLM alignment using LOGISTIC REGRESSION.
4.  Part 2: Analyze if prediction accuracy is moderated by alignment with
    specific demographic subgroups (e.g., male vs. female).
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm # <<< NEW IMPORT

# --- Configuration ---
PROJECT_ROOT = Path.cwd()
# Inputs
RESULTS_CSV_PATH = PROJECT_ROOT / "runs" / "associative_choice_results.csv"
LEXICON_PATH = PROJECT_ROOT / "runs" / "llm_association_lexicon.jsonl"
SWOW_R100_PATH = PROJECT_ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"
# Outputs
PLOTS_DIR = PROJECT_ROOT / "plots"
GENERAL_ALIGNMENT_PLOT_PATH = PLOTS_DIR / "moderator_general_alignment_regression.png" # Changed filename
DEMOGRAPHIC_ALIGNMENT_PLOT_PATH = PLOTS_DIR / "moderator_demographic_alignment.png"

# --- Helper Function ---
def wjacc(ctr1: Counter, ctr2: Counter) -> float:
    """Calculates the weighted Jaccard similarity between two Counters."""
    if not ctr1 or not ctr2: return 0.0
    all_keys = ctr1.keys() | ctr2.keys()
    intersection_sum = sum(min(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    union_sum = sum(max(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    return intersection_sum / union_sum if union_sum else 0.0

def run_moderator_analysis():
    """Main function to run the moderator analysis."""
    print("--- Running Moderator Analysis ---")
    PLOTS_DIR.mkdir(exist_ok=True)

    # --- 1. Load All Necessary Data ---
    print("\n[1/4] Loading data sources...")
    try:
        df_results = pd.read_csv(RESULTS_CSV_PATH)
        df_swow = pd.read_csv(SWOW_R100_PATH)
        
        llm_lexicon = {}
        with open(LEXICON_PATH, 'r') as f:
            for line in f:
                record = json.loads(line)
                all_words = [str(w).lower() for s in record['association_sets'] for w in s if w]
                llm_lexicon[record['word']] = Counter(all_words)
    except FileNotFoundError as e:
        print(f"❌ ERROR: A required data file was not found: {e.filename}. Please run previous scripts first.")
        return
    
    print("✅ Data loaded successfully.")

    # --- 2. Calculate Alignment Scores ---
    print("\n[2/4] Calculating alignment scores (General and Demographic)...")
    
    df_swow_clean = df_swow.dropna(subset=['cue', 'R1', 'R2', 'R3'])
    general_human_norms = {
        cue.lower(): Counter(pd.concat([grp['R1'], grp['R2'], grp['R3']]).dropna().str.lower())
        for cue, grp in df_swow_clean.groupby("cue")
    }

    df_swow['gender_norm'] = df_swow['gender'].str.strip().str.lower().str[0]
    male_df = df_swow[df_swow['gender_norm'] == 'm']
    female_df = df_swow[df_swow['gender_norm'] == 'f']

    male_human_norms = {
        cue.lower(): Counter(pd.concat([grp['R1'], grp['R2'], grp['R3']]).dropna().str.lower())
        for cue, grp in male_df.groupby("cue")
    }
    female_human_norms = {
        cue.lower(): Counter(pd.concat([grp['R1'], grp['R2'], grp['R3']]).dropna().str.lower())
        for cue, grp in female_df.groupby("cue")
    }

    alignment_scores = {}
    for word, llm_vec in llm_lexicon.items():
        alignment_scores[word] = {
            'general': wjacc(general_human_norms.get(word, Counter()), llm_vec),
            'male': wjacc(male_human_norms.get(word, Counter()), llm_vec),
            'female': wjacc(female_human_norms.get(word, Counter()), llm_vec),
        }

    for a_type in ['general', 'male', 'female']:
        df_results[f'anchor_align_{a_type}'] = df_results['anchor'].map(lambda x: alignment_scores.get(x, {}).get(a_type, 0))
        df_results[f't1_align_{a_type}'] = df_results['target1'].map(lambda x: alignment_scores.get(x, {}).get(a_type, 0))
        df_results[f't2_align_{a_type}'] = df_results['target2'].map(lambda x: alignment_scores.get(x, {}).get(a_type, 0))
        df_results[f'avg_align_{a_type}'] = df_results[[f'anchor_align_{a_type}', f't1_align_{a_type}', f't2_align_{a_type}']].mean(axis=1)

    print("✅ Alignment scores calculated and merged.")
    
    # --- 3. Part 1: General Alignment as Moderator (Logistic Regression) ---
    print("\n[3/4] Analyzing General Alignment with Logistic Regression...")
    
    # Define variables for the model
    # Dependent variable 'y' is whether the prediction was correct (1) or not (0)
    y = df_results['correct']
    # Independent variable 'X' is the continuous alignment score
    X = df_results['avg_align_general']
    # Add a constant (i.e., intercept) to the model
    X = sm.add_constant(X)
    
    # Fit the logistic regression model
    logit_model = sm.Logit(y.astype(float), X.astype(float)).fit()
    
    # Print the detailed model summary
    print("\n--- Logistic Regression Summary ---")
    print(logit_model.summary())
    
    # Visualize the logistic regression
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df_results['avg_align_general'], y=df_results['correct'].astype(float), 
                data=df_results, logistic=True, ci=95,
                scatter_kws={'alpha': 0.2, 's': 20},
                line_kws={'color': 'red'})

    plt.title('Logistic Regression of Prediction Accuracy vs. Alignment Score', fontsize=16)
    plt.xlabel('Average Triplet Alignment with General Human Norms', fontsize=12)
    plt.ylabel('Probability of Correct Prediction', fontsize=12)
    plt.savefig(GENERAL_ALIGNMENT_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ General alignment regression plot saved to '{GENERAL_ALIGNMENT_PLOT_PATH}'")
    plt.close()

    # --- 4. Part 2: Demographic Alignment as Moderator ---
    print("\n[4/4] Analyzing Demographic Alignment as a moderator...")
    
    df_male = df_results[['correct', 'avg_align_male']].copy()
    df_male.rename(columns={'avg_align_male': 'alignment'}, inplace=True)
    df_male['subgroup'] = 'Male'

    df_female = df_results[['correct', 'avg_align_female']].copy()
    df_female.rename(columns={'avg_align_female': 'alignment'}, inplace=True)
    df_female['subgroup'] = 'Female'
    
    df_demographic = pd.concat([df_male, df_female])
    
    df_demographic['align_bin'] = df_demographic.groupby('subgroup')['alignment'].transform(
        lambda x: pd.qcut(x, q=4, labels=['Lowest', 'Low', 'High', 'Highest'], duplicates='drop')
    )
    
    accuracy_by_demographic_align = df_demographic.groupby(['subgroup', 'align_bin'])['correct'].mean().reset_index()
    
    print("\n--- Prediction Accuracy by Demographic (Gender) Alignment ---")
    print(accuracy_by_demographic_align)

    plt.figure(figsize=(12, 7))
    sns.catplot(data=accuracy_by_demographic_align, x='align_bin', y='correct', hue='subgroup', 
                kind='bar', palette={'Male': 'blue', 'Female': 'orange'}, legend_out=False)
    plt.title('Prediction Accuracy Moderated by Gender-Specific Alignment', fontsize=16)
    plt.xlabel('Triplet Alignment with Subgroup Norms (Binned)', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title='Alignment Subgroup')
    plt.tight_layout()
    plt.savefig(DEMOGRAPHIC_ALIGNMENT_PLOT_PATH, dpi=300)
    print(f"✅ Demographic alignment plot saved to '{DEMOGRAPHIC_ALIGNMENT_PLOT_PATH}'")
    plt.close()

if __name__ == "__main__":
    run_moderator_analysis()
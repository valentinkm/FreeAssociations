"""Core analysis functions for human-model alignment comparisons."""
import json
import random
from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from . import settings, llm_generation
from .analysis_utils import load_lexicon, weighted_jaccard

def compare_model_alignment(models: list, nsets: int, ncues: int):
    """
    Ensures lexicons are sufficiently populated, then compares the foundational
    alignment of a specified list of models against human SWOW data.
    """
    print(f"\n--- Comparing Foundational Model Alignment vs. Human Norms ---")
    print(f"Models to compare: {', '.join(models)}")
    print(f"Ensuring at least {ncues} cues exist for each model with n_sets={nsets}")
    
    models_to_compare = models

    # 1. Load human norms and get a master list of all possible cues
    print("🔄 Loading human norms from SWOW dataset...")
    try:
        df_swow = pd.read_csv(settings.SWOW_DATA_PATH)
        human_norms = {
            cue.lower(): Counter(pd.concat([grp['R1'], grp['R2'], grp['R3']]).dropna().str.lower())
            for cue, grp in df_swow.groupby("cue")
        }
        all_human_cues = list(human_norms.keys())
    except FileNotFoundError:
        print(f"❌ ERROR: SWOW data file not found at '{settings.SWOW_DATA_PATH}'.")
        return

    # 2. Loop through models to generate missing data
    for model_name in models_to_compare:
        print(f"\n-- Checking/Generating data for model: {model_name} --")
        lexicon_path = settings.get_lexicon_path(model_name=model_name, nsets=nsets)
        
        processed_words = set()
        if lexicon_path.exists():
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        processed_words.add(json.loads(line)['word'].lower())
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        print(f"Found {len(processed_words)} existing cues in '{lexicon_path.name}'.")

        num_needed = ncues - len(processed_words)
        if num_needed > 0:
            print(f"Need to generate {num_needed} more cues to reach target of {ncues}.")
            available_to_generate = [c for c in all_human_cues if c not in processed_words]
            if not available_to_generate:
                print("⚠️ No more unique cues available from SWOW to generate.")
                continue
            
            cues_to_generate = random.sample(available_to_generate, k=min(num_needed, len(available_to_generate)))
            
            llm_generation.generate_lexicon_data(
                vocabulary=set(cues_to_generate),
                model_name=model_name,
                lexicon_path=lexicon_path,
                nsets=nsets,
                prompt_key=settings.GENERALIZE_DEFAULTS['prompt']
            )
        else:
            print("✅ Lexicon meets or exceeds the required number of cues.")

    # 3. Loop through models again to perform the final analysis
    print("\n--- Calculating Final Alignment Scores for Comparison ---")
    results = []
    for model_name in models_to_compare:
        lexicon_path = settings.get_lexicon_path(model_name=model_name, nsets=nsets)
        llm_lexicon = load_lexicon(lexicon_path)
        if not llm_lexicon: continue

        available_cues = list(llm_lexicon.keys())
        if len(available_cues) < ncues:
            print(f"⚠️ Warning: Model '{model_name}' has only {len(available_cues)} cues, less than the target {ncues}.")
            comparison_cues = available_cues
        else:
            comparison_cues = random.sample(available_cues, k=ncues)

        alignment_scores = []
        for word in comparison_cues:
            human_vec = human_norms.get(word.lower(), Counter())
            if human_vec:
                alignment_scores.append(weighted_jaccard(human_vec, llm_lexicon[word]))
        
        if alignment_scores:
            avg_alignment = np.mean(alignment_scores)
            results.append({"Model": model_name, "Alignment Score": avg_alignment})
            print(f"  {model_name}: Average Alignment = {avg_alignment:.3f}")

    if not results:
        print("\n❌ No models could be evaluated.")
        return

    # 4. Create and save the comparison plot
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=results_df, x="Model", y="Alignment Score", palette="viridis")
    ax.set_title(f"Model Alignment vs. Human Norms (n_sets={nsets}, n_cues={ncues})", fontsize=16)
    ax.set_ylabel("Average Alignment (Weighted Jaccard)")
    ax.set_xlabel("Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plot_path = settings.PLOTS_DIR / "model_alignment_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ Model alignment comparison plot saved to '{plot_path.name}'")

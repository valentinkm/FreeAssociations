"""
analysis.py
───────────
Core analysis functions for comparing human vs. LLM data.
This module now includes full analysis pipelines with regression,
moderation analysis, plotting, and result saving.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import cosine
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from . import settings

# --- Helper Functions ---

def _robust_flatten(items):
    """Recursively flattens a list of lists of strings."""
    for x in items:
        if isinstance(x, list):
            yield from _robust_flatten(x)
        elif isinstance(x, str):
            yield x.lower()

def _wjacc(ctr1: Counter, ctr2: Counter) -> float:
    """Calculates the weighted Jaccard similarity between two Counters."""
    if not ctr1 or not ctr2: return 0.0
    all_keys = ctr1.keys() | ctr2.keys()
    intersection = sum(min(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    union = sum(max(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    return intersection / union if union else 0.0

def _cosine_sim(ctr1: Counter, ctr2: Counter) -> float:
    """Calculates the cosine similarity between two Counters."""
    if not ctr1 or not ctr2: return 0.0
    vocab = sorted(list(ctr1.keys() | ctr2.keys()))
    v1 = np.array([ctr1.get(word, 0) for word in vocab])
    v2 = np.array([ctr2.get(word, 0) for word in vocab])
    if not np.any(v1) or not np.any(v2): return 0.0
    return 1 - cosine(v1, v2)

def _load_lexicon(lexicon_path: Path):
    """Loads a lexicon file into a dictionary of word Counters."""
    print(f"🔄 Loading LLM Association Lexicon from '{lexicon_path.name}'...")
    try:
        llm_lexicon = {}
        with open(lexicon_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                word = record['word']
                associations = list(_robust_flatten(record.get('association_sets', [])))
                llm_lexicon[word] = Counter(associations)
        print(f"✅ Loaded {len(llm_lexicon)} word vectors from lexicon.")
        return llm_lexicon
    except FileNotFoundError:
        print(f"❌ ERROR: Lexicon file not found at '{lexicon_path}'.")
        return None

# --- Analysis Pipelines ---

def analyze_holistic_similarity(spp_data_to_analyze: pd.DataFrame, lexicon_path: Path):
    """Performs the full LLM holistic similarity analysis for SPP."""
    print("\n--- Analyzing Holistic Similarity (LLM vs. SPP) ---")
    
    llm_lexicon = _load_lexicon(lexicon_path)
    if not llm_lexicon: return

    available_words = set(llm_lexicon.keys())
    spp_data_analyzable = spp_data_to_analyze[
        spp_data_to_analyze['prime'].str.lower().isin(available_words) &
        spp_data_to_analyze['target'].str.lower().isin(available_words)
    ].copy()

    if spp_data_analyzable.empty:
        print("❌ No SPP pairs found where both prime and target exist in the lexicon.")
        return

    results = []
    for task in tqdm(spp_data_analyzable.itertuples(), total=len(spp_data_analyzable), desc="Calculating SPP Similarity"):
        prime_vec = llm_lexicon.get(task.prime.lower())
        target_vec = llm_lexicon.get(task.target.lower())
        similarity = _cosine_sim(prime_vec, target_vec)
        results.append({ "prime": task.prime, "target": task.target, "priming_effect": task.priming_effect, "associative_cosine_similarity": similarity, })
    
    analysis_df = pd.DataFrame(results)
    if len(analysis_df) < 2:
        print("\n❌ Error: Not enough valid pairs to run a regression.")
        return

    print("🔄 Running final regression analysis...")
    y = analysis_df['priming_effect']
    X = sm.add_constant(analysis_df['associative_cosine_similarity'])
    model = sm.OLS(y, X).fit()
    
    print("\n--- SPP Holistic Similarity Regression Results ---")
    print(model.summary())
    
    nsets_val = lexicon_path.stem.split('_')[-1]
    plot_path = settings.PLOTS_DIR / f"spp_holistic_nsets_{nsets_val}.png"
    plt.figure(figsize=(10, 7))
    sns.regplot(x='associative_cosine_similarity', y='priming_effect', data=analysis_df, scatter_kws={'alpha': 0.5}, line_kws={'color': '#4C72B0'})
    plt.title(f'LLM Associative Similarity vs. Semantic Priming (n_sets={nsets_val})', fontsize=16)
    plt.xlabel('LLM Associative Cosine Similarity', fontsize=12)
    plt.ylabel('Semantic Priming Effect (Log RT Difference)', fontsize=12)
    stats_text = f'$R^2 = {model.rsquared:.3f}$\n$p = {model.pvalues["associative_cosine_similarity"]:.3g}$'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, va='top', bbox=dict(boxstyle='round', fc='#AED6F1', alpha=0.5))
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analysis complete. Plot saved to '{plot_path}'")


def analyze_3tt_similarity(ttt_data_to_analyze: pd.DataFrame, lexicon_path: Path):
    """
    Performs the full 3TT analysis, including choice prediction, moderation by
    alignment, plotting, and result saving.
    """
    print("\n--- Analyzing 3TT Choice Prediction and Moderation ---")
    
    llm_lexicon = _load_lexicon(lexicon_path)
    if not llm_lexicon: return

    # --- 1. Predict Choices ---
    print("\n[1/3] Predicting human choices using LLM associations...")
    predictions = []
    for task in tqdm(ttt_data_to_analyze.itertuples(), total=len(ttt_data_to_analyze), desc="Predicting 3TT Choices"):
        cue, t1, t2 = task.cue.lower(), task.choiceA.lower(), task.choiceB.lower()
        cue_vec, t1_vec, t2_vec = llm_lexicon.get(cue), llm_lexicon.get(t1), llm_lexicon.get(t2)
        if not all([cue_vec, t1_vec, t2_vec]): continue

        # Predict using both cosine and weighted Jaccard similarity
        sim_cos_1, sim_cos_2 = _cosine_sim(cue_vec, t1_vec), _cosine_sim(cue_vec, t2_vec)
        sim_jacc_1, sim_jacc_2 = _wjacc(cue_vec, t1_vec), _wjacc(cue_vec, t2_vec)
        
        predictions.append({
            "cue": cue, "choiceA": t1, "choiceB": t2,
            "human_choice_num": int(task.chosen),
            "human_choice_word": task.human_related_choice.lower(),
            "pred_cos": 1 if sim_cos_1 >= sim_cos_2 else 2,
            "pred_jacc": 1 if sim_jacc_1 >= sim_jacc_2 else 2,
        })
        
    analysis_df = pd.DataFrame(predictions)
    analysis_df['correct_cos'] = (analysis_df['pred_cos'] == analysis_df['human_choice_num'])
    analysis_df['correct_jacc'] = (analysis_df['pred_jacc'] == analysis_df['human_choice_num'])

    acc_cos = analysis_df['correct_cos'].mean()
    acc_jacc = analysis_df['correct_jacc'].mean()

    print(f"\n--- 3TT Prediction Accuracy ---")
    print(f"Total Triplets Evaluated: {len(analysis_df)}")
    print(f"Accuracy (Cosine Similarity): {acc_cos:.2%}")
    print(f"Accuracy (Weighted Jaccard): {acc_jacc:.2%}")

    # --- 2. Moderation Analysis ---
    print("\n[2/3] Analyzing if alignment moderates prediction accuracy...")
    try:
        df_swow = pd.read_csv(settings.SWOW_DATA_PATH)
        human_norms = {c.lower(): Counter(pd.concat([g['R1'], g['R2'], g['R3']]).dropna().str.lower()) for c, g in df_swow.groupby("cue")}
    except FileNotFoundError:
        print("❌ Could not load SWOW data for human norms. Skipping moderation analysis.")
    else:
        alignment_scores = {word: _wjacc(human_norms.get(word, Counter()), llm_vec) for word, llm_vec in llm_lexicon.items()}
        analysis_df['avg_triplet_alignment'] = analysis_df.apply(
            lambda r: np.mean([alignment_scores.get(r['cue'],0), alignment_scores.get(r['choiceA'],0), alignment_scores.get(r['choiceB'],0)]), axis=1
        )
        
        # Run logistic regression: Does alignment predict correctness?
        y = analysis_df['correct_cos']
        X = sm.add_constant(analysis_df['avg_triplet_alignment'])
        logit_model = sm.Logit(y.astype(float), X.astype(float)).fit(disp=0)
        
        print("\n--- Moderation by Alignment (Logistic Regression) ---")
        print(logit_model.summary())
        
        # Plot moderation analysis
        nsets_val = lexicon_path.stem.split('_')[-1]
        plot_path = settings.PLOTS_DIR / f"3tt_moderation_nsets_{nsets_val}.png"
        plt.figure(figsize=(10, 6))
        sns.regplot(x='avg_triplet_alignment', y='correct_cos', data=analysis_df, logistic=True, ci=95,
                    scatter_kws={'alpha': 0.2, 'color': 'darkcyan'}, line_kws={'color': 'teal'})
        plt.title(f'Prediction Accuracy Moderated by Human-LLM Alignment (n_sets={nsets_val})', fontsize=16)
        plt.xlabel('Average Triplet Alignment with Human Norms (W. Jaccard)', fontsize=12)
        plt.ylabel('Probability of Correct Prediction', fontsize=12)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Moderation plot saved to '{plot_path}'")

    # --- 3. Save Detailed Results ---
    print("\n[3/3] Saving detailed results...")
    nsets_val = lexicon_path.stem.split('_')[-1]
    results_dir = settings.OUTPUTS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"3tt_results_nsets_{nsets_val}.csv"
    analysis_df.to_csv(results_path, index=False)
    print(f"✅ Detailed 3TT results saved to '{results_path}'")


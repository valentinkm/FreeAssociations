"""
analysis.py
───────────
Core analysis functions for comparing human vs. LLM data.
"""
import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import cosine
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from . import settings, data_utils, llm_generation

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
        with open(lexicon_path, 'r', encoding='utf-8') as f:
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

# --- Main Analysis Pipelines ---

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
    
    model_name = lexicon_path.name.split('_')[1]
    nsets_val = lexicon_path.stem.split('_')[-1]
    plot_path = settings.PLOTS_DIR / f"spp_holistic_{model_name}_nsets_{nsets_val}.png"
    plt.figure(figsize=(10, 7))
    sns.regplot(x='associative_cosine_similarity', y='priming_effect', data=analysis_df, scatter_kws={'alpha': 0.5}, line_kws={'color': '#4C72B0'})
    plt.title(f'LLM Associative Similarity vs. Semantic Priming ({model_name}, n_sets={nsets_val})', fontsize=16)
    plt.xlabel('LLM Associative Cosine Similarity', fontsize=12)
    plt.ylabel('Semantic Priming Effect (Log RT Difference)', fontsize=12)
    stats_text = f'$R^2 = {model.rsquared:.3f}$\n$p = {model.pvalues["associative_cosine_similarity"]:.3g}$'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, va='top', bbox=dict(boxstyle='round', fc='#AED6F1', alpha=0.5))
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analysis complete. Plot saved to '{plot_path.name}'")

def analyze_3tt_similarity(ttt_data_to_analyze: pd.DataFrame, lexicon_path: Path):
    """
    Performs the full 3TT analysis, including choice prediction, moderation by
    alignment, plotting, and result saving.
    """
    print("\n--- Analyzing 3TT Choice Prediction and Moderation ---")
    
    llm_lexicon = _load_lexicon(lexicon_path)
    if not llm_lexicon: return

    # 1. Predict Choices
    print("\n[1/3] Predicting human choices using LLM associations...")
    predictions = []
    for task in tqdm(ttt_data_to_analyze.itertuples(), total=len(ttt_data_to_analyze), desc="Predicting 3TT Choices"):
        cue, t1, t2 = task.cue.lower(), task.choiceA.lower(), task.choiceB.lower()
        cue_vec, t1_vec, t2_vec = llm_lexicon.get(cue), llm_lexicon.get(t1), llm_lexicon.get(t2)
        if not all([cue_vec, t1_vec, t2_vec]): continue

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
    if analysis_df.empty:
        print("❌ No predictions could be made. Check for word overlap with the lexicon.")
        return
        
    analysis_df['correct_cos'] = (analysis_df['pred_cos'] == analysis_df['human_choice_num'])
    analysis_df['correct_jacc'] = (analysis_df['pred_jacc'] == analysis_df['human_choice_num'])
    acc_cos = analysis_df['correct_cos'].mean()
    acc_jacc = analysis_df['correct_jacc'].mean()

    print(f"\n--- 3TT Prediction Accuracy ---")
    print(f"Total Triplets Evaluated: {len(analysis_df)}")
    print(f"Accuracy (Cosine Similarity): {acc_cos:.2%}")
    print(f"Accuracy (Weighted Jaccard): {acc_jacc:.2%}")

    # 2. Moderation Analysis
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
        
        y = analysis_df['correct_cos']
        X = sm.add_constant(analysis_df['avg_triplet_alignment'])
        
        if len(y.unique()) > 1 and len(X) > 1:
            try:
                logit_model = sm.Logit(y.astype(float), X.astype(float)).fit(disp=0)
                print("\n--- Moderation by Alignment (Logistic Regression) ---")
                print(logit_model.summary())
                
                model_name = lexicon_path.name.split('_')[1]
                nsets_val = lexicon_path.stem.split('_')[-1]
                plot_path = settings.PLOTS_DIR / f"3tt_moderation_{model_name}_nsets_{nsets_val}.png"
                plt.figure(figsize=(10, 6))
                sns.regplot(x='avg_triplet_alignment', y='correct_cos', data=analysis_df, logistic=True, ci=95,
                            scatter_kws={'alpha': 0.2, 'color': 'darkcyan'}, line_kws={'color': 'teal'})
                plt.title(f'Prediction Accuracy Moderated by Human-LLM Alignment ({model_name}, n_sets={nsets_val})', fontsize=16)
                plt.xlabel('Average Triplet Alignment with Human Norms (W. Jaccard)', fontsize=12)
                plt.ylabel('Probability of Correct Prediction', fontsize=12)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Moderation plot saved to '{plot_path.name}'")
            except Exception as e:
                print(f"⚠️ Skipping moderation analysis due to a statistical error (e.g., perfect separation): {e}")
        else:
            print("⚠️ Skipping moderation analysis due to insufficient data variance.")

    # 3. Save Detailed Results
    print("\n[3/3] Saving detailed results...")
    model_name = lexicon_path.name.split('_')[1]
    nsets_val = lexicon_path.stem.split('_')[-1]
    results_path = settings.RESULTS_DIR / f"3tt_results_{model_name}_nsets_{nsets_val}.csv"
    analysis_df.to_csv(results_path, index=False)
    print(f"✅ Detailed 3TT results saved to '{results_path.name}'")

def compare_models_on_task(task: str):
    """Compares the performance of multiple models on a given generalization task."""
    print(f"\n--- Comparing Model Performance on Task: {task.upper()} ---")
    
    models_to_compare = ["gpt-4o", "gemini-1.5-flash-latest", "llama3", "llama3:text"]
    nsets_for_comparison = 25
    
    print(f"Loading task data for {task.upper()}...")
    if task == 'spp':
        task_data, _ = data_utils.get_spp_data_for_analysis(Path(), 500)
    else: # 3tt
        task_data, _ = data_utils.get_3tt_data_for_analysis(Path(), 500)

    if task_data is None:
        print("❌ Could not load task data. Aborting comparison.")
        return
        
    results = []
    for model_name in models_to_compare:
        print(f"\n-- Evaluating model: {model_name} --")
        lexicon_path = settings.get_lexicon_path(model_name=model_name, nsets=nsets_for_comparison)
        if not lexicon_path.exists():
            print(f"⚠️ Warning: Lexicon file not found for '{model_name}'. Skipping. Please generate it first with the 'generalize' command.")
            continue
        
        llm_lexicon = _load_lexicon(lexicon_path)
        if not llm_lexicon: continue

        score, metric_name = 0, ""
        if task == 'spp':
            y = task_data['priming_effect']
            available_data = task_data[task_data['prime'].str.lower().isin(llm_lexicon.keys()) & task_data['target'].str.lower().isin(llm_lexicon.keys())]
            if not available_data.empty:
                X_data = available_data.apply(lambda r: _cosine_sim(llm_lexicon.get(r['prime'].lower()), llm_lexicon.get(r['target'].lower())), axis=1)
                y = y[X_data.index]
                X = sm.add_constant(X_data)
                if len(X) > 1:
                    score = sm.OLS(y, X).fit().rsquared
            metric_name = "R-squared"
        elif task == '3tt':
            correct_predictions = 0
            available_data = task_data[task_data.apply(lambda r: all(w.lower() in llm_lexicon for w in [r.cue, r.choiceA, r.choiceB]), axis=1)]
            if not available_data.empty:
                for t in available_data.itertuples():
                    cue_vec, t1_vec, t2_vec = llm_lexicon.get(t.cue.lower()), llm_lexicon.get(t.choiceA.lower()), llm_lexicon.get(t.choiceB.lower())
                    sim1, sim2 = _cosine_sim(cue_vec, t1_vec), _cosine_sim(cue_vec, t2_vec)
                    pred_choice = 1 if sim1 >= sim2 else 2
                    if pred_choice == t.human_choice_num:
                        correct_predictions += 1
                score = correct_predictions / len(available_data)
            metric_name = "Accuracy"
            
        results.append({"Model": model_name, "Score": score})
        print(f"  {metric_name}: {score:.3f}")
        
    if not results:
        print("\n❌ No models could be evaluated. Ensure lexicons exist.")
        return

    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=results_df, x="Model", y="Score", palette="viridis")
    ax.set_title(f"Model Performance Comparison on {task.upper()} Task (n_sets={nsets_for_comparison})", fontsize=16)
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(settings.PLOT_MODEL_COMPARISON, dpi=300)
    print(f"\n✅ Model comparison plot saved to '{settings.PLOT_MODEL_COMPARISON.name}'")

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
        llm_lexicon = _load_lexicon(lexicon_path)
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
                alignment_scores.append(_wjacc(human_vec, llm_lexicon[word]))
        
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

"""Generalization analyses using SPP and 3TT datasets."""
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from . import settings, data_utils
from .analysis_utils import load_lexicon, cosine_sim, weighted_jaccard


def analyze_holistic_similarity(spp_data_to_analyze: pd.DataFrame, lexicon_path: Path):
    """Performs the LLM holistic similarity analysis for SPP."""
    print("\n--- Analyzing Holistic Similarity (LLM vs. SPP) ---")
    llm_lexicon = load_lexicon(lexicon_path)
    if not llm_lexicon:
        return

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
        similarity = cosine_sim(prime_vec, target_vec)
        results.append({
            "prime": task.prime,
            "target": task.target,
            "priming_effect": task.priming_effect,
            "associative_cosine_similarity": similarity,
        })

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
    """Performs the full 3TT analysis, including choice prediction and moderation."""
    print("\n--- Analyzing 3TT Choice Prediction and Moderation ---")

    llm_lexicon = load_lexicon(lexicon_path)
    if not llm_lexicon:
        return

    # 1. Predict Choices
    print("\n[1/3] Predicting human choices using LLM associations...")
    predictions = []
    for task in tqdm(ttt_data_to_analyze.itertuples(), total=len(ttt_data_to_analyze), desc="Predicting 3TT Choices"):
        cue, t1, t2 = task.cue.lower(), task.choiceA.lower(), task.choiceB.lower()
        cue_vec, t1_vec, t2_vec = llm_lexicon.get(cue), llm_lexicon.get(t1), llm_lexicon.get(t2)
        if not all([cue_vec, t1_vec, t2_vec]):
            continue

        sim_cos_1, sim_cos_2 = cosine_sim(cue_vec, t1_vec), cosine_sim(cue_vec, t2_vec)
        sim_jacc_1, sim_jacc_2 = weighted_jaccard(cue_vec, t1_vec), weighted_jaccard(cue_vec, t2_vec)

        predictions.append({
            "cue": cue,
            "choiceA": t1,
            "choiceB": t2,
            "human_choice_num": int(task.chosen),
            "human_choice_word": task.human_related_choice.lower(),
            "pred_cos": 1 if sim_cos_1 >= sim_cos_2 else 2,
            "pred_jacc": 1 if sim_jacc_1 >= sim_jacc_2 else 2,
        })

    analysis_df = pd.DataFrame(predictions)
    if analysis_df.empty:
        print("❌ No predictions could be made. Check for word overlap with the lexicon.")
        return

    analysis_df['correct_cos'] = analysis_df['pred_cos'] == analysis_df['human_choice_num']
    analysis_df['correct_jacc'] = analysis_df['pred_jacc'] == analysis_df['human_choice_num']
    acc_cos = analysis_df['correct_cos'].mean()
    acc_jacc = analysis_df['correct_jacc'].mean()

    print("\n--- 3TT Prediction Accuracy ---")
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
        alignment_scores = {word: weighted_jaccard(human_norms.get(word, Counter()), llm_vec) for word, llm_vec in llm_lexicon.items()}
        analysis_df['avg_triplet_alignment'] = analysis_df.apply(
            lambda r: np.mean([alignment_scores.get(r['cue'], 0), alignment_scores.get(r['choiceA'], 0), alignment_scores.get(r['choiceB'], 0)]),
            axis=1,
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
                sns.regplot(
                    x='avg_triplet_alignment',
                    y='correct_cos',
                    data=analysis_df,
                    logistic=True,
                    ci=95,
                    scatter_kws={'alpha': 0.2, 'color': 'darkcyan'},
                    line_kws={'color': 'teal'},
                )
                plt.title(
                    f'Prediction Accuracy Moderated by Human-LLM Alignment ({model_name}, n_sets={nsets_val})',
                    fontsize=16,
                )
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
    else:  # 3tt
        task_data, _ = data_utils.get_3tt_data_for_analysis(Path(), 500)

    if task_data is None:
        print("❌ Could not load task data. Aborting comparison.")
        return

    results = []
    for model_name in models_to_compare:
        print(f"\n-- Evaluating model: {model_name} --")
        lexicon_path = settings.get_lexicon_path(model_name=model_name, nsets=nsets_for_comparison)
        if not lexicon_path.exists():
            print(
                f"⚠️ Warning: Lexicon file not found for '{model_name}'. Skipping. Please generate it first with the 'generalize' command."
            )
            continue

        llm_lexicon = load_lexicon(lexicon_path)
        if not llm_lexicon:
            continue

        score, metric_name = 0, ""
        if task == 'spp':
            y = task_data['priming_effect']
            available_data = task_data[
                task_data['prime'].str.lower().isin(llm_lexicon.keys())
                & task_data['target'].str.lower().isin(llm_lexicon.keys())
            ]
            if not available_data.empty:
                X_data = available_data.apply(
                    lambda r: cosine_sim(llm_lexicon.get(r['prime'].lower()), llm_lexicon.get(r['target'].lower())),
                    axis=1,
                )
                y = y[X_data.index]
                X = sm.add_constant(X_data)
                if len(X) > 1:
                    score = sm.OLS(y, X).fit().rsquared
            metric_name = "R-squared"
        elif task == '3tt':
            correct_predictions = 0
            available_data = task_data[
                task_data.apply(lambda r: all(w.lower() in llm_lexicon for w in [r.cue, r.choiceA, r.choiceB]), axis=1)
            ]
            if not available_data.empty:
                for t in available_data.itertuples():
                    cue_vec, t1_vec, t2_vec = (
                        llm_lexicon.get(t.cue.lower()),
                        llm_lexicon.get(t.choiceA.lower()),
                        llm_lexicon.get(t.choiceB.lower()),
                    )
                    sim1, sim2 = cosine_sim(cue_vec, t1_vec), cosine_sim(cue_vec, t2_vec)
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

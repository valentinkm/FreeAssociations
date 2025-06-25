"""
evaluation.py
─────────────
Functions for evaluating the output of generation experiments, such as
prompt sweeps and yoked steering runs.
"""
import pandas as pd
import numpy as np
import json
import math
import hashlib
import pickle
import re
import csv
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from . import settings
from .profile_utils import age_bin, norm_gender, slugify

# --- Helper Functions ---
def _wjacc(ctr1: Counter, ctr2: Counter) -> float:
    if not ctr1 or not ctr2: return 0.0
    all_keys = ctr1.keys() | ctr2.keys()
    intersection = sum(min(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    union = sum(max(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    return intersection / union if union else 0.0

def _entropy(cnt):
    tot = sum(cnt.values())
    return -sum((c / tot) * math.log2(c / tot) for c in cnt.values()) if tot else 0.0

# --- Prompt Sweep Evaluation ---
def evaluate_prompt_sweep():
    # This function remains the same
    print("\n--- Evaluating Prompt Sweep Results ---")
    if not settings.RUNS_DIR.exists():
        print("❌ 'runs' directory not found. No data to evaluate.")
        return
    if settings.HUMAN_NORMS_CACHE.exists():
        human_norms = pickle.loads(settings.HUMAN_NORMS_CACHE.read_bytes())
    else:
        print("Building human norms cache...")
        df = pd.read_csv(settings.SWOW_DATA_PATH, usecols=["cue", "R1", "R2", "R3"])
        human_norms = { cue.lower(): Counter(w for col in ("R1", "R2", "R3") for w in grp[col].dropna()) for cue, grp in df.groupby("cue") }
        settings.HUMAN_NORMS_CACHE.write_bytes(pickle.dumps(human_norms))
    seen = set(pd.read_csv(settings.SWEEP_SCORES_CSV_PATH)["hash"].unique()) if settings.SWEEP_SCORES_CSV_PATH.exists() else set()
    rows = []
    hash8 = lambda cfg: hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    for f in tqdm(sorted(settings.RUNS_DIR.glob("grid_*.jsonl")), desc="Scoring Runs"):
        with open(f) as fh:
            cfg = None
            j_sum = rec_sum = ent_gap_sum = cue_n = 0
            for line in fh:
                rec, cue = json.loads(line), json.loads(line)["cue"].lower()
                if cue not in human_norms: continue
                hctr = human_norms[cue]
                m_words = [w.lower() for triple in rec["sets"] for w in triple]
                j_sum += _wjacc(hctr, Counter(m_words))
                top10 = {w for w, _ in hctr.most_common(10)}
                rec_sum += any(w in top10 for w in m_words)
                ent_gap_sum += abs(_entropy(hctr) - _entropy(Counter(m_words)))
                cue_n += 1
            if cue_n:
                cfg = rec["cfg"]
                h = hash8(cfg)
                if h in seen: continue
                rows.append({"hash": h, "prompt": cfg["prompt"], "jaccard": round(j_sum/cue_n, 4), "recall10": round(rec_sum/cue_n, 4), "ent_gap": round(ent_gap_sum/cue_n, 4)})
                seen.add(h)
    if rows:
        mode = "a" if settings.SWEEP_SCORES_CSV_PATH.exists() else "w"
        with open(settings.SWEEP_SCORES_CSV_PATH, mode, newline="") as fw:
            wr = csv.DictWriter(fw, fieldnames=rows[0].keys())
            if mode == "w": wr.writeheader()
            wr.writerows(rows)
        print(f"✅ Added {len(rows)} new run results to '{settings.SWEEP_SCORES_CSV_PATH.name}'")
    else:
        print("ℹ️ No new runs to evaluate.")
    if not settings.SWEEP_SCORES_CSV_PATH.exists(): return
    df = pd.read_csv(settings.SWEEP_SCORES_CSV_PATH)
    sns.set_theme(style="whitegrid")
    plt.figure()
    ax = sns.scatterplot(data=df, x="ent_gap", y="jaccard", hue="prompt", s=100)
    ax.set(xlabel="Entropy Gap (lower = better)", ylabel="Weighted Jaccard (higher = better)", title="Free-Association Alignment: Prompt Variants")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.savefig(settings.PLOT_SWEEP_SCATTER, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot saved to '{settings.PLOT_SWEEP_SCATTER.name}'")
    tbl = df.groupby("prompt").agg(jaccard_mean=("jaccard", "mean"), recall_mean=("recall10", "mean"), ent_gap_mean=("ent_gap", "mean")).sort_values("jaccard_mean", ascending=False).round(3)
    print("\n--- Prompt Ranking (by Jaccard) ---")
    print(tbl)

# --- Yoked Steering Evaluation ---
def evaluate_yoked_steering():
    """Evaluates the yoked steering experiment with robust, publication-quality plotting."""
    print("\n--- Evaluating Yoked Steering Results ---")
    print("🔄 Loading and processing all data sources...")
    try:
        df_human = pd.read_csv(settings.SWOW_DATA_PATH)
        steered_data_path = settings.YOKED_STEERED_PATH
        nonsteered_data_path = settings.YOKED_NONSTEERED_PATH
        
        UK_IRELAND = {"United Kingdom", "Ireland"}
        EUROPE = {"Germany", "Belgium", "Netherlands", "France", "Spain", "Finland", "Sweden", "Italy", "Denmark", "Norway", "Switzerland", "Poland", "Hungary", "Romania", "Portugal", "Austria", "Greece", "Turkey", "Czech Republic", "Croatia", "Serbia", "Slovakia", "Luxembourg", "Iceland", "Slovenia", "Malta", "Cyprus", "Estonia", "Latvia", "Lithuania"}
        def country_binner(country):
            if country == "United States": return "United States"
            if country in UK_IRELAND: return "UK & Ireland"
            if country == "Canada": return "Canada"
            if country in EUROPE: return "Europe"
            return "Other"
        
        edu_map = {1: "<HS", 2: "HS", 3: "Some College", 4: "Bachelor", 5: "Master+"}
        
        df_human.dropna(subset=['cue', 'age', 'gender', 'education', 'country', 'R1'], inplace=True)
        df_human['age_bin'] = df_human['age'].apply(age_bin)
        df_human['gender_bin'] = df_human['gender'].apply(norm_gender)
        df_human['education_bin'] = df_human['education'].map(edu_map).fillna('Other')
        df_human['country_bin'] = df_human['country'].apply(country_binner)
        
        human_melted = pd.melt(df_human, id_vars=['cue', 'age_bin', 'gender_bin', 'education_bin', 'country_bin'], value_vars=['R1', 'R2', 'R3'], value_name='response').dropna()
        human_melted['response'] = human_melted['response'].str.lower()
        
        steered_records = []
        with open(steered_data_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                for resp in record.get('responses', []):
                    profile_parts = re.match(r"profile_age_.+?_gender_.+?_native_.+?_country_(?P<country_bin>.+?)_edu_(?P<education_bin>.+)", resp['profile_id'])
                    if profile_parts:
                        for word in resp.get('set', []):
                            steered_records.append({'cue': record['cue'], 'response': word.lower()})
        df_steered_raw = pd.DataFrame(steered_records)
        steered_agg = {cue: Counter(g['response']) for cue, g in df_steered_raw.groupby('cue')}
        nonsteered_agg = {json.loads(line)['cue']: Counter(w.lower() for r in json.loads(line)['responses'] for w in r['set']) for line in open(nonsteered_data_path, 'r')}
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: A required data file is missing: {e.filename}. Cannot run evaluation.")
        return

    print("🔄 Calculating alignment scores by subgroup...")
    all_results = []
    axes_to_evaluate = ['country_bin', 'age_bin', 'gender_bin', 'education_bin']
    for axis in tqdm(axes_to_evaluate, desc="Processing Axes"):
        for subgroup in human_melted[axis].unique():
            if str(subgroup).lower() == 'other': continue
            human_sub_df = human_melted[human_melted[axis] == subgroup]
            human_norms = {c: Counter(g['response']) for c, g in human_sub_df.groupby('cue')}
            for cue, h_ctr in human_norms.items():
                s_ctr = steered_agg.get(cue, Counter())
                ns_ctr = nonsteered_agg.get(cue, Counter())
                if s_ctr and ns_ctr:
                    all_results.append({'axis': axis.replace('_bin', '').title(), 'subgroup': subgroup, 'cue': cue, 'Steered': _wjacc(h_ctr, s_ctr), 'Non-Steered': _wjacc(h_ctr, ns_ctr)})
    
    results_df = pd.DataFrame(all_results)
    
    print("🔄 Generating new evaluation plot...")
    plot_df = results_df.melt(id_vars=['axis', 'subgroup'], value_vars=['Steered', 'Non-Steered'], var_name='Model Type', value_name='Alignment')
    
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": '--', "grid.color": '0.9'})
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    axis_titles = ['Country', 'Age', 'Gender', 'Education']
    palette = {'Steered': '#005f99', 'Non-Steered': '#c20000'}

    for i, axis_name in enumerate(axis_titles):
        ax = axes[i]
        axis_data = plot_df[plot_df['axis'] == axis_name]
        
        if axis_data.empty:
            ax.set_title(f"{axis_name}\n(No Data)")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        order = sorted(axis_data['subgroup'].unique())
        if axis_name == 'Education':
            edu_order = ["<HS", "HS", "Some College", "Bachelor", "Master+"]
            order = [e for e in edu_order if e in axis_data['subgroup'].unique()]
        elif axis_name == 'Age':
            age_order = ["<25", "25-34", "35-44", "45-59", "60+"]
            order = [a for a in age_order if a in axis_data['subgroup'].unique()]
        elif axis_name == 'Country':
            country_order = ["United States", "UK & Ireland", "Canada", "Europe"]
            order = [c for c in country_order if c in axis_data['subgroup'].unique()]

        # Plot the individual data points (the "rain")
        sns.stripplot(data=axis_data, x='subgroup', y='Alignment', hue='Model Type', ax=ax, order=order,
                      jitter=0.2, dodge=True, alpha=0.1, palette=palette, legend=False)

        # Plot the mean markers
        # <<< FIX: Removed the invalid 'legend=False' argument >>>
        sns.pointplot(data=axis_data, x='subgroup', y='Alignment', hue='Model Type', ax=ax, order=order,
                      dodge=0.4, join=False, errwidth=0, markers="s", scale=1.0, palette=palette)

        ax.set_title(axis_name)
        ax.set_xlabel('')
        ax.set_ylabel('Alignment (W. Jaccard)' if i == 0 else '')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        # Manually create a clean legend for the first plot
        if i == 0:
            from matplotlib.lines import Line2D
            custom_handles = [Line2D([0], [0], marker='s', color='w', label='Steered Mean', markerfacecolor=palette['Steered'], markersize=10),
                              Line2D([0], [0], marker='s', color='w', label='Non-Steered Mean', markerfacecolor=palette['Non-Steered'], markersize=10)]
            ax.legend(handles=custom_handles, title='Model Type')
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    fig.suptitle('Subgroup Alignment: Steered vs. Non-Steered Language Models', fontsize=16, y=1.02)
    plt.ylim(bottom=0, top=max(0.3, plot_df['Alignment'].max() * 1.1))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(settings.PLOT_YOKED_STEERING, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Yoked steering evaluation complete. Plot saved to '{settings.PLOT_YOKED_STEERING.name}'")

#!/usr/bin/env python3
import warnings, os
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

df = pd.read_csv("sweep_scores.csv", engine="python", on_bad_lines="skip")

required = {"jaccard","recall10","ent_gap","prompt","temperature","calls_per_cue"}
missing  = required - set(df.columns)
if missing:
    raise SystemExit(f"❌ Missing columns: {missing}")

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")

def caption(ax, txt):
    ax.text(0.01, -0.18, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="dimgray", wrap=True)

# ---------- scatter ----------
ax = sns.scatterplot(data=df, x="ent_gap", y="jaccard",
                     hue="prompt", style="calls_per_cue", s=70)
ax.set(xlabel="Entropy gap  (lower = better)",
       ylabel="Weighted Jaccard  (higher = better)",
       title="Human–model alignment trade‑off")
caption(ax, "Shape encodes batch size (calls_per_cue).")
plt.savefig("plots/scatter_tradeoff.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- bar ----------
ax = sns.barplot(data=df, x="temperature", y="recall10",
                 hue="prompt", errorbar="sd")
ax.set(ylabel="Recall@10", title="Recall@10 by temperature & prompt")
caption(ax, "Proportion of cues where at least one model word "
            "appears in humans’ top‑10 responses.")
plt.savefig("plots/bar_recall.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- heatmap ----------
pivot = df.pivot_table(index="calls_per_cue", columns="temperature",
                       values="jaccard", aggfunc="mean")
ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
ax.set_title("Mean Jaccard across calls_per_cue × temperature")
caption(ax, "Higher value indicates closer content overlap.")
plt.savefig("plots/heatmap_jaccard.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Plots saved to ./plots/")

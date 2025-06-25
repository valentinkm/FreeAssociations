#!/usr/bin/env python3
"""
evaluate_and_plot.py  –  prompt-only evaluation (scatter plot)
"""

import sys, csv, json, math, hashlib, pickle, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# ── paths ───────────────────────────────────────────────────────────────
SWOW_PATH = ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"
RUN_DIR   = ROOT / "runs"
OUT_CSV   = ROOT / "sweep_scores.csv"
PLOT_DIR  = ROOT / "plots"
CACHE     = ROOT / "eval" / "human_all.pkl"

# ── load / build human counters (all participants) ───────────────────────
if CACHE.exists():
    HUMAN = pickle.loads(CACHE.read_bytes())
else:
    df = pd.read_csv(SWOW_PATH, usecols=["cue", "R1", "R2", "R3"])
    HUMAN = {
        cue.lower(): collections.Counter(
            w for col in ("R1", "R2", "R3") for w in grp[col].dropna()
        )
        for cue, grp in df.groupby("cue")
    }
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(HUMAN))

# ── metric helpers ───────────────────────────────────────────────────────
def wjacc(h_ctr, m_words):
    m_ctr = collections.Counter(m_words)
    inter = sum(min(h_ctr[w], m_ctr[w]) for w in h_ctr | m_ctr)
    union = sum(max(h_ctr[w], m_ctr[w]) for w in h_ctr | m_ctr)
    return inter / union if union else 0.0

def entropy(cnt):
    tot = sum(cnt.values())
    return -sum((c / tot) * math.log2(c / tot) for c in cnt.values()) if tot else 0.0

hash8 = lambda cfg: hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

# ── score new runs ───────────────────────────────────────────────────────
seen = set(pd.read_csv(OUT_CSV)["hash"].unique()) if OUT_CSV.exists() else set()
rows = []

for f in tqdm(sorted(RUN_DIR.glob("*.jsonl")), desc="runs"):
    with open(f) as fh:
        cfg = None
        j_sum = rec_sum = ent_gap_sum = cue_n = 0
        for line in fh:
            rec  = json.loads(line)
            cfg  = rec["cfg"]
            cue  = rec["cue"].lower()

            if cue not in HUMAN:
                continue
            hctr     = HUMAN[cue]
            m_words  = [w.lower() for triple in rec["sets"] for w in triple]

            j_sum       += wjacc(hctr, m_words)
            top10        = {w for w, _ in hctr.most_common(10)}
            rec_sum     += any(w in top10 for w in m_words)
            ent_gap_sum += abs(entropy(hctr) - entropy(collections.Counter(m_words)))
            cue_n       += 1

        if cfg and cue_n:
            h = hash8(cfg)
            if h in seen:
                continue
            rows.append({
                "hash":     h,
                "prompt":   cfg["prompt"],
                "jaccard":  round(j_sum       / cue_n, 4),
                "recall10": round(rec_sum     / cue_n, 4),
                "ent_gap":  round(ent_gap_sum / cue_n, 4),
            })
            seen.add(h)

if rows:
    OUT_CSV.parent.mkdir(exist_ok=True)
    mode = "a" if OUT_CSV.exists() else "w"
    with open(OUT_CSV, mode, newline="") as fw:
        wr = csv.DictWriter(fw, fieldnames=rows[0].keys())
        if mode == "w":
            wr.writeheader()
        wr.writerows(rows)
    print(f"✅  Added {len(rows)} new rows → {OUT_CSV}")
else:
    print("ℹ️  No new runs to append.")

# ── load full data and plot ──────────────────────────────────────────────
df = pd.read_csv(OUT_CSV)

needed = {"jaccard", "recall10", "ent_gap", "prompt"}
if missing := needed - set(df.columns):
    raise SystemExit(f"❌ Missing columns: {missing}")

PLOT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

plt.figure()
ax = sns.scatterplot(
    data=df,
    x="ent_gap", y="jaccard",
    hue="prompt", s=100
)
ax.set(
    xlabel="Entropy gap (lower = better)",
    ylabel="Weighted Jaccard (higher = better)",
    title="Free-association alignment • prompt variants"
)
ax.text(0.01, -0.15,
        "Each point = one run (5 cues × n sets).",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="dimgray")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.savefig(PLOT_DIR / "scatter_tradeoff.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"📊  Saved scatter_tradeoff.png in {PLOT_DIR}/")

# ── ranking table --------------------------------------------------------
tbl = (
    df.groupby("prompt")
      .agg(jaccard_mean=("jaccard", "mean"),
           jaccard_sd=("jaccard", "std"),
           recall_mean=("recall10", "mean"),
           ent_gap_mean=("ent_gap", "mean"))
      .sort_values("jaccard_mean", ascending=False)
      .round(3)
)
print("\n=== Prompt ranking (mean ± SD) ===")
print(tbl.to_string())

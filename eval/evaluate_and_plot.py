#!/usr/bin/env python3
"""
evaluate_and_plot.py – prompt-sweep scoring & plots
"""
import sys, csv, json, math, hashlib, pickle, collections, os
from pathlib import Path

# ── make `src` importable ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# now we can import settings if we want the official prompt list
try:
    from src.settings import SEARCH_SPACE        # optional, but handy
except ModuleNotFoundError:
    SEARCH_SPACE = {}

import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# ── PATHS ────────────────────────────────────────────────────────────────
SWOW_PATH = ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"
CACHE     = ROOT / "eval" / "human_cache.pkl"
RUN_DIR   = ROOT / "runs"
OUT_CSV   = ROOT / "sweep_scores.csv"
PLOT_DIR  = ROOT / "plots"

# ── HUMAN DISTRIBUTION (LOAD / BUILD) ────────────────────────────────────
def build_cue_counts(csv_path: Path):
    df = pd.read_csv(csv_path, usecols=["cue", "R1", "R2", "R3"])
    d = {}
    for cue, grp in df.groupby("cue"):
        ctr = collections.Counter()
        for col in ("R1", "R2", "R3"):
            ctr.update(grp[col].dropna().str.lower().str.strip())
        d[cue.lower()] = ctr
    return d

if CACHE.exists():
    cue_counts = pickle.loads(CACHE.read_bytes())
else:
    print("🥣  Building human counters …")
    cue_counts = build_cue_counts(SWOW_PATH)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(cue_counts))

# ── METRIC HELPERS ───────────────────────────────────────────────────────
def wjacc(h_ctr, m_words):
    m_ctr = collections.Counter(m_words)
    inter = sum(min(h_ctr[w], m_ctr[w]) for w in h_ctr | m_ctr)
    union = sum(max(h_ctr[w], m_ctr[w]) for w in h_ctr | m_ctr)
    return inter / union if union else 0.0

def entropy(cnt):
    tot = sum(cnt.values())
    return -sum((c / tot) * math.log2(c / tot) for c in cnt.values()) if tot else 0.0

hash8 = lambda cfg: hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

# ── SCORE NEW RUNS ───────────────────────────────────────────────────────
seen_hashes = set()
if OUT_CSV.exists():
    seen_hashes.update(pd.read_csv(OUT_CSV)["hash"].unique())

new_rows = []
for path in tqdm(sorted(RUN_DIR.glob("*.jsonl")), desc="runs"):
    with open(path) as fh:
        cfg = None
        j_sum = rec_sum = ent_gap_sum = cue_n = 0
        for line in fh:
            rec = json.loads(line)
            cfg = rec["cfg"]
            cue = rec["cue"].lower()
            if cue not in cue_counts:
                continue
            hctr     = cue_counts[cue]
            m_words  = [w.lower() for triple in rec["sets"] for w in triple]

            j_sum       += wjacc(hctr, m_words)
            top10        = {w for w, _ in hctr.most_common(10)}
            rec_sum     += any(w in top10 for w in m_words)
            ent_gap_sum += abs(entropy(hctr) - entropy(collections.Counter(m_words)))
            cue_n       += 1

        if cfg and cue_n:
            h = hash8(cfg)
            if h in seen_hashes:
                continue
            new_rows.append({
                **cfg,
                "hash":      h,
                "jaccard":   round(j_sum       / cue_n, 4),
                "recall10":  round(rec_sum     / cue_n, 4),
                "ent_gap":   round(ent_gap_sum / cue_n, 4),
            })
            seen_hashes.add(h)

if new_rows:
    OUT_CSV.parent.mkdir(exist_ok=True)
    mode = "a" if OUT_CSV.exists() else "w"
    with open(OUT_CSV, mode, newline="") as fw:
        wr = csv.DictWriter(fw, fieldnames=new_rows[0].keys())
        if mode == "w":
            wr.writeheader()
        wr.writerows(new_rows)
    print(f"✅  Added {len(new_rows)} new rows → {OUT_CSV}")
else:
    print("ℹ️  No new runs to append.")

# ── PLOTTING ─────────────────────────────────────────────────────────────
df = pd.read_csv(OUT_CSV, engine="python", on_bad_lines="skip")

required = {"jaccard", "recall10", "ent_gap", "prompt"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"❌ Missing columns: {missing}")

PLOT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

def caption(ax, txt):
    ax.text(0.01, -0.15, txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8, color="dimgray", wrap=True)

# Scatter
plt.figure()
ax = sns.scatterplot(data=df, x="ent_gap", y="jaccard",
                     hue="prompt", s=80)
ax.set(xlabel="Entropy gap (lower = better)",
       ylabel="Weighted Jaccard (higher = better)",
       title="Human–Model Alignment per Prompt")
caption(ax, "Each point = one prompt variant (5 cues, single call).")
plt.savefig(PLOT_DIR / "scatter_tradeoff.png", dpi=300, bbox_inches="tight")
plt.close()

# Bar – Recall@10
plt.figure()
order = SEARCH_SPACE.get("prompt") or sorted(df["prompt"].unique())
ax = sns.barplot(data=df, x="prompt", y="recall10",
                 order=order, errorbar="sd")
ax.set(xlabel="Prompt variant", ylabel="Recall@10",
       title="Recall@10 by Prompt")
plt.xticks(rotation=20, ha="right")
caption(ax, "Proportion of cues where ≥1 model word is in humans' top-10.")
plt.savefig(PLOT_DIR / "bar_recall.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"📊  Plots saved in {PLOT_DIR}/")

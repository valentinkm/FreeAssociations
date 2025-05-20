#!/usr/bin/env python3
"""
evaluate_and_plot.py – prompt-sweep scoring & plots
updated for age-bucket evaluation
"""
import sys, csv, json, math, hashlib, pickle, collections, os
from pathlib import Path
from functools import lru_cache

# ── make `src` importable ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.settings import SEARCH_SPACE    # handy for prompt order
except ModuleNotFoundError:
    SEARCH_SPACE = {}

import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# ── PATHS ────────────────────────────────────────────────────────────────
SWOW_PATH = ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"
CACHE     = ROOT / "eval" / "human_cache.pkl"       # stores dict[bucket] = counters
RUN_DIR   = ROOT / "runs"
OUT_CSV   = ROOT / "sweep_scores.csv"
PLOT_DIR  = ROOT / "plots"

# ── AGE-BUCKET HELPER ────────────────────────────────────────────────────
def age_bucket(age):
    if pd.isna(age):                    return "age_unknown"
    age = int(age)
    if age < 25:                        return "age_<25"
    if 25 <= age < 35:                  return "age_25-34"
    if 35 <= age < 50:                  return "age_35-49"
    if 50 <= age < 65:                  return "age_50-64"
    return "age_65+"

# ── HUMAN DISTRIBUTION (LOAD / BUILD) ────────────────────────────────────
def _build_cue_counts(df):
    """Given a SWOW dataframe (already filtered), return cue→Counter."""
    d = {}
    for cue, grp in df.groupby("cue"):
        ctr = collections.Counter()
        for col in ("R1", "R2", "R3"):
            ctr.update(grp[col].dropna().str.lower().str.strip())
        d[cue.lower()] = ctr
    return d

def _load_or_build_human_counters():
    """
    Returns dict: bucket_name → { cue → Counter }
    Builds missing buckets on demand & writes back to CACHE.
    """
    if CACHE.exists():
        try:
            store = pickle.loads(CACHE.read_bytes())
            if isinstance(store, dict):
                return store
        except Exception:
            pass
    return {}

# in-memory cache (will grow as buckets get computed)
HUMAN_COUNTS = _load_or_build_human_counters()

def get_counters_for_bucket(bucket: str):
    """Return cue→Counter for the given age bucket (or 'all')."""
    if bucket in HUMAN_COUNTS:
        return HUMAN_COUNTS[bucket]

    df = pd.read_csv(
        SWOW_PATH,
        usecols=["cue", "R1", "R2", "R3", "age"]
    )
    if bucket != "all":
        df["bucket"] = df["age"].apply(age_bucket)
        df = df[df["bucket"] == bucket]

    HUMAN_COUNTS[bucket] = _build_cue_counts(df)
    # persist to disk
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(HUMAN_COUNTS))
    return HUMAN_COUNTS[bucket]

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
            rec  = json.loads(line)
            cfg  = rec["cfg"]
            demo = cfg.get("demographic", "all")    # age bucket
            cue  = rec["cue"].lower()

            hctr_dict = get_counters_for_bucket(demo)
            if cue not in hctr_dict:
                continue
            hctr     = hctr_dict[cue]
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

required = {"jaccard", "recall10", "ent_gap", "prompt", "demographic"}
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

# Scatter – coloured by demographic, marker style = prompt
plt.figure()
ax = sns.scatterplot(
    data=df,
    x="ent_gap", y="jaccard",
    hue="demographic", style="prompt", s=80
)
ax.set(
    xlabel="Entropy gap (lower = better)",
    ylabel="Weighted Jaccard (higher = better)",
    title="Human–Model Alignment • Age buckets"
)
caption(ax, "Each point = one prompt × demographic run (5 cues, single call).")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.savefig(PLOT_DIR / "scatter_tradeoff.png", dpi=300, bbox_inches="tight")
plt.close()

# Bar – Recall@10 faceted by demographic
order = SEARCH_SPACE.get("prompt") or sorted(df["prompt"].unique())
g = sns.catplot(
    data=df, kind="bar",
    x="prompt", y="recall10",
    col="demographic", col_wrap=3,
    order=order, errorbar="sd", sharey=False
)
g.set_titles("{col_name}")
g.set_axis_labels("Prompt variant", "Recall@10")
for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    caption(ax, "≥1 model word in humans' top-10 (higher = better).")
plt.savefig(PLOT_DIR / "bar_recall.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"📊  Plots saved in {PLOT_DIR}/")

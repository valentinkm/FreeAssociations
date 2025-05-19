#!/usr/bin/env python3
"""
evaluate_runs.py  –  weighted‑Jaccard, Recall@10, Entropy gap
uses calls_per_cue (batch size) instead of the old sets_per_call knob
"""
import json, pickle, collections, csv, hashlib, math
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

SWOW_PATH = "Small World of Words/SWOW-EN.R100.20180827.csv"
CACHE     = Path("eval/human_cache.pkl")
RUN_DIR   = Path("runs")
OUT_CSV   = Path("sweep_scores.csv")

# ---------- load / cache human distributions ----------
def build_cue_counts(csv_path):
    df = pd.read_csv(csv_path, usecols=["cue", "R1", "R2", "R3"])
    d = {}
    for cue, grp in df.groupby("cue"):
        ctr = collections.Counter()
        for col in ["R1", "R2", "R3"]:
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

# ---------- metric helpers ----------
def wjacc(h_ctr, m_words):
    m_ctr = collections.Counter(m_words)
    inter = sum(min(h_ctr[w], m_ctr[w]) for w in h_ctr | m_ctr)
    union = sum(max(h_ctr[w], m_ctr[w]) for w in h_ctr | m_ctr)
    return inter / union if union else 0.0

def entropy(cnt):
    tot = sum(cnt.values())
    return -sum((c/tot) * math.log2(c/tot) for c in cnt.values()) if tot else 0

hash8 = lambda cfg: hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

# ---------- iterate runs ----------
rows = []
for path in tqdm(sorted(RUN_DIR.glob("*.jsonl")), desc="runs"):
    with open(path) as fh:
        cfg = None
        j_sum = rec_sum = e_gap_sum = cues = 0
        for line in fh:
            rec = json.loads(line)
            cfg = rec["cfg"]
            cue = rec["cue"].lower()
            if cue not in cue_counts:
                continue
            hctr = cue_counts[cue]
            m_words = [w.lower() for triple in rec["sets"] for w in triple]

            j_sum   += wjacc(hctr, m_words)
            top10    = {w for w, _ in hctr.most_common(10)}
            rec_sum += any(w in top10 for w in m_words)
            e_gap_sum += abs(entropy(hctr) - entropy(collections.Counter(m_words)))
            cues += 1
        if cfg and cues:
            rows.append({**cfg,
                         "hash": hash8(cfg),
                         "jaccard":  round(j_sum   / cues, 4),
                         "recall10": round(rec_sum / cues, 4),
                         "ent_gap":  round(e_gap_sum / cues, 4)})

if not rows:
    print("No new runs.")
    raise SystemExit

OUT_CSV.parent.mkdir(exist_ok=True)
mode = "a" if OUT_CSV.exists() else "w"
with open(OUT_CSV, mode, newline="") as fw:
    w = csv.DictWriter(fw, fieldnames=rows[0].keys())
    if mode == "w": w.writeheader()
    w.writerows(rows)
print(f"🏁  Added {len(rows)} runs → {OUT_CSV}")

# ─────────── GLOBAL CFG ───────────
CFG = {
    "model":             "gpt-4.1-nano",
    "temperature":       1.1,            # locked for prompt sweep
    "top_p":             1.0,
    "frequency_penalty": 0.0,
    "presence_penalty":  0.0,

    "calls_per_cue":     1,              # single-call only
    "prompt":            "descriptive_context",  # overridden by sweep
    "demographic":       "all",          # ← NEW default bucket
    "num_cues":          5,              # how many cues to sample
}

# ─────────── PROMPT-ENGINEERING SWEEP ───────────
SEARCH_SPACE = {
    "prompt": [
        "descriptive_context",
        "instructional_format",
        "creativity_boost",
        "memory_only",
        "category_anchor",
        "chain_of_thought",
    ],
    "demographic": [
        "all",
        "age_<25",
        "age_25-34",
        "age_35-49",
        "age_50-64",
        "age_65+",
        "gender_f",
        "gender_m",
        "gender_other",
    ],
}

# ─────────── CONSTANTS ───────────
TOTAL_SETS = 10
MAX_TOKENS = 180
DATA_PATH  = "Small World of Words/SWOW-EN.R100.20180827.csv"

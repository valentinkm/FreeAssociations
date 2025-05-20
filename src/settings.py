# src/settings.py

# ─────────── GLOBAL CFG ───────────
CFG = {
    "model":             "gpt-4.1-nano",
    "temperature":       1.1,            # locked for prompt sweep
    "top_p":             1.0,
    "frequency_penalty": 0.0,
    "presence_penalty":  0.0,

    "calls_per_cue":     1,              # single-call only
    "prompt":            "descriptive_context",  # overridden by sweep
    "num_cues":          5,              # how many cues to sample
}

# ─────────── PROMPT-ENGINEERING SWEEP ───────────
SEARCH_SPACE = {
    "prompt": [
        "descriptive_context",
        "instructional",        # instead of instructional_format
        "creativity_boost",
        "memory",               # instead of memory_only
        "categroy_anchor",      # spelling as-is, or fix both sides
        "chain_of_thought",
    ]
}

# ─────────── CONSTANTS ───────────
TOTAL_SETS = 10
MAX_TOKENS = 180
DATA_PATH  = "Small World of Words/SWOW-EN.R100.20180827.csv"

"""
settings.py
───────────
Global configuration (CFG) plus the SEARCH_SPACE the sweep iterates over.
"""

# ─────────── GLOBAL CFG ───────────
CFG = {
    "model":             "gpt-4.1-nano",
    "temperature":       1.1,
    "top_p":             1.0,
    "frequency_penalty": 0.0, 
    "presence_penalty":  0.0,

    "calls_per_cue":     1, # number of calls to make per cue
    "prompt":            "descriptive_context", # prompt template to use
    "demographic":       "all", # demographic profile to use
    "num_cues":          5, # number of cues to test
    "sets_total":        1, # default triplet responses per cue
}

# ─────────── PROMPT-ENGINEERING SWEEP ───────────
SEARCH_SPACE = {
    "prompt": [
        # plain role
        "default_question",
        "default_imperative",
        "intuition_question",
        "intuition_imperative",
        "experiential_question",
        "experiential_imperative",
        "participant_default_question",
        "participant_default_imperative",
        "participant_intuition_question",
        "participant_intuition_imperative",
        "participant_experiential_question",
        "participant_experiential_imperative",
    ],

    # demographic axis
    "demographic": ["all"],
}

# ─────────── CONSTANTS ───────────
TOTAL_SETS = 10
MAX_TOKENS = 180
DATA_PATH  = "Small World of Words/SWOW-EN.R100.20180827.csv"

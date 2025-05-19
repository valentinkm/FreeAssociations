# src/settings.py
# -------------------------------------------------
#  ❖ SINGLE‑RUN CFG
# -------------------------------------------------
CFG = {
    "model":             "gpt-4.1-nano",
    "temperature":       1.3,
    "top_p":             1.0,
    "frequency_penalty": 0.0,
    "presence_penalty":  0.0,

    # prompt key used only when calls_per_cue > 1
    "prompt": "human_json",      # or "chatml_func"

    "calls_per_cue": 1,          # 1 → 10 single prompts ; >1 → batch size
    "num_cues": 5,
}

# -------------------------------------------------
#  ❖ SWEEP search space
# -------------------------------------------------
SEARCH_SPACE = {
    "model":             ["gpt-4.1-nano"],
    "temperature":       [0.7, 1.1, 1.5],
    "top_p":             [1.0, 0.95],
    "frequency_penalty": [0.0, 0.3],
    "presence_penalty":  [0.0, 0.6],
    "calls_per_cue":     [1, 5, 10],     # batch size; total sets always 10
    "prompt":            ["human_json", "chatml_func"],
}

# -------------------------------------------------
#  ❖ CONSTANTS
# -------------------------------------------------
TOTAL_SETS = 10
MAX_TOKENS = 180
N_CUES      = 500
DATA_PATH   = "Small World of Words/SWOW-EN.R100.20180827.csv"

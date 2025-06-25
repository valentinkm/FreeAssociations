"""
settings.py
───────────
Central configuration for the project.

This file manages file paths, model parameters, and experiment settings.
It ensures that all other modules can access consistent configuration
and that data/output directories are handled correctly.
"""
from pathlib import Path

# --- Core Project Paths ---
# The root of the project is the directory containing 'main.py' and 'src/'.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Input Data Paths ---
DATA_DIR = PROJECT_ROOT / "data"
SWOW_DIR = DATA_DIR / "SWOW"
SPP_DIR = DATA_DIR / "SPP" # Renamed from 'generalize_spp'
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Specific data files
SPP_DATA_PATH = SPP_DIR / "spp_naming_data_raw.xlsx"
SWOW_DATA_PATH = SWOW_DIR / "SWOW-EN.R100.20180827.csv"
SWOW_COMPLETE_DATA_PATH = SWOW_DIR / "SWOW-EN.complete.20180827.csv"

# --- Output Paths ---
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
LEXICONS_DIR = OUTPUTS_DIR / "lexicons"

# Specific output files
LEXICON_PATH = LEXICONS_DIR / "llm_association_lexicon.jsonl"
PLOT_HOLISTIC_PATH = PLOTS_DIR / "spp_holistic_similarity_analysis.png"
PLOT_HUMAN_BASELINE_PATH = PLOTS_DIR / "spp_human_baseline_analysis.png"
YOKED_DIR = RUNS_DIR / "yoked_generation"
PROFILES_PATH = OUTPUTS_DIR / "demographic_profiles.csv"


# --- LLM & Analysis Configuration ---
CFG = {
    # Model parameters
    "model": "gpt-4-turbo", # Using a more modern default
    "temperature": 1.1,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 180,

    # Experiment parameters
    "prompt": "participant_default_question", # Default prompt template
    "demographic": "all", # Default demographic profile
    "num_cues": 10, # Number of cues to test in sweeps
    "sets_total": 5, # Default triples per cue
}

# --- Prompt Engineering Sweep Configuration ---
SEARCH_SPACE = {
    "prompt": [
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
    "demographic": ["all"],
}

# --- SPP Analysis Constants ---
SPP_CONSTANTS = {
    "NUM_PAIRS_TO_PROBE": 50,
    "RANDOM_SEED": 42,
    "RELATED_COND": 1,
    "UNRELATED_COND": 2,
    "NUM_SETS_PER_WORD": 25, # For lexicon generation
}

# --- Initialization Function (This is what main.py needs) ---
def initialize_project_paths():
    """
    Creates all necessary output directories if they don't exist.
    This function should be called once at the start of any script.
    """
    print("Initializing project directories...")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    LEXICONS_DIR.mkdir(exist_ok=True)
    YOKED_DIR.mkdir(exist_ok=True)
    print("✅ Directories initialized.")


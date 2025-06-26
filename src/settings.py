"""
settings.py
───────────
Central configuration for the project.
"""
from pathlib import Path

# --- Core Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
SWOW_DIR = DATA_DIR / "SWOW"
SPP_DIR = DATA_DIR / "SPP"
TTT_DIR = DATA_DIR / "3TT"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
LEXICONS_DIR = OUTPUTS_DIR / "lexicons"
YOKED_DIR = RUNS_DIR / "yoked_generation"
PROFILES_PATH = OUTPUTS_DIR / "demographic_profiles.csv"

# --- Yoked Generation Paths ---
YOKED_STEERED_PATH = YOKED_DIR / "yoked_steered_responses.jsonl"
YOKED_NONSTEERED_PATH = YOKED_DIR / "yoked_nonsteered_responses.jsonl"

# --- Specific Data File Paths ---
SPP_DATA_PATH = SPP_DIR / "spp_naming_data_raw.xlsx"
SWOW_DATA_PATH = SWOW_DIR / "SWOW-EN.R100.20180827.csv"
TTT_RESULTS_PATH = TTT_DIR / "Results Summary.csv"

# --- Evaluation Output Paths ---
EVAL_CACHE_DIR = OUTPUTS_DIR / ".cache"
HUMAN_NORMS_CACHE = EVAL_CACHE_DIR / "human_all.pkl"
SWEEP_SCORES_CSV_PATH = RESULTS_DIR / "prompt_sweep_scores.csv"
PLOT_SWEEP_SCATTER = PLOTS_DIR / "scatter_tradeoff.png"
PLOT_YOKED_STEERING = PLOTS_DIR / "yoked_steering_validation.png"
PLOT_MODEL_COMPARISON = PLOTS_DIR / "model_comparison_generalization.png"

# --- LLM & Analysis Configuration ---
CFG = {
    "model": "gpt-4o",
    "temperature": 1.1,
    "top_p": 1.0,
}

# --- Generalization Analysis Defaults ---
GENERALIZE_DEFAULTS = {
    "prompt": "participant_default_question",
    "random_seed": 42
}

# --- Prompt Engineering Sweep Configuration ---
SEARCH_SPACE = {
    "prompt": [ "default_question", "default_imperative", "intuition_question", "intuition_imperative", "experiential_question", "experiential_imperative", "participant_default_question", "participant_default_imperative", "participant_intuition_question", "participant_intuition_imperative", "participant_experiential_question", "participant_experiential_imperative", ],
}

# --- Path and Directory Management ---
def get_lexicon_path(model_name: str, nsets: int) -> Path:
    """Returns a unique lexicon path based on the model and number of sets."""
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    return LEXICONS_DIR / f"lexicon_{safe_model_name}_nsets_{nsets}.jsonl"

def initialize_project_paths():
    """Creates all necessary output directories if they don't exist."""
    print("Initializing project directories...")
    for path in [OUTPUTS_DIR, RUNS_DIR, PLOTS_DIR, LEXICONS_DIR, RESULTS_DIR, YOKED_DIR, EVAL_CACHE_DIR]:
        path.mkdir(exist_ok=True)
    print("✅ Directories initialized.")

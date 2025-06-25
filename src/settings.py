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
# ... (other paths remain the same) ...
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
LEXICONS_DIR = OUTPUTS_DIR / "lexicons"

# --- Evaluation Output Paths ---
EVAL_CACHE_DIR = OUTPUTS_DIR / ".cache" # Moved cache out of src
HUMAN_NORMS_CACHE = EVAL_CACHE_DIR / "human_all.pkl"
SWEEP_SCORES_CSV_PATH = RESULTS_DIR / "prompt_sweep_scores.csv"
PLOT_SWEEP_SCATTER = PLOTS_DIR / "scatter_tradeoff.png"
PLOT_YOKED_STEERING = PLOTS_DIR / "yoked_steering_validation.png"
# <<< NEW: Path for the model comparison plot >>>
PLOT_MODEL_COMPARISON = PLOTS_DIR / "model_comparison_generalization.png"


# --- LLM & Analysis Configuration ---
CFG = {
    # Default model is now gpt-4o
    "model": "gpt-4o",
    "temperature": 1.1,
    "top_p": 1.0,
    # ... (other configs remain the same) ...
}
# ... (other settings remain the same) ...

# --- Path and Directory Management ---
def get_lexicon_path(model_name: str, nsets: int) -> Path:
    """
    Returns a unique lexicon path based on the model and number of sets.
    Example: lexicon_gpt-4o_nsets_25.jsonl
    """
    # <<< CHANGE: Filename now includes the model name >>>
    safe_model_name = model_name.replace("/", "_") # Handle model names with slashes
    return LEXICONS_DIR / f"lexicon_{safe_model_name}_nsets_{nsets}.jsonl"

def initialize_project_paths():
    """Creates all necessary output directories if they don't exist."""
    print("Initializing project directories...")
    for path in [OUTPUTS_DIR, RUNS_DIR, PLOTS_DIR, LEXICONS_DIR, RESULTS_DIR, YOKED_DIR, EVAL_CACHE_DIR]:
        path.mkdir(exist_ok=True)
    print("✅ Directories initialized.")

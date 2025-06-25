"""
experiment_runners.py
─────────────────────
High-level functions for running complex experiments like prompt sweeps
and yoked generation.
"""
import itertools
import hashlib
import json
import random
import datetime as dt
import pandas as pd

# Import project-wide settings and necessary functions
from . import settings
from .llm_generation import call_model
from .prompt_loader import TEMPLATES
from .profile_utils import (
    age_bin, norm_gender, native_bin, edu_bin, slugify
)

# --- Sweep Experiment Runner ---

def cfg_hash(cfg: dict) -> str:
    """Deterministic 8-char hash for the current cfg."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

def all_combinations(space: dict):
    """Yields all combinations of parameters from a search space dictionary."""
    keys, vals = zip(*space.items())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def run_sweep(nsets: int, no_demo: bool):
    """
    Launches a prompt sweep experiment based on the SEARCH_SPACE in settings.
    """
    print("--- Launching Prompt Sweep ---")
    
    # Use a local copy of the search space to modify
    space = dict(settings.SEARCH_SPACE)
    
    # Apply CLI modifications
    if no_demo and "demographic" in space:
        space["demographic"] = ["all"]
    
    if nsets and nsets > 0:
        settings.CFG["sets_total"] = nsets
    else:
        # Ensure a default value is present
        settings.CFG.setdefault("sets_total", 1)

    # Load cues for the experiment
    try:
        cues = pd.read_csv(settings.SWOW_DATA_PATH)["cue"].dropna().unique()
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find SWOW data at {settings.SWOW_DATA_PATH}")
        return

    # --- Main Sweep Loop ---
    for overrides in all_combinations(space):
        prompt_key = overrides.get("prompt")
        if prompt_key not in TEMPLATES:
            print(f"⚠️  Prompt template '{prompt_key}' not found – skipping")
            continue

        # Update the global config with the current experiment's parameters
        current_cfg = settings.CFG.copy()
        current_cfg.update(overrides)

        h = cfg_hash(current_cfg)
        out_path = settings.RUNS_DIR / f"grid_{h}.jsonl"
        if out_path.exists():
            print(f"✅ {out_path.name} exists – skipping")
            continue

        print(f"\n▶️  Running config: {overrides}")
        print(f"   Writing data to {out_path}")
        
        sample_cues = random.sample(list(cues), k=current_cfg["num_cues"])

        with open(out_path, "w", encoding="utf-8") as fw:
            for cue in sample_cues:
                sets_acc = call_model(
                    cue=cue,
                    prompt_key=current_cfg["prompt"],
                    num_sets_to_generate=current_cfg["sets_total"]
                )
                if sets_acc:
                    fw.write(json.dumps({
                        "cue":  cue,
                        "sets": sets_acc,
                        "cfg":  current_cfg,
                    }) + "\n")
                else:
                    print(f"⏭️  No sets for cue '{cue}' – not writing")
        
        print(f"✅ Done — wrote {len(sample_cues)} cues for config {h}")

# --- Yoked Generation Runner ---

def run_yoked_generation():
    """
    Runs the yoked (matched-sample) generation experiment.
    This functionality is self-contained and complex, so it's kept together.
    """
    # This function is a placeholder for the logic from `run_yoked_generation.py`.
    # Due to its complexity, it would be refactored and placed here.
    # For now, we'll just print a message.
    print("\n--- Yoked Generation Experiment ---")
    print("NOTE: The yoked generation logic is complex and should be migrated")
    print("from the original script into this function.")
    print("To run it, you would call: python main.py generate --type=yoked")

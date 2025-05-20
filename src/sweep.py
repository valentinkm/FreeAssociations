#!/usr/bin/env python3
"""
sweep.py  – grid / random sweep launcher
---------------------------------------

Reads SEARCH_SPACE from settings.py, mutates CFG in-place, and calls
run_lm.generate_and_save for each unique parameter combo.
"""
import itertools, hashlib, json, datetime as dt
from pathlib import Path

from .settings      import CFG, SEARCH_SPACE
from .run_lm        import generate_and_save
from .prompt_loader import TEMPLATES   # just for sanity-check

RUN_DIR = Path("runs")
RUN_DIR.mkdir(exist_ok=True)

def cfg_hash(cfg: dict) -> str:
    """Deterministic 8-char hash for the current cfg."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

def all_combinations(space: dict):
    keys, vals = zip(*space.items())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    for overrides in all_combinations(SEARCH_SPACE):
        prompt_key   = overrides.get("prompt")
        demographic  = overrides.get("demographic", "all")

        # Skip missing base prompt templates early
        if prompt_key not in TEMPLATES:
            print(f"⚠️  Prompt template '{prompt_key}' not found – skipping")
            continue

        CFG.update(overrides)

        h = cfg_hash(CFG)
        out_path = RUN_DIR / f"grid_{h}.jsonl"

        if out_path.exists():
            print(f"⚠️  {out_path.name} exists – skipping")
            continue

        print(f"▶️  {overrides}")
        generate_and_save(str(out_path))

if __name__ == "__main__":
    main()

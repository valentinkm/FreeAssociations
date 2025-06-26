"""
llm_generation.py
─────────────────
Functions for generating data using the unified model interface.
"""
import time
import json
import textwrap
from pathlib import Path
from tqdm import tqdm

from . import settings
from .prompt_loader import get_prompt, render_prompt
from .model_interface import get_model_associations

def _log_prompt(cue, prompt):
    if settings.CFG.get('verbose', False):
        print(f"\n🟦 PROMPT | cue={cue!r}\n" + textwrap.indent(prompt, "    "))

def _log_reply(txt):
    if settings.CFG.get('verbose', False):
        tag = "🟩" if txt else "🟥"
        preview = "<None>" if not txt else json.dumps(txt)[:200] + "…"
        print(f"{tag} REPLY  | {preview}")

def generate_lexicon_data(vocabulary: set, model_name: str, lexicon_path: Path, nsets: int, prompt_key: str):
    """Generates LLM associations and appends them to a specific lexicon file."""
    print(f"\n--- Generating associations for {len(vocabulary)} words using '{model_name}' ---")
    if not vocabulary:
        print("✅ No new words to generate.")
        return

    base_prompt = get_prompt(prompt_key, "all")
    
    with open(lexicon_path, "a", encoding="utf-8") as f:
        for word in tqdm(sorted(list(vocabulary)), desc="Building Lexicon"):
            try:
                prompt_for_instruct_models = render_prompt(base_prompt, word, n=nsets)
                _log_prompt(word, prompt_for_instruct_models)
                
                response_data = get_model_associations(
                    instruct_prompt=prompt_for_instruct_models,
                    model_name=model_name,
                    cue_word=word,
                    nsets=nsets
                )
                _log_reply(response_data)

                if response_data and "sets" in response_data:
                    record = {"word": word, "association_sets": response_data["sets"]}
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                tqdm.write(f"❌ FAILED on cue '{word}': {e}")

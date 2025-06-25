"""
llm_generation.py
─────────────────
Functions for interacting with the LLM API and generating data.
"""
import time
import json
import re
import textwrap
from json.decoder import JSONDecodeError
from tqdm import tqdm

from openai import OpenAI

# Import config, paths, and supporting functions from other modules
from . import settings
from .prompt_loader import get_prompt, render_prompt

# Initialize the OpenAI client
# Ensure your OPENAI_API_KEY is set in your environment
try:
    client = OpenAI()
except ImportError:
    print("❌ Error: The 'openai' library is required. Please run 'pip install openai'.")
    client = None

# --- Internal Helper Functions ---

def _log_prompt(cue, n_sets, prompt):
    """Logs the formatted prompt to the console."""
    print(f"\n🟦 PROMPT | cue={cue!r} n_sets={n_sets} demographic={settings.CFG['demographic']}\n" + textwrap.indent(prompt, "    "))

def _log_reply(txt):
    """Logs the model's reply to the console."""
    tag = "🟥" if not txt else "🟩"
    preview = "<None>" if not txt else ' '.join(txt.split())[:200] + "…"
    print(f"{tag} REPLY  | {preview}")

def _repair_json(txt: str | None):
    """Attempts to repair a truncated JSON string from the model."""
    if not txt:
        return None
    # A common failure is a missing final bracket. Find the last valid structure.
    m = re.search(r'\]\s*\}', txt)
    if m:
        candidate = txt[:m.end()]
        try:
            json.loads(candidate)
            return candidate
        except JSONDecodeError:
            pass
    return None

def _call_single_set(cue: str, prompt_key: str):
    """Makes a single, robust API call to the LLM for one set of associations."""
    if not client:
        raise ConnectionError("OpenAI client not initialized. Is the library installed and API key set?")

    base_tpl = get_prompt(prompt_key, settings.CFG["demographic"])
    # The prompt to the model always asks for one set of three words.
    user_msg = render_prompt(base_tpl, cue, n=1)

    _log_prompt(cue, n_sets=1, prompt=user_msg)

    rsp = client.chat.completions.create(
        model=settings.CFG["model"],
        temperature=settings.CFG["temperature"],
        top_p=settings.CFG["top_p"],
        frequency_penalty=settings.CFG["frequency_penalty"],
        presence_penalty=settings.CFG["presence_penalty"],
        max_tokens=settings.CFG["max_tokens"],
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    if not rsp.choices:
        raise RuntimeError("Empty choices returned from API.")

    raw = rsp.choices[0].message.content
    _log_reply(raw)

    if not raw:
        raise RuntimeError("Empty payload in model reply.")

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "sets" in data and isinstance(data["sets"], list):
            return data["sets"]
        if isinstance(data, dict): # Fallback for other structures
            keys = sorted(data.keys(), key=lambda k: (len(k), k))
            return [[data[k] for k in keys][:3]]
        raise KeyError("Unrecognized reply structure.")
    except (JSONDecodeError, KeyError, TypeError):
        if repaired := _repair_json(raw):
            return json.loads(repaired)["sets"]
        raise RuntimeError(f"Failed to parse model reply.")

# --- Public-Facing Functions ---

def call_model(cue: str, prompt_key: str, num_sets_to_generate: int) -> list:
    """
    Generates a specified number of association sets for a cue by calling
    the LLM in a loop.
    """
    all_sets = []
    # Set the prompt key in the global CFG for the internal function
    settings.CFG['prompt'] = prompt_key

    for i in range(num_sets_to_generate):
        try:
            single_set = _call_single_set(cue, prompt_key)
            if single_set:
                all_sets.extend(single_set)
            # A small delay to avoid hitting rate limits on rapid calls
            time.sleep(0.1)
        except Exception as e:
            # Log the error but continue trying to get the other sets
            print(f"⚠️  Error on cue '{cue}' (attempt {i+1}/{num_sets_to_generate}): {e} – skipping one attempt")
            continue
            
    return all_sets

def generate_lexicon_data(vocabulary: set):
    """Generates LLM associations for any words not already in the central lexicon."""
    print("\n--- Generating LLM Association Lexicon (if needed) ---")
    if not vocabulary:
        print("✅ No new words to generate. All required data is already in the lexicon.")
        return

    # Ensure the parent directory exists
    settings.LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    num_sets = settings.SPP_CONSTANTS['NUM_SETS_PER_WORD']
    prompt_key = settings.CFG['prompt']
    
    with open(settings.LEXICON_PATH, "a", encoding="utf-8") as f:
        for word in tqdm(sorted(list(vocabulary)), desc="Building Lexicon"):
            response_sets = call_model(
                cue=word,
                prompt_key=prompt_key,
                num_sets_to_generate=num_sets
            )
            if response_sets:
                record = {"word": word, "association_sets": response_sets}
                f.write(json.dumps(record) + "\n")

    print("✅ Lexicon generation complete.")

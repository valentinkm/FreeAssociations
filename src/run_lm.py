#!/usr/bin/env python3
"""
run_lm.py – generator of free-association triples (zero-shot version)

(v2 - Refactored to handle looping internally and accept num_sets_to_generate)
"""
import time
import json
import textwrap
import random
import re
import datetime as dt
from pathlib import Path
from json.decoder import JSONDecodeError
import argparse

import pandas as pd
from openai import OpenAI

from .settings import CFG, MAX_TOKENS, DATA_PATH
from .prompt_loader import get_prompt, render_prompt

client = OpenAI()

# --- Internal helper for making a single API call ---
def _log_prompt(cue, n_sets, prompt):
    print(f"\n🟦 PROMPT | cue={cue!r}  n_sets={n_sets}  "
          f"demographic={CFG['demographic']}\n" + textwrap.indent(prompt, "   "))

def _log_reply(txt):
    tag = "🟥" if not txt else "🟩"
    preview = "<None>" if not txt else ' '.join(txt.split())[:200] + "…"
    print(f"{tag} REPLY  | {preview}")

def _repair_json(txt: str | None):
    if not txt:
        return None
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
    base_tpl = get_prompt(prompt_key, CFG["demographic"])
    # The user-facing prompt should always ask for one set of three words.
    # <<< FIX: Changed keyword argument from n_sets=1 to n=1 >>>
    user_msg = render_prompt(base_tpl, cue, n=1)

    _log_prompt(cue, n_sets=1, prompt=user_msg)

    rsp = client.chat.completions.create(
        model=CFG["model"],
        temperature=CFG["temperature"],
        top_p=CFG["top_p"],
        frequency_penalty=CFG["frequency_penalty"],
        presence_penalty=CFG["presence_penalty"],
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    if not rsp.choices:
        raise RuntimeError("Empty choices")

    choice = rsp.choices[0]
    raw = choice.message.content
    _log_reply(raw)

    if not raw:
        raise RuntimeError("Empty payload")

    # This parsing logic is kept from the original for compatibility
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "sets" in data:
            return data["sets"]
        if isinstance(data, dict):
            keys = sorted(data.keys(), key=lambda k: (len(k), k))
            return [[data[k] for k in keys][:3]]
        if isinstance(data, str):
            return [data.strip().split(",")[:3]]
        raise KeyError("Unrecognised reply structure")
    except (JSONDecodeError, KeyError) as e:
        repaired = _repair_json(raw)
        if repaired:
            return json.loads(repaired)["sets"]
        raise RuntimeError(f"Failed to parse model reply: {e}")

# --- Main public function ---
def call_model(cue: str, prompt_key: str, num_sets_to_generate: int) -> list:
    """
    Generates a specified number of association sets for a cue by calling
    the LLM in a loop.
    """
    all_sets = []
    # Set the prompt key in the global CFG for the internal function
    CFG['prompt'] = prompt_key

    # The main loop now lives here, where it belongs.
    for _ in range(num_sets_to_generate):
        try:
            single_set = _call_single_set(cue, prompt_key)
            if single_set:
                all_sets.extend(single_set)
            # A small delay to avoid hitting rate limits on rapid calls
            time.sleep(0.1)
        except RuntimeError as e:
            # Log the error but continue trying to get the other sets
            print(f"⚠️  Error on cue '{cue}': {e} – skipping one attempt")
            continue
            
    return all_sets

# Note: The original generate_and_save and CLI logic are kept for compatibility
# with the 'sweep.py' script.
def generate_and_save(out_path: str, seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    print(f"🟢 Writing data to {out_path}")
    cues = pd.read_csv(DATA_PATH)["cue"].dropna().unique()
    sample = random.sample(list(cues), k=CFG["num_cues"])

    with open(out_path, "w", encoding="utf-8") as fw:
        for cue in sample:
            # This now correctly calls the looping function
            sets_acc = call_model(
                cue=cue,
                prompt_key=CFG["prompt"],
                num_sets_to_generate=CFG["sets_total"]
            )
            if sets_acc:
                fw.write(json.dumps({
                    "cue":  cue,
                    "sets": sets_acc,
                    "cfg":  CFG,
                }) + "\n")
            else:
                print(f"⏭️  No sets for cue '{cue}' – not writing")

    print(f"✅ Done — wrote {len(sample)} cues")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("outfile", nargs="?", default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    out = args.outfile or f"runs/lm_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    generate_and_save(out, seed=args.seed)

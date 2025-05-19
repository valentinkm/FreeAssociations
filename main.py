#!/usr/bin/env python3
"""
flexible_cue_pipeline.py
---------------------------------
Free‑association test harness with human‑style prompts
and machine‑parsable JSON output.
"""

from __future__ import annotations
import os, time, json, sys, datetime as dt
from typing import List, Literal

import pandas as pd
import backoff
from openai import OpenAI, RateLimitError, APIError
from tqdm.auto import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────
CFG = {
    "model": "gpt-4",          # or "gpt-4-turbo"
    "temperature": 1.5,              # intentionally high
    "calls_per_cue": 1,              # single test run
    "num_cues": 10,
    "mode": "json",                  # "json" or "func"
    "swow_path": os.path.join(
        "Small World of Words", "SWOW-EN.R100.20180827.csv"
    ),
    "out_path": "results_flexible.jsonl",
}

# ── DATA ──────────────────────────────────────────────────────────────────
swow = pd.read_csv(CFG["swow_path"])
CUES = sorted(swow["cue"].dropna().unique())[: CFG["num_cues"]]
print(f"\nTesting {len(CUES)} cues: {CUES}\n")

# ── OPENAI CLIENT ─────────────────────────────────────────────────────────
client = OpenAI()
if client.api_key is None:
    sys.exit("❌  Set OPENAI_API_KEY in your environment")

# ── HUMAN‑STYLE PROMPTS ──────────────────────────────────────────────────
def category_prompt(cue: str) -> str:
    return (
        "You are doing a word‑association survey.\n\n"
        "Task 1 – Category:  When you see a cue word, you write which broad "
        "semantic category it belongs to (e.g., 'animal', 'emotion', "
        "'profession').\n\n"
        f"Cue word: {cue}\n\n"
        "→ Write **one** lowercase word for the category\n\n"
        "FORMAT:  return a JSON object like {\"category\": \"yourword\"}"
    )

def assoc_prompt(cue: str) -> str:
    return (
        "You are doing a classic *free‑association* task used in psychology.\n"
        "When you read a cue word, immediately write the first three words "
        "that pop into your mind.  Keep each word short (usually one word).\n\n"
        f"Cue word: {cue}\n\n"
        "Repeat this *independently* **ten** times so we get ten different "
        "sets of three associations.\n\n"
        "FORMAT exactly as JSON:\n"
        "{\n"
        "  \"sets\": [\n"
        "    [\"first\",\"second\",\"third\"],      # set 1\n"
        "    … nine more sets …\n"
        "  ]\n"
        "}"
    )

# ── FUNCTION SCHEMAS (for mode == 'func') ─────────────────────────────────
FUNC_CATEGORY = [{
    "name": "store_category",
    "description": "One‑word semantic category",
    "parameters": {
        "type": "object",
        "properties": {"category": {"type": "string"}},
        "required": ["category"],
    },
}]
FUNC_ASSOC = [{
    "name": "store_sets",
    "description": "Ten sets of three one‑word associations",
    "parameters": {
        "type": "object",
        "properties": {
            "sets": {
                "type": "array",
                "minItems": 10, "maxItems": 10,
                "items": {
                    "type": "array",
                    "minItems": 3, "maxItems": 3,
                    "items": {"type": "string"},
                },
            }
        },
        "required": ["sets"],
    },
}]

# ── BACK‑OFF WITH LOGGING ────────────────────────────────────────────────
def _log_retry(details):
    exc  = details["exception"]
    wait = round(details["wait"], 1)
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] retrying after {exc.__class__.__name__} – sleeping {wait}s")

@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APIError),
    max_time=120,
    on_backoff=_log_retry,
)
def chat_retry(**kw):
    return client.chat.completions.create(**kw)

# ── CALL HELPERS ──────────────────────────────────────────────────────────
def _call_json(messages: list, key: str):
    rsp = chat_retry(
        model           = CFG["model"],
        temperature     = CFG["temperature"],
        response_format = {"type": "json_object"},
        messages        = messages,
    )
    return json.loads(rsp.choices[0].message.content)[key]

def _call_func(messages: list, schema: list, key: str):
    rsp = chat_retry(
        model       = CFG["model"],
        temperature = CFG["temperature"],
        functions   = schema,
        messages    = messages,
    )
    if rsp.choices[0].finish_reason != "function_call":
        raise RuntimeError("Model did not call the function")
    args = json.loads(rsp.choices[0].message.function_call.arguments)
    return args[key]

def get_category(cue: str) -> str:
    msgs = [
        {"role": "system", "content": "You respond only with valid JSON."},
        {"role": "user",   "content": category_prompt(cue)},
    ]
    if CFG["mode"] == "json":
        return _call_json(msgs, "category")
    return _call_func(msgs, FUNC_CATEGORY, "category")

def get_assocs(cue: str):
    msgs = [
        {"role": "system", "content": "You respond only with valid JSON."},
        {"role": "user",   "content": assoc_prompt(cue)},
    ]
    if CFG["mode"] == "json":
        return _call_json(msgs, "sets")
    return _call_func(msgs, FUNC_ASSOC, "sets")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────
records = []
for cue in tqdm(CUES, desc="Cues"):
    for _ in range(CFG["calls_per_cue"]):
        try:
            category   = get_category(cue)
            assoc_sets = get_assocs(cue)
        except Exception as e:
            print(f"\n❌  Error on cue '{cue}': {e}\n")
            category, assoc_sets = None, []
        records.append({
            "cue": cue,
            "category": category,
            "associations": assoc_sets,
        })
        time.sleep(0.2)

# ── SAVE ──────────────────────────────────────────────────────────────────
with open(CFG["out_path"], "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"\n✅  {len(records)} calls saved → {CFG['out_path']}")

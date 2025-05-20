#!/usr/bin/env python3
"""
run_lm.py – robust generator of TOTAL_SETS free associations per cue
"""
import time, json, textwrap, random, re, datetime as dt
from math import ceil
from pathlib import Path
from json.decoder import JSONDecodeError

import pandas as pd
from openai import OpenAI

from .settings      import CFG, TOTAL_SETS, MAX_TOKENS, DATA_PATH
# ──────────────────────────────── NEW ↓ ─────────────────────────────────
from .prompt_loader import get_prompt, render_prompt, FUNC_SCHEMA
# (TEMPLATES is no longer needed here.)

client = OpenAI()

# ── helpers ---------------------------------------------------------------
def _log_prompt(cue, n_sets, prompt):
    print(f"\n🟦 PROMPT | cue={cue!r}  n_sets={n_sets}  demographic={CFG['demographic']}\n"
          + textwrap.indent(prompt, "   "))

def _log_reply(txt):
    if not txt:
        print("🟥 REPLY  | <None>")
    else:
        print(f"🟩 REPLY  | {' '.join(txt.split())[:200]}…")

def _repair_json(txt):
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

# ── single call with retries ---------------------------------------------
def call_model(cue: str, n_sets: int, prompt_key: str, retry=False):
    # --- build user prompt -----------------------------------------------
    base_tpl = get_prompt(prompt_key, CFG["demographic"])          # ← uses new helper
    user_msg = render_prompt(base_tpl, cue, n_sets)
    _log_prompt(cue, n_sets, user_msg)

    msgs = [
        {"role": "system", "content": "Return ONLY valid JSON."},
        {"role": "user",   "content": user_msg},
    ]
    kw = dict(
        model=CFG["model"],
        temperature=CFG["temperature"],
        top_p=CFG["top_p"],
        frequency_penalty=CFG["frequency_penalty"],
        presence_penalty=CFG["presence_penalty"],
        max_tokens=MAX_TOKENS,
        messages=msgs,
        response_format={"type": "json_object"},
    )
    if prompt_key == "chatml_func":
        kw["functions"] = [FUNC_SCHEMA(n_sets)]

    rsp = client.chat.completions.create(**kw)
    if not rsp.choices:
        if not retry:
            print("⚠️  Empty choices – retrying once")
            return call_model(cue, n_sets, prompt_key, retry=True)
        raise RuntimeError("OpenAI returned no choices twice.")

    choice = rsp.choices[0]
    raw = (choice.message.function_call.arguments
           if choice.finish_reason == "function_call"
           else choice.message.content)
    _log_reply(raw)

    if not raw:
        if prompt_key != "chatml_func" and not retry:
            print("⚠️  Empty payload – retrying via function_call")
            return call_model(cue, n_sets, "chatml_func", retry=True)
        raise RuntimeError("Received empty payload twice.")

    try:
        return json.loads(raw)["sets"][:n_sets]
    except JSONDecodeError:
        repaired = _repair_json(raw)
        if repaired:
            return json.loads(repaired)["sets"][:n_sets]
        if prompt_key != "chatml_func" and not retry:
            print("⚠️  Parse failed – retrying via function_call")
            return call_model(cue, n_sets, "chatml_func", retry=True)
        raise

# ── outer loop -----------------------------------------------------------
def generate_and_save(out_path: str, seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    print(f"🟢 Writing data to {out_path}")
    cues = pd.read_csv(DATA_PATH)["cue"].dropna().unique()
    sample = random.sample(list(cues), k=CFG["num_cues"])

    prompt_key = CFG["prompt"]
    chunk      = CFG["calls_per_cue"]
    chunks     = ceil(TOTAL_SETS / chunk)

    with open(out_path, "w", encoding="utf-8") as fw:
        for cue in sample:
            sets_acc = []
            while len(sets_acc) < TOTAL_SETS:
                batch = min(chunk, TOTAL_SETS - len(sets_acc))
                try:
                    sets_acc.extend(call_model(cue, batch, prompt_key))
                except RuntimeError as e:
                    print(f"⚠️  {cue}: {e}  – waiting 2 s then retrying once")
                    time.sleep(2)
                    try:
                        sets_acc.extend(call_model(cue, batch, prompt_key, retry=True))
                    except RuntimeError:
                        print(f"⏭️  Skipping cue '{cue}' after repeated failures")
                        break

            if sets_acc:
                fw.write(json.dumps({
                    "cue": cue,
                    "sets": sets_acc,
                    "cfg":  CFG                 # now includes 'demographic'
                }) + "\n")

    print(f"✅ Done — wrote {len(sample)} cues × ≤{TOTAL_SETS} sets")

# ── CLI ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("outfile", nargs="?", default=None,
                   help="Path for JSONL output")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for cue sampling")
    args = p.parse_args()

    out = args.outfile or f"runs/lm_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    generate_and_save(out, seed=args.seed)

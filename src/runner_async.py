# src/runner_async.py
import asyncio, json, time, math, pandas as pd
from pathlib import Path
from openai import AsyncOpenAI
from .settings import CFG, TOTAL_SETS, MAX_TOKENS, DATA_PATH
from .prompt_loader import render_prompt, TEMPLATES, FUNC_SCHEMA

# ---- tune these to your org limits ----
MAX_CONCURRENCY = 20       # simultaneous sockets
RPM_LIMIT       = 350      # requests per minute
TPM_LIMIT       = 20_000   # tokens per minute (rough estimate)

client = AsyncOpenAI()

sem     = asyncio.Semaphore(MAX_CONCURRENCY)
req_log = []               # (timestamp, tokens) per request

async def throttle(tokens_estimate: int):
    "basic token+request budgeter"
    while True:
        now = time.time()
        # drop old log entries (>60 s)
        while req_log and now - req_log[0][0] > 60:
            req_log.pop(0)
        req_cnt = len(req_log)
        tok_sum = sum(t for _, t in req_log)
        if req_cnt < RPM_LIMIT and tok_sum + tokens_estimate < TPM_LIMIT:
            req_log.append((now, tokens_estimate))
            return
        await asyncio.sleep(0.25)

async def call_one(cue, n_sets, prompt_key):
    user_msg = render_prompt(TEMPLATES[prompt_key], cue, n_sets)
    msgs = [{"role":"system","content":"Return ONLY valid JSON."},
            {"role":"user","content":user_msg}]
    kw = dict(model=CFG["model"], messages=msgs,
              temperature=CFG["temperature"], max_tokens=MAX_TOKENS)
    if prompt_key == "chatml_func":
        kw["functions"] = [FUNC_SCHEMA(n_sets)]

    est_tokens = 50 + n_sets * 15      # rough budget
    await throttle(est_tokens)

    async with sem:
        rsp = await client.chat.completions.create(**kw)
    txt = (rsp.choices[0].message.function_call.arguments
           if prompt_key == "chatml_func" else rsp.choices[0].message.content)
    return json.loads(txt)["sets"][:n_sets]

async def process_cue(cue):
    single = CFG["calls_per_cue"] == 1
    prompt_key = "human_single" if single else CFG["prompt"]
    chunk_size = 1 if single else CFG["calls_per_cue"]
    sets = []
    while len(sets) < TOTAL_SETS:
        batch = min(chunk_size, TOTAL_SETS - len(sets))
        sets.extend(await call_one(cue, batch, prompt_key))
    return {"cue": cue, "sets": sets, "cfg": CFG}

async def main(out_path: str):
    df = pd.read_csv(DATA_PATH)["cue"].dropna().unique()
    cues = sorted(df)[: CFG["num_cues"]]
    out  = Path(out_path)
    out.parent.mkdir(exist_ok=True)

    async with aiofiles.open(out, "w") as fw:   # pip install aiofiles
        tasks = [process_cue(c) for c in cues]
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            await fw.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    import aiofiles, argparse, datetime as dt
    p = argparse.ArgumentParser()
    p.add_argument("outfile", nargs="?", default=None)
    args = p.parse_args()
    opath = args.outfile or f"runs/async_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    asyncio.run(main(opath))

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
import time
import re
import pandas as pd
from tqdm import tqdm

from . import settings
from .model_interface import get_model_associations
from .prompt_loader import get_prompt, render_prompt, TEMPLATES
from .profile_utils import age_bin, norm_gender, native_bin, edu_bin, slugify

# --- Prompt Sweep Experiment Runner ---
def run_sweep(model_name: str, nsets: int, ncues: int):
    """Launches a prompt sweep experiment for a specific model."""
    print(f"--- Launching Prompt Sweep for model: {model_name} ---")
    
    space = dict(settings.SEARCH_SPACE)
    if "demographic" in space: space["demographic"] = ["all"]
    
    try:
        all_cues = pd.read_csv(settings.SWOW_DATA_PATH)["cue"].dropna().unique()
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find SWOW data at {settings.SWOW_DATA_PATH}")
        return

    for overrides in itertools.product(*space.values()):
        prompt_config = dict(zip(space.keys(), overrides))
        prompt_key = prompt_config.get("prompt")
        if prompt_key not in TEMPLATES: continue

        run_config = {"model": model_name, "prompt": prompt_key, "nsets": nsets}
        h = hashlib.md5(json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:8]
        
        out_path = settings.RUNS_DIR / f"grid_{h}.jsonl"
        if out_path.exists():
            print(f"✅ {out_path.name} exists – skipping")
            continue

        print(f"\n▶️  Running config: {run_config}")
        sample_cues = random.sample(list(all_cues), k=ncues)
        base_prompt = get_prompt(prompt_key, "all")

        with open(out_path, "w", encoding="utf-8") as fw:
            for cue in tqdm(sample_cues, desc=f"Prompt: {prompt_key}"):
                try:
                    prompt_for_call = render_prompt(base_prompt, cue, n=nsets)
                    response_data = get_model_associations(prompt_for_call, model_name)
                    if response_data and "sets" in response_data:
                        fw.write(json.dumps({"cue": cue, "sets": response_data["sets"], "cfg": run_config}) + "\n")
                except Exception as e:
                    tqdm.write(f"❌ FAILED on cue '{cue}': {e}")
        print(f"✅ Done — wrote {len(sample_cues)} cues for config {h}")

# --- Yoked Generation Runner ---
EDU_MAP = {"1": "no formal education", "2": "elementary-school education", "3": "high-school education", "4": "university bachelor degree", "5": "university master degree", "unspecified": "unspecified education"}
PROFILE_TEMPLATE = """You are answering in english in a study as a typical participant with the following characteristics:
• Age: {{age}}
• Gender: {{gender}}
• Native language: {{nativeLanguage}}
• Country: {{country}}
• Education: {{education}}

{{BASE_PROMPT}}"""

def _build_steered_prompt(profile_id: str, base_prompt_text: str) -> str:
    """Manually builds the final steered prompt from a profile_id."""
    try:
        pattern = re.compile(r"profile_age_(?P<age>.+?)_gender_(?P<gender>.+?)_native_(?P<native>.+?)_country_(?P<country>.+?)_edu_(?P<education>.+)")
        match = pattern.match(profile_id)
        if not match: raise ValueError("Regex did not match profile ID format.")
        parts = match.groupdict()
        vals = { "age": parts["age"].replace('<', 'under '), "gender": parts["gender"], "nativeLanguage": "English" if parts["native"] == "en" else "a non-English language", "country": parts["country"].replace("_", " ").title(), "education": EDU_MAP.get(parts["education"], "unspecified education") }
        final_prompt = PROFILE_TEMPLATE
        for key, value in vals.items():
            final_prompt = final_prompt.replace(f"{{{{{key}}}}}", str(value))
        return final_prompt.replace("{{BASE_PROMPT}}", base_prompt_text)
    except Exception as e:
        raise KeyError(f"Could not parse profile_id: {profile_id}. Error: {e}")

def run_yoked_generation(model_name: str, ncues: int, nsets: int):
    """Runs the yoked (matched-sample) generation experiment for a specific model."""
    print(f"\n--- Yoked Steering Run for model: {model_name} ---")
    
    print(f"🔄 Step 1: Identifying human profiles...")
    try:
        df = pd.read_csv(settings.SWOW_DATA_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find SWOW data at '{settings.SWOW_DATA_PATH}'")
        return

    df.dropna(subset=['cue', 'age', 'gender', 'nativeLanguage', 'country', 'education'], inplace=True)
    df['age_bin'] = df['age'].apply(age_bin)
    df['gender_bin'] = df['gender'].apply(norm_gender)
    df['country_bin'] = df['country'].str.strip().apply(slugify).replace('', 'unspecified')
    df['education_bin'] = df['education'].apply(edu_bin)
    df['native_bin'] = df['nativeLanguage'].apply(native_bin)
    df['profile_id'] = ("profile_age_" + df["age_bin"] + "_gender_" + df["gender_bin"] + "_native_" + df["native_bin"] + "_country_" + df["country_bin"] + "_edu_" + df["education_bin"])

    profiles_by_cue = df.groupby('cue')['profile_id'].apply(list).to_dict()
    all_cues = random.sample(list(profiles_by_cue.keys()), k=min(ncues, len(profiles_by_cue)))
    
    print(f"✅ Plan created. Will run on {len(all_cues)} cues.")

    def get_processed_cues(path):
        if not path.exists(): return set()
        with open(path, 'r') as f: return {json.loads(line)['cue'] for line in f if line.strip()}
    
    processed_steered = get_processed_cues(settings.YOKED_STEERED_PATH)
    processed_nonsteered = get_processed_cues(settings.YOKED_NONSTEERED_PATH)

    base_prompt_text = TEMPLATES[settings.GENERALIZE_DEFAULTS['prompt']]
    
    print("\n🔄 Step 2: Generating yoked LLM responses...")
    
    with open(settings.YOKED_STEERED_PATH, "a") as f_steered:
        for cue in tqdm(all_cues, desc="Steered Runs"):
            if cue in processed_steered: continue
            
            all_sets_for_cue = []
            for profile_id in profiles_by_cue[cue]:
                try:
                    full_prompt = _build_steered_prompt(profile_id, base_prompt_text)
                    prompt_for_call = render_prompt(full_prompt, cue, n=1)
                    response_data = get_model_associations(prompt_for_call, model_name)
                    if response_data and "sets" in response_data:
                        all_sets_for_cue.append({"profile_id": profile_id, "set": response_data["sets"][0]})
                except Exception as e:
                    tqdm.write(f"❌ Steered ERROR on cue '{cue}' with profile '{profile_id}': {e}")

            if all_sets_for_cue:
                f_steered.write(json.dumps({"cue": cue, "responses": all_sets_for_cue}) + "\n")

    with open(settings.YOKED_NONSTEERED_PATH, "a") as f_nonsteered:
        for cue in tqdm(all_cues, desc="Non-Steered Runs"):
            if cue in processed_nonsteered: continue
            
            all_sets_for_cue = []
            for _ in range(nsets):
                try:
                    prompt_for_call = render_prompt(base_prompt_text, cue, n=1)
                    response_data = get_model_associations(prompt_for_call, model_name)
                    if response_data and "sets" in response_data:
                        all_sets_for_cue.append({"profile_id": "all", "set": response_data["sets"][0]})
                except Exception as e:
                    tqdm.write(f"❌ Non-steered ERROR on cue '{cue}': {e}")
            
            if all_sets_for_cue:
                f_nonsteered.write(json.dumps({"cue": cue, "responses": all_sets_for_cue}) + "\n")

    print("\n--- Yoked Run Complete ---")

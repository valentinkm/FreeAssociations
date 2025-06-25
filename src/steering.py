#!/usr/bin/env python3
"""
run_yoked_generation.py - An efficient script to generate a "yoked" or
"matched-sample" set of steered and non-steered LLM responses.

(v2 - Fixes profile_id creation for participants with no country data)
"""

import json
import sys
import random
import pandas as pd
import re
import time
from pathlib import Path
from tqdm import tqdm

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Import from existing project ---
try:
    from src.settings import CFG
    from src.run_lm import client, _repair_json
    from src.prompt_loader import TEMPLATES, render_prompt
    from src.extract_profiles import age_bin, norm_gender, native_bin, edu_bin, slugify
except ImportError as e:
    print(f"❌ Error: Could not import from 'src'. Ensure required files exist.")
    print(f"   Details: {e}")
    sys.exit(1)

# --- Configuration ---
NUM_CUES_TO_RUN = 10 # Control how many cues to process in a single run
BASE_PROMPT_KEY = "participant_default_question"
HUMAN_DATA_PATH = PROJECT_ROOT / "Small World of Words" / "SWOW-EN.R100.20180827.csv"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "yoked_generation"
STEERED_OUTPUT_PATH = OUTPUT_DIR / "yoked_steered_responses.jsonl"
NONSTEERED_OUTPUT_PATH = OUTPUT_DIR / "yoked_nonsteered_responses.jsonl"


# --- Self-Contained Prompting Logic ---
# (This section is robust and remains unchanged)
EDU_MAP = {
    "1": "no formal education", "2": "elementary-school education",
    "3": "high-school education", "4": "university bachelor degree",
    "5": "university master degree", "unspecified": "unspecified education",
}

PROFILE_TEMPLATE = """You are answering in english in a study as a typical participant with the following characteristics:
• Age: {{age}}
• Gender: {{gender}}
• Native language: {{nativeLanguage}}
• Country: {{country}}
• Education: {{education}}

{{BASE_PROMPT}}"""

def build_steered_prompt(profile_id: str, base_prompt_text: str) -> str:
    """Manually and robustly builds the final steered prompt from a profile_id."""
    try:
        pattern = re.compile(
            r"profile_age_(?P<age>.+?)"
            r"_gender_(?P<gender>.+?)"
            r"_native_(?P<native>.+?)"
            r"_country_(?P<country>.+?)"
            r"_edu_(?P<education>.+)"
        )
        match = pattern.match(profile_id)
        if not match: raise ValueError("Regex did not match profile ID format.")
        parts = match.groupdict()
        vals = {
            "age": parts["age"].replace('<', 'under '), "gender": parts["gender"],
            "nativeLanguage": "English" if parts["native"] == "en" else "a non-English language",
            "country": parts["country"].replace("_", " ").title(),
            "education": EDU_MAP.get(parts["education"], "unspecified education")
        }
        final_prompt = PROFILE_TEMPLATE
        for key, value in vals.items():
            final_prompt = final_prompt.replace(f"{{{{{key}}}}}", str(value))
        return final_prompt.replace("{{BASE_PROMPT}}", base_prompt_text)
    except (ValueError, AttributeError) as e:
        raise KeyError(f"Could not parse profile_id: {profile_id}. Error: {e}")

def call_model_with_prompt(cue: str, n_sets: int, full_prompt: str):
    """A direct model call with retries and a unified, robust parser."""
    user_msg = render_prompt(full_prompt, cue, n_sets)
    for attempt in range(3):
        try:
            rsp = client.chat.completions.create(
                model=CFG.get("model", "gpt-4.1-nano"), temperature=CFG.get("temperature", 1.1), max_tokens=200,
                messages=[{"role": "system", "content": "Return ONLY valid JSON."}, {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"},
            )
            raw = rsp.choices[0].message.content
            if not raw: time.sleep(1); continue
            try:
                data = json.loads(raw)
                if isinstance(data, dict) and "sets" in data: return data["sets"][:n_sets]
                if isinstance(data, dict): return [[str(v) for v in data.values()][:3]]
                if isinstance(data, list): return data[:n_sets]
            except json.JSONDecodeError:
                if repaired := _repair_json(raw):
                    try: return json.loads(repaired)["sets"][:n_sets]
                    except (json.JSONDecodeError, KeyError): pass
            words = [w.strip() for w in raw.replace(",", " ").split() if w.strip()]
            if len(words) >= 3: return [[words[0], words[1], words[2]]]
            return None
        except Exception as e:
            tqdm.write(f"API Error on attempt {attempt + 1}: {e}. Retrying after delay...")
            time.sleep(2 ** attempt)
    raise RuntimeError("Could not get a valid response after multiple retries.")


# --- Main Functions ---

def create_and_execute_yoked_plan():
    """Identifies human profiles for each cue and runs the yoked generation."""
    print("--- Yoked Demographic Steering Run ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Step 1: Identifying human profiles for each cue...")
    df = pd.read_csv(HUMAN_DATA_PATH)
    df.dropna(subset=['cue', 'age', 'gender', 'nativeLanguage', 'country', 'education'], inplace=True)

    df['age_bin'] = df['age'].apply(age_bin)
    df['gender_bin'] = df['gender'].apply(norm_gender)
    # <<< FIX: Replace empty country slugs with 'unspecified' to prevent errors >>>
    df['country_bin'] = df['country'].str.strip().apply(slugify).replace('', 'unspecified')
    df['education_bin'] = df['education'].apply(edu_bin)
    df['native_bin'] = df['nativeLanguage'].apply(native_bin)
    df['profile_id'] = ("profile_age_" + df["age_bin"] + "_gender_" + df["gender_bin"] +
                        "_native_" + df["native_bin"] + "_country_" + df["country_bin"] +
                        "_edu_" + df["education_bin"])

    # Group profiles by cue
    profiles_by_cue = df.groupby('cue')['profile_id'].apply(list).to_dict()
    all_cues = list(profiles_by_cue.keys())
    random.shuffle(all_cues)
    cues_to_run = all_cues[:NUM_CUES_TO_RUN]
    
    print(f"✅ Plan created. Will run on {len(cues_to_run)} cues.")

    # Load existing data to avoid re-running
    processed_steered = set()
    if STEERED_OUTPUT_PATH.exists():
        with open(STEERED_OUTPUT_PATH, 'r') as f:
            for line in f: 
                try: processed_steered.add(json.loads(line)['cue'])
                except (json.JSONDecodeError, KeyError): continue

    processed_nonsteered = set()
    if NONSTEERED_OUTPUT_PATH.exists():
        with open(NONSTEERED_OUTPUT_PATH, 'r') as f:
            for line in f: 
                try: processed_nonsteered.add(json.loads(line)['cue'])
                except (json.JSONDecodeError, KeyError): continue

    base_prompt_text = TEMPLATES[BASE_PROMPT_KEY]
    
    print("\n🔄 Step 2: Generating yoked LLM responses...")
    
    # --- Generate Steered Data ---
    with open(STEERED_OUTPUT_PATH, "a") as f_steered:
        for cue in tqdm(cues_to_run, desc="Steered Runs"):
            if cue in processed_steered: continue
            
            all_sets_for_cue = []
            for profile_id in profiles_by_cue[cue]:
                try:
                    prompt = build_steered_prompt(profile_id, base_prompt_text)
                    response_set = call_model_with_prompt(cue, 1, prompt)
                    if response_set:
                        # Storing each response set with its corresponding profile
                        all_sets_for_cue.append({"profile_id": profile_id, "set": response_set[0]})
                except Exception as e:
                    tqdm.write(f"❌ Steered ERROR on cue '{cue}' with profile '{profile_id}': {e}")

            if all_sets_for_cue:
                f_steered.write(json.dumps({"cue": cue, "responses": all_sets_for_cue}) + "\n")

    # --- Generate Non-Steered Data ---
    with open(NONSTEERED_OUTPUT_PATH, "a") as f_nonsteered:
        for cue in tqdm(cues_to_run, desc="Non-Steered Runs"):
            if cue in processed_nonsteered: continue
            
            all_sets_for_cue = []
            # Generate 100 non-steered responses to match the human sample size
            for _ in range(100):
                try:
                    response_set = call_model_with_prompt(cue, 1, base_prompt_text)
                    if response_set:
                        all_sets_for_cue.append({"profile_id": "all", "set": response_set[0]})
                except Exception as e:
                    tqdm.write(f"❌ Non-steered ERROR on cue '{cue}': {e}")
            
            if all_sets_for_cue:
                f_nonsteered.write(json.dumps({"cue": cue, "responses": all_sets_for_cue}) + "\n")

    print("\n--- Run Complete ---")

if __name__ == "__main__":
    create_and_execute_yoked_plan()

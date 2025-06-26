"""
model_interface.py
──────────────────
A unified interface for calling different language models, with a simplified
two-step fallback for non-instructed base models.
"""
import os
import json
import re
import textwrap
from openai import OpenAI
import google.generativeai as genai

from . import settings

# --- Client Configurations ---
try:
    openai_client = OpenAI()
except (ImportError, Exception):
    openai_client = None
    print("⚠️ OpenAI client not configured.")

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except (ImportError, KeyError, Exception):
    genai = None
    print("⚠️ Gemini client not configured.")

try:
    local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")
except (ImportError, Exception):
    local_client = None
    print("⚠️ Local client not configured.")

# --- Prompt Templates ---
BASE_MODEL_ZERO_SHOT_PROMPT = """Cue: "{cue_word}"
Association:"""

# --- Helper Logging Function ---
def _log_verbose(title, content):
    """Prints detailed logs if verbose mode is enabled."""
    if settings.CFG.get('verbose', False):
        print(textwrap.dedent(f"""
        -----------------------------------------
        VERBOSE: {title}
        -----------------------------------------
        {content}
        -----------------------------------------"""))

# --- Model Callers ---
def call_openai_model(full_prompt: str, model_name: str):
    """Makes an API call to an OpenAI model for structured JSON."""
    if not openai_client: raise ConnectionError("OpenAI client not initialized.")
    rsp = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Return ONLY valid JSON with a 'sets' key containing a list of lists of strings."},
            {"role": "user", "content": full_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=settings.CFG.get('temperature', 1.0),
        top_p=settings.CFG.get('top_p', 1.0)
    )
    return json.loads(rsp.choices[0].message.content)

def call_gemini_model(full_prompt: str, model_name: str):
    """Makes an API call to a Google Gemini model using a structured JSON schema."""
    if not genai: raise ConnectionError("Gemini client not initialized.")
    json_schema = {"type": "OBJECT", "properties": {"sets": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "STRING"}}}}}
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    rsp = gemini_model.generate_content(
        f"Please act as a helpful assistant. Your only job is to return valid JSON. {full_prompt}",
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json", response_schema=json_schema,
            temperature=settings.CFG.get('temperature', 1.0), top_p=settings.CFG.get('top_p', 1.0)
        )
    )
    return json.loads(rsp.text)

def call_local_model_raw(prompt: str, model_name: str, max_tokens: int = 15):
    """Calls a local model for raw, unstructured text output."""
    if not local_client: raise ConnectionError("Local client not initialized.")
    _log_verbose(f"PROMPT to {model_name}", prompt)
    rsp = local_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.CFG.get('temperature', 1.0),
        top_p=settings.CFG.get('top_p', 1.0),
        max_tokens=max_tokens
    )
    raw_content = rsp.choices[0].message.content
    _log_verbose(f"RAW REPLY from {model_name}", raw_content)
    return raw_content

def call_local_model_json(prompt: str, model_name: str):
    """Calls a local, instruct-tuned model and robustly parses the JSON output."""
    raw_content = call_local_model_raw(prompt, model_name, max_tokens=150)
    try:
        # Attempt 1: Parse directly
        return json.loads(raw_content)
    except json.JSONDecodeError:
        # Attempt 2: Find a JSON block within the text
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass # Continue to next fallback
        
        words = []
        # Find lines that look like 'word1: association' or '1. association'
        for line in raw_content.splitlines():
            match = re.search(r'[:\.\s]([\w\s-]+)$', line.strip())
            if match:
                # Clean up potential leading/trailing junk
                word = match.group(1).strip().replace('"', '').replace("'", "")
                words.append(word)
        
        if len(words) >= 3:
            return {"sets": [words[:3]]}

        if settings.CFG.get('verbose', False):
            print(f"--- VERBOSE: RAW MODEL OUTPUT (FINAL PARSE FAILED) ---\n{raw_content}\n----------------------------------------------------")
        raise ValueError("No valid JSON or parsable format found in local model's response.")


def get_model_associations(model_name: str, cue_word: str, nsets: int, instruct_prompt: str):
    """
    Acts as a switchboard. Uses a simplified two-step fallback for base models.
    """
    # For base models, use the two-step fallback strategy
    if model_name.endswith(':text'):
        all_sets = []
        formatter_model = 'gpt-4o'
        for _ in range(nsets):
            # Step 1: Elicit raw, unstructured associations with the simplest prompt
            base_prompt = BASE_MODEL_ZERO_SHOT_PROMPT.format(cue_word=cue_word)
            raw_output = call_local_model_raw(base_prompt, model_name)

            # Step 2: Use the formatter model to clean up and get JSON
            formatter_prompt = f'''From the text below, extract the three most relevant single-word associations for the cue "{cue_word}".
TEXT: "{raw_output}"
Return your answer in a valid JSON object with a "sets" key, where the value is a list containing ONE list of three strings. Example: {{"sets": [["word1", "word2", "word3"]]}}'''
            
            # Use the appropriate caller for the formatter model
            formatted_json = call_openai_model(formatter_prompt, formatter_model)
            
            if formatted_json and "sets" in formatted_json and formatted_json["sets"]:
                all_sets.append(formatted_json["sets"][0])

        return {"sets": all_sets}

    # For all instruct-tuned models
    elif model_name.startswith('gpt'):
        return call_openai_model(instruct_prompt, model_name)
    elif model_name.startswith('gemini'):
        return call_gemini_model(instruct_prompt, model_name)
    else: # Assume other local models are instruct-tuned
        return call_local_model_json(instruct_prompt, model_name)

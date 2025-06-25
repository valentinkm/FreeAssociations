"""
model_interface.py
──────────────────
A unified interface for calling different language models, including
OpenAI, Google Gemini, and local models.
"""
import os
import json
from openai import OpenAI
import google.generativeai as genai

from . import settings

# --- Client Configurations ---
try:
    openai_client = OpenAI() # Assumes OPENAI_API_KEY is set in environment
except (ImportError, Exception):
    openai_client = None
    print("⚠️ OpenAI client not configured. Is the library installed and API key set?")

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini_client = genai.GenerativeModel('gemini-1.5-flash-latest')
except (ImportError, KeyError, Exception):
    gemini_client = None
    print("⚠️ Gemini client not configured. Is the library installed and GOOGLE_API_KEY set?")

try:
    local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")
except (ImportError, Exception):
    local_client = None
    print("⚠️ Local client not configured. Assumes an OpenAI-compatible server at http://localhost:11434/v1")


def call_openai_model(full_prompt: str, model_name: str):
    """Makes an API call to an OpenAI model."""
    if not openai_client: raise ConnectionError("OpenAI client not initialized.")
    rsp = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Return ONLY valid JSON with a 'sets' key containing a list of lists."},
            {"role": "user", "content": full_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=settings.CFG.get('temperature', 1.0),
        top_p=settings.CFG.get('top_p', 1.0)
    )
    return json.loads(rsp.choices[0].message.content)

def call_gemini_model(full_prompt: str):
    """Makes an API call to a Google Gemini model."""
    if not gemini_client: raise ConnectionError("Gemini client not initialized.")
    rsp = gemini_client.generate_content(
        f"Please act as a helpful assistant. Your only job is to return valid JSON. {full_prompt}",
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=settings.CFG.get('temperature', 1.0),
            top_p=settings.CFG.get('top_p', 1.0)
        )
    )
    # Gemini can wrap its JSON output in ```json ... ```, so we need to clean it
    cleaned_text = rsp.text.strip().lstrip("```json").rstrip("```")
    return json.loads(cleaned_text)

def call_local_model(full_prompt: str, model_name: str):
    """Makes an API call to a local, OpenAI-compatible model."""
    if not local_client: raise ConnectionError("Local client not initialized. Is a server running?")
    rsp = local_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Return ONLY valid JSON with a 'sets' key containing a list of lists."},
            {"role": "user", "content": full_prompt},
        ],
        temperature=settings.CFG.get('temperature', 1.0),
        top_p=settings.CFG.get('top_p', 1.0)
    )
    return json.loads(rsp.choices[0].message.content)

def get_model_associations(full_prompt: str, model_name: str):
    """Acts as a switchboard to call the correct model API."""
    if model_name.startswith('gpt'):
        return call_openai_model(full_prompt, model_name)
    elif model_name.startswith('gemini'):
        return call_gemini_model(full_prompt)
    else: # Assumes all other models are local and OpenAI-compatible
        return call_local_model(full_prompt, model_name)

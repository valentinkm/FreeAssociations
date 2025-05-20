"""
Helpers for loading base prompt templates **and**
demographic wrappers, plus a one-stop `get_prompt(...)`.

A “wrapper” lives in prompts/demographics/* and must contain
the literal token {{BASE_PROMPT}} where the base prompt will be inserted.
"""
from pathlib import Path

# ─── base templates ──────────────────────────────────────────────────────
TEMPLATES = {
    p.stem.replace("-", "_"): p.read_text(encoding="utf-8")
    for p in Path("prompts").glob("*.txt")
}

# ─── demographic wrappers ────────────────────────────────────────────────
DEMOGRAPHIC_TPL = {
    p.stem: p.read_text(encoding="utf-8")
    for p in Path("prompts/demographics").glob("*.txt")
}

def get_prompt(prompt_key: str, demographic: str) -> str:
    """
    Return the final prompt string for the model, combining:
      • the *base* template named by `prompt_key`
      • an optional *demographic* wrapper

    Raises KeyError if either component is missing.
    """
    base = TEMPLATES[prompt_key]                    # may raise KeyError

    if demographic == "all":
        return base

    wrapper = DEMOGRAPHIC_TPL[demographic]          # may raise KeyError
    return wrapper.replace("{{BASE_PROMPT}}", base)

# ─── legacy helpers still used elsewhere ────────────────────────────────
def render_prompt(tpl: str, cue: str, n: int) -> str:
    """Simple replacement for {{cue}} / {{nsets}} placeholders."""
    return tpl.replace("{{cue}}", cue).replace("{{nsets}}", str(n))

def FUNC_SCHEMA(n_sets: int) -> dict:
    """Function-call schema used in run_lm.py"""
    return {
        "name": "store_sets",
        "description": "Return sets of free associations",
        "parameters": {
            "type": "object",
            "properties": {
                "sets": {
                    "type": "array",
                    "minItems": n_sets,
                    "maxItems": n_sets,
                    "items": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "string"}
                    }
                }
            },
            "required": ["sets"]
        }
    }

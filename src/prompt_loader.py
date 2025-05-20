# src/prompt_loader.py
from pathlib import Path

# Load all prompt templates once.
# Normalize keys by replacing '-' → '_'
TEMPLATES = {
    p.stem.replace("-", "_"): p.read_text()
    for p in Path("prompts").glob("*.txt")
}

def render_prompt(tpl: str, cue: str, n: int) -> str:
    return tpl.replace("{{cue}}", cue).replace("{{nsets}}", str(n))

def FUNC_SCHEMA(n_sets: int) -> dict:
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

# src/prompt_loader.py
from pathlib import Path

# Load all prompt templates once
TEMPLATES = {
    "human_single": Path("prompts/human_single.txt").read_text(),
    "human_json":   Path("prompts/human_json.txt").read_text(),
    "chatml_func":  Path("prompts/chatml_func.txt").read_text(),
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

"""
prompt_loader.py - load base templates and render full-profile steering

(Corrected Version)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import re

# ───────────────────────────── paths ────────────────────────────────────
PROMPTS_DIR = Path("prompts")
DEMOS_DIR = PROMPTS_DIR / "demographics"
PROFILE_TPL = DEMOS_DIR / "profile_template.txt"

# ───────────────────── base templates ───────────────────────────────────
TEMPLATES: Dict[str, str] = {
    p.stem.replace("-", "_"): p.read_text("utf-8")
    for p in PROMPTS_DIR.glob("*.txt")
}

# ─────────────── master profile template ────────────────────────────────
try:
    PROFILE_TEMPLATE = PROFILE_TPL.read_text("utf-8")
    if "{{BASE_PROMPT}}" not in PROFILE_TEMPLATE:
        raise RuntimeError("profile_template.txt must contain {{BASE_PROMPT}}")
except FileNotFoundError:
    # Define a default in case the file is missing
    PROFILE_TEMPLATE = """You are answering as a typical participant with the following characteristics:
• Age: {{age}}
• Gender: {{gender}}

{{BASE_PROMPT}}"""


# ─────────────────────── profile-id regex ───────────────────────────────
# <<< FIX: This regex is more robust and correctly handles all profile ID variations >>>
_PROFILE_RE = re.compile(
    r"""^profile_
        age(?P<age>[^_]+)
        _gender_(?P<gender>[^_]+)
        (?:_native_(?P<native>[^_]+(?:_[^_]+)*?))? # Makes native optional
        (?:_country_(?P<country>[^_]+(?:_[^_]+)*?))? # Makes country optional
        (?:_edu_(?P<education>.+))?$""", # Makes education optional
    re.VERBOSE,
)

# ───────────────── education code → phrase ──────────────────────────────
EDU_MAP = {
    "1": "no formal education",
    "2": "elementary-school education",
    "3": "high-school education",
    "4": "university bachelor degree",
    "5": "university master degree",
    "unspecified": "unspecified education",
}

# ─────────────────── helper: render wrapper ─────────────────────────────
def _render_profile_wrapper(profile_id: str) -> str:
    m = _PROFILE_RE.fullmatch(profile_id)
    if not m:
        # Fallback for simplified profiles from run_smart_generation.py
        if profile_id.startswith("profile_age"):
            parts = profile_id.replace("profile_age", "").split("_gender_")
            vals = {"age": parts[0], "gender": parts[1]}
        else:
            raise KeyError(f"Unrecognised profile id: {profile_id}")
    else:
        vals = m.groupdict()

    # Clean up and format values
    vals["age"] = vals.get("age", "").replace("<", "under ")
    vals["nativeLanguage"] = "English" if vals.get("native") == "en" else "a non-English language"
    vals["country"] = vals.get("country", "").replace("_", " ").title()
    vals["education"] = EDU_MAP.get(vals.get("education"), "unspecified")
    
    # fill template
    wrapper = PROFILE_TEMPLATE
    for key, value in vals.items():
        if value: # Only replace if value exists
            wrapper = wrapper.replace(f"{{{{{key}}}}}", str(value))

    return wrapper

# ───────────────────────── public API ───────────────────────────────────
def get_prompt(prompt_key: str, demographic: str) -> str:
    """Return the final prompt, optionally wrapped for a demographic profile."""
    try:
        base = TEMPLATES[prompt_key]
    except KeyError as e:
        raise KeyError(f"Unknown base prompt: {prompt_key}") from e

    if demographic == "all":
        return base

    if demographic.startswith("profile_"):
        wrapper = _render_profile_wrapper(demographic)
        return wrapper.replace("{{BASE_PROMPT}}", base)

    raise KeyError(f"Unknown demographic key: {demographic}")

# ───────────────────── legacy helpers (unchanged) ───────────────────────
def render_prompt(tpl: str, cue: str, n: int) -> str:
    return tpl.replace("{{cue}}", cue).replace("{{nsets}}", str(n))


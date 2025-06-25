#!/usr/bin/env python3
"""
extract_profiles.py  –  mine unique demographic profiles from SWOW-EN

• Education codes 1–5 handled
• Native-language binning fixed: recognises all labels that denote English
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd

# ─────────────────── helper binning functions ───────────────────────────
def age_bin(age: float | int) -> str:
    age = int(age)
    if age < 25:  return "<25"
    if age <= 34: return "25-34"
    if age <= 44: return "35-44"
    if age <= 59: return "45-59"
    return "60+"

def norm_gender(g: str) -> str:
    g = str(g).strip().lower()
    if g.startswith("f"): return "female"
    if g.startswith("m"): return "male"
    return "other"

# --- NEW: comprehensive English label set -------------------------------
_ENGLISH_LABELS = {
    "united states", "united kingdom", "australia", "new zealand",
    "ireland", "canada", "south africa", "other_english"
}

def native_bin(lang: str) -> str:
    """
    Return 'en' if nativeLanguage denotes any English variety,
    else 'non_en'.
    """
    val = str(lang).strip().lower()
    if "english" in val or val in _ENGLISH_LABELS:
        return "en"
    return "non_en"

def edu_bin(e) -> str:
    """
    Map numeric codes 1–5 to themselves; blanks/unknown → 'unspecified'.
    """
    if pd.isna(e):
        return "unspecified"

    if isinstance(e, (int, float)):
        code = str(int(e))
    else:
        txt = str(e).strip()
        code = txt[:-2] if re.fullmatch(r"\d+\.0", txt) else txt

    return code if code in {"1", "2", "3", "4", "5"} else "unspecified"

def slugify(txt: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", txt.lower().replace(" ", "_"))

# ───────────────────────────── main ─────────────────────────────────────
def main(csv_path: Path, top_countries: int | None):
    usecols = ["age", "gender", "nativeLanguage", "country", "education"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)

    df = df.dropna(subset=["age", "gender", "nativeLanguage", "country"])
    df["age_bin"]       = df["age"].apply(age_bin)
    df["gender"]        = df["gender"].apply(norm_gender)
    df["native_bin"]    = df["nativeLanguage"].apply(native_bin)
    df["education_bin"] = df["education"].apply(edu_bin)

    if top_countries:
        top = df["country"].value_counts().nlargest(top_countries).index
        df = df[df["country"].isin(top)]

    df["country_slug"] = df["country"].str.strip().apply(slugify)

    df["profile_id"] = (
        "age"  + df["age_bin"] +
        "_gender_"  + df["gender"] +
        "_native_"  + df["native_bin"] +
        "_country_" + df["country_slug"] +
        "_edu_"     + df["education_bin"]
    )

    group_cols = [
        "profile_id", "age_bin", "gender",
        "native_bin", "country_slug", "education_bin"
    ]
    profiles = (
        df.groupby(group_cols).size()
          .reset_index(name="n_records")
          .sort_values("n_records", ascending=False)
    )

    out = Path("dem_steering/profiles.csv")
    profiles.to_csv(out, index=False)
    print(f"✅  wrote {len(profiles):,} unique profiles → {out}")

# ─────────────────────────── CLI entry ──────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path,
                        help="Path to SWOW-EN CSV file")
    parser.add_argument("--top-countries", type=int, default=20,
                        help="Keep only the N most frequent countries "
                             "(0 = keep all)")
    args = parser.parse_args()

    main(args.csv, None if args.top_countries == 0 else args.top_countries)

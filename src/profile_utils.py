"""
profile_utils.py
────────────────
Utility functions for creating and handling demographic profiles from SWOW data.
"""
import re
import pandas as pd

from . import settings

# --- Binning and Normalization Functions ---

def age_bin(age: float | int) -> str:
    """Categorizes age into predefined bins."""
    age = int(age)
    if age < 25: return "<25"
    if age <= 34: return "25-34"
    if age <= 44: return "35-44"
    if age <= 59: return "45-59"
    return "60+"

def norm_gender(g: str) -> str:
    """Normalizes gender strings to 'female', 'male', or 'other'."""
    g = str(g).strip().lower()
    if g.startswith("f"): return "female"
    if g.startswith("m"): return "male"
    return "other"

_ENGLISH_LABELS = {
    "united states", "united kingdom", "australia", "new zealand",
    "ireland", "canada", "south africa", "other_english"
}

def native_bin(lang: str) -> str:
    """Bins native language as 'en' or 'non_en'."""
    val = str(lang).strip().lower()
    if "english" in val or val in _ENGLISH_LABELS:
        return "en"
    return "non_en"

def edu_bin(e) -> str:
    """Maps education codes to clean, consistent strings."""
    if pd.isna(e):
        return "unspecified"
    code = str(int(e)) if isinstance(e, (int, float)) else str(e).strip()
    return code if code in {"1", "2", "3", "4", "5"} else "unspecified"

def slugify(txt: str) -> str:
    """Creates a URL-friendly slug from text."""
    return re.sub(r"[^a-z0-9_]+", "", txt.lower().replace(" ", "_"))

# --- Main Profile Extraction Function ---

def extract_all_profiles(top_countries: int | None):
    """
    Mines unique demographic profiles from the SWOW dataset and saves them to a CSV file.
    """
    print(f"🔄 Reading SWOW data from {settings.SWOW_COMPLETE_DATA_PATH.name}...")
    try:
        usecols = ["age", "gender", "nativeLanguage", "country", "education"]
        df = pd.read_csv(settings.SWOW_COMPLETE_DATA_PATH, usecols=lambda c: c in usecols)
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find SWOW data at {settings.SWOW_COMPLETE_DATA_PATH}")
        return

    df.dropna(subset=["age", "gender", "nativeLanguage", "country"], inplace=True)
    
    print("🔄 Binning and normalizing demographic data...")
    df["age_bin"]       = df["age"].apply(age_bin)
    df["gender"]        = df["gender"].apply(norm_gender)
    df["native_bin"]    = df["nativeLanguage"].apply(native_bin)
    df["education_bin"] = df["education"].apply(edu_bin)

    if top_countries:
        print(f"🔄 Filtering to top {top_countries} countries...")
        top = df["country"].value_counts().nlargest(top_countries).index
        df = df[df["country"].isin(top)]

    df["country_slug"] = df["country"].str.strip().apply(slugify)

    df["profile_id"] = (
        "age"      + df["age_bin"] +
        "_gender_" + df["gender"] +
        "_native_" + df["native_bin"] +
        "_country_" + df["country_slug"] +
        "_edu_"    + df["education_bin"]
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

    out_path = settings.PROFILES_PATH
    profiles.to_csv(out_path, index=False)
    print(f"✅ Wrote {len(profiles):,} unique profiles to '{out_path}'")

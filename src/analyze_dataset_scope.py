#!/usr/bin/env python3
"""
analyze_dataset_scope.py - A utility script to calculate the full scope
of the human dataset for a complete LLM comparison.

This script calculates:
- The total number of unique demographic profiles.
- The total number of unique cues.
- The total number of unique (cue, profile) pairs, which represents the
  total number of steered LLM runs needed for 100% coverage.
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- Setup Project Path ---
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import from your existing project ---
try:
    from src.extract_profiles import age_bin, norm_gender, native_bin, edu_bin, slugify
except ImportError as e:
    print(f"❌ Error: Could not import from 'src/extract_profiles.py'. Ensure file exists.")
    sys.exit(1)

# --- Paths ---
HUMAN_DATA_PATH = PROJECT_ROOT / "Small World of Words" / "SWOW-EN.complete.20180827.csv"

def analyze_scope():
    """Analyzes the SWOW dataset to determine the full scope of a complete comparison."""
    print("--- Dataset Scope Analysis ---")
    
    # 1. Load human data
    print(f"🔄 Loading human data from '{HUMAN_DATA_PATH}'...")
    try:
        id_cols = ['cue', 'age', 'gender', 'nativeLanguage', 'country', 'education']
        df = pd.read_csv(HUMAN_DATA_PATH, usecols=id_cols)
    except FileNotFoundError:
        print(f"❌ FATAL: Could not find human data file at '{HUMAN_DATA_PATH}'")
        return

    df.dropna(subset=id_cols, inplace=True)

    # 2. Create the full, correct profile_id for each row
    print("🔄 Creating demographic profiles for all participants...")
    df['age_bin'] = df['age'].apply(age_bin)
    df['gender_bin'] = df['gender'].apply(norm_gender)
    df['country_bin'] = df['country'].str.strip().apply(slugify)
    df['education_bin'] = df['education'].apply(edu_bin)
    df['native_bin'] = df['nativeLanguage'].apply(native_bin)
    
    df['profile_id'] = (
        "profile_age_" + df["age_bin"] +
        "_gender_" + df["gender_bin"] +
        "_native_" + df["native_bin"] +
        "_country_" + df["country_bin"] +
        "_edu_" + df["education_bin"]
    )

    # 3. Perform the calculations
    print("🔄 Calculating dataset statistics...")
    total_unique_cues = df['cue'].nunique()
    total_unique_profiles = df['profile_id'].nunique()
    
    # The number of unique (cue, profile) pairs is the crucial number.
    # This represents the total number of individual steered LLM runs required.
    total_cue_profile_pairs = df[['cue', 'profile_id']].drop_duplicates().shape[0]

    # 4. Print the results
    print("\n--- Full Dataset Scope ---")
    print(f"📊 Total Unique Cues: {total_unique_cues:,}")
    print(f"📊 Total Unique Demographic Profiles: {total_unique_profiles:,}")
    print("-" * 30)
    print(f"🚀 Total (Cue, Profile) Pairs: {total_cue_profile_pairs:,}")
    print("\nThis final number represents the total number of steered LLM runs")
    print("required to achieve 100% coverage of the human dataset.")


if __name__ == "__main__":
    analyze_scope()

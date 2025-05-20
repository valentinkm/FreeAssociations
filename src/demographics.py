"""
Utility functions for assigning SWOW rows to demographic buckets.
Used both during generation (for bookkeeping) and evaluation.
"""
from __future__ import annotations
import pandas as pd
from typing import Callable, Dict

# ─── individual axis helpers ─────────────────────────────────────────────
def _age_bucket(age) -> str:
    if pd.isna(age):
        return "age_unknown"
    age = int(age)
    if age < 25:          return "age_<25"
    if age < 35:          return "age_25-34"
    if age < 50:          return "age_35-49"
    if age < 65:          return "age_50-64"
    return "age_65+"

def _gender_bucket(g) -> str:
    g = str(g or "").strip().lower()
    if g.startswith("m"): return "gender_m"
    if g.startswith("f"): return "gender_f"
    return "gender_other"

BUCKET_FNS: Dict[str, Callable[[pd.Series], str]] = {
    "age":    lambda row: _age_bucket(row["age"]),
    "gender": lambda row: _gender_bucket(row["gender"]),
}

def tag_rows(df: pd.DataFrame, axes=("age", "gender")) -> pd.DataFrame:
    out = df.copy()
    for axis in axes:
        out[axis] = out.apply(BUCKET_FNS[axis], axis=1)
    return out

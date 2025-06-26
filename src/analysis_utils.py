"""Utility functions shared across analysis modules."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

from . import settings


def robust_flatten(items):
    """Recursively flattens a list of lists of strings."""
    for x in items:
        if isinstance(x, list):
            yield from robust_flatten(x)
        elif isinstance(x, str):
            yield x.lower()


def weighted_jaccard(ctr1: Counter, ctr2: Counter) -> float:
    """Calculates the weighted Jaccard similarity between two Counters."""
    if not ctr1 or not ctr2:
        return 0.0
    all_keys = ctr1.keys() | ctr2.keys()
    intersection = sum(min(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    union = sum(max(ctr1.get(k, 0), ctr2.get(k, 0)) for k in all_keys)
    return intersection / union if union else 0.0


def cosine_sim(ctr1: Counter, ctr2: Counter) -> float:
    """Calculates the cosine similarity between two Counters."""
    if not ctr1 or not ctr2:
        return 0.0
    vocab = sorted(list(ctr1.keys() | ctr2.keys()))
    v1 = np.array([ctr1.get(word, 0) for word in vocab])
    v2 = np.array([ctr2.get(word, 0) for word in vocab])
    if not np.any(v1) or not np.any(v2):
        return 0.0
    return 1 - cosine(v1, v2)


def load_lexicon(lexicon_path: Path):
    """Loads a lexicon file into a dictionary of word Counters."""
    print(f"🔄 Loading LLM Association Lexicon from '{lexicon_path.name}'...")
    try:
        llm_lexicon = {}
        with open(lexicon_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                word = record["word"]
                associations = list(robust_flatten(record.get("association_sets", [])))
                llm_lexicon[word] = Counter(associations)
        print(f"✅ Loaded {len(llm_lexicon)} word vectors from lexicon.")
        return llm_lexicon
    except FileNotFoundError:
        print(f"❌ ERROR: Lexicon file not found at '{lexicon_path}'.")
        return None

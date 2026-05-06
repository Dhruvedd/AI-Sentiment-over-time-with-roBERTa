"""
Hedging / epistemic-uncertainty scoring.

A "hedge" is a marker that the speaker is qualifying confidence: words/phrases
like "maybe", "I think", "could", "might". We use a curated lexicon and score
texts at the sentence level — a hedge in one sentence of a long comment is
informative even if surrounding sentences are confident.

We report `hedge_density` = fraction of sentences containing at least one hedge.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

# Hedging lexicon. Curated from Hyland (2005) "Metadiscourse" and computational
# linguistics work on epistemic markers. Multi-word phrases come first so they
# match before their single-word components.
HEDGES = [
    # Multi-word phrases (matched first)
    r"\bi\s+(?:think|believe|guess|suspect|reckon|suppose|assume|feel)\b",
    r"\bin\s+my\s+(?:opinion|view|experience)\b",
    r"\bit\s+(?:seems|appears|looks)\s+(?:like|that)\b",
    r"\bsort\s+of\b",
    r"\bkind\s+of\b",
    r"\bas\s+far\s+as\s+i\s+(?:know|can\s+tell)\b",
    r"\bnot\s+sure\b",
    r"\bcould\s+be\b",
    r"\bmight\s+be\b",
    r"\bmay\s+be\b",
    # Single hedge words
    r"\b(?:maybe|perhaps|possibly|probably|presumably|apparently)\b",
    r"\b(?:might|may|could|would|should)\b",
    r"\b(?:seem|seems|seemed|appear|appears|appeared)\b",
    r"\b(?:likely|unlikely)\b",
    r"\b(?:somewhat|somehow|roughly|approximately|around|about)\b",
    r"\b(?:usually|generally|typically|often|sometimes|occasionally)\b",
    r"\b(?:assume|assumed|guess|guessed|suspect|suspected)\b",
    r"\b(?:tend|tends|tended)\s+to\b",
    r"\b(?:i\.?e\.?|e\.?g\.?)\b",  # qualifying expansions
    r"\bsuggest(?:s|ed)?\b",
    r"\bindicate(?:s|d)?\b",
]

HEDGE_REGEX = re.compile("|".join(HEDGES), flags=re.IGNORECASE)

# Crude sentence splitter. Avoids the NLTK punkt download for portability.
# Splits on .!? followed by whitespace + capital letter (or end of string).
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])|(?<=[.!?])\s*$")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = SENT_SPLIT.split(text.strip())
    # Filter out empty/very-short fragments (often punctuation artifacts)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def hedge_score_text(text: str) -> dict[str, float]:
    """
    Score a single document. Returns:
      - n_sentences  : number of sentences detected
      - n_hedged     : sentences with at least one hedge match
      - hedge_density: n_hedged / n_sentences (0 if no sentences)
      - hedge_count  : total hedge matches (length-dependent, secondary metric)
    """
    sentences = split_sentences(text)
    if not sentences:
        return {"n_sentences": 0, "n_hedged": 0,
                "hedge_density": 0.0, "hedge_count": 0}
    hedged = 0
    total_matches = 0
    for s in sentences:
        matches = HEDGE_REGEX.findall(s)
        if matches:
            hedged += 1
            total_matches += len(matches)
    return {
        "n_sentences": len(sentences),
        "n_hedged": hedged,
        "hedge_density": hedged / len(sentences),
        "hedge_count": total_matches,
    }


def score_hedging(texts: Iterable[str]) -> pd.DataFrame:
    rows = [hedge_score_text(t or "") for t in texts]
    df = pd.DataFrame(rows)
    df.columns = [f"hedge_{c}" if not c.startswith("hedge") else c for c in df.columns]
    # Final column names: n_sentences, n_hedged, hedge_density, hedge_count
    df = df.rename(columns={"hedge_n_sentences": "n_sentences",
                             "hedge_n_hedged":   "n_hedged"})
    return df

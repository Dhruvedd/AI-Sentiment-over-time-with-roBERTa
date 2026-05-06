"""
Sentiment, emotion, and uncertainty scoring.

Five complementary signals:
  1. VADER         — lexicon polarity, social-media-tuned, handles negation.
  2. AFINN         — simpler lexicon polarity baseline (Nielsen 2011).
  3. Twitter-RoBERTa — transformer polarity, tuned on ~124M tweets.
  4. NRC EmoLex    — emotion categories (8 emotions + pos/neg).
  5. Hedging       — sentence-level epistemic-uncertainty density.

Each runs on the full document; lexicons normalize for length. RoBERTa truncates
to max_length tokens (default 512), which covers ~80% of HN texts in full.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.hedging import score_hedging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# --------- VADER (lexicon, social-media-tuned) -----------------------------

def score_vader(texts: list[str]) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    rows = [sia.polarity_scores(t or "") for t in texts]
    out = pd.DataFrame(rows).rename(columns={
        "neg": "vader_neg", "neu": "vader_neu",
        "pos": "vader_pos", "compound": "vader_compound",
    })
    return out


# --------- AFINN (simpler lexicon polarity baseline) ------------------------

def score_afinn(texts: list[str]) -> pd.DataFrame:
    from afinn import Afinn
    af = Afinn(language="en")
    scores, per_word = [], []
    for t in texts:
        t = t or ""
        s = af.score(t)
        n = max(len(t.split()), 1)
        scores.append(s)
        per_word.append(s / n)
    return pd.DataFrame({"afinn_sum": scores, "afinn_per_word": per_word})


# --------- Twitter-RoBERTa (transformer polarity) ---------------------------

def score_transformer(texts: list[str], model_name: str, batch_size: int = 64,
                       max_length: int = 512) -> pd.DataFrame:
    """Returns prob(neg/neu/pos) and a signed score in [-1, 1]."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {model_name} on {device} (max_length={max_length})")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    probs = np.zeros((len(texts), 3), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="roberta", unit="batch"):
            batch = [t or "" for t in texts[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs[i:i + batch_size] = torch.softmax(logits, dim=-1).cpu().numpy()

    df = pd.DataFrame(probs, columns=["rob_neg", "rob_neu", "rob_pos"])
    df["rob_signed"] = df["rob_pos"] - df["rob_neg"]
    df["rob_label"] = np.array(["negative", "neutral", "positive"])[probs.argmax(axis=1)]
    return df


# --------- NRC EmoLex (emotion categories) ----------------------------------

NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
                "sadness", "surprise", "trust", "positive", "negative"]


def _ensure_nltk_punkt():
    """NRCLex needs NLTK's punkt tokenizer — download silently if missing."""
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            log.info(f"Downloading NLTK '{pkg}'...")
            nltk.download(pkg, quiet=True)


def score_nrc(texts: list[str]) -> pd.DataFrame:
    """
    For each text, compute affect_frequencies — share of affective vocabulary
    in each NRC emotion category. This is length-invariant: a 1500-word
    comment with 5 fear words and 30 emotion-bearing words scores fear=5/30,
    not fear=5/1500.
    """
    _ensure_nltk_punkt()
    from nrclex import NRCLex
    rows = []
    for t in tqdm(texts, desc="nrc", unit="doc"):
        nrc = NRCLex(t or "")
        freqs = nrc.affect_frequencies          # dict, normalized to sum to 1
        rows.append({f"nrc_{k}": float(freqs.get(k, 0.0)) for k in NRC_EMOTIONS})
    return pd.DataFrame(rows)


# --------- Orchestration ----------------------------------------------------

def add_sentiment(df: pd.DataFrame, cfg: dict, text_col: str = "text") -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    texts = df[text_col].fillna("").tolist()
    sent_cfg = cfg.get("sentiment", {})

    if sent_cfg.get("vader", True):
        log.info("Scoring VADER...")
        df = pd.concat([df, score_vader(texts)], axis=1)

    if sent_cfg.get("afinn", True):
        log.info("Scoring AFINN...")
        df = pd.concat([df, score_afinn(texts)], axis=1)

    if sent_cfg.get("nrc", True):
        log.info("Scoring NRC EmoLex...")
        df = pd.concat([df, score_nrc(texts)], axis=1)

    if sent_cfg.get("hedging", True):
        log.info("Scoring hedging...")
        df = pd.concat([df, score_hedging(texts)], axis=1)

    model_name = sent_cfg.get("transformer_model")
    if model_name:
        log.info("Scoring transformer...")
        df = pd.concat([df, score_transformer(
            texts, model_name,
            max_length=sent_cfg.get("transformer_max_length", 512),
            batch_size=sent_cfg.get("transformer_batch_size", 64),
        )], axis=1)

    if "vader_compound" in df and "rob_signed" in df:
        df["sent_agree_vader_rob"] = np.sign(df["vader_compound"]) == np.sign(df["rob_signed"])
    if "vader_compound" in df and "afinn_per_word" in df:
        df["sent_agree_vader_afinn"] = np.sign(df["vader_compound"]) == np.sign(df["afinn_per_word"])

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Cleaned parquet from preprocess.py")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config()

    df = pd.read_parquet(args.input)
    log.info(f"Loaded {len(df):,} rows")
    df = add_sentiment(df, cfg)

    out = Path(args.output) if args.output else Path(args.input).with_name(
        Path(args.input).stem + "_sent.parquet"
    )
    df.to_parquet(out, index=False)
    log.info(f"Wrote -> {out}")


if __name__ == "__main__":
    main()

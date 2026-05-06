"""
Topic modeling with BERTopic.

We fit on the training partition only (temporal split, see split.py) to prevent
leakage, then transform the rest. Topic IDs and embeddings are persisted so
downstream sentiment-by-topic analysis is reproducible.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def fit_topics(df: pd.DataFrame, cfg: dict, text_col: str = "text",
                fit_mask: np.ndarray | None = None):
    """Fit BERTopic on rows where fit_mask is True (or all rows if None)."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    texts = df[text_col].fillna("").tolist()
    if fit_mask is None:
        fit_mask = np.ones(len(texts), dtype=bool)

    log.info(f"Embedding {fit_mask.sum():,} fit-set docs (of {len(texts):,} total)")
    embedder = SentenceTransformer(cfg["topics"]["embedding_model"])
    embeddings = embedder.encode(texts, show_progress_bar=True,
                                  batch_size=128, convert_to_numpy=True,
                                  device="cuda" if _cuda_available() else "cpu")

    umap_model = UMAP(
        n_neighbors=cfg["topics"]["n_neighbors"],
        n_components=cfg["topics"]["n_components"],
        min_dist=0.0, metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg["topics"]["min_topic_size"],
        metric="euclidean", cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2))

    log.info("Fitting BERTopic on training partition...")
    fit_texts = [t for t, m in zip(texts, fit_mask) if m]
    fit_embs = embeddings[fit_mask]

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topic_model.fit(fit_texts, embeddings=fit_embs)

    log.info("Transforming all docs...")
    topics, _ = topic_model.transform(texts, embeddings=embeddings)

    df = df.copy()
    df["topic"] = topics
    return df, topic_model, embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-out", default=None,
                        help="Directory to save BERTopic model")
    args = parser.parse_args()
    cfg = load_config()

    df = pd.read_parquet(args.input)
    log.info(f"Loaded {len(df):,} rows from {args.input}")

    # Default: fit on first 70% by date (training partition)
    df = df.sort_values("created_utc").reset_index(drop=True)
    cutoff = int(len(df) * 0.70)
    fit_mask = np.zeros(len(df), dtype=bool)
    fit_mask[:cutoff] = True
    log.info(f"Fitting on {fit_mask.sum():,} training rows (first 70% by time)")

    df_out, model, embs = fit_topics(df, cfg, fit_mask=fit_mask)

    out = Path(args.output) if args.output else Path(args.input).with_name(
        Path(args.input).stem + "_topics.parquet"
    )
    df_out.to_parquet(out, index=False)
    log.info(f"Wrote -> {out}")

    if args.model_out:
        Path(args.model_out).mkdir(parents=True, exist_ok=True)
        model.save(args.model_out, serialization="safetensors",
                   save_ctfidf=True, save_embedding_model=False)
        np.save(Path(args.model_out) / "embeddings.npy", embs)
        log.info(f"Saved BERTopic model -> {args.model_out}")


if __name__ == "__main__":
    main()

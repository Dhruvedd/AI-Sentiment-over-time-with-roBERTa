"""
Text and metadata preprocessing.

Inputs : raw posts parquet from src.collect.
Outputs: cleaned/feature-engineered parquet ready for sentiment + topic modeling.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import emoji
import numpy as np
import pandas as pd

from src.collect import sample_min_score, sample_top_n_per_day
from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
DELETED_TOKENS = {"[deleted]", "[removed]", ""}


def clean_text(s: str | None) -> str:
    if not isinstance(s, str) or s.strip() in DELETED_TOKENS:
        return ""
    s = URL_RE.sub(" ", s)
    s = emoji.replace_emoji(s, replace=" ")
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def build_text(row: pd.Series) -> str:
    """Combine title + selftext into a single field for sentiment / topic models."""
    title = clean_text(row.get("title"))
    body = clean_text(row.get("selftext"))
    if body:
        return f"{title}. {body}"
    return title


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df.apply(build_text, axis=1)
    df["title_clean"] = df["title"].map(clean_text)
    df["title_len_chars"] = df["title_clean"].str.len()
    df["title_len_words"] = df["title_clean"].str.split().str.len().fillna(0).astype(int)
    df["has_question"] = df["title_clean"].str.contains(r"\?", regex=True, na=False)
    df["has_exclaim"] = df["title_clean"].str.contains(r"!", regex=True, na=False)
    df["has_image"] = df["url"].fillna("").str.contains(
        r"\.(jpg|jpeg|png|gif|webp)$|i\.redd\.it|imgur", case=False, regex=True
    )
    df["has_link"] = (~df["is_self"].astype(bool)) & (~df["has_image"])
    # Temporal features
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df["date"] = df["created_utc"].dt.date
    df["year"] = df["created_utc"].dt.year
    df["month"] = df["created_utc"].dt.month
    df["weekday"] = df["created_utc"].dt.weekday
    df["hour"] = df["created_utc"].dt.hour
    df["is_weekend"] = df["weekday"] >= 5
    # Targets
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0).astype(int)
    df["log_score"] = np.log1p(df["score"].clip(lower=0))
    return df


def filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    mask = (df["created_utc"] >= start_ts) & (df["created_utc"] < end_ts)
    return df.loc[mask].reset_index(drop=True)


def drop_dead(df: pd.DataFrame) -> pd.DataFrame:
    """Drop deleted/removed posts and posts with empty text."""
    keep = (df["text"].str.len() > 0) & (df["author"].fillna("").str.lower() != "[deleted]")
    return df.loc[keep].reset_index(drop=True)


def run(input_path: Path, out_dir: Path, cfg: dict) -> dict[str, Path]:
    log.info(f"Loading {input_path}")
    df = pd.read_parquet(input_path)
    log.info(f"  {len(df):,} raw posts")

    df = filter_window(df, cfg["date_range"]["start"], cfg["date_range"]["end"])
    log.info(f"  {len(df):,} after date window")

    df = add_features(df)
    df = drop_dead(df)
    log.info(f"  {len(df):,} after dropping dead/empty")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {}

    full_path = out_dir / f"{input_path.stem}_clean.parquet"
    df.to_parquet(full_path, index=False)
    out_paths["full"] = full_path
    log.info(f"Wrote full clean -> {full_path}")

    top_df = sample_top_n_per_day(df, n=cfg["sampling"]["top_n_per_day"])
    top_path = out_dir / f"{input_path.stem}_topN.parquet"
    top_df.to_parquet(top_path, index=False)
    out_paths["top_n"] = top_path
    log.info(f"  top-{cfg['sampling']['top_n_per_day']}/day sample: {len(top_df):,} -> {top_path}")

    thr_df = sample_min_score(df, threshold=cfg["sampling"]["min_score_threshold"])
    thr_path = out_dir / f"{input_path.stem}_threshold.parquet"
    thr_df.to_parquet(thr_path, index=False)
    out_paths["threshold"] = thr_path
    log.info(f"  score>={cfg['sampling']['min_score_threshold']} sample: {len(thr_df):,} -> {thr_path}")

    return out_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw parquet from src.collect")
    args = parser.parse_args()
    cfg = load_config()
    run(Path(args.input), Path(cfg["paths"]["processed_dir"]), cfg)


if __name__ == "__main__":
    main()

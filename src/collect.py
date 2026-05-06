"""
Hacker News data collection via BigQuery.

Pulls AI-related stories and comments from the public dataset
`bigquery-public-data.hacker_news.full` and writes them to parquet.

Authentication
--------------
You need a GCP project with the BigQuery API enabled (free sandbox is fine).
After installing the gcloud CLI:

    gcloud auth application-default login

The default Python client picks up these credentials automatically.

Run
---
    python -m src.collect --project YOUR_PROJECT_ID
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Keywords used to filter HN items as "Generative-AI-related." Scoped
# deliberately to text/conversational GenAI: LLMs, chatbots, foundation
# models, and the surrounding policy discourse. We exclude broad terms
# ("machine learning", "neural network", "deep learning") and the dedicated
# image/video generation domain (Midjourney, DALL-E, Stable Diffusion) as
# out-of-scope. Word boundaries avoid false positives.
AI_KEYWORDS = [
    # --- OpenAI family ---
    r"chatgpt",
    r"openai",
    r"gpt-?[2345](?:\.5)?",        # GPT-2/3/3.5/4/4.5/5 with or without dash
    r"gpt-?4o",                     # GPT-4o
    r"\bo[13]\b",                   # OpenAI o1, o3 reasoning models

    # --- Anthropic ---
    r"anthropic",
    r"\bclaude\b",

    # --- Google ---
    r"\bgemini\b",
    r"\bgemma\b",
    r"\bbard\b",
    r"google ai",
    r"google deepmind",
    r"\bdeepmind\b",

    # --- Meta ---
    r"\bllama\b",
    r"meta ai",

    # --- Other major LLM families ---
    r"\bmistral\b",
    r"\bmixtral\b",
    r"\bdeepseek\b",
    r"\bqwen\b",
    r"\bgrok\b",
    r"\bphi-?[234]\b",              # Microsoft Phi models

    # --- LLM-specific concepts and techniques ---
    r"large language model",
    r"\bllms?\b",
    r"generative ai",
    r"\bgenai\b",
    r"foundation model",
    r"prompt engineer",
    r"fine-?tun(?:e|ing|ed)",
    r"\brlhf\b",
    r"chain[- ]of[- ]thought",
    r"mixture of experts",
    r"retrieval[- ]augmented",

    # --- Coding/productivity assistants (LLM-based) ---
    r"\bcopilot\b",

    # --- Policy / discourse ---
    r"\bagi\b",
    r"ai safety",
    r"ai alignment",
    r"ai ethics",
    r"ai regulation",
    r"artificial intelligence",
]

KEYWORD_REGEX = r"\b(" + "|".join(AI_KEYWORDS) + r")\b"


def build_query(start: str, end: str) -> str:
    """Construct the BigQuery SQL for the given date window."""
    return f"""
    SELECT
      id,
      type,
      `by`         AS author,
      timestamp    AS created_utc,
      title,
      text,
      url,
      score,
      descendants  AS num_comments,
      parent,
      ranking
    FROM `bigquery-public-data.hacker_news.full`
    WHERE timestamp BETWEEN TIMESTAMP('{start}') AND TIMESTAMP('{end}')
      AND (
        REGEXP_CONTAINS(LOWER(COALESCE(text,  '')), r'{KEYWORD_REGEX}')
        OR
        REGEXP_CONTAINS(LOWER(COALESCE(title, '')), r'{KEYWORD_REGEX}')
      )
      AND (deleted IS NULL OR deleted = FALSE)
      AND (dead    IS NULL OR dead    = FALSE)
      AND type IN ('story', 'comment')
    ORDER BY timestamp
    """


def fetch(project_id: str, start: str, end: str) -> pd.DataFrame:
    """Run the query against BigQuery and return a DataFrame."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    sql = build_query(start, end)
    log.info(f"Querying HN BigQuery for window {start} -> {end}")
    log.info(f"  ~{len(AI_KEYWORDS)} keyword patterns")

    job = client.query(sql)
    log.info(f"  Job ID: {job.job_id}")
    df = job.to_dataframe(progress_bar_type="tqdm")
    log.info(f"  Returned {len(df):,} rows")
    log.info(f"  Bytes processed: {job.total_bytes_processed / 1e9:.2f} GB")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the HN dataframe compatible with the downstream pipeline, which
    expects columns: created_utc, title, selftext, score, num_comments,
    author, is_self, has_image (etc).

    HN items come in two flavors:
      - story: has title, may have text (Ask HN) and url, has score
      - comment: has text only, no title, no score, has parent
    We unify by mapping `text` -> `selftext` and treating comment titles as "".
    """
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df["title"]    = df["title"].fillna("")
    df["selftext"] = df["text"].fillna("")
    df["score"]    = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0).astype(int)
    df["author"]   = df["author"].fillna("")
    df["url"]      = df["url"].fillna("")
    # Synthetic flags for downstream feature engineering compatibility
    df["is_self"]    = df["url"] == ""
    df["over_18"]    = False
    df["spoiler"]    = False
    df["link_flair_text"] = df["type"]   # repurpose: 'story' or 'comment'
    df["subreddit"]  = "hackernews"
    df["permalink"]  = "https://news.ycombinator.com/item?id=" + df["id"].astype(str)
    df = df.drop_duplicates(subset="id").sort_values("created_utc").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True,
                        help="Your GCP project ID (e.g. cs439-sentiment-447802)")
    parser.add_argument("--out", default=None,
                        help="Output parquet path (default: data/raw/hn_ai.parquet)")
    args = parser.parse_args()
    cfg = load_config()

    out = Path(args.out) if args.out else Path(cfg["paths"]["raw_dir"]) / "hn_ai.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = fetch(args.project, cfg["date_range"]["start"], cfg["date_range"]["end"])
    df = normalize(df)
    df.to_parquet(out, index=False)
    log.info(f"Wrote {len(df):,} rows -> {out}")
    log.info(f"  stories : {(df['type']=='story').sum():,}")
    log.info(f"  comments: {(df['type']=='comment').sum():,}")


# Sampling utilities (kept here so preprocess.py imports stay valid)
def sample_top_n_per_day(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Keep top-N items per UTC day by score (stories) or num_comments (comments)."""
    df = df.copy()
    df["date"] = df["created_utc"].dt.date
    # For comments (score is 0), fall back to a length-based proxy so they don't
    # all look identically "low score" — short threads get downweighted.
    df["_rank_score"] = df["score"].where(df["score"] > 0, df["selftext"].str.len() / 100)
    return (df.sort_values(["date", "_rank_score"], ascending=[True, False])
              .groupby("date", group_keys=False)
              .head(n)
              .drop(columns=["_rank_score"])
              .reset_index(drop=True))


def sample_min_score(df: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
    """Keep stories with score >= threshold. Comments are kept regardless
    (they have no score on HN)."""
    keep = (df["type"] == "comment") | (df["score"] >= threshold)
    return df.loc[keep].reset_index(drop=True).copy()


if __name__ == "__main__":
    main()

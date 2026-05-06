"""
Export slim, shareable artifacts for collaborators.

Reads the full pipeline output and writes:
  - data/processed/hn_ai_slim.parquet  : per-document scores + topic + metadata,
                                          full text dropped to keep size small.
  - results/topic_info.csv             : BERTopic topic_info (id, count, keywords).

These two files are what the analysis notebook depends on. Tracking them in git
lets a fresh clone produce all figures and tables without rerunning the
~50-minute pipeline. The full parquet and BERTopic model remain gitignored.

Run after the main pipeline:
    python -m src.export_share
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Columns to drop from the slim parquet — they're large and the notebook
# doesn't need them for plotting / aggregation. The friend can still see what
# a post was about via `title` (truncated).
HEAVY_COLS = ["text", "selftext", "title_clean", "url", "permalink"]
TITLE_MAX = 200  # truncate title to this many chars


def export_slim_parquet(proc_dir: Path) -> Path:
    src = proc_dir / "hn_ai_topN_sent_topics.parquet"
    dst = proc_dir / "hn_ai_slim.parquet"
    df = pd.read_parquet(src)
    drop = [c for c in HEAVY_COLS if c in df.columns]
    slim = df.drop(columns=drop).copy()
    if "title" in slim.columns:
        slim["title"] = slim["title"].fillna("").astype(str).str[:TITLE_MAX]
    slim.to_parquet(dst, index=False)
    log.info(f"Slim parquet -> {dst} ({len(slim):,} rows × {len(slim.columns)} cols)")
    log.info(f"  dropped: {drop}")
    return dst


def export_topic_info(model_dir: Path, results_dir: Path) -> Path:
    from bertopic import BERTopic
    m = BERTopic.load(str(model_dir))
    info = m.get_topic_info()
    # Representation column is a list of words — flatten to a comma-joined string for CSV.
    info["Representation"] = info["Representation"].apply(
        lambda r: ", ".join(r) if isinstance(r, list) else str(r)
    )
    # Drop the heavy Representative_Docs column if present
    if "Representative_Docs" in info.columns:
        info = info.drop(columns=["Representative_Docs"])
    dst = results_dir / "topic_info.csv"
    info.to_csv(dst, index=False)
    log.info(f"Topic info -> {dst} ({len(info)} topics)")
    return dst


def main():
    cfg = load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    results = Path(cfg["paths"]["results_dir"])
    model_dir = PROJECT_ROOT / "experiments" / "bertopic_hn"

    export_slim_parquet(proc)
    export_topic_info(model_dir, results)
    log.info("Done.")


if __name__ == "__main__":
    main()

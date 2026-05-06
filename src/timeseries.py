"""
Time-series aggregation and change-point detection on sentiment.

This is where the project's core question gets answered: how has public
sentiment toward AI evolved across our window, and at which points does
the sentiment regime shift?
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


def aggregate(df: pd.DataFrame, freq: str = "W",
               score_col: str = "rob_signed") -> pd.DataFrame:
    """Aggregate sentiment to weekly (or other) bins. freq follows pandas offsets."""
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df = df.set_index("created_utc")
    agg = df[score_col].resample(freq).agg(["mean", "median", "std", "count"])
    agg = agg.rename(columns=lambda c: f"{score_col}_{c}")
    agg.columns = [c.replace(f"{score_col}_", "sent_") for c in agg.columns]
    return agg.reset_index()


def aggregate_by_topic(df: pd.DataFrame, freq: str = "W",
                        score_col: str = "rob_signed") -> pd.DataFrame:
    """Per-topic time series — for the topic-stratified analysis."""
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    out = (df.set_index("created_utc")
             .groupby("topic")[score_col]
             .resample(freq)
             .agg(["mean", "count"])
             .reset_index())
    return out


def detect_changepoints(series: np.ndarray, n_bkps: int | None = None,
                          pen: float | None = 5.0) -> list[int]:
    """
    Use Pelt with RBF cost. Returns list of breakpoint indices (excluding final).
    Pass either n_bkps (exact count) or pen (penalty for auto-selection).
    """
    import ruptures as rpt
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 10:
        return []
    algo = rpt.Pelt(model="rbf").fit(s.reshape(-1, 1))
    if n_bkps is not None:
        return algo.predict(n_bkps=n_bkps)[:-1]
    return algo.predict(pen=pen)[:-1]


def event_window_test(df: pd.DataFrame, event_date: str, window_days: int = 30,
                       score_col: str = "rob_signed") -> dict:
    """
    Welch's t-test comparing sentiment in [event - window, event) vs [event, event + window].
    Returns mean, n, t-statistic, p-value.
    """
    from scipy import stats
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    e = pd.Timestamp(event_date, tz="UTC")
    pre = df[(df["created_utc"] >= e - pd.Timedelta(days=window_days))
             & (df["created_utc"] <  e)][score_col].dropna()
    post = df[(df["created_utc"] >= e)
              & (df["created_utc"] <  e + pd.Timedelta(days=window_days))][score_col].dropna()
    if len(pre) < 5 or len(post) < 5:
        return {"event": event_date, "n_pre": len(pre), "n_post": len(post),
                "mean_pre": np.nan, "mean_post": np.nan, "delta": np.nan,
                "t": np.nan, "p": np.nan}
    t, p = stats.ttest_ind(post, pre, equal_var=False)
    return {
        "event": event_date,
        "n_pre": len(pre), "n_post": len(post),
        "mean_pre": float(pre.mean()), "mean_post": float(post.mean()),
        "delta": float(post.mean() - pre.mean()),
        "t": float(t), "p": float(p),
    }


def run_event_battery(df: pd.DataFrame, cfg: dict, score_col: str = "rob_signed",
                       window_days: int = 30) -> pd.DataFrame:
    rows = []
    for ev in cfg.get("events", []):
        r = event_window_test(df, ev["date"], window_days=window_days, score_col=score_col)
        r["label"] = ev["label"]
        rows.append(r)
    out = pd.DataFrame(rows)
    # Bonferroni correction, since we test multiple events
    if len(out):
        out["p_bonferroni"] = (out["p"] * len(out)).clip(upper=1.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Sentiment-scored parquet")
    parser.add_argument("--score-col", default="rob_signed")
    parser.add_argument("--freq", default="W")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    cfg = load_config()

    df = pd.read_parquet(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Aggregating overall...")
    agg = aggregate(df, freq=args.freq, score_col=args.score_col)
    agg.to_csv(out_dir / f"sentiment_{args.freq}.csv", index=False)

    log.info("Detecting change points...")
    cps = detect_changepoints(agg["sent_mean"].values, pen=5.0)
    cp_dates = agg.iloc[cps]["created_utc"].astype(str).tolist() if cps else []
    pd.DataFrame({"changepoint_date": cp_dates}).to_csv(
        out_dir / "changepoints.csv", index=False
    )
    log.info(f"  found {len(cps)} change points: {cp_dates}")

    log.info("Running event battery...")
    events = run_event_battery(df, cfg, score_col=args.score_col)
    events.to_csv(out_dir / "event_tests.csv", index=False)

    if "topic" in df.columns:
        log.info("Aggregating per topic...")
        per_topic = aggregate_by_topic(df, freq=args.freq, score_col=args.score_col)
        per_topic.to_csv(out_dir / f"sentiment_by_topic_{args.freq}.csv", index=False)

    log.info(f"All outputs -> {out_dir}")


if __name__ == "__main__":
    main()

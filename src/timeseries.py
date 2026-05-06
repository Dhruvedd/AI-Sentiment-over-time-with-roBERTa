"""
Time-series aggregation, change-point detection, event tests, and inter-metric
correlations across all sentiment / emotion / uncertainty dimensions.

What this module computes
-------------------------
For every available metric in (polarity, NRC emotions, hedging) we:
  1. Aggregate to a chosen frequency (weekly by default), overall and per topic.
  2. Detect change points with PELT (RBF cost) at one or more penalties.
  3. Run a Welch t-test in a ±N-day window around each named event.
  4. Compute pairwise Pearson + Spearman correlations across metrics.

Outputs (in `results/`)
-----------------------
  - sentiment_{freq}.csv          : wide table, one column per metric/stat
  - changepoints.csv              : long, (metric, penalty, breakpoint_date)
  - event_tests.csv               : long, (metric, event, label, ..., p, p_bonferroni)
  - sentiment_by_topic_{freq}.csv : long, (topic, metric, week, mean, count)
  - correlations_pearson.csv      : square matrix
  - correlations_spearman.csv     : square matrix

The single-metric helpers (`aggregate`, `detect_changepoints`, `run_event_battery`)
are kept for backward compatibility with notebooks.
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


# Default metrics — only those that exist in the input parquet are actually used.
DEFAULT_METRICS = [
    # Polarity
    "rob_signed", "vader_compound", "afinn_per_word",
    # NRC emotion categories (length-normalized affect frequencies)
    "nrc_positive", "nrc_negative",
    "nrc_anger", "nrc_anticipation", "nrc_disgust", "nrc_fear",
    "nrc_joy", "nrc_sadness", "nrc_surprise", "nrc_trust",
    # Uncertainty
    "hedge_density",
]


# ---------- Single-metric helpers (kept for notebook compatibility) ----------

def aggregate(df: pd.DataFrame, freq: str = "W",
               score_col: str = "rob_signed") -> pd.DataFrame:
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df = df.set_index("created_utc")
    agg = df[score_col].resample(freq).agg(["mean", "median", "std", "count"])
    agg.columns = [f"sent_{c}" for c in agg.columns]
    return agg.reset_index()


def aggregate_by_topic(df: pd.DataFrame, freq: str = "W",
                        score_col: str = "rob_signed") -> pd.DataFrame:
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    return (df.set_index("created_utc")
              .groupby("topic")[score_col]
              .resample(freq)
              .agg(["mean", "count"])
              .reset_index())


def detect_changepoints(series: np.ndarray, n_bkps: int | None = None,
                          pen: float | None = 5.0) -> list[int]:
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
    if len(out):
        out["p_bonferroni"] = (out["p"] * len(out)).clip(upper=1.0)
    return out


# ---------- Multi-metric extensions ----------------------------------------

def _available(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    """Filter to metrics that actually exist as columns."""
    return [m for m in metrics if m in df.columns]


def aggregate_multi(df: pd.DataFrame, freq: str = "W",
                     metrics: list[str] | None = None) -> pd.DataFrame:
    """Wide aggregation: one row per `freq` bin, one column per (metric, stat)."""
    metrics = _available(df, metrics or DEFAULT_METRICS)
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df = df.set_index("created_utc")
    agg = df[metrics].resample(freq).agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


def aggregate_by_topic_multi(df: pd.DataFrame, freq: str = "W",
                              metrics: list[str] | None = None) -> pd.DataFrame:
    """Long format: (topic, metric, week, mean, count)."""
    metrics = _available(df, metrics or DEFAULT_METRICS)
    if "topic" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    rows = []
    for m in metrics:
        agg = (df.set_index("created_utc")
                 .groupby("topic")[m]
                 .resample(freq)
                 .agg(["mean", "count"])
                 .reset_index()
                 .rename(columns={"mean": "value"}))
        agg["metric"] = m
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def detect_changepoints_multi(wide_agg: pd.DataFrame,
                                metrics: list[str] | None = None,
                                penalties: list[float] = (3.0, 5.0, 10.0)) -> pd.DataFrame:
    """
    Run PELT on each metric at multiple penalties (lower = more change points).
    Reporting at multiple penalties lets us distinguish high-confidence breaks
    (still detected at high pen) from speculative ones.
    """
    metrics = metrics or DEFAULT_METRICS
    out = []
    for m in metrics:
        col = f"{m}_mean"
        if col not in wide_agg.columns:
            continue
        for pen in penalties:
            cps = detect_changepoints(wide_agg[col].values, pen=pen)
            for cp in cps:
                out.append({
                    "metric": m,
                    "penalty": pen,
                    "breakpoint_index": int(cp),
                    "breakpoint_date": wide_agg.iloc[cp]["created_utc"],
                })
    return pd.DataFrame(out)


def run_event_battery_multi(df: pd.DataFrame, cfg: dict,
                              metrics: list[str] | None = None,
                              window_days: int = 30) -> pd.DataFrame:
    """Long format: (metric, event, label, n_pre, n_post, mean_pre, mean_post, delta, t, p, p_bonferroni)."""
    metrics = _available(df, metrics or DEFAULT_METRICS)
    rows = []
    for m in metrics:
        for ev in cfg.get("events", []):
            r = event_window_test(df, ev["date"], window_days=window_days, score_col=m)
            r["label"]  = ev["label"]
            r["metric"] = m
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        # Bonferroni correction over the full grid (events × metrics)
        n_tests = out["p"].notna().sum()
        out["p_bonferroni"] = (out["p"] * n_tests).clip(upper=1.0)
    return out


def metric_correlations(df: pd.DataFrame,
                          metrics: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise correlations across metrics at the document level."""
    metrics = _available(df, metrics or DEFAULT_METRICS)
    sub = df[metrics].dropna(how="any")
    return sub.corr(method="pearson"), sub.corr(method="spearman")


# ---------- CLI ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Sentiment-scored parquet")
    parser.add_argument("--freq", default="W",
                        help="Resample frequency (W=weekly, ME=monthly)")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Subset of metrics; defaults to all available")
    parser.add_argument("--penalties", nargs="*", type=float, default=[3.0, 5.0, 10.0])
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    cfg = load_config()

    df = pd.read_parquet(args.input)
    log.info(f"Loaded {len(df):,} rows from {args.input}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = args.metrics or DEFAULT_METRICS
    available = _available(df, metrics)
    log.info(f"Metrics available: {len(available)}/{len(metrics)} -> {available}")

    log.info("Aggregating overall...")
    wide = aggregate_multi(df, freq=args.freq, metrics=available)
    wide.to_csv(out_dir / f"sentiment_{args.freq}.csv", index=False)
    log.info(f"  wide aggregation: {len(wide)} bins, {len(wide.columns)} cols")

    log.info("Detecting change points across metrics × penalties...")
    cps = detect_changepoints_multi(wide, metrics=available, penalties=args.penalties)
    cps.to_csv(out_dir / "changepoints.csv", index=False)
    log.info(f"  found {len(cps)} change-point detections "
             f"({cps['metric'].nunique() if len(cps) else 0} metrics had any)")

    log.info("Running event battery across metrics × events...")
    ev = run_event_battery_multi(df, cfg, metrics=available, window_days=args.window_days)
    ev.to_csv(out_dir / "event_tests.csv", index=False)
    sig = ev[ev["p_bonferroni"] < 0.05]
    log.info(f"  ran {len(ev)} tests; {len(sig)} significant after Bonferroni")
    if len(sig):
        for _, r in sig.iterrows():
            log.info(f"    {r['metric']:<20s} {r['event']} ({r['label']}): "
                     f"delta={r['delta']:+.4f}, p_bonf={r['p_bonferroni']:.4g}")

    if "topic" in df.columns:
        log.info("Aggregating per topic across metrics...")
        per_topic = aggregate_by_topic_multi(df, freq=args.freq, metrics=available)
        per_topic.to_csv(out_dir / f"sentiment_by_topic_{args.freq}.csv", index=False)
        log.info(f"  {len(per_topic)} rows, {per_topic['topic'].nunique()} topics, "
                 f"{per_topic['metric'].nunique()} metrics")

    log.info("Computing inter-metric correlations...")
    pearson, spearman = metric_correlations(df, metrics=available)
    pearson.to_csv(out_dir / "correlations_pearson.csv")
    spearman.to_csv(out_dir / "correlations_spearman.csv")
    log.info(f"  saved {pearson.shape[0]}x{pearson.shape[0]} correlation matrices")

    log.info(f"All outputs -> {out_dir}")


if __name__ == "__main__":
    main()

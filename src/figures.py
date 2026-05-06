"""
Generate all paper figures from the pipeline outputs.

Reads from:
  - data/processed/hn_ai_topN_sent_topics.parquet  (per-document scores + topics)
  - results/sentiment_W.csv                         (wide weekly aggregation)
  - results/event_tests.csv                         (long event battery)
  - results/changepoints.csv                        (long change-points)
  - results/sentiment_by_topic_W.csv                (long per-topic-week)
  - results/correlations_pearson.csv                (square matrix)
  - experiments/bertopic_hn/                        (BERTopic model for topic labels)

Writes PNG files into:  results/figures/
Run:                    python -m src.figures
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Publication-style defaults
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
})

NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear",
                "joy", "sadness", "surprise", "trust"]

POLARITY_METRICS = [("rob_signed", "RoBERTa", "tab:blue"),
                     ("vader_compound", "VADER", "tab:orange"),
                     ("afinn_per_word", "AFINN", "tab:green")]


# --------- Helpers --------------------------------------------------------

def annotate_events(ax, events, max_labels=None):
    """Draw vertical event lines + rotated labels above the axis."""
    ymin, ymax = ax.get_ylim()
    label_y = ymax + (ymax - ymin) * 0.02
    for i, ev in enumerate(events):
        if max_labels and i >= max_labels:
            break
        date = pd.Timestamp(ev["date"], tz="UTC")
        ax.axvline(date, color="firebrick", alpha=0.35, linestyle="--", linewidth=0.7, zorder=0)
        ax.text(date, label_y, ev["label"], rotation=60, fontsize=7,
                ha="left", va="bottom", alpha=0.75, color="firebrick")


def add_changepoints(ax, cps_df, metric, penalty=10.0, color="seagreen"):
    """Vertical lines at change-point dates for a given metric/penalty."""
    sub = cps_df[(cps_df["metric"] == metric) & (cps_df["penalty"] == penalty)]
    for _, r in sub.iterrows():
        ax.axvline(pd.Timestamp(r["breakpoint_date"]), color=color,
                   alpha=0.75, linestyle=":", linewidth=1.4, zorder=1)


def topic_labels_from_model(model_path: Path, n: int = 25) -> dict[int, str]:
    """Map topic_id -> short human-readable label using top-3 words."""
    from bertopic import BERTopic
    m = BERTopic.load(str(model_path))
    info = m.get_topic_info().head(n)
    out = {}
    for _, row in info.iterrows():
        rep = row["Representation"][:3] if row["Representation"] else []
        out[int(row["Topic"])] = "/".join(rep) if rep else str(row["Topic"])
    return out


# --------- Figures --------------------------------------------------------

def fig01_headline(wide, events, cps, out):
    """Headline: RoBERTa weekly mean with event markers + change points."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    se = wide["rob_signed_std"] / np.sqrt(wide["rob_signed_count"])
    ax.plot(wide["created_utc"], wide["rob_signed_mean"],
            color="tab:blue", lw=1.6, label="weekly mean (RoBERTa signed)")
    ax.fill_between(wide["created_utc"],
                    wide["rob_signed_mean"] - se,
                    wide["rob_signed_mean"] + se, alpha=0.18, color="tab:blue", label="±1 SE")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(ax.get_ylim())  # freeze before annotation
    annotate_events(ax, events)
    add_changepoints(ax, cps, "rob_signed", penalty=10.0)
    ax.set_title("HN sentiment toward Generative AI, 2022–2026 (RoBERTa weekly mean)")
    ax.set_ylabel("Signed sentiment (pos − neg)")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig02_polarity_comparison(wide, events, cps, out):
    """Three polarity scorers side by side — shows VADER's saturation pathology."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, (col, label, color) in zip(axes, POLARITY_METRICS):
        mean_col = f"{col}_mean"
        if mean_col not in wide.columns:
            continue
        ax.plot(wide["created_utc"], wide[mean_col], color=color, lw=1.4)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(ax.get_ylim())
        annotate_events(ax, events) if ax is axes[0] else None
        add_changepoints(ax, cps, col, penalty=5.0)
        ax.set_ylabel(label)
        ax.set_title(f"{label}: weekly mean ({mean_col})", loc="left", fontsize=10)
    axes[0].set_title("Polarity scorers compared — VADER saturates positive on long-form text",
                       fontsize=12, loc="left")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig03_emotion_grid(wide, events, cps, out):
    """8-panel grid of NRC emotions over time."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()
    for ax, emo in zip(axes, NRC_EMOTIONS):
        col = f"nrc_{emo}_mean"
        if col not in wide.columns:
            ax.set_visible(False)
            continue
        ax.plot(wide["created_utc"], wide[col], color=f"C{NRC_EMOTIONS.index(emo)}", lw=1.3)
        ax.set_ylim(ax.get_ylim())
        annotate_events(ax, events, max_labels=3)
        add_changepoints(ax, cps, f"nrc_{emo}", penalty=5.0)
        ax.set_title(f"NRC {emo}", loc="left", fontsize=10)
    fig.suptitle("Emotion trajectories on HN AI discourse, weekly affect frequencies",
                  fontsize=13, y=1.00)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig04_hedging(wide, events, cps, out):
    """Hedge density over time — uncertainty trajectory."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(wide["created_utc"], wide["hedge_density_mean"],
            color="purple", lw=1.6, label="weekly mean hedge density")
    se = wide["hedge_density_std"] / np.sqrt(wide["hedge_density_count"])
    ax.fill_between(wide["created_utc"],
                    wide["hedge_density_mean"] - se,
                    wide["hedge_density_mean"] + se, alpha=0.18, color="purple")
    ax.set_ylim(ax.get_ylim())
    annotate_events(ax, events)
    add_changepoints(ax, cps, "hedge_density", penalty=10.0)
    ax.set_title("Epistemic uncertainty: fraction of HN sentences containing a hedge")
    ax.set_ylabel("Hedge density")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig05_event_forest(events_tests, out):
    """Forest plot of significant event × metric effect sizes."""
    df = events_tests.dropna(subset=["delta", "p_bonferroni"]).copy()
    sig = df[df["p_bonferroni"] < 0.10].sort_values("delta")
    if len(sig) == 0:
        log.warning("No Bonferroni-significant events; skipping fig05")
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(sig))))
    y = np.arange(len(sig))
    colors = ["firebrick" if r["delta"] < 0 else "seagreen" for _, r in sig.iterrows()]
    ax.barh(y, sig["delta"].values, color=colors, alpha=0.75)
    labels = [f"{r['label']}  ·  {r['metric']}" for _, r in sig.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Δ (post − pre, ±30-day window)")
    ax.set_title("Significant event-induced shifts (Bonferroni p < 0.10)")
    for i, (_, r) in enumerate(sig.iterrows()):
        ax.text(r["delta"], i, f" p={r['p_bonferroni']:.2g}",
                va="center", ha="left" if r["delta"] >= 0 else "right",
                fontsize=8, color="black")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig06_correlation_heatmap(pearson, out):
    """Inter-metric Pearson correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pearson, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Inter-metric document-level correlations (Pearson)")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig07_topic_emotion_heatmap(df, topic_labels, out, n_topics=15):
    """Top-N topics × NRC emotions: which topics elicit which emotions?"""
    df = df[df["topic"] != -1]
    sizes = df["topic"].value_counts().head(n_topics)
    keep_topics = sizes.index.tolist()
    sub = df[df["topic"].isin(keep_topics)]
    cols = [f"nrc_{e}" for e in NRC_EMOTIONS if f"nrc_{e}" in sub.columns]
    means = sub.groupby("topic")[cols].mean()
    means = means.reindex(keep_topics)
    means.index = [f"{t}: {topic_labels.get(t, t)}" for t in keep_topics]
    means.columns = [c.replace("nrc_", "") for c in means.columns]
    # Z-score within emotion (column) so we see relative emphasis
    z = (means - means.mean()) / means.std()
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(means))))
    sns.heatmap(z, ax=ax, cmap="RdBu_r", center=0, annot=means.values,
                fmt=".3f", annot_kws={"size": 7}, cbar_kws={"label": "z-score (within emotion)"})
    ax.set_title(f"Topic × emotion: which discourse areas elicit which feelings (top {len(means)} topics)")
    ax.set_xlabel("NRC emotion")
    ax.set_ylabel("Topic (id: top-3 keywords)")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig08_topic_polarity_trajectories(per_topic, topic_labels, out, n=8):
    """Top-N topics' polarity trajectories — which topics drove which shifts."""
    sub = per_topic[per_topic["metric"] == "rob_signed"].copy()
    sub = sub[sub["topic"] != -1]
    counts = sub.groupby("topic")["count"].sum().sort_values(ascending=False)
    top = counts.head(n).index.tolist()
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(top):
        s = sub[sub["topic"] == t].sort_values("created_utc")
        # Smooth a bit — 4-week rolling mean for legibility
        s["smoothed"] = s["value"].rolling(4, min_periods=1).mean()
        label = f"{t}: {topic_labels.get(t, t)}"
        ax.plot(pd.to_datetime(s["created_utc"], utc=True), s["smoothed"],
                lw=1.5, label=label, color=cmap(i))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Sentiment trajectory by topic (top {n}, RoBERTa, 4-week smoothed)")
    ax.set_ylabel("Signed sentiment")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig09_polarity_vs_hedging_scatter(df, out):
    """Document-level scatter: hedge_density vs RoBERTa signed polarity."""
    sub = df[(df["n_sentences"] >= 3)].sample(min(8000, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(sub["rob_signed"], sub["hedge_density"],
              gridsize=40, mincnt=1, cmap="viridis", bins="log")
    ax.set_xlabel("RoBERTa signed polarity")
    ax.set_ylabel("Hedge density")
    ax.axvline(0, color="white", lw=0.5)
    r = sub[["rob_signed", "hedge_density"]].corr().iloc[0, 1]
    ax.set_title(f"Hedging vs polarity (n≈{len(sub):,}, Pearson r = {r:.3f})")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --------- Driver --------------------------------------------------------

def main():
    cfg = load_config()
    project_root = Path(cfg["paths"]["raw_dir"]).parent.parent
    proc = Path(cfg["paths"]["processed_dir"])
    results = Path(cfg["paths"]["results_dir"])
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    df = pd.read_parquet(proc / "hn_ai_topN_sent_topics.parquet")
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    wide = pd.read_csv(results / "sentiment_W.csv", parse_dates=["created_utc"])
    wide["created_utc"] = pd.to_datetime(wide["created_utc"], utc=True)
    events_tests = pd.read_csv(results / "event_tests.csv")
    cps = pd.read_csv(results / "changepoints.csv", parse_dates=["breakpoint_date"])
    cps["breakpoint_date"] = pd.to_datetime(cps["breakpoint_date"], utc=True)
    per_topic = pd.read_csv(results / "sentiment_by_topic_W.csv",
                             parse_dates=["created_utc"])
    per_topic["created_utc"] = pd.to_datetime(per_topic["created_utc"], utc=True)
    pearson = pd.read_csv(results / "correlations_pearson.csv", index_col=0)

    events = cfg.get("events", [])
    log.info("Loading topic labels...")
    try:
        topic_labels = topic_labels_from_model(project_root / "experiments/bertopic_hn", n=30)
    except Exception as e:
        log.warning(f"Couldn't load BERTopic model: {e}; using numeric labels only")
        topic_labels = {}

    log.info("Building figures...")
    fig01_headline(wide, events, cps, figs / "01_headline_sentiment.png")
    fig02_polarity_comparison(wide, events, cps, figs / "02_polarity_comparison.png")
    fig03_emotion_grid(wide, events, cps, figs / "03_emotion_grid.png")
    fig04_hedging(wide, events, cps, figs / "04_hedging.png")
    fig05_event_forest(events_tests, figs / "05_event_forest.png")
    fig06_correlation_heatmap(pearson, figs / "06_correlation_heatmap.png")
    fig07_topic_emotion_heatmap(df, topic_labels, figs / "07_topic_emotion_heatmap.png")
    fig08_topic_polarity_trajectories(per_topic, topic_labels,
                                        figs / "08_topic_polarity_trajectories.png")
    fig09_polarity_vs_hedging_scatter(df, figs / "09_hedging_vs_polarity_scatter.png")

    log.info(f"All figures -> {figs}")
    for p in sorted(figs.glob("*.png")):
        log.info(f"  {p.name}")


if __name__ == "__main__":
    main()

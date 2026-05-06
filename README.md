# Public Sentiment Toward Generative AI on Hacker News (2022–2026)

CS439 final project. We measure how the Hacker News community's discussion of generative AI has evolved between January 2022 and April 2026 along five orthogonal dimensions: transformer-based polarity (RoBERTa), two lexicon-based polarity baselines (VADER + AFINN), eight NRC emotion categories, and sentence-level epistemic hedging. We anchor findings to ten named AI events (ChatGPT launch, GPT-4, OpenAI board crisis, DeepSeek R1, etc.) and to 78 BERTopic-derived discourse clusters.

> **Headline claim.** GenAI discourse on HN has shifted from speculative enthusiasm toward concrete, less-hedged, increasingly negative technical evaluation. Hedging fell ~50%, polarity drifted negative, anticipation/joy/surprise declined while trust held stable, and every Bonferroni-significant event-induced shift was negative.

For the full analysis, see `notebooks/analysis.ipynb`. For the report-author brief in plain English, see `REPORT_BRIEF.md`.

## Pipeline

```
collect.py ──► preprocess.py ──► sentiment.py ──► topics.py ──► timeseries.py ──► figures.py
   raw           cleaned + sampled    + 5 dimensions     + BERTopic     aggregates,           PNG figures
                                                                         change points,
                                                                         event tests,
                                                                         correlations
```

Each stage reads parquet, writes parquet (or CSV at the end). Stages are independent — rerun any one without redoing the rest.

| Stage | Input | Output | Wall time on RTX 4060 |
|---|---|---|---|
| `collect.py` | HN BigQuery | `data/raw/hn_ai.parquet` | 1-3 min |
| `preprocess.py` | raw parquet | `data/processed/hn_ai_*.parquet` (×3 samples) | ~6 min |
| `sentiment.py` | top-N parquet | `..._sent.parquet` | ~30 min |
| `topics.py` | sent parquet | `..._sent_topics.parquet` + BERTopic model | ~7-12 min |
| `timeseries.py` | sent_topics parquet | `results/*.csv` | ~1 min |
| `figures.py` | results CSVs | `results/figures/*.png` | ~30 s |

## Setup

### 1. Python environment

```powershell
cd C:\Projects\cs439-ai-sentiment
python -m venv .venv
.venv\Scripts\activate
```

### 1a. PyTorch with CUDA (GPU users — recommended)

`pip install torch` installs the CPU-only build silently on Windows. Install the CUDA wheel first:

```powershell
nvidia-smi    # confirm driver-supported CUDA version
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 1b. Remaining requirements

```powershell
pip install -r requirements.txt
```

### 2. Google Cloud / BigQuery (one-time, free)

The Hacker News corpus comes from `bigquery-public-data.hacker_news.full`, a public dataset on Google BigQuery. The free sandbox tier (no billing required) gives 1 TB of query processing per month — far more than this project needs.

1. Create a free GCP account at [console.cloud.google.com](https://console.cloud.google.com) and a project.
2. Open the BigQuery console — accept the sandbox prompt.
3. Install the gcloud CLI: <https://cloud.google.com/sdk/docs/install>
4. Authenticate the Python clients:

   ```powershell
   gcloud auth application-default login
   ```

5. Copy `.env.example` to `.env` and fill in `GCP_PROJECT_ID=your-project-id`. Or pass `--project YOUR_PROJECT_ID` on the CLI for each invocation.

## Run the pipeline

```powershell
python -m src.collect    --project YOUR_PROJECT_ID
python -m src.preprocess --input data/raw/hn_ai.parquet
python -m src.sentiment  --input data/processed/hn_ai_topN.parquet
python -m src.topics     --input data/processed/hn_ai_topN_sent.parquet --model-out experiments/bertopic_hn
python -m src.timeseries --input data/processed/hn_ai_topN_sent_topics.parquet
python -m src.figures
```

Or chained as one command:

```powershell
python -m src.collect    --project YOUR_PROJECT_ID ; `
python -m src.preprocess --input data/raw/hn_ai.parquet ; `
python -m src.sentiment  --input data/processed/hn_ai_topN.parquet ; `
python -m src.topics     --input data/processed/hn_ai_topN_sent.parquet --model-out experiments/bertopic_hn ; `
python -m src.timeseries --input data/processed/hn_ai_topN_sent_topics.parquet ; `
python -m src.figures
```

## Five measurement dimensions

| # | Module | Tool | Output |
|---|---|---|---|
| 1 | RoBERTa polarity | `cardiffnlp/twitter-roberta-base-sentiment-latest` | signed score in [−1, +1] |
| 2 | VADER polarity | VADER (Hutto & Gilbert 2014) | compound score in [−1, +1] |
| 3 | AFINN polarity | AFINN-165 (Nielsen 2011) | per-word integer mean |
| 4 | NRC emotion | NRC EmoLex (Mohammad & Turney) | 8 emotion categories + pos/neg, length-normalized |
| 5 | Hedge density | custom 22-pattern lexicon, sentence-level | fraction of sentences with ≥1 hedge |

We picked these for **methodological diversity**: one transformer-based polarity scorer paired with two lexicon-based polarity baselines (the disagreement reveals VADER's saturation pathology on long-form text — a finding in its own right), one length-normalized emotion lexicon for richer affect dimensions, and a custom hedging signal that's empirically orthogonal to polarity.

## Project layout

```
cs439-ai-sentiment/
├── README.md                  # this file
├── REPORT_BRIEF.md            # plain-English brief for the report author
├── config.yaml                # date range, sampling, named events, model paths
├── requirements.txt
├── .env.example               # GCP project ID template
├── .gitignore
├── .gitattributes
├── data/
│   ├── raw/                   # raw HN parquet from BigQuery (gitignored)
│   └── processed/             # cleaned + sentiment-scored (gitignored)
├── experiments/               # BERTopic checkpoint (gitignored)
├── notebooks/
│   └── analysis.ipynb         # the only tracked notebook — figures + commentary
├── results/
│   ├── figures/               # 9 paper figures (PNG, tracked)
│   └── *.csv                  # aggregations, event tests, change points (gitignored)
└── src/
    ├── config.py              # config loader
    ├── collect.py             # HN BigQuery -> parquet
    ├── preprocess.py          # cleaning, feature engineering, sampling
    ├── sentiment.py           # VADER + AFINN + RoBERTa + NRC + hedging
    ├── hedging.py             # hedging lexicon + sentence-level scorer
    ├── topics.py              # BERTopic with temporal-split fit
    ├── timeseries.py          # multi-metric aggregation, change points, event tests
    └── figures.py             # generate all paper figures
```

## Key data and methodology notes

- **Corpus filter.** ~39 generative-AI keyword regexes covering models (GPT, Claude, Gemini, Llama, Mistral, DeepSeek, Gemma, Qwen, Grok, Phi), companies (OpenAI, Anthropic, Google AI, DeepMind, Meta AI), techniques (LLM, RAG, RLHF, fine-tuning, chain-of-thought), and policy discourse (AI safety/alignment/ethics/regulation, AGI). We deliberately **exclude** image-generation tools and broad pre-LLM ML terms.
- **HTML entity decoding.** HN's BigQuery export preserves HTML entities (`&#x27;`, `&quot;`) as literal strings. `preprocess.clean_text` decodes them before any scoring — without this, topic clusters get polluted with `x27` / `x2f` / `quot` tokens.
- **Length normalization.** RoBERTa truncates to 512 tokens (covers ~80% of HN texts in full). NRC uses `affect_frequencies` (share of affective vocabulary), which is length-invariant. Hedging is sentence-level density. AFINN reports per-word mean rather than sum.
- **Leakage prevention.** BERTopic is fit only on the first 70% of items by date; the last 30% receive topic assignments without contributing to the model.
- **Sampling robustness.** Two parallel samples — `top-30/day` (47k items, primary) and `score ≥ 20` (588k items, robustness check). Findings should hold on both.
- **Multiple-comparisons control.** 140 event tests (10 events × 14 metrics) get Bonferroni-corrected; 8 survive at p < 0.10.

## Reproducibility

The 9 figures in `results/figures/` were generated by the chained pipeline above with no manual editing. Re-running on a fresh machine should reproduce all results modulo small numerical noise (BERTopic uses random initialization with `random_state=42`).

## Working with the data without rerunning the pipeline

For collaborators (e.g., the report author) who want to make new figures or pull custom numbers without spending an hour rerunning sentiment scoring and topic modeling, the repo ships two slim shareable artifacts:

- **`data/processed/hn_ai_slim.parquet`** (~4 MB) — one row per HN item, 55 columns. Includes timestamp, author, truncated title, score, topic id, and **every** sentiment/emotion/hedging metric. Full text is dropped to keep the file small; if you need raw post text, rerun `src/collect.py` and `src/preprocess.py`.
- **`results/topic_info.csv`** — the BERTopic catalog: topic id, item count, top-10 representative keywords. Lets you label topic numbers without loading the BERTopic model.

These are produced by:

```powershell
python -m src.export_share
```

The analysis notebook (`notebooks/analysis.ipynb`) loads from these files only — clone the repo, install dependencies, open the notebook, and you can play with the data immediately. Section 13 of the notebook has copy-paste recipes for common slices (most-fearful posts, per-topic trajectories, custom event windows, etc.).

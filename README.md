# CS439 — Public Sentiment Toward AI on Hacker News (2022–2026)

How has sentiment toward AI within the Hacker News community evolved between January 2022 and April 2026, and which named events (ChatGPT launch, GPT-4, EU AI Act, etc.) coincide with statistically significant regime shifts?

**Data source:** Hacker News full archive via Google BigQuery — `bigquery-public-data.hacker_news.full`. Free public dataset, updated daily, contains every story and comment since 2006.

## Pipeline

```
collect.py  ──►  preprocess.py  ──►  sentiment.py  ──►  topics.py  ──►  timeseries.py
   raw            cleaned + sampled    + VADER/RoBERTa    + BERTopic    aggregates,
                                                                        change points,
                                                                        event tests
```

Each stage reads parquet, writes parquet. Stages are independent — rerun any one without redoing the rest.

## Setup

### 1. Python environment

```powershell
cd C:\Projects\cs439-ai-sentiment
python -m venv .venv
.venv\Scripts\activate
```

### 1a. PyTorch with CUDA (GPU users)

`pip install torch` installs the CPU-only build on Windows — silently. To use
your NVIDIA GPU, install torch separately *before* the rest of requirements,
matching your CUDA version:

```powershell
# Check your CUDA version (top right of `nvidia-smi` output)
nvidia-smi

# Then install the matching wheel. Common cases:
# CUDA 12.1+:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU):
pip install torch==2.3.0
```

Verify GPU is visible:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### 1b. Remaining requirements

```powershell
pip install -r requirements.txt
```

### 2. Google Cloud / BigQuery (one-time, free)

1. Create a free GCP account at [console.cloud.google.com](https://console.cloud.google.com).
2. Create a new project (e.g. `cs439-sentiment`). Note the **Project ID**.
3. Open the BigQuery console — accept the sandbox prompt (no billing required).
4. Install the gcloud CLI: <https://cloud.google.com/sdk/docs/install>
5. Authenticate locally:

   ```powershell
   gcloud auth application-default login
   ```

6. Copy `.env.example` to `.env` and set `GCP_PROJECT_ID=your-project-id`.

The BigQuery sandbox gives you 1 TB of query processing per month — far more than this project needs.

## Run the pipeline

```powershell
# 1. Collect AI-related HN stories + comments from BigQuery -> parquet
python -m src.collect --project YOUR_PROJECT_ID

# 2. Clean text, engineer features, produce two parallel samples
python -m src.preprocess --input data/raw/hn_ai.parquet

# 3. Score sentiment with both VADER and Twitter-RoBERTa
python -m src.sentiment --input data/processed/hn_ai_topN.parquet

# 4. Fit BERTopic on the training partition (first 70% by date) for leakage-free topics
python -m src.topics --input data/processed/hn_ai_topN_sent.parquet `
    --model-out experiments/bertopic_hn

# 5. Aggregate, detect change points, run event-window t-tests
python -m src.timeseries --input data/processed/hn_ai_topN_sent_topics.parquet
```

Then open the notebooks in `notebooks/` for the headline figures.

## Project layout

```
cs439-ai-sentiment/
├── config.yaml               # date range, sampling, named events
├── requirements.txt
├── .env.example              # GCP project ID template
├── data/
│   ├── raw/                  # raw HN parquet from BigQuery (gitignored)
│   └── processed/            # cleaned + sentiment-scored (gitignored)
├── experiments/              # model checkpoints (BERTopic, etc.)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_sentiment_over_time.ipynb
├── results/                  # CSVs and figures for the report
└── src/
    ├── config.py
    ├── collect.py            # HN BigQuery -> parquet
    ├── preprocess.py         # cleaning, feature engineering, sampling
    ├── sentiment.py          # VADER + Twitter-RoBERTa
    ├── topics.py             # BERTopic with temporal-split fit
    └── timeseries.py         # aggregation, change points, event tests
```

## Scoping decisions

- **Population.** Hacker News, technical-leaning community. We frame this explicitly as "sentiment within an engaged tech community" — not as a proxy for general public sentiment. This becomes a Limitations bullet in the report.
- **AI keyword filter.** ~25 patterns covering models, companies, technical terms, and policy concepts (see `AI_KEYWORDS` in `src/collect.py`). Tuned on the EDA query before final collection.
- **Item types.** Both stories and comments. Stories give us topical anchors; comments give us sentiment density. Reported separately and combined.
- **Sampling.** We keep top-N-per-day and score-threshold samples in parallel and report both for robustness.
- **Sentiment models.** Twitter-RoBERTa primary, VADER lexicon baseline. Inter-model agreement is itself an evaluation metric.

# Brief for the Report Author

This document gives you everything you need to write the final report. It's written assuming you haven't worked on the code side, so I've kept it in plain language. Where I had to use a technical term I explain it on first use. The report's required sections (Abstract, Introduction, Related Work, Methodology, Experiments & Results, Conclusion) and the rubric tell you the structure — this brief gives you the *content*.

The companion notebook is `notebooks/analysis.ipynb`. It contains the same figures plus the underlying data tables, viewable directly on GitHub.

---

## 1. The 30-second pitch

We measured how the **Hacker News** community has been talking about **generative AI** (ChatGPT, GPT-4, Claude, Gemini, etc.) every week from January 2022 through April 2026. We didn't just ask "is the sentiment positive or negative?" — we measured five different aspects of the discussion at the same time, then looked for shifts around major AI events (model launches, the OpenAI board crisis, etc.) and across different conversation topics.

The headline finding: **HN's stance on AI has cooled measurably**. Sentiment drifts negative, hedging drops by half, and excitement (joy, anticipation, surprise) all decline — while trust stays stable. Capability releases like GPT-4 specifically *reduce* hedging, meaning people stop speculating once they have something concrete to evaluate.

---

## 2. What is Hacker News and why use it?

**Hacker News** (news.ycombinator.com) is a discussion site run by the startup accelerator Y Combinator. Its audience is mostly software engineers, startup founders, and tech-adjacent professionals. People post links to articles or write text-only "Ask HN" threads, and others respond with threaded comments. It's known for long-form, often skeptical/analytical discussion.

We chose HN because:
- The **full archive is publicly available** through Google BigQuery (Google's data warehouse) at zero cost. No scraping, no API rate limits, no access friction.
- The audience is technically engaged and opinionated about AI — every model release, controversy, and policy debate gets discussed in depth.
- Comments are **long-form** (median ~248 words), giving sentiment models real text to work with rather than 280-character tweets.

**The honest limitation** (worth flagging in the Limitations section): HN is *not* representative of the general public. It's tech-leaning, English-speaking, and Western-skewed. We frame the project as measuring sentiment in "an engaged technical community" rather than claiming we speak for everyone.

---

## 3. What data we collected

From Google BigQuery's public Hacker News dataset (`bigquery-public-data.hacker_news.full`), we pulled every story or comment posted between **2022-01-01** and **2026-04-30** that mentions any of ~39 generative-AI keywords. The keyword list covers:

- **Model names**: ChatGPT, GPT-3/4/5, GPT-4o, Claude, Gemini (when qualified — see below), Llama, Mistral, DeepSeek, Gemma, Bard, Qwen, Grok, Phi, etc.
- **Companies**: OpenAI, Anthropic, Google AI, DeepMind, Meta AI
- **Concepts**: large language model, LLM, generative AI, foundation model, prompt engineering, fine-tuning, RLHF, chain of thought
- **Policy/discourse**: AI safety, AI alignment, AI ethics, AI regulation, AGI

We *deliberately excluded* image-generation tools (Midjourney, Stable Diffusion, DALL-E) and broad ML terms (machine learning, neural network, deep learning) because they cover different domains and weren't the focus.

| | Count |
|---|---|
| Total items collected | **656,980** |
| Stories (link or "Ask HN" posts) | 77,304 |
| Comments | 579,676 |
| Items in the **top-30/day** sample we analyze | **46,387** |
| Median post length | 248 words |
| 95th-percentile post length | 596 words |

**Why a top-30/day sample instead of everything?** The full corpus would take 10× longer to score with our transformer model and add little signal — sentiment of *visible* posts is what shapes community perception. We keep both samples and verify our findings replicate on the larger threshold sample (`score ≥ 20`, 587,900 items) for robustness.

**One technical detail worth mentioning** (helps with rubric's data-handling points): Hacker News stores text with HTML entities (`&#x27;` for apostrophe, `&quot;` for quote, etc.) preserved as literal strings. Our preprocessing decodes these before any analysis — without that step, topic clusters get polluted with "x27", "x2f", "quot" tokens. We caught this on a first pass and fixed it.

**Another:** "Gemini" is ambiguous (it's both Google's AI model and a popular non-AI internet protocol discussed on HN). We tightened the keyword filter to require qualifiers — `gemini pro`, `gemini api`, `google gemini` — to exclude the protocol false positives.

---

## 4. What we measured: five complementary signals

A single sentiment score is a thin signal. We measured five orthogonal dimensions:

| # | Dimension | What it captures | Tool |
|---|---|---|---|
| 1 | **RoBERTa polarity** | Pos/neg sentiment from a deep learning model fine-tuned on 124M tweets | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| 2 | **VADER polarity** | Lexicon-based sentiment, designed for social media | VADER (Hutto & Gilbert 2014) |
| 3 | **AFINN polarity** | Simpler lexicon baseline (per-word integer scores) | AFINN (Nielsen 2011) |
| 4 | **NRC emotion** | 8 emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust + positive/negative | NRC EmoLex (Mohammad & Turney 2013) |
| 5 | **Hedge density** | Fraction of sentences containing an epistemic-uncertainty marker ("maybe", "I think", "could be") | Custom 22-pattern regex lexicon |

**Why three different polarity scorers?** Because they disagree, and that disagreement is itself a finding (see Figure 2). VADER pinned at high positive (mean 0.51). RoBERTa centered near neutral (mean −0.06). AFINN slightly positive (mean 0.03). The "tone of HN AI discourse" depends entirely on which scorer you ask. **We use RoBERTa as our primary signal because VADER's compound-score formula saturates on long-form text** (a known issue when applying tweet-tuned tools to paragraph-length comments). Reporting all three is a methodological contribution worth highlighting.

**Why NRC?** Polarity is one-dimensional. NRC gives us eight specific emotions, so we can answer questions like "did fear spike around the OpenAI board crisis?" (it didn't significantly — positivity dropped instead). NRC's `affect_frequencies` are length-normalized, so a 1500-word comment with 5 fear words doesn't get drowned out by all the neutral words.

**Why hedge density?** Hedging tracks epistemic confidence, which is *not* the same as polarity. A confident negative comment is different from a hedged neutral one. Measuring it separately gives us a second axis.

---

## 5. What we did with the measurements

Every post and comment got scored on all five dimensions. Then we ran four downstream analyses:

### 5a. Time-series aggregation
Group posts by week, take the mean of each metric. We get a 224-week-long trajectory for each of 14 metrics (3 polarity scorers + 8 NRC emotions + 2 NRC aggregates + hedge density).

### 5b. Change-point detection
We use a method called PELT (Pruned Exact Linear Time, Killick et al. 2012) to automatically find the dates where each trajectory has a statistically meaningful regime shift. We run it at three "penalty" levels — finding the same break date at higher penalty means it's more robust.

### 5c. Event-window tests
We picked 10 named events (ChatGPT public release, GPT-4, Sora, OpenAI board crisis, EU AI Act, DeepSeek R1, etc.) and for each one ran a Welch t-test comparing the 30 days before vs. the 30 days after, on every metric. That's 14 metrics × 10 events = 140 tests. We applied **Bonferroni correction** to control for multiple comparisons.

### 5d. Topic modeling
We grouped the corpus into 78 thematic clusters using **BERTopic** — an unsupervised method that:
1. Embeds each post as a 384-dimensional vector using a small transformer (sentence-transformers all-MiniLM-L6-v2)
2. Reduces dimensions with UMAP
3. Clusters with HDBSCAN

Importantly, we fit the topic model only on the **first 70% of the data by date** to prevent "data leakage" — the model never sees future posts when assigning topic IDs to past ones. This matters for rigor and is mentioned in the rubric.

The 19 largest non-noise topics cover the full landscape of AI discourse: GPT models / coding tools, Google search & competition, AGI debates, ChatGPT/Bing Sydney, copyright & fair use, AI safety, the AI art controversy, education, RAG/embeddings, NVIDIA infrastructure, voice AI, and so on.

---

## 6. The findings — what to claim in the paper

Ranked by importance:

### Finding 1: HN sentiment toward GenAI is sliding
The 2025-2026 trough is the most negative period of the entire window. RoBERTa weekly mean drifts from ~0 in 2022 to roughly −0.15 to −0.20 in 2026. **(Figure 1)**

### Finding 2: Hedging collapsed by ~50%
The single largest quantitative effect in the corpus. Mean hedge density was ~0.38 in early 2022 and ~0.18 by mid-2026. As capabilities became concrete, the community stopped speaking in hypotheticals. **(Figure 4)**

### Finding 3: Three polarity scorers, three different stories
VADER is saturated at +0.4–0.6 the entire time — it would say HN has been uniformly enthusiastic. RoBERTa says HN has been neutral and slowly turning critical. AFINN sits in the middle. This is a documented failure mode of lexicon-based sentiment on long text and is itself a methodological contribution. **(Figure 2)**

### Finding 4: Eight Bonferroni-significant event shifts — all negative
Out of 140 tests, eight survived correction. **All eight deltas were negative.** No event in our list significantly *increased* sentiment, hedging, or any positive emotion.

The strongest individual effects:
- **GPT-4 release reduced hedging** (−0.059, p ≈ 3×10⁻⁵). The strongest single effect in the corpus.
- **OpenAI board crisis reduced positivity** (−0.041 in NRC positive, p ≈ 10⁻⁴; also −0.10 in VADER). Both signals agree.
- **GPT-4o announcement reduced VADER polarity** (−0.10).
- **DeepSeek R1 reduced disgust, sadness, *and* hedging** — the only event that lowered negative emotions. The community welcomed it.

**Pattern**: capability releases convert speculation into evaluation. **(Figure 5)**

### Finding 5: The "hype cycle settling" arc is measurable
Anticipation, joy, and surprise all decline over the four years. Trust stays high. Sadness rises slightly. Anger and fear are stable. This is the canonical Gartner hype-cycle "trough of disillusionment" pattern, observed quantitatively week-by-week. **(Figure 3)**

### Finding 6: Topic-level emotional structure is rich
Aggregate signals mask topic-specific emotional fingerprints. From the topic × emotion heatmap:
- **Copyright / fair-use discussions** are angry and disgusted.
- **AI art** carries elevated disgust and anger.
- **Education impact** is high in joy and trust.
- **Engineer / job-market discourse** is calm and trust-rich.
- **AGI / consciousness threads** show elevated fear.

The headline shift in mood is partly *which topics dominate when*, not uniform mood movement. **(Figures 7, 8)**

### Finding 7: Multi-dimensional measurement is justified
Polarity scorers correlate only modestly with each other (r ≈ 0.32–0.59). Hedging is essentially orthogonal to polarity (r = −0.13). Emotions cluster sensibly (anger × disgust = 0.42). No single number captures the discourse — and we have correlation evidence to back the claim. **(Figure 6)**

---

## 7. The central claim

> **Generative AI discourse on Hacker News has shifted from speculative enthusiasm toward concrete, less-hedged, increasingly negative technical evaluation between 2022 and 2026, with measurable per-event and per-topic signatures.**

This is the sentence to anchor the abstract and the conclusion.

---

## 8. Mapping content to the required report sections

The rubric requires Abstract, Introduction, Related Work, Methodology, Experiments & Results, and Conclusion. Here's where the content above maps:

### Abstract (~150-200 words)
Compress sections 1, 6 (top three findings), and 7 of this brief.

### Introduction
- Open with the question: how has tech-community sentiment toward AI evolved post-ChatGPT?
- Motivate it: AI is reshaping software, jobs, IP, education; understanding how informed observers respond is socially relevant.
- State four contributions: (a) a 656k-item HN corpus filtered for generative AI; (b) a multi-dimensional measurement framework combining transformer + lexicon polarity, NRC emotion, and a hedge lexicon; (c) event-anchored tests across 10 major AI events; (d) a topic-level emotional structure analysis using BERTopic.
- End with the 7 findings as bullet points.

### Related Work
Group prior work into three buckets:
1. **Sentiment analysis on social media corpora** (Twitter studies — Hutto & Gilbert 2014 on VADER, Pang & Lee 2008, Cardiff NLP Twitter-RoBERTa)
2. **Public opinion on AI** (Pew surveys 2022-2026, Stanford AI Index public-opinion chapters, BBVA AI sentiment via GDELT). Note these are *survey-based or media-tone-based*; ours is text-content-based on user-generated discussion.
3. **Topic modeling and discourse analysis** (Blei et al. 2003 on LDA, Grootendorst 2022 on BERTopic, Mohammad & Turney 2013 on NRC EmoLex)
4. **Hedging and epistemic markers** (Hyland 2005 on metadiscourse, Tan et al. 2014 on persuasion)

The gap our work fills: existing public-opinion work is survey-based or aggregates pre-computed sentiment; existing computational work is single-dimensional (just polarity) or limited to short windows around single events. We do continuous multi-dimensional measurement across the full post-ChatGPT era.

### Methodology
Heavy section. Use the rubric — it explicitly asks about binary mapping, one-hot encoding, feature scaling, train/test split, and architectural choices. Cover:

1. **Data source and collection.** BigQuery query, keyword regex filter, type filter (stories + comments only), the deletion/dead-flag filter.
2. **Preprocessing.** HTML entity decoding (mention this — it's a real data-quality finding), URL stripping, emoji handling, whitespace normalization, the "deleted/removed" filter.
3. **Sampling strategy.** Top-30/day plus the score≥20 robustness sample. Why both. Temporal split for topic-model fitting.
4. **Feature engineering.** Log-score transformation, temporal features (hour, weekday, weekend), structural flags (has_image, is_self), title statistics.
5. **Polarity scorers.** Each one, why we picked it, what the disagreement implies. Note `max_length=512` for RoBERTa to cover ~80% of texts in full.
6. **NRC scoring.** Length-normalization via `affect_frequencies` so scores are comparable across long and short documents.
7. **Hedging.** Sentence-level binary detection, then density per document. Hedge lexicon is curated from Hyland (2005).
8. **Topic modeling.** SBERT embeddings → UMAP → HDBSCAN → BERTopic. Train on first 70% by date.
9. **Time series.** Weekly resampling, PELT change-point detection (multiple penalty levels), Welch t-tests for events with Bonferroni correction.

### Experiments & Results
Walk through the figures in order. Each figure → 2-3 paragraphs of interpretation. Use the brief's section 6 as the spine.

Make sure you include:
- The polarity-scorer disagreement (Fig 2) framed as a methodological *result*, not a problem.
- The 8 Bonferroni-significant events with effect sizes and p-values (Fig 5).
- The PELT change-point dates as a separate table.
- The topic × emotion heatmap explicitly described.
- Inter-metric correlations as a justification of the multi-dimensional approach.
- An ablation: comparing top-30/day sample vs. score≥20 sample to show robustness.

### Conclusion
Restate central claim. List limitations honestly:
- HN ≠ general public.
- Keyword-based filter has known false-positives/negatives.
- Hedging lexicon is curated, not learned.
- Aggregate sentiment masks topic-level structure (we discuss this; we don't claim to fully resolve it).
- We don't establish causation between events and shifts — only co-occurrence within 30-day windows.

Future work: stance detection, expansion to other communities (Reddit, Bluesky), fine-tuning a domain-specific sentiment classifier on labeled HN data, causal-inference techniques for event impact.

---

## 9. Where to find each thing

| You need | It's at |
|---|---|
| All 9 paper figures (PNG) | `results/figures/01_*.png` through `09_*.png` |
| Time-series CSVs to build new figures | `results/sentiment_W.csv` (overall), `results/sentiment_by_topic_W.csv` (per-topic) |
| Event test results | `results/event_tests.csv` |
| Change-point dates | `results/changepoints.csv` |
| Inter-metric correlation matrices | `results/correlations_pearson.csv`, `results/correlations_spearman.csv` |
| Topic catalog (top words per cluster) | Run cell 11 of `notebooks/analysis.ipynb`, or load BERTopic model from `experiments/bertopic_hn/` |
| Per-document scored corpus | `data/processed/hn_ai_topN_sent_topics.parquet` |
| The pipeline code | `src/*.py` — collect, preprocess, sentiment, topics, timeseries, figures |

If you want a number for the report and you can't find it in the figures or this brief, ask — most things are one Pandas one-liner away from the parquet.

---

## 10. Glossary (in case anything's unclear)

- **Corpus**: the full collection of texts we're analyzing.
- **Polarity / sentiment**: how positive or negative something is, on a continuous scale.
- **Lexicon**: a curated list of words and their associated sentiment / emotion scores.
- **Transformer model**: a type of deep-learning architecture that processes language by attending to context. RoBERTa is one such model.
- **Embedding**: a fixed-length vector that represents a piece of text in a way that captures meaning.
- **PELT**: Pruned Exact Linear Time — an algorithm that finds the dates where a time series changes its statistical behavior.
- **Bonferroni correction**: when running many statistical tests, multiply each p-value by the number of tests to avoid false-positive findings from luck. We did 140 tests; only the 8 that survived correction count as significant.
- **HDBSCAN**: a clustering algorithm that doesn't require you to specify the number of clusters in advance and can label points as "noise" if they don't belong anywhere.
- **UMAP**: a dimensionality-reduction algorithm — turns 384-dimensional embeddings into 5-dimensional ones we can cluster on.
- **BERTopic**: a topic-modeling library that combines embeddings + UMAP + HDBSCAN into one pipeline.
- **VADER pathology / saturation**: VADER's mathematical formula was tuned for short text. On long text, the score gets pulled toward ±1 even when the meaning is mixed.
- **Affect frequencies**: NRC's length-normalized score — fraction of emotion-bearing words in the text that fell into each category.
- **Hedge / epistemic marker**: a word or phrase that signals uncertainty about a claim ("maybe", "I think", "seems like").

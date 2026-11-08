# Tallie 1.0

<p align="center">
  <strong>A language model rebuilt to read discussions, extract opinions, and turn thousands of comments into ranked answers.</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/pateluday/tallie"><img alt="Hugging Face model" src="https://img.shields.io/badge/Model-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Model size" src="https://img.shields.io/badge/Parameters-1.33B-111111">
</p>

Tallie began with a question: **could a language model read an entire discussion once and answer questions about what people recommended, disliked, or mentioned—without generating prose for every comment?**

I built Tallie to do exactly that. Given a collection of comments and a question such as _“podcasts people recommend”_ or _“most hated books,”_ it extracts named items, classifies their type and the writer's opinion, decides which mentions answer the question, and tallies the results into a ranked list.

```text
"podcasts people recommend" + 24,000 Hacker News comments
                              ↓
  1. Planet Money             16 positive mentions
  2. Radiolab                 13 positive mentions
  3. EconTalk                 13 positive mentions
  4. This American Life       13 positive mentions
  5. Darknet Diaries          14 positive, 1 neutral, 1 mixed
```

The trained model is published on **[Hugging Face](https://huggingface.co/pateluday/tallie)**. This repository contains the complete engineering work behind it: data collection and labeling, model architecture, training, evaluation, inference, ranking, tests, and the hand-labeled benchmark.

---

## What I built

Tallie is a **1.33B-parameter, non-generative, bidirectional transformer** derived from Qwen3-1.7B. It processes each comment in one forward pass and produces structured evidence rather than generated text.

For every relevant mention, Tallie returns:

- the exact text span that names an item;
- one of 11 item types, including books, movies, games, podcasts, software, products, and people;
- an opinion label: positive, negative, neutral, or mixed; and
- a relevance score conditioned on the user's question.

Ordinary deterministic code then normalizes, filters, counts, and ranks those mentions. Every returned item is grounded in the source text, so Tallie cannot invent an item that nobody mentioned.

## Rebuilding a writer into a reader

I started from Qwen3-1.7B and changed its role rather than simply prompting it:

```text
Qwen3-1.7B                           28-layer generative language model
    │
    ├── removed the language head   no token-by-token generation
    ├── retained 20 layers          smaller and faster
    ├── changed causal attention    bidirectional context
    └── trained four task heads     mentions · relevance · opinion · type
                                      │
                                      ▼
Tallie 1.0                          1.33B-parameter extraction model
```

The four heads share one transformer pass:

| Head | What it learns |
|---|---|
| **Mention** | BIO token tags identifying exact named-item spans |
| **Relevance** | Whether a span answers the supplied question |
| **Opinion** | Positive, negative, neutral, or mixed sentiment toward the item |
| **Type** | One of 11 supported item categories |

I trained the transformer with LoRA while training the task heads directly. The question and comment are encoded together, allowing relevance to be evaluated in context rather than through keyword matching.

## Results

I evaluated Tallie on a hand-labeled set of **150 Hacker News comments, 224 named items, and 165 questions**. One third of the comments came from video-game discussions, a topic excluded from training to provide a held-out domain test.

| Evaluation | Tallie 1.0 |
|---|---:|
| Named-item extraction F1 | **0.86** |
| ↳ precision / recall | 0.79 / **0.94** |
| Item-type accuracy | **0.91** |
| End-to-end question-answering F1 | **0.68** |
| ↳ familiar topics / held-out games | 0.70 / 0.65 |
| Opinion accuracy | 0.64 |
| Throughput on one NVIDIA L4 | **130 comments/second** |

These values match the published [Tallie model card](https://huggingface.co/pateluday/tallie). The benchmark labels are included in [`data/gold_test_labels.jsonl`](data/gold_test_labels.jsonl); the original public comments can be reconstructed by ID through the Hacker News API.

## How the project evolved

Tallie's largest improvements came from diagnosing the data rather than making the architecture more complicated.

| Version | What I changed | Named-item F1 | End-to-end F1 | What I learned |
|---|---|---:|---:|---|
| 0.1 | Qwen3-8B teacher labels, 16 retained layers, one epoch | ≈0.50 | 0.50 | The first teacher reached only 0.55 F1 against human labels, limiting the student. |
| 0.2 | Stronger teacher, cleaner spans, junk filters, 20 layers | 0.85 | 0.59 | Improving labels raised the ceiling more than changing the model. |
| **1.0** | Type head, wrong-type negatives, balanced opinion loss, validation-tuned cutoff, three epochs | **0.86** | **0.68** | Explicit item types removed many otherwise plausible but incorrect answers. |

The full sequence of experiments, failures, diagnoses, and fixes is documented in **[`docs/JOURNEY.md`](docs/JOURNEY.md)**.

## Design decisions

### Non-generative extraction

Generating an answer separately for every comment would repeatedly pay the cost of autoregressive decoding. Tallie instead reads each comment once and emits structured predictions in parallel. This makes large discussion archives practical to process and keeps every result tied to a source span.

### Distillation plus a human benchmark

An LLM teacher labeled roughly 6,000 public Hacker News comments for names, types, opinions, and topics. I then built a separate hand-labeled test set because agreement with a teacher is not the same as correctness. That benchmark exposed the original teacher as the main bottleneck and guided the relabeling work behind v0.2.

### A type-aware relevance model

Relevance alone could return a well-matched item of the wrong kind—for example, a book in response to a movie question. Adding a type head let the ranking pipeline enforce the requested category and produced the largest improvement in Tallie 1.0.

### Model-only inference

I also tested a Wikipedia-assisted lookup stage. The combined system scored 0.67 end-to-end and the lookup by itself scored 0.44, compared with **0.68 for Tallie alone**. I removed the extra dependency and shipped the simpler, stronger system.

## Repository map

```text
tallie/                    model, inference, filtering, and ranking package
  model.py                 Qwen3 surgery, LoRA configuration, and four task heads
  data.py                  tokenization and span-label alignment
  infer.py                 batched extraction, loading, and export
  ranking.py               question parsing, normalization, tallying, and ranking
  pipeline.py              high-level Tallie interface
scripts/
  scrape_hn.py             Hacker News discussion collection
  label_teacher.py         LLM-assisted training-label generation
  build_dataset.py         examples, splits, and held-out-domain construction
  train.py                 training, validation, cutoff tuning, and checkpoints
  evaluate.py              test-split and throughput evaluation
  eval_gold.py             evaluation against the human-labeled benchmark
notebooks/train_tallie.ipynb
                           reproducible Tallie 1.0 training notebook
data/gold_test_labels.jsonl
                           150 annotated comments represented by IDs and labels
tests/                     CPU tests built with tiny random models
docs/JOURNEY.md            detailed experimental history
```

## Reproducing the work

The repository preserves the complete pipeline rather than redistributing raw forum text or multi-gigabyte model weights.

```bash
pip install -r requirements.txt

# CPU-only correctness tests; no model download required
python tests/test_ranking.py
python tests/test_pipeline.py
python tests/smoke_test.py
```

The main experimental stages are represented by the scripts below. Exact Tallie 1.0 settings and execution order are captured in [`notebooks/train_tallie.ipynb`](notebooks/train_tallie.ipynb).

```text
scrape_hn.py → label_teacher.py → build_dataset.py
                                      ↓
                                  train.py
                                      ↓
                         evaluate.py + eval_gold.py
```

For a small inference example using the published model, see [`examples/example.py`](examples/example.py).

## Limitations

- Tallie was trained on Hacker News and is strongest on software, books, and technology culture.
- Music, television, and games are less represented in the training labels.
- Opinion labels skew positive, and sarcasm remains difficult; measured opinion accuracy is 0.64.
- Questions requiring outside knowledge—such as genres or story topics—are less reliable than type-and-opinion questions.
- Tallie currently supports English only.
- Ranked results summarize the views expressed by a source community; they are not ground truth about item quality.

## Data, model, and license

- **Training data:** approximately 6,000 public Hacker News comments collected through the Algolia HN API and labeled by an LLM teacher. Raw comments are not redistributed.
- **Evaluation data:** IDs and original human annotations are included; comment text can be retrieved from the public API with [`scripts/fetch_gold_text.py`](scripts/fetch_gold_text.py).
- **Base model:** [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), licensed under Apache 2.0.
- **Tallie code and model:** Apache License 2.0.

---

<p align="center">
  <a href="https://huggingface.co/pateluday/tallie"><strong>Model on Hugging Face</strong></a>
  &nbsp;·&nbsp;
  <a href="https://pateluday.com"><strong>Portfolio</strong></a>
</p>

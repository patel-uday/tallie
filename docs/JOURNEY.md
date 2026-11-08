# Building Tallie: the full log

Every version, what went wrong, how it was diagnosed, and what fixed it. All numbers are from actual runs.

---

## 0. The idea

Asking "what do people on this forum recommend or hate?" is easy for a big LLM, but expensive: every comment means a long prompt plus a generated answer, token by token. The goal was a model that **reads** a comment once and outputs structure (item, type, opinion), with the counting done by ordinary code.

## 1. Design

- **Base: Qwen3-1.7B.** Small enough to train on a single cloud GPU, but with the world knowledge the task needs. A 150M encoder was considered and left for later as a size/accuracy experiment.
- **Surgery:** remove the language-model head, keep 20 of 28 layers, switch attention from causal to bidirectional so every token sees the whole comment and the question.
- **Heads:** mention (BIO tags), relevance (conditioned on the question), opinion (4 classes); a type head was added in 1.0.
- **Training:** LoRA on the backbone, full training of the heads.
- **Data:** Hacker News threads collected through the Algolia API. The query list covered books, movies, TV, podcasts, music and tools; **video-game threads were held out entirely** to test generalization. That gave 24,023 comments.

## 2. Version 0.1: it trains, but the answers are junk

**Setup:** 6,000 comment chunks labeled by **Qwen3-8B** (AWQ, vLLM) on a free Colab T4. 8 of 6,000 outputs were unparseable. Training: 16 layers, 1 epoch.

**Engineering problems on the way:**
- Colab ships an old `torchao` that makes `peft` refuse to load. Fix: uninstall it.
- The T4 reports bfloat16 as "supported", but it's emulated and **5× slower**. Fix: pick fp16 based on the GPU's compute capability, with a gradient scaler.
- Colab disconnects. Fix: resumable labeling (every 1,000 chunks) and resumable training (checkpoint every 300 steps).

**Result:** end-to-end F1 **0.50** against the teacher's labels, 0.45 on held-out games, 25 comments/s on a T4.

**But the demo showed the real problem.** "Books people loved" returned `books`, `book`, `The`, `it`, `,`.

**Diagnosis:**
1. **A matching bug.** The dataset builder located each labeled name in the comment with a case-insensitive fallback and no word boundaries. When the teacher labeled the novel *IT*, every "it" in that comment became a book title.
2. **Teacher junk.** The 8B teacher labeled generic words ("books", "wafer", "startups", "salaries") as named items despite instructions not to.

**Error analysis on 1,500 test examples** made it concrete:
- Most "boundary errors" were an artifact of the analysis itself (a leading space), not model mistakes.
- About **70% of "missed" items were teacher junk** that the model had correctly ignored.
- Many "false positives" were **real names the teacher had missed** (*Seveneves*, *Call of Duty*, *Mahabharata*, *Kubrick*).

**Conclusion:** the scoreboard was measuring agreement with a sloppy teacher, not the model's quality.

## 3. Building an honest yardstick

150 comments were **hand-labeled**: 224 named items with types and opinions, plus 165 list-style questions with correct answers. A third of them come from the held-out games threads, and a third contain no items at all, to catch invented answers.

Scoring the teachers on it:

| Labels | Names F1 | Precision | Recall | Opinion acc. |
|---|---|---|---|---|
| Qwen3-8B (v0.1 teacher) | **0.55** | 0.51 | 0.60 | 0.72 |
| Stronger LLM teacher (v0.2+) | **0.88** | 0.83 | 0.92 | 0.68 |

The 8B teacher also **invented items from thread titles** (labeling "Firefox" in comments that never mention it) and merged items ("Noughts & Crosses by Malorie Blackman"). **The student can't beat its teacher.** v0.1's ceiling was its labels.

## 4. Version 0.2: better labels

**Relabeling with a stronger teacher.** All 5,992 chunks were relabeled, with every output validated (exact-substring names, allowed types and opinions). Questions are no longer written by the teacher: early teacher-written questions were repetitive templates, so training questions are now generated from the labels. Each item also carries 1–5 specific attributes and topics, which feed those generated questions.

**Code fixes:**
- Junk filter (generic words, pronouns, punctuation) and whole-word matching.
- Spans that start or end mid-word are snapped to whole words; spans inside URLs are dropped.
- Extra loss weight on name tokens (names are a small fraction of all tokens).
- 20 layers instead of 16, 2 epochs.
- Inference sorts comments by length to cut padding: **2.4× faster**.

**Result (hand-labeled test):** names F1 **0.85**, end-to-end **0.59** (games 0.55), opinion 0.65; 60 comments/s on a T4.

**New weakness:** at corpus scale, the model **ignored the topic** of a question. "Funny movies" returned well-liked *books*, and "books about King Arthur" returned *Three-Body Problem*. It had no notion of what kind of thing each item was.

## 5. Experiment: a Wikipedia knowledge lookup (dropped)

Idea: look each unique item up on Wikipedia (description + categories) to check its type and genre.

- **Engineering:** Wikipedia's search API heavily rate-limits Colab's shared IPs (about a third of requests got HTTP 429). That was replaced with **batched exact-title lookups** (50 titles per request), and descriptions and categories were fetched in two passes. 3,097 of 4,524 frequent names were matched.
- **Strict mode** (only items Wikipedia can confirm): precision **0.875**, but recall 0.27.
- **Hybrid** (Wikipedia decides when it knows the item, the model otherwise) helped on familiar topics but hurt on unseen ones.
- Common words linked to obscure same-named pages ("France" → *France (film)*, "Mac" → *Mac (novel)*).

Final comparison on the hand-labeled test with the 1.0 model: **model alone 0.684**, hybrid 0.669, strict 0.442. The lookup was removed.

## 6. Version 1.0: teach the model what things are

**Changes:**
- **Type head** trained from the types already in the teacher's labels (11 classes).
- **Wrong-type negatives:** more training questions whose correct answer is *nothing* (asking for movies in a comment about books), and several names per type ("films", "apps", "novels").
- **Balanced opinion loss** (inverse-frequency weights; the labels were 64% positive).
- **Relevance cutoff tuned on validation:** 0.40 beat 0.50 (0.760 vs 0.752 on validation).
- **Vocabulary trimming fixed:** v0.2's slim export kept only tokens seen in training, so questions with new words were misread (0.37 vs 0.59). The fix keeps every ASCII token: 0.70 slim vs 0.68 full.
- 3 epochs, on an NVIDIA L4.

**Results:**

| | v0.2 | **v1.0** |
|---|---|---|
| End-to-end F1, hand-labeled (all / familiar / games) | 0.59 / 0.61 / 0.55 | **0.68 / 0.70 / 0.65** |
| Names F1 (hand-labeled) | 0.85 | **0.86** |
| Type accuracy (hand-labeled) | — | **0.91** |
| Opinion accuracy (hand-labeled) | 0.65 | 0.64 |
| End-to-end F1 vs teacher labels (test / held-out games) | 0.66 / 0.61 | **0.70 / 0.66** |
| Speed | 60/s (T4) | **130/s (L4)** |

Opinion accuracy didn't move. The balanced loss helped on teacher-labeled data, but against human labels the teacher's positive lean is the ceiling. Better opinion labels are on the roadmap.

## 7. Shipping, and a near-loss

Google Drive's 15 GB filled up during the 1.0 run. Colab **appeared** to save the model folders, but the 2.5 GB weight files never reached Drive. The small files did, so nothing looked wrong until upload time.

**Recovery:** the 80 MB training checkpoint (LoRA adapters + heads) had survived. The model was rebuilt by re-applying the same surgery to a fresh Qwen3-1.7B download, loading the checkpoint (302 tensors, 0 missing) and merging the adapters. It was verified against the hand-labeled test (names F1 0.861 vs 0.859 originally) before publishing.

## Lessons

1. **Evaluate against humans early.** Teacher-graded scores hid a teacher at 0.55.
2. **Data beats architecture.** Relabeling moved names F1 from about 0.5 to 0.85; the type head then fixed the remaining failure *mode*.
3. **Read the outputs, not just the metrics.** Every major bug (the matching bug, missing types, the slim-vocabulary bug) showed up first in a demo list.
4. **Validate generated labels early.** Check the first batch before labeling everything; teachers take shortcuts.
5. **Measure the fancy idea against the simple one.** The Wikipedia lookup was clever; the model alone was better.
6. **Check that large files actually landed.** Silent storage failures are real; keep small checkpoints that can rebuild a model.

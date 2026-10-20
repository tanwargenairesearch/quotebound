# Extractive QA: answers bound to the passage

Teach Qwen3.5-4B to answer a question with the shortest stretch of a passage that answers it, then decode under a
schema that makes any other answer impossible to write. The data is
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) (CC BY-SA 4.0).

```sh
hf download Qwen/Qwen3.5-4B --local-dir models/qwen3.5-4b-16bit
recipes/extractive-qa/run.sh models/qwen3.5-4b-16bit work/extractive-qa
```

## What it does

1. **Prepare** (`prepare.py`, standard library only).
   - `train.jsonl`: 2,000 questions from SQuAD's training articles.
   - `dev.jsonl`: 200 questions from a tenth of the training articles, held out from training. The trainer
     reports its loss on these.
   - `eval.jsonl` and `gold.jsonl`: 300 questions from SQuAD's official dev set, with its reference answers.
   - The user turn is `Passage:\n…\n\nQuestion: …`. The assistant turn is `{"answer":"…"}`, compact JSON with UTF-8
     unescaped, which is exactly what the constrained decoder writes.
   - A training question is kept only when its answer stands in the passage as whole words, because that is the
     only kind of answer the decoder can write. This check is looser than the decoder's on numbers: the decoder
     also refuses a number cut from a wider one, such as `5` from `5.5` or `33` from `33%`. One of the 2,000
     training answers is such a quote (`33`, from "33% of the population"). The receipts recipe's `stands()` is the decoder's exact rule.
2. **Train** a LoRA adapter for one epoch (141 steps), with the dev loss every 40 steps.
3. **Generate** the 300 eval answers four ways: the base model and the adapter, each with and without exact
   drafts (`--draft-train train.jsonl --draft-k 8`). Every run uses `--schema schema.json --quote-source user`, so
   each answer is a JSON object whose `answer` is a stretch of the user turn.
4. **Score** (`score.py`): SQuAD exact match and F1, how many answers parse and stand in their passage, time per
   question, and decode speed. Then a byte-for-byte check that drafts changed no answer.

## The schema

```json
{"type": "object", "additionalProperties": false, "required": ["answer"],
 "properties": {"answer": {"$ref": "#/$defs/quote"}},
 "$defs": {"quote": {"type": "string", "minLength": 1, "maxLength": 200}}}
```

A string is bound to the input when its key is `quote`, or when it is a `$ref` to a definition named `quote`.
With `--quote-source user`, the decoder allows only the bytes that keep the string a stretch of the last user
turn. When the string closes, it must stand as whole words, letter case aside. The model chooses where the quote
starts and stops, but it cannot write a word that is not in the passage.

## Measured

One run on 20 Oct 2026: an Apple M4 Pro (48 GB), macOS 26.6.2, on mains power, with `run.sh`'s settings and
the trainer's defaults (LoRA rank 16, alpha 32, learning rate 1e-4, 4,096 tokens a batch).

**Training.**
- 2,000 questions, one epoch: 141 steps in 43 min 24 s, including five dev-loss passes of about 64 s each.
- 220 tokens a second (median step; 184 to 229), with 13.5 GB peak memory.
- Dev loss 0.795 at step 0, then 0.115 at step 40, 0.112 at step 80, 0.105 at step 120 and 0.105 at step 141.
  `run/best` is step 120.

**Answers.** 300 SQuAD dev questions, each run with `--schema schema.json --quote-source user --max-tokens 64`:

| | exact match | F1 | answers that stand in the passage | p50 / p95 per question | decode tokens a second |
|---|---|---|---|---|---|
| base Qwen3.5-4B | 15.3 | 25.2 | 300 / 300 | 1.04 / 2.01 s | 23.5 |
| base, exact drafts | 15.3 | 25.2 | 300 / 300 | 0.64 / 1.29 s | 45.1 (1.90 tokens a pass) |
| LoRA adapter | 89.3 | 94.5 | 300 / 300 | 0.65 / 1.08 s | 30.8 |
| LoRA adapter, exact drafts | 89.3 | 94.5 | 300 / 300 | 0.53 / 0.69 s | 51.2 (2.46 tokens a pass) |

**Exactness.** With drafts, all 300 answers are byte-identical to plain decoding, for both the base model and the
adapter (`score.py --same`).

**Overrides.** An override is a step where the constraint replaced the model's own top token.
- The base model, untrained for this format, was overridden in all 300 answers, 953 steps in all. Often its quote
  became the question itself, because the question is part of the user turn. So the base row measures an
  untrained model forced into the format, not how well it reads.
- The adapter was overridden in 1 answer of 300, at 1 step. That answer (`zip`, where the reference is `"zip" the
  mouth shut`) is wrong. A caller that withholds overridden answers would have withheld it.

**What these numbers do not show.**
- They come from one run on one machine. The question-level timings include each question's prefill.
- SQuAD has been public since 2016, so Qwen3.5 may have seen it in pretraining. The base and adapter rows share
  that exposure.
- The 300 questions are a sample of SQuAD's dev set (`prepare.py --seed 0`), not the whole set.

# Receipts: many fields, every value copied from the document

Teach Qwen3.5-4B to read a receipt into JSON: each item's name, count and price, and the subtotal, tax, service
charge, discount, total, cash, change and card payment. A field the receipt does not show is `null`. Then decode
under a schema that lets every value be only a stretch of the receipt's own text. The data is the labels of
[CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC BY 4.0), Indonesian shop and restaurant
receipts.

```sh
pip install pyarrow huggingface_hub
hf download Qwen/Qwen3.5-4B --local-dir models/qwen3.5-4b-16bit
recipes/receipts/run.sh models/qwen3.5-4b-16bit work/receipts
```

## What it does

1. **Prepare** (`prepare.py`).
   - It reads only the `ground_truth` column of CORD v2's parquet files at a pinned revision, through HTTP range
     reads, so the 2.3 GB of receipt images are not downloaded.
   - Each receipt becomes text: its labelled words, grouped into CORD's printed rows, top to bottom and left to right.
   - The answer is built from CORD's labels: `menu.nm`, `menu.cnt` and `menu.price` for each item, and
     `sub_total.*` and `total.*` for the rest.
   - `train.jsonl` (CORD train) and `dev.jsonl` (CORD validation) keep a receipt only when every value stands in its
     text under the decoder's quote rule, and no field holds two values.
   - `eval.jsonl` and `gold.jsonl` (CORD test) keep a receipt whenever no field holds two values. A gold value that
     the decoder cannot quote stays in, and counts as a miss.
2. **Train** a LoRA adapter for two epochs, with the dev loss every 25 steps.
3. **Generate** the test receipts four ways:
   - the base model, constrained;
   - the adapter, constrained;
   - the adapter, constrained, with exact drafts;
   - the adapter with no schema at all, to count the values the constraint would have prevented.
4. **Score** (`score.py`):
   - precision, recall and F1 over (field, value) pairs;
   - receipts with every value and every null right;
   - values that do not stand in the receipt;
   - time per receipt and decode speed;
   - a byte-for-byte check that drafts changed no answer.

The scores are this recipe's own. They are not CORD's official metric, which also scores fields this recipe leaves
out.

## The schema

`schema.json` has `items` (each with a `name` and a nullable `count` and `price`) and eight nullable totals. Every
string is a `$ref` to the definition named `quote`, either directly or through `maybe` (`anyOf` a quote or
`null`). With `--quote-source user`, each one is a stretch of the receipt that stands alone: not inside a word,
and not inside a wider number. So `"2"` cannot be cut from `2.00`, and `"000"` cannot be cut from `60.000`.

`prepare.py`'s `stands()` is the same rule in Python. Every one of the 727 train and 93 dev answers passes the
decoder's own constraint, checked through the C library (`libquotebound_constrain`).

## Measured

One run on 20 Oct 2026: an Apple M4 Pro (48 GB), macOS 26.6.2, on mains power, with `run.sh`'s settings and
the trainer's defaults (LoRA rank 16, alpha 32, learning rate 1e-4, 4,096 tokens a batch).

**Training.**
- 727 receipts, two epochs: 106 steps in 35 min 13 s, including six dev-loss passes of about 31 s each.
- 207 tokens a second (median step; 167 to 217), with 13.5 GB peak memory.
- Dev loss 0.700 at step 0, then 0.019 at step 25, 0.011 at step 50, 0.0089 at step 75, 0.0081 at step 100 and
  0.0081 at step 106. `run/best` is step 100.

**Answers.** 96 CORD test receipts, with `--max-tokens 768`. "Constrained" means `--schema schema.json
--quote-source user`.

| | precision | recall | F1 | receipts all right | values not on the receipt | p50 / p95 per receipt | decode tokens a second |
|---|---|---|---|---|---|---|---|
| base Qwen3.5-4B, constrained | 67.7 | 39.1 | 49.6 | 6 | 0 of 600 | 2.47 / 8.08 s | 21.5 |
| LoRA adapter, constrained | 94.6 | 95.3 | 95.0 | 65 | 0 of 1,045 | 4.27 / 10.41 s | 24.4 |
| LoRA adapter, constrained, exact drafts | 94.6 | 95.3 | 95.0 | 65 | 0 of 1,045 | 1.35 / 3.53 s | 80.4 (4.73 tokens a pass) |
| LoRA adapter, no constraint | 94.6 | 95.3 | 95.0 | 65 | 5 of 1,045 | 4.31 / 10.41 s | 24.2 |

**Exactness.** With drafts, all 96 answers are byte-identical to the plain constrained decoder's
(`score.py --same`). Drafts help more here than on SQuAD: the answers are long, and much of their text (item
names, prices, the JSON keys) repeats the prompt or the training answers.

**What the constraint changed.** An override is a step where the constraint replaced the model's own top token.
- The base model, untrained for this format, was overridden in all 96 answers, 1,504 steps in all.
- The adapter, constrained, was overridden in 4 answers at 5 steps. Those are exactly the 5 values that the same
  adapter, unconstrained, wrote but the receipt does not print, in the same 4 receipts:
  - **Four item names that wrap onto the next printed line**, joined into words the receipt does not print. For
    example, `Nasi Cap Cay Porsi B` and `esar` become `Nasi Cap Cay Porsi Besar`. The same happens with
    `(untuk 1 Po` and `rsi)`, `(14 Pcs` and `)`, and a list of flavours.
  - **One figure with a digit dropped.** The receipt prints `Kepiting 7-8.4ons - Chilly`, and the unconstrained
    answer writes `7-8.ons`.

  Constrained, each name stops at the line break and the figure is written as printed. None of the five matches the
  reference, which joins a wrapped name with a space (`Porsi B esar`), so the scores are equal. The difference is
  that every constrained value is the receipt's own text, and the 5 places where the model wanted to depart from it
  are each recorded as an override.

**What these numbers do not show.**
- They come from one run on one machine.
- 96 of CORD's 100 test receipts are scored. Four have a field with two values, which this answer shape cannot
  hold.
- 14 of the 1,038 reference values cannot be quoted, for example a name that wraps across lines, and they count
  as misses.
- The scores are this recipe's own, not CORD's official metric.
- The base model writes fewer values (600 against the adapter's 1,045), which is why its time per receipt is
  shorter.

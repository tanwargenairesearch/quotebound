# quotebound

Fine-tune and run Qwen3.5 on a Mac, natively in C++ on [MLX](https://github.com/ml-explore/mlx), with a decoder
that can only quote what it was given.

quotebound is one C++ program for Apple silicon. There is no Python in training or inference.

- **Trains LoRA adapters** on Qwen3.5-4B and -9B in bf16 (the base stays frozen). It has a hand-written Metal kernel
  for the backward pass of Qwen3.5's Gated DeltaNet recurrence.
- **Generates** greedily with a KV cache, token for token what mlx-lm generates (checked by parity tests).
- **Constrains an answer to a JSON Schema and binds every quote to the input.** With `--quote-source user`, every
  string at a `quote` key must be a stretch of the user's message, standing as whole words. The model cannot write
  a quote that is not in its input.
- **Reports the values it forced.** Every step where the constraint overrode the model's own top choice is
  recorded as a byte span, so a caller can withhold a value the model did not choose itself.
- **Decodes faster without changing a byte.** With exact speculative decoding, tokens are drafted from the prompt
  and from the training set's n-grams, then verified in one pass by kernels that round exactly as the one-token
  step does. The answers are byte-identical to plain greedy decoding.
- **Ships the constraint as a small C library** without MLX (`make constrain-lib`; `build_constrain_lib.sh` on
  Linux), so another decoder can mask its logits with exactly the same allowed tokens.

It was built for extracting facts from free text where a wrong value costs more than a missing one. If the answer
must be the input's own words, the decoder enforces it rather than hoping for it.

## Requirements

- An Apple silicon Mac on macOS 26. Tested on macOS 26.6.2, M4 Pro, 48 GB; the 4B trains in about 14 GB.
- Apple clang with C++20 (the Xcode command line tools).
- A Python with the `mlx==0.32.3` wheel. It is used only for the wheel's headers and `libmlx.dylib`; nothing runs
  in Python.
- simdjson 4.x (`brew install simdjson`). Tested with 4.6.3; 3.x lacks a type the schema compiler reads, and 5.x is
  untested.
- A checkpoint from Hugging Face: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) or
  [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) in bf16 safetensors, as published. Qwen3 checkpoints
  (`Qwen3ForCausalLM`) also load. Their tokenizer, chat template and architecture are tested, the architecture on tiny
  random models; unlike Qwen3.5, no test compares a real Qwen3 checkpoint with mlx-lm.

## Build

```sh
python3 -m pip install mlx==0.32.3
brew install simdjson
make                      # build/quotebound
```

Override any of `PYTHON`, `MLX`, `SIMDJSON`, `MODELS`, `DENY` or `MACOS_MIN` on the command line (`make
MLX=/path/to/site-packages/mlx`) or in a `config.mk` beside the Makefile. The Makefile's header explains each one.

## Quick start

```sh
hf download Qwen/Qwen3.5-4B --local-dir models/qwen3.5-4b-16bit
M=models/qwen3.5-4b-16bit

# a chat answer
build/quotebound generate --model $M --user "Name three moons of Jupiter."

# an answer that must be JSON under a schema, its quote taken from the message
build/quotebound generate --model $M --schema recipes/extractive-qa/schema.json --quote-source user \
  --system "Answer from the passage with JSON {\"answer\": ...}." \
  --user "Passage: The Eiffel Tower was completed in 1889 for the World's Fair. Question: When was it completed?"

# LoRA training: JSONL {"messages": [...]}, trained on the assistant turns
build/quotebound train --model $M --train train.jsonl --dev dev.jsonl --out runs/first --epochs 1

# many rows in one load, with the adapter, exact drafts on
build/quotebound generate --model $M --adapter runs/first/best --batch eval.jsonl --out out.jsonl \
  --schema schema.json --quote-source user --draft-train train.jsonl --draft-k 8
```

`build/quotebound` with no arguments prints every flag.

## Recipes

Each recipe is a runnable sequence on a Mac, on public data, with the numbers of a run we actually made.

- [`recipes/extractive-qa`](recipes/extractive-qa/README.md): SQuAD v1.1, answering a question with words quoted
  from the passage. LoRA on Qwen3.5-4B, quote-bound decoding, and the byte-exactness check of exact drafts against
  plain greedy decoding.
- [`recipes/receipts`](recipes/receipts/README.md): CORD v2, a receipt's items and totals as JSON, with nulls for
  what the receipt does not show. Many fields per document, every value copied from the receipt. It also counts
  what the same adapter writes with the constraint off.

More recipes are planned: a classification task with enum output, question writing from structured input, and
plain instruction tuning.

## Measured

Each figure comes from one run of each recipe on an Apple M4 Pro (48 GB), macOS 26.6.2, on mains power, on 20 Oct
2026. The recipes show how each was run, and what the numbers do not show.

| | extractive QA (SQuAD v1.1, 300 questions) | receipts (CORD v2, 96 receipts) |
|---|---|---|
| training | 2,000 examples, 1 epoch, 43 min | 727 examples, 2 epochs, 35 min |
| trainer speed, peak memory | 220 tokens/s, 13.5 GB | 207 tokens/s, 13.5 GB |
| base Qwen3.5-4B, constrained | exact match 15.3, F1 25.2 | field F1 49.6 |
| LoRA adapter, constrained | exact match 89.3, F1 94.5 | field F1 95.0, 65 of 96 receipts all right |
| values quoted from the input | 300 / 300 | 1,045 / 1,045 |
| the adapter with the constraint off | | 5 values not on the receipt, at exactly the 5 steps the constrained run records as overrides |
| p50 per input, plain → exact drafts | 0.65 → 0.53 s | 4.27 → 1.35 s |
| byte-identical with drafts | 300 / 300 | 96 / 96 |

Base models score low partly because they have not learnt the answer format: the constraint overrides them on
every answer. SQuAD has been public since 2016, so Qwen3.5 may have seen it in pretraining.

## What is exact, and what is not

- **Exact:** decoding with exact drafts (`--draft-train`, `--draft-k`) gives the same bytes as plain greedy
  decoding, on the same model, adapter and machine. The test suite checks this on random models in float32 and
  bfloat16, and the recipe checks it on a trained adapter.
- **Exact:** adapters are applied unmerged, in the same low-rank path that training ran.
- **Not exact:** `--merge-lora` folds the adapter into the bf16 weights at load. It is faster, but rounding the merged
  weights once can change some answers.
- **Not exact:** `--quantize 8` runs 8-bit weights. It is faster still, and answers change.
- Use the inexact flags only after measuring that the change is acceptable for your task. Nothing here claims that
  a quantised or merged model matches the bf16 one.

## Keeping a held-out set out of reach

`QUOTEBOUND_DENY=dir1:dir2` (at run time) or `make DENY=dir1:dir2` (compiled in) makes every input and output flag
refuse a path under those directories, after resolving `..` and symlinks. Use it to keep an evaluation set away
from training and tuning runs by accident.

## Tests

```sh
hf download Qwen/Qwen3.5-4B --local-dir models/qwen3.5-4b-16bit
hf download mlx-community/Qwen3-8B-8bit --local-dir models/qwen3-8b-8bit   # its tokenizer is read
make test-quick           # tokenizers, chat rendering, data, the recurrence and its gradients, tiny models,
                          # training, batch generation, constrained decoding, exact drafts (no real weights)
make test-qwen35-4b       # the 4B against mlx-lm fixtures (also test-qwen35-9b)
```

The fixtures in `testdata/` were dumped once from the Python references (`transformers`, `tokenizers`, `mlx-lm`);
[testdata/README.md](testdata/README.md) has the commands.

## Layout

| file | job |
|---|---|
| `src/unicode.*`, `src/pretokenize.*`, `src/tokenizer.*` | UTF-8 and character classes, the Qwen split patterns, byte-level BPE and the chat templates |
| `src/model.*`, `src/deltanet.*` | config and safetensors loading, attention, MLP, KV cache; Qwen3.5's Gated DeltaNet |
| `src/delta_grad.*` | the recurrence's backward: chunk checkpoints and a Metal kernel |
| `src/generate.*`, `src/batch.*` | prefill and greedy decoding; `generate --batch` with resume |
| `src/schema.*`, `src/constrain.*` | a JSON Schema compiled for decoding; the token trie, the byte automaton, quotes bound to the input |
| `src/draft.*`, `src/exact.*` | drafts from n-grams and the prompt; multi-row decode steps that round like the one-token step |
| `src/lora.*`, `src/data.*`, `src/loss.*`, `src/adamw.*`, `src/train.*` | adapters, training data and batch plans, loss, optimizer, the training loop with checkpoints and resume |
| `src/constrain_capi.*` | the constraint behind a C interface, without MLX |
| `src/files.h`, `src/provenance.*`, `src/json.h` | synced writes and the deny list; checkpoint identity; JSON lookups |

## Limits

- Qwen3 and Qwen3.5, text only. The vision tower and the multi-token-prediction head are not read.
- LoRA on a bf16 base only, on one Mac.
- Greedy decoding, one sequence at a time.
- macOS 26 and the mlx 0.32.3 wheel; other versions are untested.
- Schemas use the keyword subset listed in `src/schema.h`. Any other keyword is refused by name, never ignored.

## License

Apache License 2.0 (see [LICENSE](LICENSE)). Parts follow or adapt MLX and mlx-lm (MIT); [NOTICE](NOTICE) lists
them with their notices.

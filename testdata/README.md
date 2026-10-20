# Test oracles

Frozen outputs of the Python reference implementations, dumped once. Nothing here is read by the build, by
`mlxtune` or by any training or inference path; only the tests read these files.

| file | from | what the test checks |
|---|---|---|
| `tokenizer_qwen3.json`, `tokenizer_qwen35.json` | `tokenizers` 0.23.2, `transformers` 5.19.0 | ids for 313 strings; chat template text and ids for 13 prompts; the class (letter, number, whitespace, other) the split regex gives each of the 1,112,064 code points; NFC of 3,481 decomposable inputs. Pinned to the sha256 of the `tokenizer.json` it was dumped from |
| `chats_qwen3.json`, `chats_qwen35.json` | `transformers` 5.19.0 | 6 whole conversations for training (system, user and assistant turns, up to three exchanges, whitespace around turns, non-ASCII): the text `apply_chat_template` renders with no generation prompt, Hugging Face's ids for it, and each answer as the template writes it plus `<\|im_end\|>` (the loss spans `render_chat` must mark). No turn holds added-token text, so Hugging Face's ids are the template's control tokens plus plain BPE. Pinned to the sha256 of `tokenizer.json` |
| `parity_qwen35_4b.json`, `parity_qwen35_9b.json` | `mlx` 0.32.3, `mlx-lm` 0.31.3 | for 7 chat prompts (`short 1` to `short 5`, `prompt300`, `long_extract`): prompt ids, argmax at every position, top-10 log-softmax at listed positions, the 32-token greedy continuation (ids and raw bytes); for information, mlx-lm's own `stream_generate` ids and its timing. Pinned to the mlx version and the checkpoint (sha256 of `model.safetensors.index.json`, each shard's name and size) |
| `prompt300.txt`, `long_system.txt`, `long_user.txt` | `dump_fixtures.py texts` | the measurement prompts: a 300-token gardening question, and a 2,464-token (Qwen3) / 2,512-token (Qwen3.5) extraction prompt, a JSON-schema system prompt for an event booking and an organiser's informal notes |

## Content

All text in the fixtures is synthetic, written for these tests, and domain-neutral. It holds no personal data;
URLs and email addresses use `example.org` and `example.com`.
- **Chat prompts:** a general assistant's system prompt, and five short questions (baking, astronomy,
  programming, gardening), one with no system prompt and one with curly quotes and an em dash.
- **`prompt300`:** a user's ~300-token question about planning a vegetable garden, with measurements and
  temperatures.
- **`long_extract`:** about 2,450 tokens, so the prefill crosses one 2,048-token chunk boundary. The system
  prompt asks for exactly one JSON object and describes its schema (enums, nested objects, arrays, null rules);
  the user message is an organiser's long, informal notes for a society's weekend away, with conflicting figures
  and open questions.
- **Strings:** short sentences with numbers, units, percentages, dates and currency; number groupings (`62k`,
  `1,234,567.89`, `₹5,00,000`, `€1.000,50`); contractions and quote marks; Latin with combining marks and NFC/NFD
  pairs; Greek, Cyrillic, CJK, Hangul, Arabic and Hebrew (right-to-left, with and without vowel points),
  Devanagari and Thai; full-width forms, superscripts, fractions and Roman numerals; emoji with skin tones, ZWJ
  sequences, keycaps, flags and variation selectors; tabs, newlines, CRLF, runs of spaces, zero-width and other
  Unicode spaces, BOM, NEL, line and paragraph separators and C0 controls; code, JSON, SQL, a URL, an email
  address, a Windows path and HTML; long repeats; 11 strings holding added-token text; a paragraph about baking
  and three copies of it; the chat prompts; and 80 random strings drawn from 18 character pools.
- **Conversations:** a baking question and answer, a JSON extraction, a three-exchange cooking chat, single
  letters, turns padded with tabs, spaces and newlines, and a French and Japanese exchange with an emoji.

## Commands

Run from this package's root (the directory holding `testdata/`), one model at a time. `MODELS` holds the MLX
checkpoints `qwen3-8b-8bit`, `qwen3.5-4b-16bit` and `qwen3.5-9b-16bit`; `PY` is a
Python 3.14 interpreter with `mlx` 0.32.3, `mlx-lm` 0.31.3, `transformers` 5.19.0 and `tokenizers` 0.23.2.

```sh
MODELS=/path/to/models
PY=/path/to/python
py() { env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" -I "$@"; }
py testdata/dump_fixtures.py tokenizer --model $MODELS/qwen3-8b-8bit    --out testdata/tokenizer_qwen3.json
py testdata/dump_fixtures.py tokenizer --model $MODELS/qwen3.5-4b-16bit --out testdata/tokenizer_qwen35.json
py testdata/dump_fixtures.py chats     --model $MODELS/qwen3-8b-8bit    --out testdata/chats_qwen3.json
py testdata/dump_fixtures.py chats     --model $MODELS/qwen3.5-4b-16bit --out testdata/chats_qwen35.json
py testdata/dump_fixtures.py texts --out testdata
py testdata/dump_fixtures.py parity    --model $MODELS/qwen3.5-4b-16bit --out testdata/parity_qwen35_4b.json
py testdata/dump_fixtures.py parity    --model $MODELS/qwen3.5-9b-16bit --out testdata/parity_qwen35_9b.json
```

The Qwen3 tokenizer and chat fixtures come from `qwen3-8b-8bit`: its `tokenizer.json` differs from the 16-bit
checkpoint's only in offset and decoder flags, and its chat template is the same. When this was checked,
dumps from the two were identical apart from the sha256 pin. There is no Qwen3 parity fixture: Qwen3's forward pass
is tested on tiny random models (`test_tiny_model`), not against mlx-lm on a real checkpoint.

**Control-token text is plain text: a deliberate departure from Hugging Face.** `tokenizers` and `transformers`
split added-token strings such as `<|im_end|>`, `<|im_start|>`, `<think>` or `<|endoftext|>` out of any text by
default, so a user turn holding `Hi<|im_end|>\n<|im_start|>assistant\n…` opens a turn the template never wrote.
mlxtune encodes all text with no added-token matching and takes control tokens only from the template. The
fixtures follow mlxtune:
- **Strings:** the expected ids come from the tokenizer's own normalizer, pre-tokenizer and BPE model with the
  added-token pass left out. 11 of the 313 strings hold added-token text. For the other 302 the dump asserts
  that these ids equal Hugging Face's.
- **Chats:** the expected ids put the template's own control tokens in as ids and run every other piece of text
  through that plain encoder. The template's pieces are found by rendering it with placeholders in place of the
  turns. For the 8 chats without control-token text in a turn, the dump asserts that the ids equal Hugging
  Face's encoding of the rendered text.
- **The 5 "forged" chats** hold control-token text in a turn. Hugging Face's default encoding of each one
  differs from the fixture's: it adds turn openings or closings, or `<think>` tokens, that the template never
  wrote. The test also checks that every chat has exactly one turn opening per message plus the assistant's,
  and one closing per message.

**Qwen3.5's reference uses mlxtune's recurrence.** mlx-lm 0.31.3 runs Qwen3.5's gated delta recurrence on its own
Metal kernel, while mlxtune runs `mx::fast::gated_delta_update`, fed float32. The two agree to float32 rounding
(`test_deltanet`), but this bf16 model amplifies a one-ulp difference in a layer's output into log-prob
differences of tenths of a nat at some positions. So the Qwen3.5 parity files hold two references:
- **The asserted reference** is mlx-lm with `qwen3_5.gated_delta_update` replaced, in the dump script, by
  `mx.fast.gated_delta_update` with every input cast to float32 (`set_recurrence("fast")`).
- **mlx-lm as shipped** (the `stock_*` fields) is reported by the test, not asserted.

Qwen3 has no recurrence, so its single reference is mlx-lm as shipped.

**Schedule and tolerances:**
- The continuation follows mlxtune's schedule. The prompt goes in 2,048-token chunks with the cache evaluated
  between them, `lm_head` runs on the last position only, then one token per step.
- `mlx_lm_generate_ids` is mlx-lm's own loop, which prefills all but the last prompt token. The test reports it
  and does not assert it.
- Each prompt's `tolerance` is 2 bf16 ulps of the reference's own largest |logit|.
- `tight_tolerance` (1e-4 on log-softmax) is the same-version bound. The fixture rounds log-probs to 6
  decimals.

sha256 (`shasum -a 256 testdata/{chats,parity,tokenizer}_*.json testdata/*.txt`):

```
da07c3338a16d6d41044cd2e9bce67abc48f0f71615414249dadfc034a04aec1  testdata/chats_qwen3.json
964d53a06b58621651caf747a1de642a004a670b0166cab073fd0382a3308455  testdata/chats_qwen35.json
cd22eddf1c1d1d1a1a3f10f2f0c28a3cd9ff28a9e164c61fd41730d81b34e137  testdata/parity_qwen35_4b.json
0503ae90c8fee3f868c4a933241cfbd9adad58ba60c5631fc6d8040ca7ea1061  testdata/parity_qwen35_9b.json
f9191c2fa6d460621a1f31b533433db3e59a687ae85e026bd053aee06cab4f6a  testdata/tokenizer_qwen3.json
6ddcd5e85ac4a30f467c6578b987e2b29103c5fce96060ffe26e8f41259f2bbc  testdata/tokenizer_qwen35.json
bb9af6fd12b60b6a41c8aebded0653797d7e6c84223f6df94add012367301c3b  testdata/long_system.txt
da5ca2f82597503990b9ee7bb2a1b8284fb6232ce341679372226a4f0f82b6a2  testdata/long_user.txt
822309c7e4d4f8bac7c5aae5e93ddff0c066efe98a9840db933459ad236191c2  testdata/prompt300.txt
```

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "batch.h"
#include "constrain.h"
#include "exact.h"
#include "schema.h"
#include "files.h"
#include "generate.h"
#include "json.h"
#include "lora.h"
#include "model.h"
#include "provenance.h"
#include "tokenizer.h"
#include "train.h"

static int usage() {
  std::fprintf(stderr,
               "usage: quotebound generate --model DIR [--adapter DIR [--merge-lora]] [--quantize 8]\n"
               "                        (--user U [--system S] | --messages FILE)\n"
               "                        [--max-tokens N] [--repeat N] [--schema FILE [--quote-source user]]\n"
               "                        [--token-confidence]\n"
               "  greedy chat completion; text to stdout, one JSON timing line per run to stderr.\n"
               "  --messages FILE is JSON {\"messages\": [{\"role\", \"content\"}, ...]} ending in a user turn.\n"
               "  --max-tokens defaults to 256; --repeat N generates N times in one process (default 1).\n"
               "  --adapter reads DIR/adapters.safetensors and DIR/adapter_config.json, as train writes them;\n"
               "  --merge-lora folds them into the weights at load (W + scale B A in float32, rounded once to bf16).\n"
               "  --quantize 8 runs 8-bit weights (MLX affine, groups of 64 inputs), quantized at load: every\n"
               "  projection but the DeltaNet's in_proj_b/in_proj_a, and the output head; the embedding lookup, norms\n"
               "  and the recurrence stay as they are. With an adapter it needs --merge-lora (merged, then quantized).\n"
               "  --schema FILE constrains the answer to compact JSON (json.dumps' separators) the JSON Schema\n"
               "  accepts, closing it within --max-tokens; src/schema.h lists the keywords read, any other is\n"
               "  refused. --quote-source user makes every string at a quote key (or a $ref to a definition named\n"
               "  quote) a stretch of the last user turn standing as whole words, letter case aside.\n"
               "  --token-confidence adds \"tokens\": [[start, end, logprob, margin], ...] to the JSON line: each\n"
               "  generated token's bytes of the text, its log-probability under the model's whole unmasked\n"
               "  vocabulary and the top logit less the second; the text is the same as without it.\n"
               "usage: quotebound generate --model DIR [--adapter DIR [--merge-lora]] [--quantize 8] --batch IN --out OUT\n"
               "                        [--resume] [--max-tokens N]\n"
               "                        [--schema FILE] [--quote-source user] [--token-confidence]\n"
               "                        [--draft-train FILE [--draft-k 8]]\n"
               "  one load, many conversations: IN is JSONL {\"id\", \"messages\"} (- is stdin); OUT gets one JSON line\n"
               "  per row, {\"id\", \"text\", \"prompt_tokens\", \"gen_tokens\", \"prefill_ms\", \"decode_ms\",\n"
               "  \"total_ms\", \"truncated\"}, in order, each synced as written (- is stdout). --resume continues an\n"
               "  OUT that exists, skipping the ids it holds. A row's own \"schema\" (a JSON Schema object)\n"
               "  constrains it in place of --schema; constrained rows add \"mask_ms\" and \"budget_cut\".\n"
               "  --draft-train FILE (a training set, JSONL {\"messages\"}) decodes the constrained rows by exact\n"
               "  speculative decoding: up to --draft-k tokens drafted from its assistant turns' n-grams and the\n"
               "  prompt, verified in one pass; the answers are the plain decoder's, and each row adds \"passes\"\n"
               "  and \"drafted\". Not with --token-confidence.\n"
               "usage: quotebound train --model DIR --train FILE --dev FILE --out DIR (--steps N | --epochs N)\n"
               "                     [--rank 16] [--alpha 32] [--targets a,b,...] [--lr 1e-4] [--batch-tokens 4096]\n"
               "                     [--max-len 2048] [--seed 0] [--eval-every 50] [--save-every 100] [--warmup N]\n"
               "                     [--weight-decay 0] [--clip 1] [--dev-max 200] [--gen-dev 16] [--keep 5]\n"
               "                     [--patience 0] [--resume]\n"
               "  LoRA on a frozen bf16 model. FILE is JSONL, one {\"messages\": [{\"role\", \"content\"}, ...]} per\n"
               "  line, trained on the assistant turns. One JSON line per step and per dev evaluation to stdout and\n"
               "  DIR/train.log. --targets defaults to q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,\n"
               "  gate_proj,up_proj,down_proj; --warmup to 5%% of the steps. --resume continues from\n"
               "  DIR/state.safetensors.\n");
  return 2;
}

[[noreturn]] static void bad(const std::string& flag, const char* value, const char* want) {
  throw std::runtime_error(flag + " " + json_quote(value) + ": want " + want);
}

// A whole decimal integer >= lo.
static int whole(const std::string& flag, const char* s, int lo = 1) {
  errno = 0;
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (errno || end == s || *end != '\0' || v < lo || v > INT_MAX)
    bad(flag, s, lo == 0 ? "a whole number >= 0" : "a whole number >= 1");
  return int(v);
}

// A finite number >= 0.
static float number(const std::string& flag, const char* s) {
  errno = 0;
  char* end = nullptr;
  float v = std::strtof(s, &end);
  if (errno || end == s || *end != '\0' || !std::isfinite(v) || v < 0) bad(flag, s, "a finite number >= 0");
  return v;
}

static void wire_weights() {
  // As mlx-lm does: keep the weights resident instead of letting Metal page them.
  auto info = mx::device_info();
  if (auto it = info.find("max_recommended_working_set_size"); it != info.end())
    mx::set_wired_limit(std::get<size_t>(it->second));
}

static std::vector<Turn> read_messages(const std::string& path) {
  simdjson::dom::parser parser;
  std::vector<Turn> turns;
  for (simdjson::dom::element m : get<simdjson::dom::array>(load_json(parser, path), "messages", path))
    turns.push_back({std::string(get<std::string_view>(m, "role", path + " message")),
                     std::string(get<std::string_view>(m, "content", path + " message"))});
  return turns;
}

// generate --batch: every row read and its prompt rendered and checked before the weights are read; the answers in
// input order, one JSON line each, appended to OUT and synced as each finishes (or written to stdout).
static int generate_batch(const std::string& model_dir, const std::string& adapter_dir, const std::string& in_path,
                          const std::string& out_path, bool resume, int max_tokens, const std::string& schema_path,
                          bool quote_user, bool confidence, bool merge, const std::string& draft_train, int draft_k,
                          int quant_bits) {
  if (out_path.empty()) throw std::runtime_error("--batch needs --out (a file, or - for stdout)");
  if (!draft_train.empty() && confidence) throw std::runtime_error("--draft-train does not go with --token-confidence");
  if (!draft_train.empty() && quant_bits) throw std::runtime_error("--draft-train runs bf16 weights: not with --quantize");
  if (!draft_train.empty() && (draft_k < 1 || draft_k >= kMaxRows))
    throw std::runtime_error("--draft-k " + std::to_string(draft_k) + " outside 1.." + std::to_string(kMaxRows - 1));
  refuse_denied("--draft-train", draft_train);
  if (resume && out_path == "-") throw std::runtime_error("--resume needs --out FILE");
  refuse_denied("--model", model_dir), refuse_denied("--adapter", adapter_dir);
  if (in_path != "-") refuse_denied("--batch", in_path);
  if (out_path != "-") refuse_denied("--out", out_path);

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  Tokenizer tok = load_tokenizer(model_dir);
  const double tok_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  std::vector<BatchRow> rows;
  if (in_path == "-") {
    rows = read_batch(std::cin, "stdin");
  } else {
    std::ifstream f(in_path);
    if (!f) throw std::runtime_error("cannot read " + in_path);
    rows = read_batch(f, in_path);
  }
  const int max_positions = load_config(model_dir + "/config.json").max_positions;
  BatchConstraints constraints;
  std::optional<Schema> fallback;
  std::optional<VocabTrie> trie;
  if (!schema_path.empty()) fallback = load_schema(schema_path), constraints.fallback = &*fallback;
  simdjson::dom::parser schema_parser;
  for (const BatchRow& r : rows)
    if (!r.schema.empty() && !constraints.by_text.count(r.schema))
      constraints.by_text.emplace(r.schema, compile_schema(schema_parser.parse(r.schema), "row " + json_quote(r.id) +
                                                                                           " schema"));
  const bool constrained = fallback || !constraints.by_text.empty();
  if (quote_user && !constrained) throw std::runtime_error("--quote-source needs --schema or rows with a schema");
  if (constrained) trie = build_trie(tok), constraints.trie = &*trie, constraints.quote_user = quote_user;
  std::optional<BatchDrafts> drafts;
  double draft_ms = 0;
  if (!draft_train.empty()) {
    if (!constrained) throw std::runtime_error("--draft-train needs --schema or rows with a schema");
    const auto d0 = clock::now();
    drafts.emplace(BatchDrafts{training_tables(tok, draft_train), draft_k});
    draft_ms = std::chrono::duration<double, std::milli>(clock::now() - d0).count();
  }
  std::vector<std::vector<int>> prompts;
  for (const BatchRow& r : rows) {
    try {
      prompts.push_back(chat_ids(tok, r.turns));
    } catch (const std::exception& e) {
      throw std::runtime_error("row " + json_quote(r.id) + ": " + e.what());
    }
    if (prompts.back().size() + size_t(max_tokens) > size_t(max_positions))
      throw std::runtime_error("row " + json_quote(r.id) + ": prompt (" + std::to_string(prompts.back().size()) +
                               " tokens) + --max-tokens (" + std::to_string(max_tokens) +
                               ") exceeds max_position_embeddings (" + std::to_string(max_positions) + ")");
  }
  std::set<std::string> done;
  if (out_path != "-" && std::filesystem::exists(out_path)) {
    if (!resume) throw std::runtime_error("--out " + out_path + " exists: pass --resume to continue it");
    done = resume_ids(out_path, rows);
  }
  t0 = clock::now();
  Model m = load_model(model_dir);
  if (!adapter_dir.empty()) load_adapters(adapter_dir, m, identify_base(model_dir, tok));
  const int merged = merge ? merge_lora(m) : 0;
  const int quantized = quant_bits ? quantize_weights(m, quant_bits, 64) : 0;
  const double load_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  wire_weights();
  std::string extra = merge ? ",\"merged_lora\":" + std::to_string(merged) : "";
  if (quant_bits)
    extra += ",\"quantized\":" + std::to_string(quantized) + ",\"quant_bits\":" + std::to_string(quant_bits) +
             ",\"quant_group\":64";
  if (drafts) {
    char d[128];
    std::snprintf(d, sizeof d, ",\"draft_k\":%d,\"draft_tables\":%zu,\"draft_ms\":%.0f", draft_k, drafts->tables.size(),
                  draft_ms);
    extra += ",\"draft_train\":" + json_quote(draft_train) + d;
  }
  std::fprintf(stderr,
               "{\"mlx\":\"%s\",\"load_ms\":%.0f,\"tokenizer_ms\":%.0f,\"rows\":%zu,\"resumed\":%zu,"
               "\"max_tokens\":%d%s}\n",
               mx::version(), load_ms, tok_ms, rows.size(), done.size(), max_tokens, extra.c_str());
  std::optional<LineFile> file;
  if (out_path != "-") file.emplace(out_path);
  size_t written = 0;
  t0 = clock::now();
  const BatchStats stats = run_batch(m, tok, rows, prompts, done, max_tokens, [&](const BatchResult& r) {
    const std::string line = batch_line(r);
    if (file) {
      file->append(line);
    } else {
      std::fwrite(line.data(), 1, line.size(), stdout);
      std::fflush(stdout);
    }
    ++written;
  }, constrained ? &constraints : nullptr, confidence, drafts ? &*drafts : nullptr);
  std::string end;
  if (drafts)
    end = ",\"draft_tokens\":" + std::to_string(stats.tokens) + ",\"passes\":" + std::to_string(stats.passes) +
           ",\"drafted\":" + std::to_string(stats.drafted);
  std::fprintf(stderr, "{\"generated\":%zu,\"ms\":%.0f,\"peak_memory_gb\":%.3f%s}\n", written,
               std::chrono::duration<double, std::milli>(clock::now() - t0).count(), mx::get_peak_memory() / 1e9,
               end.c_str());
  return 0;
}

static int generate(int argc, char** argv) {
  std::string model_dir, adapter_dir, messages, batch, out, schema_path, quote_source;
  std::optional<std::string> system, user;
  std::string draft_train;
  int max_tokens = 256, repeat = 1, draft_k = 8, quant_bits = 0;
  bool resume = false, repeat_given = false, confidence = false, merge = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--resume" || a == "--token-confidence" || a == "--merge-lora") {
      (a == "--resume" ? resume : a == "--merge-lora" ? merge : confidence) = true;
      continue;
    }
    if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
    const char* value = argv[++i];
    if (a == "--model") model_dir = value;
    else if (a == "--adapter") adapter_dir = value;
    else if (a == "--system") system = value;
    else if (a == "--user") user = value;
    else if (a == "--messages") messages = value;
    else if (a == "--batch") batch = value;
    else if (a == "--out") out = value;
    else if (a == "--max-tokens") max_tokens = whole(a, value);
    else if (a == "--repeat") repeat = whole(a, value), repeat_given = true;
    else if (a == "--schema") schema_path = value;
    else if (a == "--quote-source") quote_source = value;
    else if (a == "--draft-train") draft_train = value;
    else if (a == "--draft-k") draft_k = whole(a, value);
    else if (a == "--quantize") quant_bits = whole(a, value);
    else throw std::runtime_error("unknown flag " + a);
  }
  if (model_dir.empty()) throw std::runtime_error("--model is required");
  if (!quote_source.empty() && quote_source != "user") bad("--quote-source", quote_source.c_str(), "user");
  if (merge && adapter_dir.empty()) throw std::runtime_error("--merge-lora needs --adapter");
  if (quant_bits && quant_bits != 8) bad("--quantize", std::to_string(quant_bits).c_str(), "8");
  if (quant_bits && !adapter_dir.empty() && !merge) throw std::runtime_error("--quantize with --adapter needs --merge-lora");
  if (!schema_path.empty()) refuse_denied("--schema", schema_path);
  if (!batch.empty()) {
    if (user || system || !messages.empty() || repeat_given)
      throw std::runtime_error("--batch takes no --user, --system, --messages or --repeat");
    return generate_batch(model_dir, adapter_dir, batch, out, resume, max_tokens, schema_path, !quote_source.empty(),
                          confidence, merge, draft_train, draft_k, quant_bits);
  }
  if (!out.empty() || resume || !draft_train.empty())
    throw std::runtime_error("--out, --resume and --draft-train go with --batch");
  if (messages.empty() == !user) throw std::runtime_error("give --user (with an optional --system) or --messages");
  if (!messages.empty() && system) throw std::runtime_error("--system goes in --messages' system turn");
  refuse_denied("--model", model_dir), refuse_denied("--adapter", adapter_dir), refuse_denied("--messages", messages);

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  Tokenizer tok = load_tokenizer(model_dir);
  double tok_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  const std::vector<Turn> turns = messages.empty() ? prompt_turns(system, *user) : read_messages(messages);
  const std::vector<int> prompt = chat_ids(tok, turns);
  if (!quote_source.empty() && schema_path.empty()) throw std::runtime_error("--quote-source needs --schema");
  std::optional<Schema> schema;
  std::optional<VocabTrie> trie;
  std::optional<QuoteText> quotes;
  if (!schema_path.empty()) {
    schema = load_schema(schema_path), trie = build_trie(tok);
    if (!quote_source.empty()) quotes = quote_text(turns.back().content);
  }
  // Refuse a request the model cannot hold before reading the weights (greedy checks it again).
  const int max_positions = load_config(model_dir + "/config.json").max_positions;
  if (prompt.size() + size_t(max_tokens) > size_t(max_positions))
    throw std::runtime_error("prompt (" + std::to_string(prompt.size()) + " tokens) + --max-tokens (" +
                             std::to_string(max_tokens) + ") exceeds max_position_embeddings (" +
                             std::to_string(max_positions) + ")");
  t0 = clock::now();
  Model m = load_model(model_dir);
  if (!adapter_dir.empty()) load_adapters(adapter_dir, m, identify_base(model_dir, tok));
  if (merge) merge_lora(m);
  if (quant_bits) quantize_weights(m, quant_bits, 64);
  double load_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  wire_weights();

  std::vector<int> stop = m.cfg.eos;
  stop.push_back(token_id(tok, "<|im_end|>"));
  stop.push_back(token_id(tok, "<|endoftext|>"));
  for (int run = 1; run <= repeat; ++run) {
    std::optional<Constraint> c;
    if (schema) c.emplace(*schema, tok, *trie, stop, quotes ? &*quotes : nullptr);
    Generation g = greedy(m, prompt, max_tokens, stop, [&](int id) {
      if (std::find(stop.begin(), stop.end(), id) != stop.end()) return;
      std::string piece = decode(tok, {id});  // raw bytes: a split UTF-8 sequence completes on the next write
      std::fwrite(piece.data(), 1, piece.size(), stdout);
      std::fflush(stdout);
    }, c ? &*c : nullptr, confidence);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    size_t n = g.ids.size();
    const bool stopped = n && std::find(stop.begin(), stop.end(), g.ids.back()) != stop.end();
    const std::string tokens = confidence ? tokens_json(token_records(tok, g, stopped)) : "";
    std::fprintf(stderr,
                 "{\"run\":%d,\"mlx\":\"%s\",\"load_ms\":%.0f,\"tokenizer_ms\":%.0f,\"prompt_tokens\":%zu,"
                 "\"prefill_ms\":%.1f,\"prefill_tok_s\":%.1f,\"gen_tokens\":%zu,\"decode_tok_s\":%.2f,"
                 "\"peak_memory_bytes\":%zu,\"peak_memory_gb\":%.3f%s%s}\n",
                 run, mx::version(), load_ms, tok_ms, prompt.size(), g.prefill_ms,
                 prompt.size() / (g.prefill_ms / 1000.0), n, n > 1 ? (n - 1) / (g.decode_ms / 1000.0) : 0.0,
                 mx::get_peak_memory(), mx::get_peak_memory() / 1e9,
                 c ? (",\"mask_ms\":" + std::to_string(g.mask_ms) + ",\"budget_cut\":" +
                      (c->budget_cut ? "true" : "false") + overrides_json(g.overridden)).c_str()
                   : "",
                 tokens.c_str());
  }
  return 0;
}

static int train(int argc, char** argv) {
  TrainArgs a;
  a.lora.targets = default_targets();
  bool resume = false;
  for (int i = 2; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--resume") {
      resume = true;
      continue;
    }
    if (i + 1 >= argc) throw std::runtime_error(k + " needs a value");
    const char* value = argv[++i];
    if (k == "--model") a.model_dir = value;
    else if (k == "--train") a.train_path = value;
    else if (k == "--dev") a.dev_path = value;
    else if (k == "--out") a.out_dir = value;
    else if (k == "--rank") a.lora.rank = whole(k, value);
    else if (k == "--alpha") a.lora.alpha = number(k, value);
    else if (k == "--lr") a.lr = number(k, value);
    else if (k == "--steps") a.steps = whole(k, value);
    else if (k == "--epochs") a.epochs = whole(k, value);
    else if (k == "--batch-tokens") a.batch_tokens = whole(k, value);
    else if (k == "--max-len") a.max_len = whole(k, value);
    else if (k == "--eval-every") a.eval_every = whole(k, value);
    else if (k == "--save-every") a.save_every = whole(k, value);
    else if (k == "--warmup") a.warmup = whole(k, value, 0);
    else if (k == "--weight-decay") a.adamw.weight_decay = number(k, value);
    else if (k == "--clip") a.adamw.clip = number(k, value);
    else if (k == "--dev-max") a.dev_max = whole(k, value);
    else if (k == "--gen-dev") a.gen_dev = whole(k, value, 0);
    else if (k == "--keep") a.keep = whole(k, value, 0);
    else if (k == "--patience") a.patience = whole(k, value, 0);
    else if (k == "--seed") a.seed = uint64_t(whole(k, value, 0));
    else if (k == "--targets") {
      a.lora.targets.clear();
      for (std::string rest = value; !rest.empty();) {
        size_t comma = rest.find(',');
        if (comma != 0) a.lora.targets.push_back(rest.substr(0, comma));
        rest = comma == std::string::npos ? "" : rest.substr(comma + 1);
      }
      if (a.lora.targets.empty()) bad(k, value, "projection names separated by commas");
    } else {
      throw std::runtime_error("unknown flag " + k);
    }
  }
  for (const auto& [flag, path] : {std::pair<std::string, std::string>{"--model", a.model_dir},
                                   {"--train", a.train_path}, {"--dev", a.dev_path}, {"--out", a.out_dir}}) {
    if (path.empty()) throw std::runtime_error(flag + " is required");
    refuse_denied(flag, path);
  }
  check_args(a);

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  Tokenizer tok = load_tokenizer(a.model_dir);
  Dataset train = load_examples(a.train_path, tok, a.max_len), dev = load_examples(a.dev_path, tok, a.max_len);
  const BaseModel base = identify_base(a.model_dir, tok);
  Model m = load_model(a.model_dir);
  const double load_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  wire_weights();
  Trainer t = make_trainer(std::move(m), std::move(train), std::move(dev), a, base, &tok);
  start_or_resume(t, resume);
  std::filesystem::create_directories(a.out_dir);
  std::FILE* file = std::fopen((a.out_dir + "/train.log").c_str(), "a");
  if (!file) throw std::runtime_error("cannot write " + a.out_dir + "/train.log");
  size_t params = 0;
  for (const mx::array& p : adapter_arrays(t.model)) params += p.size();
  auto ids = [](const std::vector<std::string>& v) {
    std::string out = "[";
    for (const std::string& id : v) out += (out.size() > 1 ? "," : "") + json_quote(id);
    return out + "]";
  };
  char counts[512];
  std::snprintf(counts, sizeof counts,
                "{\"mlx\":\"%s\",\"load_ms\":%.0f,\"train_examples\":%zu,\"dev_examples\":%zu,\"dev_scored\":%zu,"
                "\"adapter_params\":%zu,\"total_steps\":%d,\"warmup\":%d,\"resumed_at\":%d,",
                mx::version(), load_ms, t.train.size(), t.dev_rows, t.dev.size(), params, t.total_steps, t.warmup,
                t.step);
  // The training rows left out, by id; run.json holds them too, with the dev rows scored and generated.
  const std::string head = std::string(counts) + "\"train_too_long\":" + ids(t.train_meta.too_long) +
                           ",\"train_no_answer\":" + ids(t.train_meta.no_answer) + "}\n";
  for (std::FILE* f : {stdout, file}) std::fputs(head.c_str(), f), std::fflush(f);
  run_training(t, {stdout, file});
  std::fclose(file);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  try {
    if (std::strcmp(argv[1], "generate") == 0) return generate(argc, argv);
    if (std::strcmp(argv[1], "train") == 0) return train(argc, argv);
    return usage();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "quotebound: %s\n", e.what());
    return 1;
  }
}

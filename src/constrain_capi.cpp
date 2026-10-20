#include "constrain_capi.h"

#include <simdjson.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "constrain.h"
#include "schema.h"
#include "tokenizer.h"

struct qb_vocab {
  Tokenizer tok;
  VocabTrie trie;
};

// Constraint holds references to its schema, tokenizer, trie and quote text: the schema and quotes live here beside
// it, declared first so they outlive it; the vocabulary must outlive every constraint made from it.
struct qb_constraint {
  Schema schema;
  std::optional<QuoteText> quotes;
  std::unique_ptr<Constraint> c;
};

namespace {

void set_err(char* err, int errlen, const char* what) {
  if (!err || errlen <= 0) return;
  std::strncpy(err, what, size_t(errlen) - 1);
  err[errlen - 1] = '\0';
}

// Runs f, turning any exception into `fail` and its message into err.
template <class F, class R>
R guarded(char* err, int errlen, R fail, F&& f) {
  try {
    return f();
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
  } catch (...) {
    set_err(err, errlen, "unknown error");
  }
  return fail;
}

}  // namespace

extern "C" {

qb_vocab* qb_vocab_load(const char* model_dir, char* err, int errlen) {
  return guarded(err, errlen, static_cast<qb_vocab*>(nullptr), [&] {
    auto v = std::make_unique<qb_vocab>();
    v->tok = load_tokenizer(model_dir);
    v->trie = build_trie(v->tok);
    return v.release();
  });
}

int qb_vocab_size(const qb_vocab* v) { return int(v->tok.id_bytes.size()); }

int qb_vocab_token_bytes(const qb_vocab* v, int32_t id, char* out, int cap) {
  if (id < 0 || size_t(id) >= v->tok.id_bytes.size()) return -1;
  const std::string& b = v->tok.id_bytes[size_t(id)];
  std::memcpy(out, b.data(), std::min(b.size(), size_t(std::max(cap, 0))));
  return int(b.size());
}

void qb_vocab_free(qb_vocab* v) { delete v; }

qb_constraint* qb_constraint_new(const qb_vocab* v, const char* schema_json, const int32_t* stop, int nstop,
                                 const char* quote_text, char* err, int errlen) {
  return guarded(err, errlen, static_cast<qb_constraint*>(nullptr), [&] {
    auto c = std::make_unique<qb_constraint>();
    simdjson::dom::parser parser;
    c->schema = compile_schema(parser.parse(std::string(schema_json)), "schema");
    if (quote_text) c->quotes = ::quote_text(quote_text);
    c->c = std::make_unique<Constraint>(c->schema, v->tok, v->trie, std::vector<int>(stop, stop + nstop),
                                        c->quotes ? &*c->quotes : nullptr);
    return c.release();
  });
}

int qb_constraint_allowed(qb_constraint* c, int remaining, uint8_t* mask, int n, char* err, int errlen) {
  return guarded(err, errlen, -1, [&] {
    std::memset(mask, 0, size_t(std::max(n, 0)));
    const std::vector<int> allow = c->c->allowed(remaining);
    for (int id : allow) {
      if (id < 0 || id >= n) throw std::runtime_error("allowed id " + std::to_string(id) + " is outside the mask");
      mask[id] = 1;
    }
    return int(allow.size());
  });
}

int qb_constraint_accept(qb_constraint* c, int32_t id, char* err, int errlen) {
  return guarded(err, errlen, -1, [&] {
    c->c->accept(id);
    return 0;
  });
}

int qb_constraint_complete(const qb_constraint* c) { return c->c->complete() ? 1 : 0; }
int qb_constraint_min_close(const qb_constraint* c) { return c->c->min_close(); }
int qb_constraint_budget_cut(const qb_constraint* c) { return c->c->budget_cut ? 1 : 0; }
int64_t qb_constraint_bytes(const qb_constraint* c) { return int64_t(c->c->bytes()); }
void qb_constraint_free(qb_constraint* c) { delete c; }

}  // extern "C"

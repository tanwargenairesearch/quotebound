#pragma once
// quotebound's schema- and quote-constrained decoding (constrain.h) behind a C interface, for a decoder outside
// quotebound: a CUDA or CPU decoder can mask its logits with exactly the tokens quotebound's own constraint allows, so
// the two decoders differ only in the model's arithmetic, never in what an answer may be.
//
// Built without MLX, as build/libquotebound_constrain.{dylib,so} (make constrain-lib); on Linux with
// QUOTEBOUND_NO_COREFOUNDATION (unicode.cpp), since only tokenizing needs CoreFoundation and this never tokenizes.
// Every call that can fail returns NULL or -1 and writes why into err (cut to errlen, NUL-terminated); nothing throws
// across it.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qb_vocab qb_vocab;            // a checkpoint's tokenizer and the trie of its token bytes
typedef struct qb_constraint qb_constraint;  // one answer's constraint, its schema and quote text held inside

// The tokenizer of model_dir (tokenizer.json and its chat template), as quotebound generate loads it, and its trie.
qb_vocab* qb_vocab_load(const char* model_dir, char* err, int errlen);
// The tokenizer's id table size: ids at or past it are never allowed.
int qb_vocab_size(const qb_vocab* v);
// The bytes of token id into out (up to cap), their length returned; -1 for an id outside the table.
int qb_vocab_token_bytes(const qb_vocab* v, int32_t id, char* out, int cap);
void qb_vocab_free(qb_vocab* v);

// A constraint under the JSON Schema in schema_json (compiled as quotebound --schema compiles a file), its answer ending
// only once complete, with one of stop[0, nstop). quote_text: quotes are stretches of it as --quote-source user takes
// the user turn; NULL leaves quote strings free.
qb_constraint* qb_constraint_new(const qb_vocab* v, const char* schema_json, const int32_t* stop, int nstop,
                                 const char* quote_text, char* err, int errlen);
// mask[0, n) set to 1 at each id that may come next with `remaining` tokens left (this one included) and 0 elsewhere,
// n at least the vocabulary's size (ids past it stay 0); the count allowed returned (0: none can continue), -1 on error.
int qb_constraint_allowed(qb_constraint* c, int remaining, uint8_t* mask, int n, char* err, int errlen);
// Advances by token id: 0, or -1 when its bytes leave the schema.
int qb_constraint_accept(qb_constraint* c, int32_t id, char* err, int errlen);
int qb_constraint_complete(const qb_constraint* c);
// The fewest bytes that complete the answer from here (0 when complete).
int qb_constraint_min_close(const qb_constraint* c);
// 1 once some step left a token out only for the token budget.
int qb_constraint_budget_cut(const qb_constraint* c);
// The answer's bytes so far.
int64_t qb_constraint_bytes(const qb_constraint* c);
void qb_constraint_free(qb_constraint* c);

#ifdef __cplusplus
}
#endif

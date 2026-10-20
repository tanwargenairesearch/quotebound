#include "constrain.h"

#include <algorithm>
#include <bit>
#include <climits>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "json.h"
#include "unicode.h"

namespace {

constexpr int INF = INT_MAX / 4;
enum : uint8_t { K_OBJ, K_ARR, K_STR, K_NUM, K_LIT };
enum : uint8_t { O_OPEN, O_FIRST, O_KEYQ, O_KEY, O_COLON, O_VALUE, O_NEXT };
enum : uint8_t { A_OPEN, A_FIRST, A_ITEM, A_NEXT };
enum : uint8_t { S_OPEN, S_BODY, S_ESC, S_U };
enum : uint8_t { N_START, N_MINUS, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP };

int add(int a, int b) { return (a >= INF || b >= INF) ? INF : std::min(INF, a + b); }
bool digit(uint8_t b) { return b >= '0' && b <= '9'; }
constexpr uint64_t kFnv0 = 1469598103934665603ull;
uint64_t fnv(uint64_t h, uint8_t b) { return (h ^ b) * 1099511628211ull; }

int num_next(uint8_t ph, uint8_t b, const Shape& sh) {
  const bool whole = sh.type == JType::Integer;
  const bool exp = !whole && (b == 'e' || b == 'E');
  switch (ph) {
    case N_START:
      if (b == '-' && !sh.nonneg) return N_MINUS;
      [[fallthrough]];
    case N_MINUS: return b == '0' ? N_ZERO : (b >= '1' && b <= '9') ? N_INT : -1;
    case N_ZERO: return (!whole && b == '.') ? N_DOT : exp ? N_E : -1;
    case N_INT: return digit(b) ? N_INT : (!whole && b == '.') ? N_DOT : exp ? N_E : -1;
    case N_DOT: return digit(b) ? N_FRAC : -1;
    case N_FRAC: return digit(b) ? N_FRAC : exp ? N_E : -1;
    case N_E: return (b == '+' || b == '-') ? N_ESIGN : digit(b) ? N_EXP : -1;
    case N_ESIGN:
    case N_EXP: return digit(b) ? N_EXP : -1;
  }
  return -1;
}
bool num_accepting(uint8_t ph) { return ph == N_ZERO || ph == N_INT || ph == N_FRAC || ph == N_EXP; }

// Python's str.isspace, which str.strip() removes.
bool py_space(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool word(char32_t c) { return c < 128 && (std::isalnum(int(c)) || c == '_'); }
bool adigit(char32_t c) { return c >= '0' && c <= '9'; }

// The fewest bytes one character of the class takes inside a JSON string.
int min_char_bytes(const CharClass& c) {
  if (c.any) return 1;
  for (char32_t x = 0x20; x < 0x7F; ++x)
    if (x != '"' && x != '\\' && in_class(c, x)) return 1;
  for (char32_t x = 0x80; x < 0x800; ++x)
    if (in_class(c, x)) return 2;
  return 3;
}

// Whether the class holds some code point in [lo, hi].
bool class_meets(const CharClass& c, char32_t lo, char32_t hi) {
  if (c.any) return true;
  for (auto [a, b] : c.ranges) {
    if (c.negated && a <= lo && b >= hi) return false;  // one range excludes them all
    if (!c.negated && a <= hi && b >= lo) return true;
  }
  return c.negated;
}

bool char_done(Frame& f, const Shape& sh, char32_t cp) {
  if (!in_class(sh.cls, cp)) return false;
  return ++f.chars <= sh.max_len;
}

// One byte of a string's body, as json.dumps writes it: UTF-8, its short escapes, \u00XX for the other control
// characters, the class and the length checked as each character completes.
bool string_byte(Frame& f, const Shape& sh, uint8_t b) {
  switch (f.phase) {
    case S_BODY:
      if (f.need) {
        if ((b & 0xC0) != 0x80) return false;
        if ((f.lead == 0xE0 && b < 0xA0) || (f.lead == 0xED && b > 0x9F) || (f.lead == 0xF0 && b < 0x90) ||
            (f.lead == 0xF4 && b > 0x8F))
          return false;
        f.lead = 0;
        f.cp = (f.cp << 6) | (b & 0x3F);
        return --f.need ? true : char_done(f, sh, f.cp);
      }
      if (b < 0x20 || b == '"') return false;
      if (b == '\\') {
        f.phase = S_ESC;
        return f.chars < sh.max_len;  // an escape begun at maxLength could never end within it
      }
      if (b < 0x80) return char_done(f, sh, b);
      if (f.chars >= sh.max_len) return false;
      f.lead = (b == 0xE0 || b == 0xED || b == 0xF0 || b == 0xF4) ? b : 0;
      if (b >= 0xC2 && b <= 0xDF) f.need = 1, f.cp = b & 0x1F;
      else if (b >= 0xE0 && b <= 0xEF) f.need = 2, f.cp = b & 0x0F;
      else if (b >= 0xF0 && b <= 0xF4) f.need = 3, f.cp = b & 0x07;
      else return false;
      {  // and only if the class holds some character this byte can begin
        const int bits = 6 * f.need;
        const char32_t lo = std::max<char32_t>(f.cp << bits, f.need == 1 ? 0x80 : f.need == 2 ? 0x800 : 0x10000);
        const char32_t hi = std::min<char32_t>(((f.cp + 1) << bits) - 1, 0x10FFFF);
        if (!class_meets(sh.cls, lo, hi)) return false;
      }
      return true;
    case S_ESC: {
      char32_t cp;
      switch (b) {
        case '"': cp = '"'; break;
        case '\\': cp = '\\'; break;
        case 'n': cp = '\n'; break;
        case 'r': cp = '\r'; break;
        case 't': cp = '\t'; break;
        case 'b': cp = '\b'; break;
        case 'f': cp = '\f'; break;
        case 'u': return f.phase = S_U, f.u = 0, f.cp = 0, true;
        default: return false;
      }
      f.phase = S_BODY;
      return char_done(f, sh, cp);
    }
    case S_U: {
      // only \u00XX, lower-case hex, for a control character with no short escape
      const int h = digit(b) ? b - '0' : (b >= 'a' && b <= 'f') ? b - 'a' + 10 : -1;
      if (h < 0 || (f.u < 2 && h != 0) || (f.u == 2 && h > 1)) return false;
      f.cp = f.cp * 16 + char32_t(h);
      if (++f.u < 4) return true;
      if (f.cp == 8 || f.cp == 9 || f.cp == 10 || f.cp == 12 || f.cp == 13) return false;
      f.phase = S_BODY;
      return char_done(f, sh, f.cp);
    }
  }
  return false;
}

// A free string's bytes to its close: the character in progress, the characters minLength still needs, the quote.
int free_string_cost(const Frame& f, const Shape& sh) {
  const int pending = f.phase == S_ESC ? 1 : f.phase == S_U ? 4 - f.u : f.need;
  const int more = std::max(0, sh.min_len - f.chars - (pending > 0 ? 1 : 0));
  return add(pending + 1, more ? more * min_char_bytes(sh.cls) : 0);
}

}  // namespace

VocabTrie build_trie(const Tokenizer& t) {
  std::vector<bool> added(t.id_bytes.size(), false);
  for (const auto& [text, id] : t.added)
    if (id >= 0 && size_t(id) < added.size()) added[id] = true;
  std::vector<int32_t> ids;
  for (int32_t i = 0; i < int32_t(t.id_bytes.size()); ++i)
    if (!added[i] && !t.id_bytes[i].empty()) ids.push_back(i);
  std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) {
    const int c = t.id_bytes[a].compare(t.id_bytes[b]);
    return c ? c < 0 : a < b;
  });
  VocabTrie v;
  auto node = [&](uint8_t b) {
    v.first_child.push_back(-1), v.next_sibling.push_back(-1), v.byte.push_back(b);
    v.tok_begin.push_back(0), v.tok_end.push_back(0);
    return int32_t(v.byte.size()) - 1;
  };
  node(0);
  std::vector<int32_t> last_child(1, -1), path{0};
  std::string prev;
  for (int32_t id : ids) {
    const std::string& w = t.id_bytes[id];
    size_t k = 0;
    while (k < w.size() && k < prev.size() && w[k] == prev[k]) ++k;
    path.resize(k + 1);
    for (size_t d = k; d < w.size(); ++d) {
      const int32_t n = node(uint8_t(w[d]));
      last_child.push_back(-1);
      const int32_t parent = path.back();
      if (last_child[parent] < 0) v.first_child[parent] = n;
      else v.next_sibling[last_child[parent]] = n;
      last_child[parent] = n;
      path.push_back(n);
    }
    const int32_t at = path.back();
    if (v.tok_end[at] != int32_t(v.ids.size())) v.tok_begin[at] = int32_t(v.ids.size());
    v.ids.push_back(id);
    v.tok_end[at] = int32_t(v.ids.size());
    v.max_depth = std::max(v.max_depth, int(w.size()));
    prev = w;
  }
  return v;
}

QuoteText quote_text(std::string_view text) {
  std::vector<char32_t> s;
  std::vector<uint32_t> offs;
  utf8_decode(text, s, offs);
  const int n = int(s.size());
  QuoteText q;
  for (int i = 0; i < n; ++i) {
    q.byte_of.push_back(int32_t(q.e.size()));
    std::string raw;
    if (s[i] == 0xFFFFFFFF) raw = std::string(1, text[offs[i]]);
    else utf8_append(raw, s[i]);
    std::string enc = json_string_body(raw);
    if (s[i] >= 'A' && s[i] <= 'Z') enc[0] = char(enc[0] + 32);
    for (size_t k = 0; k < enc.size(); ++k) q.char_at.push_back(i);
    q.e += enc;
  }
  q.byte_of.push_back(int32_t(q.e.size()));
  q.char_at.push_back(n);
  // A quote stands alone: not inside a word, and a number not inside a wider one
  auto left_ok = [&](int i) {
    return !(i > 0 && ((word(s[i - 1]) && word(s[i])) ||
                       (adigit(s[i]) && (s[i - 1] == ',' || s[i - 1] == '.' || s[i - 1] == U'£'))));
  };
  auto right_ok = [&](int j) {
    return !(j < n && ((word(s[j]) && word(s[j - 1])) ||
                       (adigit(s[j - 1]) && (s[j] == '%' || ((s[j] == ',' || s[j] == '.') && j + 1 < n &&
                                                              adigit(s[j + 1]))))));
  };
  q.end_ok.assign(q.e.size() + 1, 0);
  for (int j = 1; j <= n; ++j)
    if (right_ok(j) && !py_space(s[j - 1])) q.end_ok[q.byte_of[j]] = 1;
  q.next_end.assign(q.e.size() + 2, INF);
  for (int p = int(q.e.size()); p >= 0; --p) q.next_end[p] = q.end_ok[p] ? p : q.next_end[p + 1];
  for (int i = 0; i < n; ++i)
    if (left_ok(i) && !py_space(s[i])) q.sa.push_back(q.byte_of[i]);
  const std::string_view e(q.e);
  std::sort(q.sa.begin(), q.sa.end(), [&](int32_t a, int32_t b) { return e.substr(a) < e.substr(b); });
  return q;
}

Constraint::Constraint(const Schema& s, const Tokenizer& tok, const VocabTrie& trie, std::vector<int> stop,
                       const QuoteText* quotes)
    : s_(s), tok_(tok), trie_(trie), stop_(std::move(stop)), q_(quotes) {
  mv_node_.assign(s.nodes.size(), -1);
  mv_shape_.assign(s.shapes.size(), -1);
  for (size_t n = 0; n < s.nodes.size(); ++n) mv_of_node(int(n));
  bufs_.resize(size_t(trie.max_depth) + 2);
  if (!push_value(Stack{}, s.root, state_))
    throw std::runtime_error(std::string("the schema admits no answer") + (quotes ? " quoting this text" : ""));
}

int Constraint::mv_of_node(int n) {
  if (mv_node_[n] >= 0) return mv_node_[n];
  int best = INF;
  for (int a : s_.nodes[n].alts) best = std::min(best, mv_of_shape(a));
  return mv_node_[n] = best;
}

int Constraint::mv_of_shape(int i) {
  if (mv_shape_[i] >= 0) return mv_shape_[i];
  const Shape& sh = s_.shapes[i];
  int v = INF;
  if (sh.literal) {
    for (const std::string& l : sh.lits) v = std::min(v, int(l.size()));
  } else if (sh.type == JType::Object) {
    mv_shape_[i] = INF;  // a value inside itself never ends
    for (int p : sh.props) mv_of_node(p);
    v = add(close_cost(sh, 0, true), 1);
  } else if (sh.type == JType::Array) {
    mv_shape_[i] = INF;
    v = 2;
    for (int k = 0; k < sh.min_items; ++k) {
      const int item = k < int(sh.prefix.size()) ? sh.prefix[k] : sh.items;
      v = add(v, add(item < 0 ? INF : mv_of_node(item), k ? 1 : 0));
    }
  } else if (sh.type == JType::String) {
    if (sh.quote && q_) {
      for (int32_t start : q_->sa) {
        const int cs = q_->char_at[start], minc = std::max(1, sh.min_len);
        if (cs + minc >= int(q_->byte_of.size())) continue;
        const int e = q_->next_end[std::max(start, q_->byte_of[cs + minc])];
        if (e < INF && q_->char_at[e] - cs <= sh.max_len) v = std::min(v, e - start + 2);
      }
    } else {
      v = add(2, sh.min_len ? sh.min_len * min_char_bytes(sh.cls) : 0);
    }
  } else {
    v = 1;
  }
  return mv_shape_[i] = v;
}

// The bytes from an object with `seen` written (first: nothing yet) to its close: the required keys still missing,
// enough cheapest other keys for minProperties, the commas, and the brace.
int Constraint::close_cost(const Shape& sh, uint64_t seen, bool first) const {
  auto key_cost = [&](int k) { return add(int(sh.key_bytes[k].size()) + 2, mv_node_[sh.props[k]]); };
  int total = 0, n = 0;
  uint64_t have = seen;
  for (int k : sh.required)
    if (!(seen >> k & 1)) total = add(total, key_cost(k)), ++n, have |= 1ull << k;
  int extra = sh.min_props - std::popcount(have);
  if (extra > 0) {
    std::vector<int> costs;
    for (int k = 0; k < int(sh.keys.size()); ++k)
      if (!(have >> k & 1)) costs.push_back(key_cost(k));
    std::sort(costs.begin(), costs.end());
    if (int(costs.size()) < extra) return INF;
    for (int k = 0; k < extra; ++k) total = add(total, costs[k]), ++n;
  }
  return add(total, (first ? std::max(0, n - 1) : n) + 1);
}

bool Constraint::push_value(const Stack& c, int node, std::vector<Stack>& out) const {
  bool any = false;
  for (int a : s_.nodes[node].alts) {
    if (mv_shape_[a] >= INF) continue;
    const Shape& sh = s_.shapes[a];
    Stack x = c;
    if (x.depth == kMaxDepth) throw std::runtime_error("constrained decoding: values nested deeper than 12");
    Frame f;
    f.shape = int16_t(a);
    if (sh.literal) {
      f.kind = K_LIT, f.lo = 0, f.hi = int16_t(sh.lits.size());
      if (x.depth && x.f[x.depth - 1].kind == K_ARR && s_.shapes[x.f[x.depth - 1].shape].unique)
        f.used = x.f[x.depth - 1].used;
    } else if (sh.type == JType::Object) {
      f.kind = K_OBJ, f.phase = O_OPEN;
    } else if (sh.type == JType::Array) {
      f.kind = K_ARR, f.phase = A_OPEN;
    } else if (sh.type == JType::String) {
      f.kind = K_STR, f.phase = S_OPEN, f.hash = kFnv0;
      if (sh.quote && q_) f.qlo = 0, f.qhi = int32_t(q_->sa.size());
    } else {
      f.kind = K_NUM, f.phase = N_START;
    }
    x.f[x.depth++] = f;
    out.push_back(x);
    any = true;
  }
  return any;
}

// Pops the finished top value into its parent (lit: the literal it was, for a unique array's record).
bool Constraint::pop_value(Stack& c, int lit) const {
  const Frame done = c.f[--c.depth];
  if (c.depth == 0) {
    c.done = true;
    return true;
  }
  Frame& p = c.f[c.depth - 1];
  if (p.kind == K_OBJ) {
    p.seen |= 1ull << p.key;
    p.phase = O_NEXT;
  } else {
    if (done.kind == K_LIT && lit >= 0 && s_.shapes[p.shape].unique) p.used |= 1ull << lit;
    ++p.count;
    p.phase = A_NEXT;
  }
  return true;
}

bool Constraint::quote_alive(const Frame& f, const Shape& sh) const {
  for (int32_t x = f.qlo; x < f.qhi; ++x) {
    const int32_t start = q_->sa[x], p = start + f.qlen;
    const int cs = q_->char_at[start], minc = std::max(1, sh.min_len);
    if (cs + minc >= int(q_->byte_of.size())) continue;
    const int e = q_->next_end[std::max(p, q_->byte_of[cs + minc])];
    if (e < INF && q_->char_at[e] - cs <= sh.max_len) return true;
  }
  return false;
}

int Constraint::quote_cost(const Frame& f, const Shape& sh) const {
  int best = INF;
  for (int32_t x = f.qlo; x < f.qhi && best > 1; ++x) {
    const int32_t start = q_->sa[x], p = start + f.qlen;
    const int cs = q_->char_at[start], minc = std::max(1, sh.min_len);
    if (cs + minc >= int(q_->byte_of.size())) continue;
    const int e = q_->next_end[std::max(p, q_->byte_of[cs + minc])];
    if (e < INF && q_->char_at[e] - cs <= sh.max_len) best = std::min(best, e - p + 1);
  }
  return best;
}

bool Constraint::quote_closable(const Frame& f) const {
  for (int32_t x = f.qlo; x < f.qhi; ++x)
    if (q_->end_ok[q_->sa[x] + f.qlen]) return true;
  return false;
}

void Constraint::feed(Stack c, uint8_t b, std::vector<Stack>& out) const {
  for (;;) {
    if (c.done || c.depth == 0) return;  // nothing follows a complete answer
    Frame& f = c.f[c.depth - 1];
    const Shape& sh = s_.shapes[f.shape];
    switch (f.kind) {
      case K_LIT: {
        const int pos = f.pos;
        auto at = [&](int x) { return int(sh.lits[x].size()) > pos ? int(uint8_t(sh.lits[x][pos])) : -1; };
        int lo = f.lo, hi = f.hi;
        while (lo < hi && at(lo) < b) ++lo;
        int end = lo;
        while (end < hi && at(end) == b) ++end;
        bool live = false;
        for (int x = lo; x < end && !live; ++x) live = !(f.used >> x & 1);
        if (live) {
          f.lo = int16_t(lo), f.hi = int16_t(end), ++f.pos;
          out.push_back(c);
          return;
        }
        if (!(int(sh.lits[f.lo].size()) == pos && !(f.used >> f.lo & 1))) return;
        pop_value(c, f.lo);
        continue;  // the byte belongs to the parent
      }
      case K_NUM: {
        const int ph = num_next(f.phase, b, sh);
        if (ph >= 0) {
          // a part's digits: at most 15 before and after the point (a double's precision), 2 in the exponent, so
          // every number reads back finite
          const bool more = ph == f.phase && (ph == N_INT || ph == N_FRAC || ph == N_EXP);
          f.count = int16_t(more ? f.count + 1 : 1);
          if (f.count > (ph == N_EXP ? 2 : 15)) return;
          f.phase = uint8_t(ph);
          out.push_back(c);
          return;
        }
        if (!num_accepting(f.phase)) return;
        pop_value(c, -1);
        continue;
      }
      case K_STR: {
        const bool quoted = sh.quote && q_;
        if (f.phase == S_OPEN) {
          if (b != '"') return;
          f.phase = S_BODY;
          if (quoted && !quote_alive(f, sh)) return;
          out.push_back(c);
          return;
        }
        if (f.phase == S_BODY && f.need == 0 && b == '"') {
          if (f.chars < sh.min_len || (quoted && !quote_closable(f))) return;
          if (c.nhash >= 0 && c.depth >= 2 && c.f[c.depth - 2].kind == K_ARR && s_.shapes[c.f[c.depth - 2].shape].unique) {
            if (std::find(c.hashes, c.hashes + c.nhash, f.hash) != c.hashes + c.nhash) return;
            if (c.nhash == 8) return;
            c.hashes[c.nhash++] = f.hash;
          }
          pop_value(c, -1);
          out.push_back(c);
          return;
        }
        if (quoted) {
          const uint8_t key = (f.phase == S_BODY && f.need == 0 && b >= 'A' && b <= 'Z') ? uint8_t(b + 32) : b;
          const std::string& e = q_->e;
          auto at = [&](int32_t x) {
            const int32_t p = q_->sa[x] + f.qlen;
            return p < int32_t(e.size()) ? int(uint8_t(e[p])) : -1;
          };
          // the starts sharing the bytes so far are sorted by their next byte (a start the text ends at first)
          int32_t L = f.qlo, H = f.qhi;
          while (L < H) {
            const int32_t m = (L + H) / 2;
            if (at(m) < key) L = m + 1;
            else H = m;
          }
          const int32_t lo = L;
          H = f.qhi;
          while (L < H) {
            const int32_t m = (L + H) / 2;
            if (at(m) <= key) L = m + 1;
            else H = m;
          }
          const int32_t hi = L;
          if (lo >= hi) return;
          f.qlo = lo, f.qhi = hi, ++f.qlen;
        }
        if (!string_byte(f, sh, b)) return;
        f.hash = fnv(f.hash, b);
        if (quoted && !quote_alive(f, sh)) return;
        out.push_back(c);
        return;
      }
      case K_OBJ: {
        auto avail = [&](int k) { return !(f.seen >> k & 1) && mv_node_[sh.props[k]] < INF; };
        auto any_avail = [&](int lo, int hi) {
          for (int k = lo; k < hi; ++k)
            if (avail(k)) return true;
          return false;
        };
        const bool closable = [&] {
          for (int k : sh.required)
            if (!(f.seen >> k & 1)) return false;
          return std::popcount(f.seen) >= sh.min_props;
        }();
        switch (f.phase) {
          case O_OPEN:
            if (b != '{') return;
            f.phase = O_FIRST;
            break;
          case O_FIRST:
          case O_NEXT:
            if (b == '}') {
              if (!closable) return;
              pop_value(c, -1);
              out.push_back(c);
              return;
            }
            if (f.phase == O_NEXT ? b != ',' : b != '"') return;
            if (!any_avail(0, int(sh.keys.size()))) return;
            if (f.phase == O_NEXT) {
              f.phase = O_KEYQ;
              break;
            }
            f.phase = O_KEY, f.lo = 0, f.hi = int16_t(sh.keys.size()), f.pos = 0;
            break;
          case O_KEYQ:
            if (b != '"') return;
            f.phase = O_KEY, f.lo = 0, f.hi = int16_t(sh.keys.size()), f.pos = 0;
            break;
          case O_KEY: {
            const int pos = f.pos;
            auto at = [&](int k) {
              return int(sh.key_bytes[k].size()) > pos ? int(uint8_t(sh.key_bytes[k][pos])) : -1;
            };
            int lo = f.lo, hi = f.hi;
            while (lo < hi && at(lo) < b) ++lo;
            int end = lo;
            while (end < hi && at(end) == b) ++end;
            if (!any_avail(lo, end)) return;
            f.lo = int16_t(lo), f.hi = int16_t(end), f.pos = int16_t(pos + 1);
            if (int(sh.key_bytes[lo].size()) == pos + 1) f.key = int16_t(lo), f.phase = O_COLON;
            break;
          }
          case O_COLON: {
            if (b != ':') return;
            f.phase = O_VALUE;
            push_value(c, sh.props[f.key], out);
            return;
          }
          default: return;
        }
        out.push_back(c);
        return;
      }
      case K_ARR: {
        auto item = [&](int i) { return i < int(sh.prefix.size()) ? sh.prefix[i] : sh.items; };
        switch (f.phase) {
          case A_OPEN:
            if (b != '[') return;
            f.phase = A_FIRST;
            if (sh.unique && c.nhash >= 0) {
              const int it = item(0);
              if (it >= 0 && !s_.shapes[s_.nodes[it].alts[0]].literal) c.nhash = 0;
            }
            out.push_back(c);
            return;
          case A_FIRST:
          case A_NEXT: {
            if (b == ']') {
              if (f.count < sh.min_items) return;
              pop_value(c, -1);
              out.push_back(c);
              return;
            }
            if (f.phase == A_NEXT && b != ',') return;
            const int it = item(f.count);
            if (f.count >= sh.max_items || it < 0 || mv_node_[it] >= INF) return;
            f.phase = A_ITEM;
            if (f.count == 0) {  // no comma: the byte starts the first item
              std::vector<Stack> starts;
              push_value(c, it, starts);
              for (const Stack& x : starts) feed(x, b, out);
              return;
            }
            push_value(c, it, out);
            return;
          }
          default: return;
        }
      }
    }
    return;
  }
}

int Constraint::frame_cost(const Stack& c, int i) const {
  const Frame& f = c.f[i];
  const Shape& sh = s_.shapes[f.shape];
  const bool top = i == c.depth - 1;
  switch (f.kind) {
    case K_LIT: {
      int best = INF;
      for (int x = f.lo; x < f.hi; ++x)
        if (!(f.used >> x & 1)) best = std::min(best, int(sh.lits[x].size()) - f.pos);
      return best;
    }
    case K_NUM: return num_accepting(f.phase) ? 0 : 1;
    case K_STR:
      if (f.phase == S_OPEN) return mv_shape_[f.shape];
      return (sh.quote && q_) ? quote_cost(f, sh) : free_string_cost(f, sh);
    case K_OBJ: {
      auto key_cost = [&](int k) { return add(int(sh.key_bytes[k].size()) + 1, mv_node_[sh.props[k]]); };
      switch (f.phase) {
        case O_OPEN: return mv_shape_[f.shape];
        case O_FIRST: return close_cost(sh, f.seen, true);
        case O_NEXT: return close_cost(sh, f.seen, false);
        case O_KEYQ:
        case O_KEY: {
          int best = INF;
          const int lo = f.phase == O_KEY ? f.lo : 0, hi = f.phase == O_KEY ? f.hi : int(sh.keys.size());
          for (int k = lo; k < hi; ++k)
            if (!(f.seen >> k & 1)) {
              const int rest = f.phase == O_KEY ? key_cost(k) - f.pos : add(key_cost(k), 1);
              best = std::min(best, add(rest, close_cost(sh, f.seen | 1ull << k, false)));
            }
          return best;
        }
        case O_COLON:
          return add(add(1, mv_node_[sh.props[f.key]]), close_cost(sh, f.seen | 1ull << f.key, false));
        default:  // O_VALUE, the value above it in progress
          return top ? INF : close_cost(sh, f.seen | 1ull << f.key, false);
      }
    }
    case K_ARR: {
      if (f.phase == A_OPEN) return mv_shape_[f.shape];
      const bool first = f.phase == A_FIRST;
      const int count = f.phase == A_ITEM ? f.count + 1 : f.count;
      int total = 1, n = 0;
      for (int k = count; k < sh.min_items; ++k, ++n) {
        const int it = k < int(sh.prefix.size()) ? sh.prefix[k] : sh.items;
        total = add(total, it < 0 ? INF : mv_node_[it]);
      }
      return add(total, first ? std::max(0, n - 1) : n);
    }
  }
  return INF;
}

int Constraint::cost(const Stack& c) const {
  if (c.done) return 0;
  int total = 0;
  for (int i = 0; i < c.depth; ++i) total = add(total, frame_cost(c, i));
  return total;
}

int Constraint::min_cost(const std::vector<Stack>& st) const {
  int best = INF;
  for (const Stack& c : st) best = std::min(best, cost(c));
  return best;
}

bool Constraint::lazy_complete(const Stack& c) const {
  if (c.done) return true;
  if (c.depth != 1) return false;
  const Frame& f = c.f[0];
  if (f.kind == K_NUM) return num_accepting(f.phase);
  if (f.kind == K_LIT) return int(s_.shapes[f.shape].lits[f.lo].size()) == f.pos && !(f.used >> f.lo & 1);
  return false;
}

bool Constraint::complete() const {
  for (const Stack& c : state_)
    if (lazy_complete(c)) return true;
  return false;
}

int Constraint::min_close() const { return min_cost(state_); }

bool Constraint::free_string_top(const std::vector<Stack>& st) const {
  if (st.size() != 1 || st[0].done || st[0].depth == 0) return false;
  const Frame& f = st[0].f[st[0].depth - 1];
  const Shape& sh = s_.shapes[f.shape];
  return f.kind == K_STR && f.phase != S_OPEN && !(sh.quote && q_);
}

void Constraint::take_tokens(int node, int cost) {
  if (cost <= remaining_ - 2) {
    out_.insert(out_.end(), trie_.ids.begin() + trie_.tok_begin[node], trie_.ids.begin() + trie_.tok_end[node]);
  } else {
    budget_cut = true;
  }
}

void Constraint::walk(int node, int level) {
  const std::vector<Stack>& cur = bufs_[level];
  for (int ch = trie_.first_child[node]; ch >= 0; ch = trie_.next_sibling[ch]) {
    std::vector<Stack>& next = bufs_[level + 1];
    next.clear();
    for (const Stack& c : cur) feed(c, trie_.byte[ch], next);
    if (next.empty()) continue;
    if (trie_.tok_begin[ch] < trie_.tok_end[ch]) take_tokens(ch, min_cost(next));
    if (trie_.first_child[ch] < 0) continue;
    if (free_string_top(next)) {
      const Stack& c = next[0];
      int rest = 0;
      for (int i = 0; i + 1 < c.depth; ++i) rest = add(rest, frame_cost(c, i));
      walk_free(ch, level + 1, c, c.f[c.depth - 1], rest);
    } else {
      walk(ch, level + 1);
    }
  }
}

// Inside a free string most bytes stay inside it: they are checked on the string's own state, and only a closing
// quote goes back through the whole automaton.
void Constraint::walk_free(int node, int level, const Stack& base, Frame fs, int rest) {
  const Shape& sh = s_.shapes[fs.shape];
  for (int ch = trie_.first_child[node]; ch >= 0; ch = trie_.next_sibling[ch]) {
    const uint8_t b = trie_.byte[ch];
    if (fs.phase == S_BODY && fs.need == 0 && b == '"') {
      Stack c = base;
      c.f[c.depth - 1] = fs;
      std::vector<Stack>& next = bufs_[level + 1];  // base lives at a shallower level, never this one
      next.clear();
      feed(c, b, next);
      if (next.empty()) continue;
      if (trie_.tok_begin[ch] < trie_.tok_end[ch]) take_tokens(ch, min_cost(next));
      if (trie_.first_child[ch] >= 0) walk(ch, level + 1);
      continue;
    }
    Frame g = fs;
    if (!string_byte(g, sh, b)) continue;
    g.hash = fnv(g.hash, b);
    if (trie_.tok_begin[ch] < trie_.tok_end[ch]) take_tokens(ch, add(rest, free_string_cost(g, sh)));
    if (trie_.first_child[ch] >= 0) walk_free(ch, level + 1, base, g, rest);
  }
}

// The FreeTable of a class: a walk of the whole trie from a clean string body, as walk_free walks it, with maxLength
// left unchecked (it is checked per step from what each token completes).
const FreeTable& Constraint::free_table(const Shape& sh) {
  std::string key(1, char(sh.cls.any)), b(sizeof(char32_t) * 2, '\0');
  key += char(sh.cls.negated);
  for (auto [lo, hi] : sh.cls.ranges) std::memcpy(b.data(), &lo, 4), std::memcpy(b.data() + 4, &hi, 4), key += b;
  std::lock_guard<std::mutex> lock(trie_.free->mu);
  std::shared_ptr<const FreeTable>& slot = trie_.free->by_class[key];
  if (slot) return *slot;
  auto t = std::make_shared<FreeTable>();
  Shape open = sh;
  open.min_len = 0, open.max_len = INT_MAX;
  struct Body {
    int32_t id;
    int16_t complete;
    uint8_t pending;
  };
  std::vector<Body> body;
  std::string path;
  std::function<void(int, int, const Frame&)> dfs = [&](int node, int level, const Frame& fs) {
    for (int ch = trie_.first_child[node]; ch >= 0; ch = trie_.next_sibling[ch]) {
      const uint8_t b = trie_.byte[ch];
      if (fs.phase == S_BODY && fs.need == 0 && b == '"') {
        t->closes.push_back({ch, level + 1, fs.chars, path});
        continue;
      }
      Frame g = fs;
      if (!string_byte(g, open, b)) continue;
      const int pending = g.phase == S_ESC ? 1 : g.phase == S_U ? 4 - g.u : g.need;
      for (int32_t k = trie_.tok_begin[ch]; k < trie_.tok_end[ch]; ++k)
        body.push_back({trie_.ids[k], int16_t(g.chars), uint8_t(pending)});
      if (trie_.first_child[ch] >= 0) {
        path.push_back(char(b));
        dfs(ch, level + 1, g);
        path.pop_back();
      }
    }
  };
  Frame start;
  start.kind = K_STR, start.phase = S_BODY;
  dfs(0, 0, start);
  std::sort(body.begin(), body.end(), [](const Body& x, const Body& y) { return x.id < y.id; });
  for (const Body& x : body) {
    t->ids.push_back(x.id), t->complete.push_back(x.complete), t->pending.push_back(x.pending);
    t->max_started = std::max(t->max_started, x.complete + (x.pending ? 1 : 0));
    t->max_pending = std::max(t->max_pending, int(x.pending));
  }
  t->min_char_bytes = min_char_bytes(sh.cls);
  slot = t;
  return *slot;
}

// walk_free's result for a clean string body, from the table: the body tokens that fit maxLength and the budget, then
// each close resumed through the automaton exactly as walk_free resumes it. False when the step needs the walk (a
// character in progress, or a stop id among the table's tokens).
bool Constraint::from_table(const Stack& base, int rest) {
  const Frame& fs = base.f[base.depth - 1];
  if (fs.phase != S_BODY || fs.need != 0) return false;
  const Shape& sh = s_.shapes[fs.shape];
  const FreeTable& t = free_table(sh);
  if (&t != table_) {
    table_ = &t, stop_in_table_ = 0;
    for (int id : stop_)
      if (std::binary_search(t.ids.begin(), t.ids.end(), id)) stop_in_table_ = 1;
  }
  if (stop_in_table_) return false;
  const int c0 = fs.chars;
  auto cost_of = [&](int complete, int pending) {  // free_string_cost of the state a token leaves
    const int more = std::max(0, sh.min_len - c0 - complete - (pending > 0 ? 1 : 0));
    return add(rest, add(pending + 1, more ? more * t.min_char_bytes : 0));
  };
  const int worst = add(rest, add(t.max_pending + 1, std::max(0, sh.min_len - c0) * t.min_char_bytes));
  body_.clear();
  body_all_ = c0 + t.max_started <= sh.max_len && worst <= remaining_ - 2;
  if (!body_all_)
    for (size_t i = 0; i < t.ids.size(); ++i) {
      if (c0 + t.complete[i] + (t.pending[i] ? 1 : 0) > sh.max_len) continue;  // its bytes leave maxLength
      if (cost_of(t.complete[i], t.pending[i]) <= remaining_ - 2) body_.push_back(t.ids[i]);
      else budget_cut = true;
    }
  for (const FreeTable::Close& e : t.closes) {
    if (c0 + e.chars > sh.max_len) continue;
    Stack c = base;
    Frame& g = c.f[c.depth - 1];
    g.chars = c0 + e.chars;
    for (char b : e.prefix) g.hash = fnv(g.hash, uint8_t(b));
    std::vector<Stack>& next = bufs_[e.level];
    next.clear();
    feed(c, '"', next);
    if (next.empty()) continue;
    if (trie_.tok_begin[e.node] < trie_.tok_end[e.node]) take_tokens(e.node, min_cost(next));
    if (trie_.first_child[e.node] >= 0) walk(e.node, e.level);
  }
  return true;
}

std::vector<int> Constraint::allowed(int remaining) {
  remaining_ = remaining;
  out_.clear();
  bool table = false;
  if (remaining >= 2) {
    bufs_[0] = state_;
    if (free_string_top(state_)) {
      const Stack& c = state_[0];
      int rest = 0;
      for (int i = 0; i + 1 < c.depth; ++i) rest = add(rest, frame_cost(c, i));
      table = free_tables && from_table(c, rest);
      if (!table) walk_free(0, 0, c, c.f[c.depth - 1], rest);
    } else {
      walk(0, 0);
    }
  }
  // a stop id is never text, even when the vocabulary also spells it (a model whose EOS is an ordinary token)
  out_.erase(std::remove_if(out_.begin(), out_.end(),
                            [&](int id) { return std::find(stop_.begin(), stop_.end(), id) != stop_.end(); }),
             out_.end());
  if (remaining >= 1 && complete()) out_.insert(out_.end(), stop_.begin(), stop_.end());
  std::sort(out_.begin(), out_.end());
  out_.erase(std::unique(out_.begin(), out_.end()), out_.end());
  if (table) {  // the table's body tokens, sorted and free of stop ids, merged in
    const std::vector<int32_t>& body = body_all_ ? table_->ids : body_;
    std::vector<int> merged(out_.size() + body.size());
    merged.erase(std::set_union(out_.begin(), out_.end(), body.begin(), body.end(), merged.begin()), merged.end());
    out_.swap(merged);
  }
  return out_;
}

void Constraint::accept(int id) {
  if (std::find(stop_.begin(), stop_.end(), id) != stop_.end()) {
    if (!complete()) throw std::runtime_error("constrained decoding: a stop token before the JSON is complete");
    return;
  }
  if (id < 0 || size_t(id) >= tok_.id_bytes.size() || tok_.added.count(tok_.id_bytes[id]))
    throw std::runtime_error("constrained decoding: token " + std::to_string(id) + " is not text");
  std::vector<Stack> cur = state_, next;
  for (char ch : tok_.id_bytes[id]) {
    next.clear();
    for (const Stack& c : cur) feed(c, uint8_t(ch), next);
    if (next.empty())
      throw std::runtime_error("constrained decoding: token " + std::to_string(id) + " " +
                               json_quote(tok_.id_bytes[id]) + " leaves the schema");
    cur.swap(next);
  }
  state_ = std::move(cur);
  bytes_ += tok_.id_bytes[id].size();
}

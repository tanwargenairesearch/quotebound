#include "pretokenize.h"

#include "unicode.h"

const char* const kSplitRegexQwen3 =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
const char* const kSplitRegexQwen35 =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

namespace {
enum : uint8_t { L = 1, N = 2, S = 4, CRLF = 8, M = 16 };
}

// Oniguruma semantics: at each position the alternatives are tried in order and the first that matches wins,
// with the greedy quantifiers backtracking as described per branch. "Word" below is \p{L} for Qwen3 and
// [\p{L}\p{M}] for Qwen3.5; the optional lead char excludes only CR, LF, \p{L} and \p{N} in both.
std::vector<std::string_view> pretokenize(std::string_view s, bool marks_join_letters) {
  std::vector<char32_t> cp;
  std::vector<uint32_t> off;
  utf8_decode(s, cp, off);
  const size_t n = cp.size();
  std::vector<uint8_t> cls(n + 1, 0);  // cls[n] = 0 makes every look-ahead past the end fail
  for (size_t i = 0; i < n; ++i) {
    char32_t c = cp[i];
    cls[i] = is_letter(c) ? L
             : is_number(c) ? N
             : is_space(c)  ? (S | ((c == '\r' || c == '\n') ? CRLF : 0))
             : is_mark(c)   ? M
                            : 0;
  }
  const uint8_t word = marks_join_letters ? (L | M) : L;
  auto has = [&](size_t i, uint8_t f) { return (cls[i] & f) != 0; };
  auto other = [&](size_t i) { return i < n && !has(i, L | N | S | (word & M)); };  // [^\s\p{L}(\p{M})\p{N}]
  // The (?i:...) group only folds ASCII here: the reference keeps 'ſ (U+017F) out of the 's branch.
  auto lower = [&](size_t i) -> char32_t { return i < n && cp[i] < 0x80 ? (cp[i] | 0x20) : 0; };

  auto match = [&](size_t i) -> size_t {
    if (cp[i] == '\'') {
      char32_t a = lower(i + 1), b = lower(i + 2);
      if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
    if (has(i, word) || (!has(i, L | N | CRLF) && has(i + 1, word))) {  // [^\r\n\p{L}\p{N}]?word+
      size_t j = has(i, word) ? i : i + 1;
      while (has(j, word)) ++j;
      return j;
    }
    if (has(i, N)) return i + 1;
    if (size_t k = i + (cp[i] == ' '); other(k)) {  //  ?[^\s...]+[\r\n]*
      while (other(k)) ++k;
      while (has(k, CRLF)) ++k;
      return k;
    }
    size_t j = i;  // i is whitespace from here on
    while (has(j, S)) ++j;
    for (size_t k = j; k > i; --k)  // \s*[\r\n]+ ends just after the run's last CR/LF
      if (has(k - 1, CRLF)) return k;
    if (j == n || j - i == 1) return j;  // \s+(?!\S) takes the whole run at the end of input; else \s+
    return j - 1;                        // \s+(?!\S) leaves the run's last char to lead the next piece
  };

  std::vector<std::string_view> out;
  for (size_t i = 0; i < n;) {
    size_t j = match(i);
    out.push_back(s.substr(off[i], off[j] - off[i]));
    i = j;
  }
  return out;
}

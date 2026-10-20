#include "unicode.h"

#ifndef QUOTEBOUND_NO_COREFOUNDATION
#include <CoreFoundation/CoreFoundation.h>
#else
#include <stdexcept>
#endif

void utf8_decode(std::string_view s, std::vector<char32_t>& cps, std::vector<uint32_t>& offs) {
  cps.clear();
  offs.clear();
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  size_t n = s.size(), i = 0;
  while (i < n) {
    offs.push_back(uint32_t(i));
    unsigned char b = p[i];
    int len = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
    char32_t c = len == 1 ? b : len == 2 ? (b & 0x1F) : len == 3 ? (b & 0x0F) : (b & 0x07);
    bool ok = len > 0 && i + len <= n;
    for (int k = 1; ok && k < len; ++k) {
      ok = (p[i + k] & 0xC0) == 0x80;
      c = (c << 6) | (p[i + k] & 0x3F);
    }
    static const char32_t min_for_len[] = {0, 0, 0x80, 0x800, 0x10000};
    ok = ok && c >= min_for_len[len] && c <= 0x10FFFF && (c < 0xD800 || c > 0xDFFF);
    cps.push_back(ok ? c : 0xFFFFFFFF);
    i += ok ? len : 1;
  }
  offs.push_back(uint32_t(n));
}

void utf8_append(std::string& out, char32_t c) {
  if (c < 0x80) {
    out += char(c);
  } else if (c < 0x800) {
    out += char(0xC0 | (c >> 6));
    out += char(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    out += char(0xE0 | (c >> 12));
    out += char(0x80 | ((c >> 6) & 0x3F));
    out += char(0x80 | (c & 0x3F));
  } else {
    out += char(0xF0 | (c >> 18));
    out += char(0x80 | ((c >> 12) & 0x3F));
    out += char(0x80 | ((c >> 6) & 0x3F));
    out += char(0x80 | (c & 0x3F));
  }
}

#ifndef QUOTEBOUND_NO_COREFOUNDATION
namespace {

enum : uint8_t { OTHER = 0, LETTER = 1, NUMBER = 2, MARK = 3 };

// Where this Mac's CoreFoundation disagrees with the Unicode 16 tables of the reference tokenizer's regex engine:
// CF already has Unicode 17's new letters, digits and marks, and does not class Tangut or Todhri as letters. Found by
// the every-code-point sweep in tests/test_tokenizer.cpp, which fails again if a macOS update moves CF.
struct Fix {
  char32_t lo, hi;
  uint8_t cls;
};
constexpr Fix kFixes[] = {
    {0x088F, 0x088F, OTHER},   {0x0C5C, 0x0C5C, OTHER},   {0x0CDC, 0x0CDC, OTHER},   {0xA7CE, 0xA7CF, OTHER},
    {0xA7D2, 0xA7D2, OTHER},   {0xA7D4, 0xA7D4, OTHER},   {0xA7F1, 0xA7F1, OTHER},   {0x105C0, 0x105F3, LETTER},
    {0x10940, 0x10959, OTHER}, {0x10EC5, 0x10EC7, OTHER}, {0x11DB0, 0x11DDB, OTHER}, {0x11DE0, 0x11DE9, OTHER},
    {0x16EA0, 0x16EB8, OTHER}, {0x16EBB, 0x16ED3, OTHER}, {0x16FF2, 0x16FF6, OTHER}, {0x17000, 0x187F7, LETTER},
    {0x18D00, 0x18D08, LETTER}, {0x18D80, 0x18DF2, OTHER}, {0x1E6C0, 0x1E6DE, OTHER}, {0x1E6E0, 0x1E6E2, OTHER},
    {0x1E6E4, 0x1E6E5, OTHER}, {0x1E6E7, 0x1E6ED, OTHER}, {0x1E6F0, 0x1E6F4, OTHER}, {0x1E6FE, 0x1E6FF, OTHER},
    {0x2B73A, 0x2B73F, OTHER}, {0x2CEA2, 0x2CEAD, OTHER}, {0x323B0, 0x33479, OTHER},
    // Unicode 17 combining marks (they only matter where marks join letters, as in Qwen3.5's regex)
    {0x1ACF, 0x1ADD, OTHER},   {0x1AE0, 0x1AEB, OTHER},   {0x10EFA, 0x10EFB, OTHER}, {0x11B60, 0x11B67, OTHER},
    {0x1E6E3, 0x1E6E3, OTHER}, {0x1E6E6, 0x1E6E6, OTHER}, {0x1E6EE, 0x1E6EF, OTHER}, {0x1E6F5, 0x1E6F5, OTHER},
};

// CoreFoundation has no plain \p{L} or \p{N} set: its "letter" set is L+M, "nonBase" is M and "alphanumeric"
// is L+M+N, so L = letter - nonBase and N = alphanumeric - letter.
struct Categories {
  CFCharacterSetRef letter_or_mark = CFCharacterSetGetPredefined(kCFCharacterSetLetter);
  CFCharacterSetRef mark = CFCharacterSetGetPredefined(kCFCharacterSetNonBase);
  CFCharacterSetRef alnum = CFCharacterSetGetPredefined(kCFCharacterSetAlphaNumeric);
  std::vector<uint8_t> bmp;

  Categories() : bmp(0x10000) {
    for (char32_t c = 0; c < 0x10000; ++c) bmp[c] = lookup(c);
  }
  uint8_t lookup(char32_t c) const {
    if (c > 0x10FFFF) return OTHER;
    for (const Fix& f : kFixes)
      if (c >= f.lo && c <= f.hi) return f.cls;
    if (CFCharacterSetIsLongCharacterMember(mark, c)) return MARK;
    if (CFCharacterSetIsLongCharacterMember(letter_or_mark, c)) return LETTER;
    return CFCharacterSetIsLongCharacterMember(alnum, c) ? NUMBER : OTHER;
  }
  uint8_t get(char32_t c) const { return c < 0x10000 ? bmp[c] : lookup(c); }
};

const Categories& categories() {
  static const Categories c;
  return c;
}

}  // namespace

bool is_letter(char32_t c) { return categories().get(c) == LETTER; }
bool is_number(char32_t c) { return categories().get(c) == NUMBER; }
bool is_mark(char32_t c) { return categories().get(c) == MARK; }
#else
// Built without CoreFoundation (QUOTEBOUND_NO_COREFOUNDATION: the constraint library on Linux, constrain_capi.h), which
// never tokenizes: the categories and NFC that only encode() reads refuse rather than differ from the Mac's.
static std::logic_error no_cf(const char* what) {
  return std::logic_error(std::string(what) + ": built without CoreFoundation, this build constrains and never tokenizes");
}

bool is_letter(char32_t) { throw no_cf("is_letter"); }
bool is_number(char32_t) { throw no_cf("is_number"); }
bool is_mark(char32_t) { throw no_cf("is_mark"); }
#endif

bool is_space(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
         c == 0x3000;
}

#ifndef QUOTEBOUND_NO_COREFOUNDATION
std::string nfc(std::string_view s) {
  // Code points below U+0300 (lead byte < 0xCC) are NFC-stable and never compose with a neighbour.
  bool stable = true;
  for (unsigned char b : s) stable = stable && b < 0xCC;
  if (stable) return std::string(s);
  std::vector<char32_t> cps;
  std::vector<uint32_t> offs;
  utf8_decode(s, cps, offs);
  std::vector<UniChar> u16;
  for (char32_t c : cps) {
    if (c == 0xFFFFFFFF) return std::string(s);  // not valid UTF-8: the byte-level BPE round-trips it as is
    if (c < 0x10000) {
      u16.push_back(UniChar(c));
    } else {
      u16.push_back(UniChar(0xD800 + ((c - 0x10000) >> 10)));
      u16.push_back(UniChar(0xDC00 + ((c - 0x10000) & 0x3FF)));
    }
  }
  // Built from UTF-16 code units because CFStringCreateWithBytes drops a leading U+FEFF as a byte-order mark.
  CFMutableStringRef m = CFStringCreateMutable(nullptr, 0);
  CFStringAppendCharacters(m, u16.data(), CFIndex(u16.size()));
  CFStringNormalize(m, kCFStringNormalizationFormC);
  u16.resize(size_t(CFStringGetLength(m)));
  CFStringGetCharacters(m, CFRangeMake(0, CFIndex(u16.size())), u16.data());
  CFRelease(m);
  std::string out;
  for (size_t i = 0; i < u16.size(); ++i) {
    char32_t c = u16[i];
    if (c >= 0xD800 && c < 0xDC00 && i + 1 < u16.size()) c = 0x10000 + ((c - 0xD800) << 10) + (u16[++i] - 0xDC00);
    utf8_append(out, c);
  }
  return out;
}
#else
std::string nfc(std::string_view) { throw no_cf("nfc"); }
#endif

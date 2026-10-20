#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Decodes UTF-8 into code points; offs gets one start offset per code point plus a final s.size().
// A byte that is not valid UTF-8 becomes one code point (0xFFFFFFFF) so byte offsets stay exact.
void utf8_decode(std::string_view s, std::vector<char32_t>& cps, std::vector<uint32_t>& offs);
void utf8_append(std::string& out, char32_t c);

bool is_letter(char32_t c);  // \p{L}
bool is_number(char32_t c);  // \p{N}
bool is_mark(char32_t c);    // \p{M}
bool is_space(char32_t c);   // \s as Oniguruma defines it for Unicode
std::string nfc(std::string_view s);

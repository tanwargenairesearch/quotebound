#pragma once
#include <string_view>
#include <vector>

// The two Split regexes this file implements by hand. Qwen3.5's differs from Qwen3's only in letting combining
// marks (\p{M}) join letter runs; the tokenizer refuses a tokenizer.json that declares anything else.
extern const char* const kSplitRegexQwen3;
extern const char* const kSplitRegexQwen35;

// Splits text into the pieces the regex matches (behaviour "Isolated"), as views into s.
std::vector<std::string_view> pretokenize(std::string_view s, bool marks_join_letters);

#include "tokenizer.h"

#include <climits>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "json.h"
#include "pretokenize.h"
#include "unicode.h"

namespace {

// FNV-1a of the exact chat template text for the two templates hand-coded in render().
constexpr uint64_t kQwen3Template = 0x2bfdb7aa9754484eULL;
constexpr uint64_t kQwen35Template = 0xf07c510f47fbaca0ULL;

uint64_t pair_key(int a, int b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); }

uint64_t fnv1a(std::string_view s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ULL;
  return h;
}

// Jinja's |trim, which is Python's str.strip(): \s plus the separators U+001C..U+001F.
std::string py_strip(std::string_view s) {
  std::vector<char32_t> cp;
  std::vector<uint32_t> off;
  utf8_decode(s, cp, off);
  auto ws = [&](size_t i) { return is_space(cp[i]) || (cp[i] >= 0x1C && cp[i] <= 0x1F); };
  size_t a = 0, b = cp.size();
  while (a < b && ws(a)) ++a;
  while (b > a && ws(b - 1)) --b;
  return std::string(s.substr(off[a], off[b] - off[a]));
}

std::vector<int> bpe(const Tokenizer& t, std::string_view piece) {
  std::vector<int> sym;
  sym.reserve(piece.size());
  for (unsigned char b : piece) sym.push_back(t.byte_id[b]);
  // Lowest-rank pair first, every occurrence left to right; equivalent to the reference's heap because a
  // merge can only create pairs of higher rank.
  while (sym.size() > 1) {
    int best_rank = INT_MAX, merged = -1;
    uint64_t best = 0;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      auto it = t.merges.find(pair_key(sym[i], sym[i + 1]));
      if (it != t.merges.end() && it->second.first < best_rank) {
        best_rank = it->second.first;
        merged = it->second.second;
        best = it->first;
      }
    }
    if (merged < 0) break;
    size_t w = 0;
    for (size_t i = 0; i < sym.size();) {
      if (i + 1 < sym.size() && pair_key(sym[i], sym[i + 1]) == best) {
        sym[w++] = merged;
        i += 2;
      } else {
        sym[w++] = sym[i++];
      }
    }
    sym.resize(w);
  }
  return sym;
}

void parse_tokenizer_json(Tokenizer& t, const std::string& path) {
  auto need = [&](bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(path + ": unsupported or malformed " + what);
  };
  using simdjson::dom::array, simdjson::dom::element, simdjson::dom::object;
  using sv = std::string_view;
  simdjson::dom::parser parser;
  element doc = load_json(parser, path);
  need(get<sv>(get<element>(doc, "normalizer", path), "type", path + " normalizer") == "NFC",
       "normalizer (only NFC is implemented)");
  element pre = get<element>(doc, "pre_tokenizer", path);
  need(get<sv>(pre, "type", path + " pre_tokenizer") == "Sequence", "pre_tokenizer (want a Sequence)");
  array steps = get<array>(pre, "pretokenizers", path + " pre_tokenizer");
  need(steps.size() == 2, "pre_tokenizer sequence (want Split then ByteLevel)");
  element split = steps.at(0), bytelevel = steps.at(1);
  const std::string w_split = path + " pre_tokenizer Split", w_bl = path + " pre_tokenizer ByteLevel";
  sv regex = get<sv>(get<element>(split, "pattern", w_split), "Regex", w_split + " pattern");
  need(get<sv>(split, "type", w_split) == "Split" && (regex == kSplitRegexQwen3 || regex == kSplitRegexQwen35),
       "split regex (only Qwen3's and Qwen3.5's are implemented)");
  t.marks_join_letters = regex == kSplitRegexQwen35;
  need(get<sv>(split, "behavior", w_split) == "Isolated" && !get<bool>(split, "invert", w_split), "split behaviour");
  need(get<sv>(bytelevel, "type", w_bl) == "ByteLevel" && !get<bool>(bytelevel, "add_prefix_space", w_bl) &&
           !get<bool>(bytelevel, "use_regex", w_bl),
       "ByteLevel pre-tokenizer");
  need(get<sv>(get<element>(doc, "decoder", path), "type", path + " decoder") == "ByteLevel", "decoder");
  element model = get<element>(doc, "model", path);
  const std::string wm = path + " model";
  need(get<sv>(model, "type", wm) == "BPE" && !get<bool>(model, "byte_fallback", wm) &&
           !get<bool>(model, "ignore_merges", wm) && !has_key(model, "dropout"),
       "BPE model options");
  for (const char* k : {"continuing_subword_prefix", "end_of_word_suffix"})
    need(!has_key(model, k) || get<sv>(model, k, wm).empty(), k);

  // GPT-2's byte <-> printable code point map that byte-level vocab strings are written in.
  char32_t b2u[256];
  std::unordered_map<char32_t, unsigned char> u2b;
  for (int b = 0, extra = 0; b < 256; ++b) {
    bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
    b2u[b] = printable ? char32_t(b) : char32_t(256 + extra++);
    u2b[b2u[b]] = (unsigned char)b;
  }

  std::unordered_map<sv, int> vocab;
  std::vector<char> seen;
  auto define = [&](int64_t id, std::string bytes) {
    need(id >= 0 && id < (1 << 24), "token id " + std::to_string(id));
    if (size_t(id) >= t.id_bytes.size()) t.id_bytes.resize(id + 1), seen.resize(id + 1);
    need(!seen[id], "vocabulary (id " + std::to_string(id) + " defined twice)");
    seen[id] = 1;
    t.id_bytes[id] = std::move(bytes);
  };
  std::vector<char32_t> cps;
  std::vector<uint32_t> offs;
  for (auto [key, value] : get<object>(model, "vocab", wm)) {
    int64_t id = int64_t(value);
    vocab[key] = int(id);
    utf8_decode(key, cps, offs);
    std::string raw;
    for (char32_t c : cps) {
      auto it = u2b.find(c);
      need(it != u2b.end(), "vocab entry outside the byte-level alphabet");
      raw += char(it->second);
    }
    define(id, std::move(raw));
  }
  for (int b = 0; b < 256; ++b) {
    std::string s;
    utf8_append(s, b2u[b]);
    auto it = vocab.find(s);
    need(it != vocab.end(), "vocab without all 256 byte tokens");
    t.byte_id[b] = it->second;
  }
  int rank = 0;
  for (element m : get<array>(model, "merges", wm)) {
    // Qwen3 writes merges as ["left", "right"], Qwen3.5 as "left right"; byte-level tokens never hold a space.
    sv a, b;
    if (m.is_array()) {
      a = m.at(0), b = m.at(1);
    } else {
      sv s = m;
      size_t sp = s.find(' ');
      need(sp != sv::npos && s.find(' ', sp + 1) == sv::npos, "merge entry");
      a = s.substr(0, sp), b = s.substr(sp + 1);
    }
    auto ia = vocab.find(a), ib = vocab.find(b), im = vocab.find(std::string(a) + std::string(b));
    need(ia != vocab.end() && ib != vocab.end() && im != vocab.end(), "merge with tokens missing from vocab");
    t.merges.emplace(pair_key(ia->second, ib->second), std::make_pair(rank++, im->second));
  }
  for (element a : get<array>(doc, "added_tokens", path)) {
    const std::string wa = path + " added_tokens";
    need(!get<bool>(a, "lstrip", wa) && !get<bool>(a, "rstrip", wa) && !get<bool>(a, "single_word", wa) &&
             !get<bool>(a, "normalized", wa),
         "added token flags");
    int64_t id = get<int64_t>(a, "id", wa);
    std::string content(get<sv>(a, "content", wa));
    define(id, content);
    t.added.emplace(std::move(content), int(id));
  }
  for (size_t id = 0; id < seen.size(); ++id) need(seen[id], "vocabulary (no entry for id " + std::to_string(id) + ")");
}

// The chat as the template lays it out: control tokens (id >= 0) and the text between them. Loss marks the
// assistant's tokens: a control token with loss, or the text from byte loss_from on.
struct Piece {
  int id = -1;
  std::string text;
  bool loss = false;
  size_t loss_from = std::string::npos;
};

bool tool_response(std::string_view s) {
  return s.starts_with("<tool_response>") && s.ends_with("</tool_response>");
}

// Both templates for system, user and assistant turns with no tools, enable_thinking=False. Qwen3.5's trims every
// turn; Qwen3's inserts them as given and lstrips newlines from an answer that follows a think block.
std::vector<Piece> render(const Tokenizer& t, const std::vector<Turn>& turns, bool generation_prompt) {
  const bool q35 = t.chat == ChatTemplate::Qwen35;
  auto turn = [&](const std::string& s) { return q35 ? py_strip(s) : s; };
  const int start = token_id(t, "<|im_start|>"), end = token_id(t, "<|im_end|>");
  const int think = token_id(t, "<think>"), unthink = token_id(t, "</think>");
  std::vector<Piece> p;
  auto control = [&](int id, bool loss = false) { p.push_back({id, {}, loss}); };
  auto text = [&](const std::string& s, size_t loss_from = std::string::npos) {
    p.push_back({-1, s, false, loss_from});
  };
  // The template's last query: the last user turn that is not a tool response.
  int last_query = -1;
  for (int i = int(turns.size()) - 1; i >= 0 && last_query < 0; --i)
    if (turns[i].role == "user" && !tool_response(turn(turns[i].content))) last_query = i;
  if (last_query < 0) throw std::runtime_error("chat: no user turn");
  for (size_t i = 0; i < turns.size(); ++i) {
    const std::string& role = turns[i].role;
    const std::string content = turn(turns[i].content);
    if (role == "system") {
      if (i != 0) throw std::runtime_error("chat: a system turn must come first");
      control(start), text("system\n" + content), control(end), text("\n");
    } else if (role == "user") {
      control(start), text("user\n" + content), control(end), text("\n");
    } else if (role == "assistant") {
      if (content.find("</think>") != std::string::npos)
        throw std::runtime_error("chat: assistant content holds </think>, which the template splits into reasoning");
      // Qwen3.5 opens every answer after the last query with the think block, Qwen3 only the conversation's last.
      if (int(i) > last_query && (q35 || i + 1 == turns.size())) {
        std::string answer = content;
        if (!q35) answer.erase(0, std::min(answer.find_first_not_of('\n'), answer.size()));
        control(start), text("assistant\n"), control(think), text("\n\n"), control(unthink);
        text("\n\n" + answer, 2);
      } else {
        const std::string head = "assistant\n";
        control(start), text(head + content, head.size());
      }
      control(end, true), text("\n");
    } else {
      throw std::runtime_error("chat: unsupported role \"" + role + "\" (system, user and assistant only)");
    }
  }
  if (generation_prompt) {
    control(start), text("assistant\n"), control(think), text("\n\n");
    control(unthink), text("\n\n");
  }
  return p;
}

// Text pieces are encoded whole, as the reference encodes the rendered string; a token is trained on when any of its
// bytes is at or past loss_from.
Rendered render_chat_pieces(const Tokenizer& t, const std::vector<Piece>& pieces) {
  Rendered r;
  for (const Piece& p : pieces) {
    if (p.id >= 0) {
      r.ids.push_back(p.id), r.loss.push_back(p.loss);
      continue;
    }
    size_t at = 0;
    for (int id : encode(t, p.text)) {
      at += t.id_bytes[id].size();
      r.ids.push_back(id), r.loss.push_back(p.loss_from != std::string::npos && at > p.loss_from);
    }
  }
  return r;
}

std::vector<Piece> render_prompt(const Tokenizer& t, const std::vector<Turn>& turns) {
  if (turns.empty() || turns.back().role != "user")
    throw std::runtime_error("chat: a prompt must end with a user turn");
  return render(t, turns, true);
}

}  // namespace

std::vector<Turn> prompt_turns(const std::optional<std::string>& system, const std::string& user) {
  std::vector<Turn> turns;
  if (system) turns.push_back({"system", *system});
  turns.push_back({"user", user});
  return turns;
}

Tokenizer load_tokenizer(const std::string& model_dir) {
  Tokenizer t;
  const std::string jinja = model_dir + "/chat_template.jinja", config = model_dir + "/tokenizer_config.json";
  std::string tmpl, tmpl_from = jinja;
  if (std::ifstream f(jinja, std::ios::binary); f) {
    tmpl.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  } else {
    simdjson::dom::parser parser;
    tmpl = std::string(get<std::string_view>(load_json(parser, config), "chat_template", config));
    tmpl_from = config;
  }
  const uint64_t h = fnv1a(tmpl);
  if (h != kQwen3Template && h != kQwen35Template) {
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
    throw std::runtime_error(tmpl_from + ": chat template is not one of the two hand-coded ones (Qwen3, Qwen3.5); "
                             "its FNV-1a is " + hex);
  }
  t.chat = h == kQwen3Template ? ChatTemplate::Qwen3 : ChatTemplate::Qwen35;
  t.template_fnv1a = h;

  const std::string path = model_dir + "/tokenizer.json";
  try {
    parse_tokenizer_json(t, path);
  } catch (const simdjson::simdjson_error& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
  return t;
}

std::vector<int> encode(const Tokenizer& t, std::string_view text) {
  std::vector<int> ids;
  if (text.empty()) return ids;
  std::string norm = nfc(text);
  for (std::string_view piece : pretokenize(norm, t.marks_join_letters)) {
    std::vector<int> p = bpe(t, piece);
    ids.insert(ids.end(), p.begin(), p.end());
  }
  return ids;
}

std::string decode(const Tokenizer& t, const std::vector<int>& ids) {
  std::string out;
  for (int id : ids) {
    if (id < 0 || size_t(id) >= t.id_bytes.size())
      throw std::runtime_error("token id " + std::to_string(id) + " is outside the tokenizer's " +
                               std::to_string(t.id_bytes.size()) + " ids");
    out += t.id_bytes[id];
  }
  return out;
}

int token_id(const Tokenizer& t, std::string_view added_token) {
  auto it = t.added.find(std::string(added_token));
  if (it == t.added.end()) throw std::runtime_error("tokenizer has no added token " + std::string(added_token));
  return it->second;
}

std::string chat_text(const Tokenizer& t, const std::vector<Turn>& turns) {
  std::string s;
  for (const Piece& p : render_prompt(t, turns)) s += p.id >= 0 ? t.id_bytes[p.id] : p.text;
  return s;
}

std::vector<int> chat_ids(const Tokenizer& t, const std::vector<Turn>& turns) {
  return render_chat_pieces(t, render_prompt(t, turns)).ids;
}

Rendered render_chat(const Tokenizer& t, const std::vector<Turn>& turns) {
  return render_chat_pieces(t, render(t, turns, false));
}

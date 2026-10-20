#pragma once
// simdjson lookups whose errors name the file and the key, not just simdjson's error code.
#include <simdjson.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

inline simdjson::dom::element load_json(simdjson::dom::parser& parser, const std::string& path) {
  simdjson::dom::element e;
  if (auto err = parser.load(path).get(e)) throw std::runtime_error(path + ": " + simdjson::error_message(err));
  return e;
}

inline bool has_key(simdjson::dom::element obj, std::string_view key) {
  simdjson::dom::element e;
  return !obj[key].get(e) && !e.is_null();
}

// obj[key] as T (int64_t, double, bool, std::string_view, dom::array, dom::object, dom::element). `where` names
// the file and the object, e.g. "DIR/config.json text_config".
template <typename T>
T get(simdjson::dom::element obj, std::string_view key, const std::string& where) {
  simdjson::dom::element e;
  if (obj[key].get(e)) throw std::runtime_error(where + ": missing key \"" + std::string(key) + "\"");
  T value;
  if (e.get(value)) throw std::runtime_error(where + ": key \"" + std::string(key) + "\" has the wrong type");
  return value;
}

// s as a JSON string literal, quotes included.
inline std::string json_quote(std::string_view s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\', out += char(c);
    } else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", c);
      out += buf;
    } else {
      out += char(c);
    }
  }
  return out + "\"";
}

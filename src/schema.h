#pragma once
// A JSON Schema compiled for constrained decoding. Only the keywords listed below are read;
// any other keyword is refused by name, with where it is, so a schema never means less than it says.
//
// Read: type (one name), properties, additionalProperties (false only), required, minProperties, enum, const,
// items (a schema or false), prefixItems, minItems, maxItems, uniqueItems (items that are enum/const values, or
// strings), minLength, maxLength, pattern (^[...]+$ or ^[^...]+$, + or *), minimum (0 only: no minus sign), anyOf,
// oneOf (read as anyOf: the decoder takes the first alternative a byte allows, so overlapping alternatives can
// both match), $ref to #/$defs/NAME or #/definitions/NAME (its sibling keywords apply too, as allOf), $defs.
// Ignored as annotations: $schema, $id, $comment, title, description, examples, default.
// Beyond the schema, a number has at most 15 digits before and after its point and 2 in its exponent, so it reads
// back finite; a string is json.dumps' escaping of its text, and no other.
#include <simdjson.h>

#include <climits>
#include <cstdint>
#include <string>
#include <vector>

enum class JType : uint8_t { Object, Array, String, Number, Integer, Boolean, Null };

// A code-point class from a pattern: ranges, or everything outside them.
struct CharClass {
  bool any = true, negated = false;
  std::vector<std::pair<char32_t, char32_t>> ranges;
};
bool in_class(const CharClass& c, char32_t cp);

// One alternative of a value.
struct Shape {
  bool literal = false;            // the value is one of `lits`, each as its compact JSON bytes, sorted
  std::vector<std::string> lits;
  JType type = JType::Null;
  // object: its properties by name (sorted), each name's JSON bytes after the opening quote, closing quote included
  std::vector<std::string> keys, key_bytes;
  std::vector<int> props;          // the node of each key's value
  std::vector<int> required;       // key indices
  int min_props = 0;
  // array
  std::vector<int> prefix;         // nodes of the leading items
  int items = -1;                  // node of every later item, -1: none
  int min_items = 0, max_items = INT_MAX;
  bool unique = false;
  // string
  int min_len = 0, max_len = INT_MAX;  // in code points
  CharClass cls;
  bool quote = false;              // at a key named quote, or a $ref to a definition named quote
  // number
  bool nonneg = false;
};

// A value: one of its shapes.
struct Node {
  std::vector<int> alts;
};

struct Schema {
  std::vector<Shape> shapes;
  std::vector<Node> nodes;
  int root = -1;
};

// Throws naming the path of the first keyword it cannot honour.
Schema compile_schema(simdjson::dom::element doc, const std::string& where);
Schema load_schema(const std::string& path);

// A value as Python's json.dumps(v, ensure_ascii=False, separators=(",", ":")) writes it.
std::string canonical_json(simdjson::dom::element v, const std::string& where);
// A string's bytes as json.dumps writes them between the quotes.
std::string json_string_body(std::string_view s);

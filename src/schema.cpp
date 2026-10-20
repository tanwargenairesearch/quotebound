#include "schema.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

#include "json.h"
#include "unicode.h"

using simdjson::dom::array;
using simdjson::dom::element;
using simdjson::dom::object;

bool in_class(const CharClass& c, char32_t cp) {
  if (c.any) return true;
  bool in = false;
  for (auto [lo, hi] : c.ranges) in = in || (cp >= lo && cp <= hi);
  return in != c.negated;
}

std::string json_string_body(std::string_view s) {
  std::string out;
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += char(c);
        }
    }
  }
  return out;
}

std::string canonical_json(element v, const std::string& where) {
  switch (v.type()) {
    case simdjson::dom::element_type::STRING: return "\"" + json_string_body(std::string_view(v)) + "\"";
    case simdjson::dom::element_type::INT64: return std::to_string(int64_t(v));
    case simdjson::dom::element_type::UINT64: return std::to_string(uint64_t(v));
    case simdjson::dom::element_type::BIGINT:
    case simdjson::dom::element_type::DOUBLE:
      throw std::runtime_error(where + ": a non-integer number in enum or const (its JSON text is not pinned)");
    case simdjson::dom::element_type::BOOL: return bool(v) ? "true" : "false";
    case simdjson::dom::element_type::NULL_VALUE: return "null";
    case simdjson::dom::element_type::ARRAY: {
      std::string out = "[";
      for (element e : array(v)) out += (out.size() > 1 ? "," : "") + canonical_json(e, where);
      return out + "]";
    }
    case simdjson::dom::element_type::OBJECT: {
      std::string out = "{";
      for (auto [k, e] : object(v))
        out += (out.size() > 1 ? "," : "") + ("\"" + json_string_body(k) + "\":") + canonical_json(e, where);
      return out + "}";
    }
  }
  throw std::runtime_error(where + ": unreadable value");
}

namespace {

const std::set<std::string_view> kAnnotations = {"$schema", "$id", "$comment", "title", "description", "examples",
                                                 "default"};
const std::set<std::string_view> kRead = {"type",     "properties", "additionalProperties", "required",
                                          "minProperties", "enum",  "const",     "items",   "prefixItems",
                                          "minItems", "maxItems",   "uniqueItems", "minLength", "maxLength",
                                          "pattern",  "minimum",    "anyOf",     "oneOf",   "$ref",
                                          "$defs",    "definitions"};

// One schema object's own keywords, before its $ref and anyOf are applied. Untyped when it gives no type, enum or
// const: then it only constrains a typed alternative it is merged with.
struct Part {
  std::optional<JType> type;
  std::optional<std::vector<std::string>> lits;
  bool has_props = false, closed = false;
  std::vector<std::pair<std::string, int>> props;
  std::vector<std::string> required;
  int min_props = 0;
  std::vector<int> prefix;
  bool has_items = false;
  int items = -1;
  int min_items = 0, max_items = INT_MAX;
  bool unique = false;
  int min_len = 0, max_len = INT_MAX;
  std::optional<CharClass> cls;
  bool nonneg = false;
};

JType json_type(char first) {
  switch (first) {
    case '"': return JType::String;
    case '{': return JType::Object;
    case '[': return JType::Array;
    case 't': case 'f': return JType::Boolean;
    case 'n': return JType::Null;
    default: return JType::Number;
  }
}

bool fits_type(const std::string& lit, JType t) {
  const JType have = json_type(lit[0]);
  if (t == JType::Integer) return have == JType::Number && lit.find_first_of(".eE") == std::string::npos;
  return have == t;
}

char32_t one_char(std::string_view s, size_t& i, const std::string& where) {
  std::vector<char32_t> cps;
  std::vector<uint32_t> offs;
  const size_t len = (uint8_t(s[i]) < 0x80) ? 1 : (uint8_t(s[i]) >= 0xF0) ? 4 : (uint8_t(s[i]) >= 0xE0) ? 3 : 2;
  if (i + len > s.size()) throw std::runtime_error(where + ": pattern ends inside a character");
  utf8_decode(s.substr(i, len), cps, offs);
  if (cps.size() != 1 || cps[0] == 0xFFFFFFFF) throw std::runtime_error(where + ": pattern is not UTF-8");
  i += len;
  return cps[0];
}

// ^[...]+$, ^[^...]+$, with * for +: one class, ranges a-b, \ escaping the next character. Returns the class and
// whether at least one character is needed.
std::pair<CharClass, bool> read_pattern(std::string_view p, const std::string& where) {
  auto refuse = [&] {
    throw std::runtime_error(where + ": pattern " + json_quote(p) +
                             " is not of the forms ^[...]+$, ^[^...]+$ (or * for +), the only ones read");
  };
  if (p.size() < 6 || p.substr(0, 2) != "^[" || p.back() != '$') refuse();
  const char rep = p[p.size() - 2];
  if ((rep != '+' && rep != '*') || p[p.size() - 3] != ']') refuse();
  std::string_view body = p.substr(2, p.size() - 5);
  CharClass c;
  c.any = false;
  size_t i = 0;
  if (!body.empty() && body[0] == '^') c.negated = true, i = 1;
  if (i >= body.size()) refuse();
  while (i < body.size()) {
    if (body[i] == '\\') ++i;
    if (i >= body.size()) refuse();
    char32_t lo = one_char(body, i, where), hi = lo;
    if (i + 1 < body.size() && body[i] == '-') {
      ++i;
      if (body[i] == '\\') ++i;
      hi = one_char(body, i, where);
      if (hi < lo) refuse();
    }
    c.ranges.push_back({lo, hi});
  }
  return {c, rep == '+'};
}

struct Compiler {
  element doc;
  std::string where;
  Schema s;
  int depth = 0;

  [[noreturn]] void fail(const std::string& path, const std::string& what) const {
    throw std::runtime_error(where + " " + (path.empty() ? "#" : path) + ": " + what);
  }

  int add_node(std::vector<int> alts) {
    s.nodes.push_back({std::move(alts)});
    return int(s.nodes.size()) - 1;
  }
  int add_shape(Shape sh) {
    s.shapes.push_back(std::move(sh));
    return int(s.shapes.size()) - 1;
  }

  int whole(element e, const std::string& path, const char* key) {
    int64_t v;
    if (e.get(v) || v < 0 || v > INT_MAX) fail(path, std::string(key) + " is not a whole number");
    return int(v);
  }

  Part part_of(object o, const std::string& path, bool quote) {
    Part p;
    for (auto [k, v] : o) {
      if (kAnnotations.count(k)) continue;
      if (!kRead.count(k)) fail(path, "keyword \"" + std::string(k) + "\" is not supported by constrained decoding");
    }
    element e;
    if (!o["type"].get(e)) {
      std::string_view t;
      if (e.get(t)) fail(path, "type must be one name (a list of types is not supported)");
      static const std::map<std::string_view, JType> names = {
          {"object", JType::Object}, {"array", JType::Array},     {"string", JType::String}, {"number", JType::Number},
          {"integer", JType::Integer}, {"boolean", JType::Boolean}, {"null", JType::Null}};
      auto it = names.find(t);
      if (it == names.end()) fail(path, "unknown type " + json_quote(t));
      p.type = it->second;
    }
    if (!o["enum"].get(e)) {
      array a;
      if (e.get(a)) fail(path, "enum is not a list");
      std::vector<std::string> lits;
      for (element v : a) lits.push_back(canonical_json(v, where + " " + path + "/enum"));
      p.lits = lits;
    }
    if (!o["const"].get(e)) {
      std::vector<std::string> one{canonical_json(e, where + " " + path + "/const")};
      if (p.lits && std::find(p.lits->begin(), p.lits->end(), one[0]) == p.lits->end()) one.clear();
      p.lits = one;
    }
    if (!o["properties"].get(e)) {
      object props;
      if (e.get(props)) fail(path, "properties is not an object");
      p.has_props = true;
      for (auto [k, v] : props)
        p.props.push_back({std::string(k), node_of(v, path + "/properties/" + std::string(k), k == "quote")});
    }
    if (!o["additionalProperties"].get(e)) {
      bool b;
      if (e.get(b) || b) fail(path, "additionalProperties other than false is not supported");
      p.closed = true;
    }
    if (!o["required"].get(e)) {
      array a;
      if (e.get(a)) fail(path, "required is not a list");
      for (element v : a) {
        std::string_view k;
        if (v.get(k)) fail(path, "required holds a non-string");
        p.required.push_back(std::string(k));
      }
    }
    if (!o["minProperties"].get(e)) p.min_props = whole(e, path, "minProperties");
    if (!o["prefixItems"].get(e)) {
      array a;
      if (e.get(a)) fail(path, "prefixItems is not a list");
      int i = 0;
      for (element v : a) p.prefix.push_back(node_of(v, path + "/prefixItems/" + std::to_string(i++), false));
    }
    if (!o["items"].get(e)) {
      p.has_items = true;
      bool b;
      if (!e.get(b)) {
        if (b) fail(path, "items: true is not supported (give the items' schema)");
        p.items = -1;
      } else {
        p.items = node_of(e, path + "/items", false);
      }
    }
    if (!o["minItems"].get(e)) p.min_items = whole(e, path, "minItems");
    if (!o["maxItems"].get(e)) p.max_items = whole(e, path, "maxItems");
    if (!o["uniqueItems"].get(e)) {
      bool b;
      if (e.get(b)) fail(path, "uniqueItems is not a boolean");
      p.unique = b;
    }
    if (!o["minLength"].get(e)) p.min_len = whole(e, path, "minLength");
    if (!o["maxLength"].get(e)) p.max_len = whole(e, path, "maxLength");
    if (!o["pattern"].get(e)) {
      std::string_view pat;
      if (e.get(pat)) fail(path, "pattern is not a string");
      auto [cls, plus] = read_pattern(pat, where + " " + path);
      p.cls = cls;
      if (plus) p.min_len = std::max(p.min_len, 1);
    }
    if (!o["minimum"].get(e)) {
      double v;
      if (e.get(v) || v != 0) fail(path, "minimum other than 0 is not supported");
      p.nonneg = true;
    }
    (void)quote;
    return p;
  }

  // The typed or literal shape a part makes, merged into `base` when one is given (a $ref target's or an anyOf
  // alternative's shape). nullopt: no value fits both.
  std::optional<Shape> merge(const Part& p, const Shape* base, const std::string& path) {
    Shape sh;
    if (base) {
      sh = *base;
    } else if (p.lits) {
      sh.literal = true;
    } else if (p.type) {
      sh.type = *p.type;
    } else {
      fail(path, "a schema needs a type, enum, const, $ref or anyOf");
    }
    // literals
    if (p.lits) {
      if (sh.literal) {
        if (base) {
          std::vector<std::string> keep;
          for (const std::string& l : sh.lits)
            if (std::find(p.lits->begin(), p.lits->end(), l) != p.lits->end()) keep.push_back(l);
          sh.lits = keep;
        } else {
          sh.lits = *p.lits;
        }
      } else {
        std::vector<std::string> keep;
        for (const std::string& l : *p.lits)
          if (literal_fits(l, sh, path)) keep.push_back(l);
        Shape lit;
        lit.literal = true, lit.lits = keep;
        sh = lit;
      }
    }
    if (sh.literal) {
      if (p.type) {
        std::vector<std::string> keep;
        for (const std::string& l : sh.lits)
          if (fits_type(l, *p.type)) keep.push_back(l);
        sh.lits = keep;
      }
      std::sort(sh.lits.begin(), sh.lits.end());
      sh.lits.erase(std::unique(sh.lits.begin(), sh.lits.end()), sh.lits.end());
      if (sh.lits.empty()) return std::nullopt;
      return sh;
    }
    if (p.type && *p.type != sh.type) {
      if (*p.type == JType::Integer && sh.type == JType::Number) sh.type = JType::Integer;
      else if (!(*p.type == JType::Number && sh.type == JType::Integer)) return std::nullopt;
    }
    // object keywords
    if (p.has_props || p.closed || !p.required.empty() || p.min_props) {
      if (sh.type != JType::Object) return sh;  // they apply to objects only
      std::map<std::string, int> props;
      for (size_t i = 0; i < sh.keys.size(); ++i) props[sh.keys[i]] = sh.props[i];
      const bool base_closed = base != nullptr;  // a finished object shape is always closed
      std::map<std::string, int> merged;
      if (base_closed) {
        merged = props;
        for (auto& [k, n] : p.props)
          if (merged.count(k)) merged[k] = intersect_nodes(merged[k], n, path + "/properties/" + k);
      } else {
        for (auto& [k, n] : p.props) merged[k] = n;
      }
      std::set<std::string> req;
      for (int r : sh.required) req.insert(sh.keys[r]);
      for (const std::string& r : p.required) req.insert(r);
      sh.keys.clear(), sh.key_bytes.clear(), sh.props.clear(), sh.required.clear();
      std::vector<std::tuple<std::string, std::string, int>> sorted;  // by the key's bytes, as the decoder reads them
      for (auto& [k, n] : merged) sorted.emplace_back(json_string_body(k) + "\"", k, n);
      std::sort(sorted.begin(), sorted.end());
      for (auto& [bytes, k, n] : sorted) {
        sh.keys.push_back(k);
        sh.key_bytes.push_back(bytes);
        sh.props.push_back(n);
      }
      for (const std::string& r : req) {
        auto it = std::find(sh.keys.begin(), sh.keys.end(), r);
        if (it == sh.keys.end()) return std::nullopt;  // required, and no key allows it
        sh.required.push_back(int(it - sh.keys.begin()));
      }
      sh.min_props = std::max(sh.min_props, p.min_props);
      if (!base_closed && !p.closed) fail(path, "an object needs additionalProperties: false");
      if (sh.keys.size() > 64) fail(path, "more than 64 properties");
    } else if (sh.type == JType::Object && !base) {
      fail(path, "an object needs properties and additionalProperties: false");
    }
    // array keywords
    if (sh.type == JType::Array) {
      if (!p.prefix.empty() || p.has_items) {
        if (base && (!sh.prefix.empty() || sh.items >= 0)) fail(path, "items given twice for one array");
        sh.prefix = p.prefix, sh.items = p.items;
      } else if (!base) {
        fail(path, "an array needs items (a schema or false) or prefixItems");
      }
      sh.min_items = std::max(sh.min_items, p.min_items);
      sh.max_items = std::min(sh.max_items, p.max_items);
      sh.unique = sh.unique || p.unique;
      if (sh.unique) check_unique(sh, path);
    }
    if (sh.type == JType::String) {
      sh.min_len = std::max(sh.min_len, p.min_len);
      sh.max_len = std::min(sh.max_len, p.max_len);
      if (p.cls) {
        if (!sh.cls.any) fail(path, "two patterns on one string");
        sh.cls = *p.cls;
      }
      if (sh.min_len > sh.max_len) return std::nullopt;
    }
    if (sh.type == JType::Number || sh.type == JType::Integer) sh.nonneg = sh.nonneg || p.nonneg;
    if (sh.type == JType::Boolean || sh.type == JType::Null) {
      Shape lit;
      lit.literal = true;
      lit.lits = sh.type == JType::Null ? std::vector<std::string>{"null"} : std::vector<std::string>{"false", "true"};
      return lit;
    }
    return sh;
  }

  // Unique items the decoder can hold apart: enum/const values, or strings (at most 8 of them).
  void check_unique(const Shape& sh, const std::string& path) {
    std::vector<int> item_nodes = sh.prefix;
    if (sh.items >= 0) item_nodes.push_back(sh.items);
    for (int n : item_nodes) {
      const Node& node = s.nodes[n];
      const bool lits = node.alts.size() == 1 && s.shapes[node.alts[0]].literal &&
                        s.shapes[node.alts[0]].lits.size() <= 64;
      bool strings = true;
      for (int a : node.alts) strings = strings && !s.shapes[a].literal && s.shapes[a].type == JType::String;
      if (!lits && !(strings && sh.max_items <= 8))
        fail(path, "uniqueItems is supported for items that are one enum/const (at most 64 values) or strings "
                   "(maxItems at most 8)");
    }
    if (!sh.prefix.empty()) fail(path, "uniqueItems with prefixItems is not supported");
  }

  bool literal_fits(const std::string& lit, const Shape& sh, const std::string& path) {
    if (!fits_type(lit, sh.type)) return false;
    if (sh.type == JType::String) {
      if (sh.quote) fail(path, "an enum or const on a quote string is not supported");
      std::vector<char32_t> cps;
      std::vector<uint32_t> offs;
      // the literal's body without its quotes; escapes count one character each as JSON reads them
      std::string_view body(lit.data() + 1, lit.size() - 2);
      int n = 0;
      bool ok = true;
      for (size_t i = 0; i < body.size();) {
        char32_t cp;
        if (body[i] == '\\') {
          cp = body[i + 1] == 'u' ? char32_t(std::stoi(std::string(body.substr(i + 2, 4)), nullptr, 16))
                                  : char32_t(body[i + 1] == 'n' ? '\n' : body[i + 1] == 't' ? '\t' : body[i + 1]);
          i += body[i + 1] == 'u' ? 6 : 2;
        } else {
          cp = one_char(body, i, path);
        }
        ok = ok && in_class(sh.cls, cp);
        ++n;
      }
      return ok && n >= sh.min_len && n <= sh.max_len;
    }
    return true;
  }

  int intersect_nodes(int a, int b, const std::string& path) {
    std::vector<int> alts;
    for (int x : s.nodes[a].alts)
      for (int y : s.nodes[b].alts)
        if (auto sh = intersect_shapes(s.shapes[x], s.shapes[y], path)) alts.push_back(add_shape(*sh));
    if (alts.empty()) fail(path, "no value fits both schemas");
    return add_node(alts);
  }

  std::optional<Shape> intersect_shapes(const Shape& x, const Shape& y, const std::string& path) {
    if (x.literal || y.literal) {
      const Shape& lit = x.literal ? x : y;
      const Shape& other = x.literal ? y : x;
      Shape out = lit;
      out.lits.clear();
      for (const std::string& l : lit.lits) {
        const bool ok = other.literal ? std::binary_search(other.lits.begin(), other.lits.end(), l)
                                      : literal_fits(l, other, path);
        if (ok) out.lits.push_back(l);
      }
      if (out.lits.empty()) return std::nullopt;
      return out;
    }
    if (x.type != y.type) return std::nullopt;
    if (x.type == JType::Object || x.type == JType::Array)
      fail(path, "merging two object or array schemas below a $ref is not supported");
    Shape out = x;
    out.min_len = std::max(x.min_len, y.min_len), out.max_len = std::min(x.max_len, y.max_len);
    if (!y.cls.any) {
      if (!x.cls.any) fail(path, "two patterns on one string");
      out.cls = y.cls;
    }
    out.quote = x.quote || y.quote;
    out.nonneg = x.nonneg || y.nonneg;
    return out;
  }

  element resolve(std::string_view ref, const std::string& path, std::string& name) {
    std::string_view section;
    if (ref.substr(0, 8) == "#/$defs/") section = "$defs", name = std::string(ref.substr(8));
    else if (ref.substr(0, 14) == "#/definitions/") section = "definitions", name = std::string(ref.substr(14));
    else fail(path, "$ref " + json_quote(ref) + ": only #/$defs/NAME and #/definitions/NAME are supported");
    element defs, target;
    if (doc[section].get(defs) || defs[name].get(target)) fail(path, "$ref " + json_quote(ref) + " is not defined");
    return target;
  }

  int node_of(element e, const std::string& path, bool quote) {
    if (++depth > 64) fail(path, "nested more than 64 deep (a recursive $ref is not supported)");
    object o;
    if (e.get(o)) fail(path, "a schema must be an object");
    const Part own = part_of(o, path, quote);
    std::vector<int> alts;
    element ref, any_of, one_of;
    const bool has_any_of = !o["anyOf"].get(any_of), has_one_of = !o["oneOf"].get(one_of);
    if (has_any_of && has_one_of) fail(path, "anyOf and oneOf together are not supported");
    const bool has_any = has_any_of || has_one_of;
    const element any = has_any_of ? any_of : one_of;
    if (!o["$ref"].get(ref)) {
      if (has_any) fail(path, "$ref with anyOf is not supported");
      std::string_view r;
      if (ref.get(r)) fail(path, "$ref is not a string");
      std::string name;
      element target = resolve(r, path, name);
      const int n = node_of(target, std::string(r.substr(1)), quote || name == "quote");
      for (int a : s.nodes[n].alts) {
        const Shape base = s.shapes[a];
        if (auto sh = merge(own, &base, path)) alts.push_back(add_shape(*sh));
      }
    } else if (has_any) {
      array a;
      if (any.get(a)) fail(path, "anyOf is not a list");
      int i = 0;
      for (element alt : a) {
        const int n = node_of(alt, path + "/anyOf/" + std::to_string(i++), quote);
        for (int x : s.nodes[n].alts) {
          const Shape base = s.shapes[x];
          if (auto sh = merge(own, &base, path)) alts.push_back(add_shape(*sh));
        }
      }
    } else if (auto sh = merge(own, nullptr, path)) {
      alts.push_back(add_shape(*sh));
    }
    if (alts.empty()) fail(path, "no value fits this schema");
    if (quote)
      for (int& a : alts)
        if (!s.shapes[a].literal && s.shapes[a].type == JType::String && !s.shapes[a].quote) {
          Shape q = s.shapes[a];
          q.quote = true;
          a = add_shape(q);
        }
    --depth;
    return add_node(alts);
  }
};

}  // namespace

Schema compile_schema(element doc, const std::string& where) {
  Compiler c{doc, where, {}, 0};
  c.s.root = c.node_of(doc, "", false);
  return std::move(c.s);
}

Schema load_schema(const std::string& path) {
  simdjson::dom::parser parser;
  return compile_schema(load_json(parser, path), path);
}

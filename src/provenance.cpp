#include "provenance.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

#include "json.h"

namespace {

std::string hex(const unsigned char* digest) {
  std::string out;
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
    char b[3];
    std::snprintf(b, sizeof b, "%02x", digest[i]);
    out += b;
  }
  return out;
}

}  // namespace

std::string sha256_hex(const void* data, size_t n) {
  CC_SHA256_CTX ctx;
  CC_SHA256_Init(&ctx);
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t at = 0; at < n;) {
    const size_t chunk = std::min<size_t>(n - at, 1u << 30);
    CC_SHA256_Update(&ctx, p + at, CC_LONG(chunk));
    at += chunk;
  }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &ctx);
  return hex(digest);
}

std::string sha256_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  CC_SHA256_CTX ctx;
  CC_SHA256_Init(&ctx);
  std::vector<char> buf(1 << 20);
  while (f.read(buf.data(), std::streamsize(buf.size())) || f.gcount() > 0)
    CC_SHA256_Update(&ctx, buf.data(), CC_LONG(f.gcount()));
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &ctx);
  return hex(digest);
}

BaseModel identify_base(const std::string& dir, const Tokenizer& t) {
  BaseModel b;
  const std::string index = dir + "/model.safetensors.index.json";
  b.config_sha256 = sha256_file(dir + "/config.json");
  b.index_sha256 = sha256_file(index);
  simdjson::dom::parser parser;
  std::set<std::string> files;
  for (auto [name, file] : get<simdjson::dom::object>(load_json(parser, index), "weight_map", index))
    files.emplace(std::string_view(file));
  for (const std::string& f : files) b.shards.emplace_back(f, uint64_t(std::filesystem::file_size(dir + "/" + f)));
  char fnv[17];
  std::snprintf(fnv, sizeof fnv, "%016llx", static_cast<unsigned long long>(t.template_fnv1a));
  b.chat_template = fnv;
  return b;
}

std::string base_json(const BaseModel& b) {
  std::string shards;
  for (const auto& [name, size] : b.shards)
    shards += (shards.empty() ? "[" : ", [") + json_quote(name) + ", " + std::to_string(size) + "]";
  return "{\"config_sha256\": " + json_quote(b.config_sha256) + ", \"index_sha256\": " +
         json_quote(b.index_sha256) + ", \"shards\": [" + shards + "], \"chat_template_fnv1a\": " +
         json_quote(b.chat_template) + "}";
}

BaseModel parse_base(simdjson::dom::element e, const std::string& where) {
  BaseModel b;
  b.config_sha256 = std::string(get<std::string_view>(e, "config_sha256", where));
  b.index_sha256 = std::string(get<std::string_view>(e, "index_sha256", where));
  b.chat_template = std::string(get<std::string_view>(e, "chat_template_fnv1a", where));
  for (simdjson::dom::element s : get<simdjson::dom::array>(e, "shards", where)) {
    std::string_view name;
    int64_t size = 0;
    if (s.at(0).get(name) || s.at(1).get(size)) throw std::runtime_error(where + ": a shard is not [name, size]");
    b.shards.emplace_back(std::string(name), uint64_t(size));
  }
  return b;
}

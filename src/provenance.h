#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <simdjson.h>

#include "tokenizer.h"

// Lowercase hex sha256 of bytes, and of a file's bytes.
std::string sha256_hex(const void* data, size_t n);
std::string sha256_file(const std::string& path);

// The checkpoint a run trains on, pinned as the parity fixtures pin one: the sha256 of config.json and of the
// safetensors index, each shard's name and size (hashing the shards themselves would read tens of GB), and the
// chat template's FNV-1a.
struct BaseModel {
  std::string config_sha256, index_sha256, chat_template;
  std::vector<std::pair<std::string, uint64_t>> shards;  // sorted by name
  bool operator==(const BaseModel&) const = default;
};
BaseModel identify_base(const std::string& model_dir, const Tokenizer& t);
std::string base_json(const BaseModel& b);
BaseModel parse_base(simdjson::dom::element e, const std::string& where);

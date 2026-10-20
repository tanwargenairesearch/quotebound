#pragma once
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Flushes a file, or a directory's entries, to stable storage. macOS's fsync stops at the drive's cache;
// F_FULLFSYNC goes through it. A file system that refuses F_FULLFSYNC gets fsync.
inline void sync_path(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path + " to sync it");
  int r = ::fcntl(fd, F_FULLFSYNC);
  if (r != 0) r = ::fsync(fd);
  ::close(fd);
  if (r != 0) throw std::runtime_error("cannot sync " + path);
}

// Makes the finished file `tmp` the file `path`: synced, renamed over it, and the rename synced, so after a crash
// `path` is the old file or the new one, whole.
inline void commit_file(const std::string& tmp, const std::string& path) {
  sync_path(tmp);
  std::filesystem::rename(tmp, path);
  const std::filesystem::path dir = std::filesystem::path(path).parent_path();
  sync_path(dir.empty() ? "." : dir.string());
}

inline void write_file(const std::string& path, const std::string& bytes) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), std::streamsize(bytes.size()));
    if (!f.flush()) throw std::runtime_error("cannot write " + tmp);
  }
  commit_file(tmp, path);
}

inline std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Directories nothing here reads or writes, such as a held-out evaluation set: the colon-separated list in the
// environment variable QUOTEBOUND_DENY, plus the list this build was compiled with (make DENY=..., empty by default).
// A leading "~/" is the home directory.
#ifndef QUOTEBOUND_DEFAULT_DENY
#define QUOTEBOUND_DEFAULT_DENY ""
#endif

inline std::vector<std::string> split_deny_list(const std::string& list) {
  std::vector<std::string> dirs;
  const char* home = std::getenv("HOME");
  size_t start = 0;
  while (start <= list.size()) {
    size_t end = list.find(':', start);
    if (end == std::string::npos) end = list.size();
    std::string d = list.substr(start, end - start);
    if (d.rfind("~/", 0) == 0 && home && *home) d = std::string(home) + d.substr(1);
    if (!d.empty()) dirs.push_back(d);
    start = end + 1;
  }
  return dirs;
}

inline std::vector<std::string> deny_dirs() {
  std::vector<std::string> dirs = split_deny_list(QUOTEBOUND_DEFAULT_DENY);
  if (const char* env = std::getenv("QUOTEBOUND_DENY"))
    for (std::string& d : split_deny_list(env)) dirs.push_back(std::move(d));
  return dirs;
}

// Refuses `path` when it is one of `dirs` or under one. Symlinks and ".." are resolved first, as far as the path exists.
inline void refuse_denied(const std::string& flag, const std::string& path, const std::vector<std::string>& dirs) {
  namespace fs = std::filesystem;
  if (path.empty()) return;
  auto resolve = [](const fs::path& p) {
    std::error_code ec;
    fs::path r = fs::weakly_canonical(fs::absolute(p), ec);
    return ec ? fs::absolute(p).lexically_normal() : r;
  };
  const fs::path p = resolve(path);
  for (const std::string& dir : dirs) {
    const fs::path d = resolve(dir);
    if (std::mismatch(d.begin(), d.end(), p.begin(), p.end()).first == d.end())
      throw std::runtime_error(flag + " " + path + ": refused, it is under " + dir +
                               " (a denied directory: QUOTEBOUND_DENY or the build's DENY)");
  }
}

inline void refuse_denied(const std::string& flag, const std::string& path) { refuse_denied(flag, path, deny_dirs()); }

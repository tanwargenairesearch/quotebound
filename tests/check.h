#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "../src/provenance.h"  // sha256_file, to pin a fixture to the file it was dumped from

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n  ", __FILE__, __LINE__, #cond);   \
      std::fprintf(stderr, __VA_ARGS__);                                       \
      std::fputc('\n', stderr);                                                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

inline double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

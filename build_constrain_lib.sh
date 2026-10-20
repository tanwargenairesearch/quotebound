#!/usr/bin/env bash
# Builds build/libquotebound_constrain.so on Linux: quotebound's constraint alone (src/constrain_capi.h), no MLX and no
# CoreFoundation (QUOTEBOUND_NO_COREFOUNDATION: only tokenizing needs it, and the library never tokenizes). The Mac builds
# the same library with `make constrain-lib`.
#
#   build_constrain_lib.sh [--simdjson-tarball FILE]
#
# simdjson is compiled in at the Mac's version, 4.6.3: Debian's and Ubuntu's packages are 3.x, which lacks the BIGINT
# element type schema.cpp reads. Its release tarball is fetched from GitHub (or FILE) and must match the pinned
# sha256; its singleheader/simdjson.h is byte-identical to the Mac's Homebrew header (sha256 37e6b2a7...).
# Needs g++ (C++20) and curl.
set -euo pipefail
cd "$(dirname "$0")"
SIMDJSON_VERSION=4.6.3
SIMDJSON_SHA256=bde0c42f43899c4c5c48be826c09abd22500ed537b89f16b3cced5eec477c263
SIMDJSON_H_SHA256=37e6b2a7b8822df77da87d6dae0b812cb331512525c5a8ca556e227cec51e855
TARBALL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --simdjson-tarball) TARBALL="$2"; shift 2 ;;
    *) echo "build_constrain_lib: unknown argument $1" >&2; exit 2 ;;
  esac
done
CXX="${CXX:-g++}"
DEP="build/simdjson-$SIMDJSON_VERSION"
mkdir -p "$DEP"
if [[ ! -f "$DEP/simdjson.cpp" ]]; then
  if [[ -z "$TARBALL" ]]; then
    TARBALL="$DEP/v$SIMDJSON_VERSION.tar.gz"
    curl -fsSL -o "$TARBALL" "https://github.com/simdjson/simdjson/archive/refs/tags/v$SIMDJSON_VERSION.tar.gz"
  fi
  echo "$SIMDJSON_SHA256  $TARBALL" | sha256sum -c --quiet -
  tar -xzf "$TARBALL" -C "$DEP" --strip-components=2 \
    "simdjson-$SIMDJSON_VERSION/singleheader/simdjson.h" "simdjson-$SIMDJSON_VERSION/singleheader/simdjson.cpp"
  echo "$SIMDJSON_H_SHA256  $DEP/simdjson.h" | sha256sum -c --quiet -
fi
"$CXX" -std=c++20 -O2 -Wall -Wextra -fPIC -shared -DQUOTEBOUND_NO_COREFOUNDATION -isystem "$DEP" \
  src/constrain_capi.cpp src/constrain.cpp src/schema.cpp src/tokenizer.cpp src/pretokenize.cpp src/unicode.cpp \
  "$DEP/simdjson.cpp" -o build/libquotebound_constrain.so
echo "built $(pwd)/build/libquotebound_constrain.so ($("$CXX" --version | head -n1); simdjson $SIMDJSON_VERSION)"

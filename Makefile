# quotebound: build and test on Apple silicon.
#
# make               -> build/quotebound
# make test-quick    -> tokenizers, chat rendering, the training-data loader and batch plans, the gated-delta
#                       recurrence (3 ways + 2 gradients), tiny random models, exact multi-row decode steps, LoRA
#                       training on them (no real weights; the tokenizers are read from the checkpoints' directories),
#                       generate --batch on them, and schema-constrained decoding (testdata/example_schema.json)
# make bench         -> the recurrence's backward, the fast op's vjp against gated_delta_trainable's (several GB)
# make test          -> test-quick, then parity against mlx-lm for each Qwen3.5 checkpoint, one model loaded at a time
# make constrain-lib -> build/libquotebound_constrain.dylib, the constraint alone with no MLX
#
# Settings, each overridable on the command line (make MLX=...) or in an optional config.mk beside this file:
#   PYTHON    a Python with the mlx wheel installed (tested: mlx 0.32.3); MLX defaults to that wheel's directory
#   MLX       the mlx package directory holding include/ and lib/libmlx.dylib
#   SIMDJSON  a simdjson install with include/ and lib/ (tested: 4.6.3; 3.x lacks the BIGINT type schema.cpp reads)
#   MODELS    the directory holding the checkpoints the tests read (see README.md)
#   DENY      directories the binary refuses to read or write, colon-separated ("~/" is the home directory)
#   MACOS_MIN the minimum macOS the binary targets (the mlx 0.32.3 wheel is built for macOS 26)
-include config.mk

PYTHON    ?= python3
MLX       ?= $(shell $(PYTHON) -c 'import pathlib, mlx.core; print(pathlib.Path(mlx.core.__file__).parent)' 2>/dev/null)
SIMDJSON  ?= $(shell brew --prefix simdjson 2>/dev/null)
MODELS    ?= models
DENY      ?=
MACOS_MIN ?= 26.2
# the tokenizer and chat tests read only the tokenizer and template: the 8-bit checkpoint's are the same ids
QWEN3_TOK ?= $(MODELS)/qwen3-8b-8bit
QWEN35_4B ?= $(MODELS)/qwen3.5-4b-16bit
QWEN35_9B ?= $(MODELS)/qwen3.5-9b-16bit

ifeq ($(MLX),)
$(error MLX not found: pip install mlx==0.32.3 into $(PYTHON), or pass MLX=/path/to/site-packages/mlx)
endif
ifeq ($(SIMDJSON),)
$(error simdjson not found: brew install simdjson, or pass SIMDJSON=/path/to/simdjson)
endif

CXX      := clang++
MACOS    := -mmacosx-version-min=$(MACOS_MIN)
CONFIG_H := build/quotebound_config.h
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -MMD -MP $(MACOS) -include $(CONFIG_H) \
            -isystem $(MLX)/include -isystem $(MLX)/include/metal_cpp -isystem $(SIMDJSON)/include
LDFLAGS  := $(MACOS) -L$(MLX)/lib -lmlx -Wl,-rpath,$(MLX)/lib -L$(SIMDJSON)/lib -lsimdjson \
            -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate -framework CoreFoundation

# DENY is compiled in: rewrite the header only when it changes, so every object that includes it rebuilds then.
$(shell mkdir -p build && printf '#define QUOTEBOUND_DEFAULT_DENY "%s"\n' '$(DENY)' > build/.config.tmp && \
        (cmp -s build/.config.tmp $(CONFIG_H) || mv build/.config.tmp $(CONFIG_H)); rm -f build/.config.tmp)

OBJ := $(patsubst %,build/%.o,unicode pretokenize tokenizer provenance model deltanet delta_grad exact generate lora \
         data loss adamw train batch schema constrain draft)

all: build/quotebound

build/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
build/test_%.o: tests/test_%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
build/quotebound: build/main.o $(OBJ)
	$(CXX) $^ $(LDFLAGS) -o $@
build/test_%: build/test_%.o $(OBJ)
	$(CXX) $^ $(LDFLAGS) -o $@
build/bench_%.o: tests/bench_%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
build/bench_%: build/bench_%.o $(OBJ)
	$(CXX) $^ $(LDFLAGS) -o $@
build/smoke_data.o: tests/smoke_data.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
build/smoke_data: build/smoke_data.o $(OBJ)
	$(CXX) $^ $(LDFLAGS) -o $@

# The constraint alone, no MLX, for a decoder outside quotebound (src/constrain_capi.h); on Linux build it with
# build_constrain_lib.sh, which needs no CoreFoundation.
CAPI_OBJ := $(patsubst %,build/%.o,unicode pretokenize tokenizer schema constrain constrain_capi)
build/libquotebound_constrain.dylib: $(CAPI_OBJ)
	$(CXX) -dynamiclib $^ $(MACOS) -L$(SIMDJSON)/lib -lsimdjson -Wl,-rpath,$(SIMDJSON)/lib -framework CoreFoundation -o $@
constrain-lib: build/libquotebound_constrain.dylib

test-quick: build/test_tokenizer build/test_chat build/test_data build/test_deltanet build/test_tiny_model \
            build/test_exact build/test_train build/test_batch build/test_constrain
	build/test_tokenizer $(QWEN3_TOK) testdata/tokenizer_qwen3.json
	build/test_tokenizer $(QWEN35_4B) testdata/tokenizer_qwen35.json
	build/test_chat $(QWEN3_TOK) testdata/chats_qwen3.json
	build/test_chat $(QWEN35_4B) testdata/chats_qwen35.json
	build/test_data $(QWEN35_4B)
	build/test_deltanet
	build/test_tiny_model
	build/test_exact
	build/test_train $(QWEN35_4B)
	build/test_batch $(QWEN35_4B)
	build/test_constrain $(QWEN35_4B) testdata/example_schema.json
bench: build/bench_grad
	build/bench_grad
test-qwen35-4b: build/test_parity
	build/test_parity $(QWEN35_4B) testdata/parity_qwen35_4b.json
test-qwen35-9b: build/test_parity
	build/test_parity $(QWEN35_9B) testdata/parity_qwen35_9b.json
test: test-quick
	$(MAKE) test-qwen35-4b
	$(MAKE) test-qwen35-9b
clean:
	rm -rf build

.PHONY: all constrain-lib test test-quick test-qwen35-4b test-qwen35-9b bench clean
.SECONDARY:
-include $(wildcard build/*.d)

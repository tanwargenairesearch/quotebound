// A real checkpoint against fixtures dumped once from mlx-lm (testdata/parity_<model>.json): prompt ids, top-1 at
// every prompt position, log-softmax at the reference's top-k, and the 32-token greedy continuation. For Qwen3.5
// the reference runs quotebound's recurrence primitive; mlx-lm as shipped ("stock") is reported, not asserted.
#include <simdjson.h>

#include <cmath>
#include <filesystem>
#include <set>
#include <string>

#include "../src/generate.h"
#include "../src/model.h"
#include "../src/tokenizer.h"
#include "check.h"

static std::string text_of(simdjson::dom::element e) { return std::string(std::string_view(e)); }

static std::vector<int> ints(simdjson::dom::element e) {
  std::vector<int> v;
  for (auto x : simdjson::dom::array(e)) v.push_back(int(int64_t(x)));
  return v;
}

// A list of rows, flattened.
static std::vector<int> ints2(simdjson::dom::element e) {
  std::vector<int> v;
  for (auto row : simdjson::dom::array(e))
    for (auto x : simdjson::dom::array(row)) v.push_back(int(int64_t(x)));
  return v;
}

static std::vector<float> floats2(simdjson::dom::element e) {
  std::vector<float> v;
  for (auto row : simdjson::dom::array(e))
    for (auto x : simdjson::dom::array(row)) v.push_back(float(double(x)));
  return v;
}

// The fixture must come from this MLX and from these weights: the index file's sha256 and each shard's size.
static void check_pins(simdjson::dom::element fx, const std::string& dir, const char* fixture) {
  const std::string mlx = text_of(fx["versions"]["mlx"]);
  CHECK(mlx == mx::version(), "%s was dumped with mlx %s; this build links mlx %s", fixture, mlx.c_str(),
        mx::version());
  const std::string index = dir + "/model.safetensors.index.json";
  const std::string want = text_of(fx["checkpoint"]["index_sha256"]), got = sha256_file(index);
  CHECK(got == want, "%s is not the checkpoint %s was dumped from (sha256 %s, fixture %s)", index.c_str(), fixture,
        got.c_str(), want.c_str());
  for (auto s : simdjson::dom::array(fx["checkpoint"]["shards"])) {
    const std::string path = dir + "/" + text_of(s.at(0));
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    CHECK(!ec && int64_t(size) == int64_t(s.at(1)), "%s: size %lld, fixture %lld", path.c_str(),
          ec ? -1LL : (long long)size, (long long)int64_t(s.at(1)));
  }
}

int main(int argc, char** argv) {
  CHECK(argc == 3, "usage: test_parity MODEL_DIR testdata/parity_<model>.json");
  const std::string dir = argv[1];
  simdjson::dom::parser parser;
  simdjson::dom::element fx = parser.load(argv[2]);
  check_pins(fx, dir, argv[2]);

  const Tokenizer tok = load_tokenizer(dir);
  auto t0 = std::chrono::steady_clock::now();
  Model m = load_model(dir);
  double load_ms = ms_since(t0);
  std::vector<int> stop = m.cfg.eos;
  stop.push_back(token_id(tok, "<|im_end|>"));
  stop.push_back(token_id(tok, "<|endoftext|>"));

  const int k = int(int64_t(fx["topk"]));
  const float tight = float(double(fx["tight_tolerance"]));
  int prompts = 0, positions = 0, top1_fail = 0, cont_fail = 0, generate_equal = 0;
  int stock_prompts = 0, stock_top1 = 0, stock_cont_equal = 0;
  float worst = 0, worst_ratio = 0, stock_worst = 0;
  for (auto p : simdjson::dom::array(fx["prompts"])) {
    const std::string name = text_of(p["name"]), system = text_of(p["system"]), user = text_of(p["user"]);
    const std::vector<int> ids = ints(p["ids"]);
    const int T = int(ids.size());
    CHECK(chat_ids(tok, prompt_turns(system.empty() ? std::nullopt : std::optional(system), user)) == ids,
          "prompt ids differ from mlx-lm's for prompt %s", name.c_str());

    const std::vector<int> at = ints(p["lp_positions"]), ref_ids = ints2(p["top_ids"]);
    const std::vector<float> ref_lp = floats2(p["top_logprobs"]);
    const int P = int(at.size());
    CHECK(int(ref_ids.size()) == P * k && int(ref_lp.size()) == P * k, "fixture shape (prompt %s)", name.c_str());

    mx::array logits = forward(m, mx::array(ids.data(), {1, T}, mx::int32));
    mx::array lp = logits - mx::logsumexp(logits, -1, true);
    mx::array top1 = mx::astype(mx::argmax(lp, -1), mx::int32);
    mx::array mine = mx::take_along_axis(mx::take(lp, mx::array(at.data(), {P}, mx::int32), 1),
                                         mx::array(ref_ids.data(), {1, P, k}, mx::int32), -1);
    mx::array diff = mx::max(mx::abs(mine - mx::array(ref_lp.data(), {1, P, k}, mx::float32)));
    std::optional<mx::array> stock_diff;
    if (p["stock_top_logprobs"].error() == simdjson::SUCCESS) {
      std::vector<float> s = floats2(p["stock_top_logprobs"]);
      stock_diff = mx::max(mx::abs(mine - mx::array(s.data(), {1, P, k}, mx::float32)));
    }
    mx::eval({top1, diff});
    const float d = diff.item<float>(), tol = float(double(p["tolerance"]));
    worst = std::max(worst, d);
    worst_ratio = std::max(worst_ratio, d / tol);
    const int32_t* t1 = top1.data<int32_t>();
    const std::vector<int> ref_top1 = ints(p["argmax"]);
    int fails = 0;
    for (int t = 0; t < T; ++t) fails += t1[t] != ref_top1[t];
    top1_fail += fails;
    positions += T;

    Generation g = greedy(m, ids, 32, stop);
    std::vector<int> want = ints(p["continuation_ids"]);
    std::string hex = text_of(p["continuation_hex"]), want_bytes;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) want_bytes += char(std::stoi(hex.substr(i, 2), nullptr, 16));
    const bool same = g.ids == want && decode(tok, g.ids) == want_bytes;
    cont_fail += !same;
    generate_equal += g.ids == ints(p["mlx_lm_generate_ids"]);
    ++prompts;

    std::string stock;
    if (stock_diff) {
      const std::vector<int> stock_top1_ref = ints(p["stock_argmax"]);
      int agree = 0;
      for (int t = 0; t < T; ++t) agree += t1[t] == stock_top1_ref[t];
      const float sd = stock_diff->item<float>();
      const bool cont = g.ids == ints(p["stock_continuation_ids"]);
      ++stock_prompts, stock_top1 += agree, stock_worst = std::max(stock_worst, sd), stock_cont_equal += cont;
      char buf[160];
      std::snprintf(buf, sizeof buf, "; stock mlx-lm: top-1 agrees %d/%d, max |dlogprob| %.3g, continuation %s", agree,
                    T, sd, cont ? "same" : "differs");
      stock = buf;
    }
    std::printf("  %s: %d tokens, top-1 differs at %d, max |dlogprob| %.3g at %d positions (tolerances %.3g and %.3g, "
                "max|logit| %.1f), continuation %s%s\n",
                name.c_str(), T, fails, d, P, tight, tol, double(p["max_abs_logit"]), same ? "identical" : "DIFFERS",
                stock.c_str());
  }
  std::printf(
      "parity %s (%s): load %.0f ms; %d prompts, %d positions, top-1 differs at %d; max |dlogprob| over top-%d %.3g "
      "(tight tolerance %.3g; %.2g of the bf16 tolerance); greedy 32-token continuations identical %d/%d; equal to "
      "mlx_lm.generate's %d/%d\n",
      argv[2], text_of(fx["reference"]).c_str(), load_ms, prompts, positions, top1_fail, k, worst, tight, worst_ratio,
      prompts - cont_fail, prompts, generate_equal, prompts);
  if (stock_prompts)
    std::printf("  against mlx-lm as shipped: top-1 agrees at %d of %d positions, max |dlogprob| %.3g, continuations "
                "the same %d/%d\n",
                stock_top1, positions, stock_worst, stock_cont_equal, stock_prompts);
  CHECK(prompts == 7, "expected 7 prompts, have %d", prompts);
  CHECK(top1_fail == 0 && worst <= tight && worst_ratio <= 1 && cont_fail == 0, "parity");
  return 0;
}

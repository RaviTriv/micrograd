#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gpt/GPTConfig.h"
#include "gpt/Model.h"
#include "micrograd/Tensor.h"

using namespace micrograd;
using micrograd::gpt::GPTConfig;
using micrograd::gpt::Model;

namespace {

GPTConfig TinyConfig() {
  GPTConfig config;
  config.depth = 1;
  config.n_layer = 1;
  config.n_embd = 16;
  config.n_head = 2;
  config.block_size = 8;
  return config;
}

bool HasParameter(Model &model, const std::string &fragment) {
  for (const auto &[name, tensor] : model.named_parameters()) {
    if (name.find(fragment) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::shared_ptr<Tensor> Tokens(size_t batch, size_t seq_len) {
  std::vector<scalar_t> values(batch * seq_len);
  for (size_t i = 0; i < values.size(); i++) {
    values[i] = static_cast<scalar_t>((i * 7) % 11);
  }
  return Tensor::from_values({batch, seq_len}, std::move(values));
}

}  // namespace

TEST(GptArch, DefaultUsesLearnedPositions) {
  Model model(11, TinyConfig());
  EXPECT_TRUE(HasParameter(model, "position_embedding"));
  EXPECT_FALSE(HasParameter(model, "q_norm"));
}

TEST(GptArch, RotaryDropsPositionEmbedding) {
  GPTConfig config = TinyConfig();
  config.arch.rotary = true;
  Model model(11, config);
  EXPECT_FALSE(HasParameter(model, "position_embedding"));
}

TEST(GptArch, QkNormAddsPerHeadGains) {
  GPTConfig config = TinyConfig();
  config.arch.qk_norm = true;
  Model model(11, config);
  EXPECT_TRUE(HasParameter(model, "attn.q_norm"));
  EXPECT_TRUE(HasParameter(model, "attn.k_norm"));
}

TEST(GptArch, RotaryScoresDependOnlyOnOffset) {
  constexpr size_t kSeq = 6;
  constexpr size_t kDim = 8;
  std::vector<scalar_t> row = {0.3f, -1.2f, 0.7f, 2.0f,
                               -0.4f, 0.9f, 1.5f, -0.8f};
  std::vector<scalar_t> values;
  for (size_t t = 0; t < kSeq; t++) {
    values.insert(values.end(), row.begin(), row.end());
  }
  auto x = Tensor::from_values({1, kSeq, kDim}, std::move(values));
  auto rotated = x->rotary_embedding();
  auto scores = rotated->matmul(rotated->transpose(1, 2));
  auto s = scores->data();
  for (size_t m = 0; m + 1 < kSeq; m++) {
    for (size_t n = 0; n + 1 < kSeq; n++) {
      EXPECT_NEAR(s[(m * kSeq) + n], s[((m + 1) * kSeq) + n + 1], 1e-4f)
          << "m=" << m << " n=" << n;
    }
  }
}

TEST(GptArch, RotarySeqDimMatchesPermutedLayout) {
  std::vector<scalar_t> values(2 * 5 * 3 * 8);
  for (size_t i = 0; i < values.size(); i++) {
    values[i] = std::sin(0.37f * static_cast<scalar_t>(i));
  }
  auto direct = Tensor::from_values({2, 5, 3, 8}, values);
  auto permuted = Tensor::from_values({2, 5, 3, 8}, values);
  direct->set_requires_grad(true);
  permuted->set_requires_grad(true);
  auto a = direct->rotary_embedding(10000.0f, 1);
  auto b = permuted->permute({0, 2, 1, 3})
               ->rotary_embedding()
               ->permute({0, 2, 1, 3});
  auto av = a->data();
  auto bv = b->data();
  for (size_t i = 0; i < av.size(); i++) {
    EXPECT_NEAR(av[i], bv[i], 1e-6f) << i;
  }
  a->mul(a)->sum()->backward();
  b->mul(b)->sum()->backward();
  auto ag = direct->grad();
  auto bg = permuted->grad();
  for (size_t i = 0; i < ag.size(); i++) {
    EXPECT_NEAR(ag[i], bg[i], 1e-5f) << i;
  }
}

TEST(GptArch, HeadNormRotaryMatchesSeparateOps) {
  constexpr size_t kHeads = 3;
  constexpr size_t kDim = 8;
  std::vector<scalar_t> values(2 * 5 * kHeads * kDim);
  for (size_t i = 0; i < values.size(); i++) {
    values[i] = std::sin(0.37f * static_cast<scalar_t>(i));
  }
  auto fused_in = Tensor::from_values({2, 5, kHeads * kDim}, values);
  auto split_in = Tensor::from_values({2, 5, kHeads * kDim}, values);
  auto fused_gain = Tensor::from_values(
      {kDim}, {0.5f, 1.0f, 1.5f, 2.0f, -0.5f, 0.8f, 1.2f, 0.3f});
  auto split_gain = Tensor::from_values(
      {kDim}, {0.5f, 1.0f, 1.5f, 2.0f, -0.5f, 0.8f, 1.2f, 0.3f});
  for (const auto &t : {fused_in, split_in, fused_gain, split_gain}) {
    t->set_requires_grad(true);
  }
  auto a = fused_in->head_norm_rotary(kHeads, fused_gain, 1e-5f, true);
  auto b = split_in->reshape({2, 5, kHeads, kDim})
               ->rms_norm({kDim}, split_gain, 1e-5f)
               ->rotary_embedding(10000.0f, 1)
               ->reshape({2, 5, kHeads * kDim});
  for (size_t i = 0; i < a->size(); i++) {
    EXPECT_NEAR(a->data()[i], b->data()[i], 1e-5f) << i;
  }
  a->mul(a)->sum()->backward();
  b->mul(b)->sum()->backward();
  for (size_t i = 0; i < fused_in->size(); i++) {
    EXPECT_NEAR(fused_in->grad()[i], split_in->grad()[i], 1e-4f) << i;
  }
  for (size_t i = 0; i < kDim; i++) {
    EXPECT_NEAR(fused_gain->grad()[i], split_gain->grad()[i], 1e-4f) << i;
  }
}

TEST(GptArch, ReluSquaredMatchesReluThenSquare) {
  auto fused =
      Tensor::from_values({2, 3}, {-1.5f, -0.2f, 0.0f, 0.3f, 1.1f, 2.5f});
  auto split =
      Tensor::from_values({2, 3}, {-1.5f, -0.2f, 0.0f, 0.3f, 1.1f, 2.5f});
  fused->set_requires_grad(true);
  split->set_requires_grad(true);
  auto a = fused->relu_squared();
  auto b = split->relu()->pow(2.0f);
  for (size_t i = 0; i < 6; i++) {
    EXPECT_FLOAT_EQ(a->data()[i], b->data()[i]);
  }
  a->sum()->backward();
  b->sum()->backward();
  for (size_t i = 0; i < 6; i++) {
    EXPECT_FLOAT_EQ(fused->grad()[i], split->grad()[i]);
  }
}

TEST(GptArch, AllSwitchesTrainEndToEnd) {
  GPTConfig config = TinyConfig();
  config.arch = {.rotary = true, .qk_norm = true, .relu_squared = true};
  Model model(11, config);
  auto logits = model.forward(Tokens(2, 8));
  ASSERT_EQ(logits->shape(), (std::vector<size_t>{2, 8, 11}));
  logits->pow(2.0f)->sum()->backward();
  for (const auto &[name, tensor] : model.named_parameters()) {
    if (name.find("q_norm") == std::string::npos) {
      continue;
    }
    double total = 0.0;
    for (scalar_t g : tensor->grad()) {
      total += std::abs(g);
    }
    EXPECT_GT(total, 0.0) << name;
  }
}

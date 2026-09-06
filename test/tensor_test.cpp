#include "micrograd/Tensor.h"

#include <gtest/gtest.h>

#include <functional>
#include <memory>

#include "micrograd/NN.h"

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"

#define SKIP_WITHOUT_METAL()                       \
  do {                                             \
    auto &ctx = MetalContext::instance();          \
    if (!ctx.isAvailable() && !ctx.initialize()) { \
      GTEST_SKIP() << "no Metal device available"; \
    }                                              \
  } while (0)
#else
#define SKIP_WITHOUT_METAL() GTEST_SKIP() << "built without Metal support"
#endif

using namespace micrograd;

auto scalar(scalar_t val) {
  return std::make_shared<Tensor>(std::vector<size_t>{1},
                                  std::vector<scalar_t>{val});
}

auto vec(const std::vector<scalar_t> &values) {
  return std::make_shared<Tensor>(std::vector<size_t>{values.size()}, values);
}

void expect_grad_matches_numeric(
    const std::vector<scalar_t> &values,
    const std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)> &f) {
  auto x = vec(values);
  f(x)->backward();
  const std::vector<scalar_t> analytic(x->grad().begin(), x->grad().end());

  constexpr scalar_t h = 1e-2f;
  for (size_t i = 0; i < values.size(); i++) {
    auto shifted = values;

    shifted[i] = values[i] + h;
    const scalar_t up = f(vec(shifted))->data()[0];

    shifted[i] = values[i] - h;
    const scalar_t down = f(vec(shifted))->data()[0];

    EXPECT_NEAR(analytic[i], (up - down) / (2 * h), 1e-3) << "at index " << i;
  }
}

void expect_backends_agree(const std::vector<std::vector<scalar_t>> &inputs,
                           const std::function<std::shared_ptr<Tensor>(
                               std::vector<std::shared_ptr<Tensor>>)> &f) {
  std::vector<std::shared_ptr<Tensor>> cpu_in, gpu_in;
  for (const auto &values : inputs) {
    cpu_in.push_back(vec(values));
    gpu_in.push_back(vec(values));
    gpu_in.back()->to(Backend::Metal);
  }

  auto cpu_out = f(cpu_in);
  cpu_out->backward();

  auto gpu_out = f(gpu_in);
  gpu_out->backward();
  gpu_out->to(Backend::CPU);

  ASSERT_EQ(cpu_out->size(), gpu_out->size());
  for (size_t i = 0; i < cpu_out->size(); i++) {
    EXPECT_NEAR(cpu_out->data()[i], gpu_out->data()[i], 1e-4) << "value " << i;
  }

  for (size_t arg = 0; arg < inputs.size(); arg++) {
    gpu_in[arg]->to(Backend::CPU);
    for (size_t i = 0; i < inputs[arg].size(); i++) {
      EXPECT_NEAR(cpu_in[arg]->grad()[i], gpu_in[arg]->grad()[i], 1e-4)
          << "grad of arg " << arg << " at " << i;
    }
  }
}

// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
const std::vector<scalar_t> kLhs = {1.5, -2.0, 0.5, 3.0};
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
const std::vector<scalar_t> kRhs = {2.0, 4.0, -1.5, 0.25};

TEST(TensorTest, SanityCheck) {
  auto x = scalar(-4.0);

  auto z = x->mul(2.0)->add(2.0)->add(x);
  auto q = z->relu()->add(z->mul(x));
  auto h = z->mul(z)->relu();
  auto y = h->add(q)->add(q->mul(x));

  y->backward();

  EXPECT_NEAR(y->data()[0], -20.0, 1e-9);
  EXPECT_NEAR(x->grad()[0], 46.0, 1e-9);
}

TEST(TensorTest, Add) {
  auto a = scalar(4);
  auto b = scalar(5);
  auto c = a->add(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], 9.0, 1e-9);
  EXPECT_NEAR(a->grad()[0], 1, 1e-9);
  EXPECT_NEAR(b->grad()[0], 1, 1e-9);
}

TEST(TensorTest, Sub) {
  auto a = scalar(4);
  auto b = scalar(7);
  auto c = a->sub(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], -3.0, 1e-9);
  EXPECT_NEAR(a->grad()[0], 1, 1e-9);
  EXPECT_NEAR(b->grad()[0], -1, 1e-9);
}

TEST(TensorTest, Mul) {
  auto a = scalar(4);
  auto b = scalar(5);
  auto c = a->mul(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], 20.0, 1e-9);
  EXPECT_NEAR(a->grad()[0], 5, 1e-9);
  EXPECT_NEAR(b->grad()[0], 4, 1e-9);
}

TEST(TensorTest, Div) {
  auto a = scalar(4);
  auto b = scalar(5);
  auto c = a->div(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], 0.80, 1e-6);
  EXPECT_NEAR(a->grad()[0], 0.20, 1e-6);
  EXPECT_NEAR(b->grad()[0], -0.16, 1e-6);
}

TEST(TensorTest, Pow) {
  auto a = scalar(4);
  auto b = a->pow(3.0);

  b->backward();

  EXPECT_NEAR(b->data()[0], 64, 1e-9);
  EXPECT_NEAR(a->grad()[0], 48, 1e-9);
}

TEST(TensorTest, ChainRule) {
  auto x = scalar(3);
  auto a = x->pow(3.0);
  auto b = a->add(1.0);
  auto y = b->pow(2.0);

  y->backward();

  EXPECT_NEAR(y->data()[0], 784, 1e-9);
  EXPECT_NEAR(x->grad()[0], 1512, 1e-9);
}

TEST(TensorTest, NumericGradRelu) {
  expect_grad_matches_numeric({-2.0f, 0.5f, 3.0f},
                              [](const auto &x) { return x->relu()->sum(); });
}

TEST(TensorTest, NumericGradSigmoid) {
  expect_grad_matches_numeric(
      {-1.5f, 0.3f, 2.0f}, [](const auto &x) { return x->sigmoid()->sum(); });
}

TEST(TensorTest, NumericGradTanh) {
  expect_grad_matches_numeric({-1.5f, 0.3f, 2.0f},
                              [](const auto &x) { return x->tanh()->sum(); });
}

TEST(TensorTest, NumericGradPow) {
  expect_grad_matches_numeric(
      {0.5f, 1.5f, 2.5f}, [](const auto &x) { return x->pow(3.0f)->sum(); });
}

TEST(TensorTest, NumericGradDiv) {
  expect_grad_matches_numeric({1.0f, 2.0f}, [](const auto &x) {
    return x->div(vec({4.0f, 5.0f}))->sum();
  });
}

TEST(TensorTest, NumericGradComposite) {
  expect_grad_matches_numeric({0.4f, -0.7f, 1.2f}, [](const auto &x) {
    return x->mul(x)->add(x->tanh())->sigmoid()->sum();
  });
}

TEST(TensorTest, MatmulNonSquare) {
  auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
                                    std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
  auto b = std::make_shared<Tensor>(
      std::vector<size_t>{3, 4},
      std::vector<scalar_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  auto c = a->matmul(b);
  c->sum()->backward();

  ASSERT_EQ(c->shape(), (std::vector<size_t>{2, 4}));
  EXPECT_NEAR(c->data()[0], 38.0, 1e-9);
  EXPECT_NEAR(c->data()[3], 56.0, 1e-9);
  EXPECT_NEAR(c->data()[7], 128.0, 1e-9);

  EXPECT_NEAR(a->grad()[0], 10.0, 1e-9);
  EXPECT_NEAR(a->grad()[1], 26.0, 1e-9);
  EXPECT_NEAR(a->grad()[2], 42.0, 1e-9);

  EXPECT_NEAR(b->grad()[0], 5.0, 1e-9);
  EXPECT_NEAR(b->grad()[4], 7.0, 1e-9);
  EXPECT_NEAR(b->grad()[8], 9.0, 1e-9);
}

TEST(TensorTest, InvalidShapesThrow) {
  EXPECT_THROW(vec({1.0, 2.0})->add(vec({1.0})), std::invalid_argument);
  EXPECT_THROW(vec({1.0, 2.0})->mul(vec({1.0})), std::invalid_argument);
  EXPECT_THROW(vec({1.0, 2.0})->matmul(vec({1.0, 2.0})), std::invalid_argument);
  EXPECT_THROW(std::make_shared<Tensor>(std::vector<size_t>{2, 2},
                                        std::vector<scalar_t>{1.0}),
               std::invalid_argument);

  auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
                                    std::vector<scalar_t>(6, 1.0));
  EXPECT_THROW(a->matmul(a), std::invalid_argument);
}

TEST(TensorTest, LinearSgdReducesLoss) {
  Linear layer(2, 1);
  SGD optimizer({layer.weights(), layer.bias()}, 0.1f);

  auto input = std::make_shared<Tensor>(std::vector<size_t>{1, 2},
                                        std::vector<scalar_t>{0.5, -0.5});
  auto target = std::make_shared<Tensor>(std::vector<size_t>{1, 1},
                                         std::vector<scalar_t>{1.0});

  const scalar_t before = mse_loss(layer.forward(input), target)->at({0});

  for (int step = 0; step < 50; step++) {
    auto loss = mse_loss(layer.forward(input), target);
    optimizer.zero_grad();
    loss->backward();
    optimizer.step();
  }

  const scalar_t after = mse_loss(layer.forward(input), target)->at({0});
  EXPECT_LT(after, before);
  EXPECT_NEAR(after, 0.0, 1e-3);
}

TEST(TensorTest, GraphIsFreed) {
  std::weak_ptr<Tensor> w;
  {
    auto x = scalar(2.0);
    auto y = x->mul(3.0)->relu();
    y->backward();
    w = y;
  }
  EXPECT_TRUE(w.expired());
}

TEST(TensorTest, MetalMatchesCpuAdd) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({kLhs, kRhs},
                        [](const auto &in) { return in[0]->add(in[1]); });
}

TEST(TensorTest, MetalMatchesCpuSub) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({kLhs, kRhs},
                        [](const auto &in) { return in[0]->sub(in[1]); });
}

TEST(TensorTest, MetalMatchesCpuMul) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({kLhs, kRhs},
                        [](const auto &in) { return in[0]->mul(in[1]); });
}

TEST(TensorTest, MetalMatchesCpuDiv) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({kLhs, kRhs},
                        [](const auto &in) { return in[0]->div(in[1]); });
}

TEST(TensorTest, MetalMatchesCpuScalarOps) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({kLhs}, [](const auto &in) { return in[0]->add(2.5); });
  expect_backends_agree({kLhs},
                        [](const auto &in) { return in[0]->sub(1.25); });
  expect_backends_agree({kLhs}, [](const auto &in) { return in[0]->mul(3.0); });
  expect_backends_agree({kLhs}, [](const auto &in) { return in[0]->div(4.0); });
}

TEST(TensorTest, MetalMatchesCpuPow) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({{0.5, 1.5, 2.0, 3.0}},
                        [](const auto &in) { return in[0]->pow(2.0); });
}

TEST(TensorTest, MetalMatchesCpuRelu) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({kLhs}, [](const auto &in) { return in[0]->relu(); });
}

TEST(TensorTest, MetalMatchesCpuSigmoid) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({kLhs},
                        [](const auto &in) { return in[0]->sigmoid(); });
}

TEST(TensorTest, MetalMatchesCpuTanh) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({kLhs}, [](const auto &in) { return in[0]->tanh(); });
}

TEST(TensorTest, MetalMatchesCpuSum) {
  SKIP_WITHOUT_METAL();
  expect_backends_agree({kLhs}, [](const auto &in) { return in[0]->sum(); });
}

TEST(TensorTest, MetalMatchesCpuMatmul) {
  SKIP_WITHOUT_METAL();
  const std::vector<scalar_t> a_data = {1, 2, 3, 4, 5, 6};
  const std::vector<scalar_t> b_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  auto make = [](const std::vector<size_t> &shape,
                 const std::vector<scalar_t> &data) {
    return std::make_shared<Tensor>(shape, data);
  };

  auto cpu_a = make({2, 3}, a_data);
  auto cpu_b = make({3, 4}, b_data);
  auto cpu_c = cpu_a->matmul(cpu_b);
  cpu_c->sum()->backward();

  auto gpu_a = make({2, 3}, a_data);
  auto gpu_b = make({3, 4}, b_data);
  gpu_a->to(Backend::Metal);
  gpu_b->to(Backend::Metal);
  auto gpu_c = gpu_a->matmul(gpu_b);
  gpu_c->sum()->backward();

  gpu_c->to(Backend::CPU);
  gpu_a->to(Backend::CPU);
  gpu_b->to(Backend::CPU);

  for (size_t i = 0; i < cpu_c->size(); i++) {
    EXPECT_NEAR(cpu_c->data()[i], gpu_c->data()[i], 1e-4) << "value " << i;
  }
  for (size_t i = 0; i < a_data.size(); i++) {
    EXPECT_NEAR(cpu_a->grad()[i], gpu_a->grad()[i], 1e-4) << "grad a " << i;
  }
  for (size_t i = 0; i < b_data.size(); i++) {
    EXPECT_NEAR(cpu_b->grad()[i], gpu_b->grad()[i], 1e-4) << "grad b " << i;
  }
}

TEST(TensorTest, SumMetalTimeComparison) {
  SKIP_WITHOUT_METAL();
  std::vector<scalar_t> data(10000);
  double expected = 0.0;
  for (int i = 0; i < 10000; i++) {
    data[static_cast<size_t>(i)] = static_cast<scalar_t>(i + 1);
    expected += static_cast<double>(i + 1);
  }

  auto x = std::make_shared<Tensor>(std::vector<size_t>{10000}, data);
  x->to(Backend::Metal);

  auto y = x->sum();

  EXPECT_NEAR(y->data()[0], expected, 1e-1);
}

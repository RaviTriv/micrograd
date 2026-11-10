#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

namespace {

struct Problem {
  size_t batch;
  size_t m;
  size_t k;
  size_t n;
  bool lhs_batched;
  bool rhs_batched;
  bool rhs_transposed;
};

struct Reference {
  std::vector<double> out;
  std::vector<double> lhs_grad;
  std::vector<double> rhs_grad;
};

std::vector<scalar_t> Uniform(size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<scalar_t> dist(-1.0f, 1.0f);
  std::vector<scalar_t> values(count);
  for (auto &value : values) {
    value = dist(rng);
  }
  return values;
}

std::vector<size_t> LhsShape(const Problem &p) {
  if (p.lhs_batched) {
    return {p.batch, p.m, p.k};
  }
  return {p.m, p.k};
}

std::vector<size_t> RhsShape(const Problem &p) {
  if (p.rhs_transposed) {
    return {p.n, p.k};
  }
  if (p.rhs_batched) {
    return {p.batch, p.k, p.n};
  }
  return {p.k, p.n};
}

Reference Naive(const Problem &p, const std::vector<scalar_t> &lhs,
                const std::vector<scalar_t> &rhs,
                const std::vector<scalar_t> &seed) {
  Reference ref{.out = std::vector<double>(p.batch * p.m * p.n),
                .lhs_grad = std::vector<double>(lhs.size()),
                .rhs_grad = std::vector<double>(rhs.size())};
  for (size_t b = 0; b < p.batch; b++) {
    const size_t lo = p.lhs_batched ? b * p.m * p.k : 0;
    const size_t ro = p.rhs_batched ? b * p.k * p.n : 0;
    for (size_t i = 0; i < p.m; i++) {
      for (size_t j = 0; j < p.n; j++) {
        const size_t o = (b * p.m * p.n) + (i * p.n) + j;
        for (size_t q = 0; q < p.k; q++) {
          const size_t li = lo + (i * p.k) + q;
          const size_t ri =
              ro + (p.rhs_transposed ? (j * p.k) + q : (q * p.n) + j);
          ref.out[o] += static_cast<double>(lhs[li]) * rhs[ri];
          ref.lhs_grad[li] += static_cast<double>(seed[o]) * rhs[ri];
          ref.rhs_grad[ri] += static_cast<double>(lhs[li]) * seed[o];
        }
      }
    }
  }
  return ref;
}

double Tolerance(size_t reduction) {
  return 1e-5 + (1e-6 * static_cast<double>(reduction));
}

void ExpectClose(std::span<const scalar_t> actual,
                 const std::vector<double> &expected, double tolerance,
                 const std::string &what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  for (size_t i = 0; i < expected.size(); i++) {
    ASSERT_NEAR(actual[i], expected[i], tolerance) << what << " at " << i;
  }
}

std::shared_ptr<Tensor> Run(const Problem &p,
                            const std::shared_ptr<Tensor> &lhs,
                            const std::shared_ptr<Tensor> &rhs) {
  return p.rhs_transposed ? lhs->matmul_nt(rhs) : lhs->matmul(rhs);
}

void CheckAgainstNaive(const Problem &p) {
  SCOPED_TRACE("batch=" + std::to_string(p.batch) + " m=" +
               std::to_string(p.m) + " k=" + std::to_string(p.k) +
               " n=" + std::to_string(p.n));
  const auto lhs_shape = LhsShape(p);
  const auto rhs_shape = RhsShape(p);
  const auto lhs_values = Uniform(lhs_shape.size() == 3 ? p.batch * p.m * p.k
                                                        : p.m * p.k,
                                  1);
  const auto rhs_values =
      Uniform(rhs_shape.size() == 3 ? p.batch * p.k * p.n : p.k * p.n, 2);
  const auto seed_values = Uniform(p.batch * p.m * p.n, 3);

  auto lhs = leaf(lhs_shape, lhs_values);
  auto rhs = leaf(rhs_shape, rhs_values);
  auto out = Run(p, lhs, rhs);
  out->backward(*Tensor::from_values(out->shape(), seed_values));

  const Reference ref = Naive(p, lhs_values, rhs_values, seed_values);
  const size_t lhs_reduction = p.n * (p.lhs_batched ? 1 : p.batch);
  const size_t rhs_reduction = p.m * (p.rhs_batched ? 1 : p.batch);
  ExpectClose(out->data(), ref.out, Tolerance(p.k), "out");
  ExpectClose(lhs->grad(), ref.lhs_grad, Tolerance(lhs_reduction), "lhs grad");
  ExpectClose(rhs->grad(), ref.rhs_grad, Tolerance(rhs_reduction), "rhs grad");
}

Problem TwoD(size_t m, size_t k, size_t n) {
  return {.batch = 1, .m = m, .k = k, .n = n, .lhs_batched = false,
          .rhs_batched = false, .rhs_transposed = false};
}

Problem Batched(size_t batch, size_t m, size_t k, size_t n) {
  return {.batch = batch, .m = m, .k = k, .n = n, .lhs_batched = true,
          .rhs_batched = true, .rhs_transposed = false};
}

Problem FoldRows(size_t batch, size_t m, size_t k, size_t n) {
  return {.batch = batch, .m = m, .k = k, .n = n, .lhs_batched = true,
          .rhs_batched = false, .rhs_transposed = false};
}

Problem BroadcastLhs(size_t batch, size_t m, size_t k, size_t n) {
  return {.batch = batch, .m = m, .k = k, .n = n, .lhs_batched = false,
          .rhs_batched = true, .rhs_transposed = false};
}

Problem NT(size_t batch, size_t m, size_t k, size_t n) {
  return {.batch = batch, .m = m, .k = k, .n = n, .lhs_batched = batch > 1,
          .rhs_batched = false, .rhs_transposed = true};
}

}  // namespace

TEST(Matmul, TwoDMatchesNaiveBelowThreshold) {
  for (const auto &p : {TwoD(1, 1, 1), TwoD(3, 5, 2), TwoD(1, 7, 1),
                        TwoD(2, 1, 3), TwoD(8, 8, 8), TwoD(65, 3, 66)}) {
    CheckAgainstNaive(p);
  }
}

TEST(Matmul, TwoDMatchesNaiveAboveThreshold) {
  for (const auto &p : {TwoD(200, 150, 97), TwoD(2, 600, 700),
                        TwoD(1, 300, 1000), TwoD(1000, 1, 300),
                        TwoD(700, 400, 1)}) {
    CheckAgainstNaive(p);
  }
}

TEST(Matmul, BatchedMatchesNaive) {
  for (const auto &p : {Batched(1, 1, 1, 1), Batched(3, 4, 5, 6),
                        Batched(64, 8, 8, 8), Batched(5, 70, 65, 33),
                        Batched(47, 33, 40, 50), Batched(9, 1, 200, 300)}) {
    CheckAgainstNaive(p);
  }
}

TEST(Matmul, FoldedRowsMatchNaive) {
  for (const auto &p : {FoldRows(2, 3, 4, 5), FoldRows(3, 50, 40, 70),
                        FoldRows(4, 1, 1, 1)}) {
    CheckAgainstNaive(p);
  }
}

TEST(Matmul, BroadcastLhsMatchesNaive) {
  for (const auto &p : {BroadcastLhs(3, 2, 4, 5), BroadcastLhs(6, 40, 50, 60)}) {
    CheckAgainstNaive(p);
  }
}

TEST(Matmul, TransposedRhsMatchesNaive) {
  for (const auto &p : {NT(1, 1, 1, 1), NT(1, 3, 5, 7), NT(1, 1, 300, 9),
                        NT(1, 130, 90, 70), NT(1, 300, 1, 400),
                        NT(2, 40, 65, 50), NT(3, 5, 6, 7),
                        NT(1, 64, 513, 65)}) {
    CheckAgainstNaive(p);
  }
}

TEST(Matmul, SkipsGradientForOperandThatDoesNotRequireGrad) {
  for (const bool transposed : {false, true}) {
    auto weight = Tensor::from_values({20, 30}, Uniform(600, 4));
    auto input = leaf({10, transposed ? size_t{30} : size_t{20}},
                      Uniform(transposed ? 300 : 200, 5));
    auto out = transposed ? input->matmul_nt(weight) : input->matmul(weight);
    out->backward(*Tensor::from_values(
        out->shape(), std::vector<scalar_t>(out->size(), 1.0f)));
    EXPECT_TRUE(input->has_grad());
    EXPECT_FALSE(weight->has_grad());

    auto frozen = Tensor::from_values(input->shape(), Values(input));
    auto trainable = leaf({20, 30}, Values(weight));
    auto out2 =
        transposed ? frozen->matmul_nt(trainable) : frozen->matmul(trainable);
    out2->backward(*Tensor::from_values(
        out2->shape(), std::vector<scalar_t>(out2->size(), 1.0f)));
    EXPECT_TRUE(trainable->has_grad());
    EXPECT_FALSE(frozen->has_grad());
  }
}

TEST(Matmul, BackwardAccumulatesIntoExistingGrad) {
  constexpr size_t kM = 300;
  constexpr size_t kK = 200;
  constexpr size_t kN = 100;
  auto lhs = leaf({kM, kK}, Uniform(kM * kK, 6));
  auto rhs = leaf({kK, kN}, Uniform(kK * kN, 7));
  const auto seed =
      Tensor::from_values({kM, kN}, std::vector<scalar_t>(kM * kN, 1.0f));
  lhs->matmul(rhs)->backward(*seed);
  const std::vector<scalar_t> lhs_once(lhs->grad().begin(), lhs->grad().end());
  const std::vector<scalar_t> rhs_once(rhs->grad().begin(), rhs->grad().end());
  lhs->matmul(rhs)->backward(*seed);
  for (size_t i = 0; i < lhs_once.size(); i++) {
    ASSERT_NEAR(lhs->grad()[i], 2.0f * lhs_once[i], Tolerance(kN));
  }
  for (size_t i = 0; i < rhs_once.size(); i++) {
    ASSERT_NEAR(rhs->grad()[i], 2.0f * rhs_once[i], Tolerance(kM));
  }
}

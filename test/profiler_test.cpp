#include "micrograd/Profiler.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "micrograd/Tensor.h"

using namespace micrograd;

namespace {

const profiler::Entry *Find(const profiler::Summary &summary,
                            const std::string &name, const std::string &phase) {
  for (const auto &entry : summary.entries) {
    if (entry.name == name && entry.phase == phase) {
      return &entry;
    }
  }
  return nullptr;
}

std::shared_ptr<Tensor> Leaf(std::vector<size_t> shape,
                             std::vector<scalar_t> values) {
  auto leaf = Tensor::from_values(std::move(shape), std::move(values));
  leaf->set_requires_grad(true);
  return leaf;
}

}  // namespace

TEST(Profiler, RecordsNothingWhileDisabled) {
  profiler::Enable();
  profiler::Disable();

  auto a = Leaf({2, 2}, {1, 2, 3, 4});
  auto b = a->add(a);

  EXPECT_FALSE(profiler::Enabled());
  EXPECT_TRUE(profiler::Collect().entries.empty());
}

TEST(Profiler, CountsForwardAndBackwardOps) {
  profiler::Enable();
  auto a = Leaf({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = Leaf({3, 2}, {1, 0, 0, 1, 1, 1});
  auto loss = a->matmul(b)->sum();
  loss->backward();
  const profiler::Summary summary = profiler::Collect();
  profiler::Disable();

  const auto *matmul_fwd = Find(summary, "matmul", "fwd");
  const auto *matmul_bwd = Find(summary, "matmul", "bwd");
  const auto *sum_fwd = Find(summary, "sum", "fwd");
  const auto *sum_bwd = Find(summary, "sum", "bwd");
  ASSERT_NE(matmul_fwd, nullptr);
  ASSERT_NE(matmul_bwd, nullptr);
  ASSERT_NE(sum_fwd, nullptr);
  ASSERT_NE(sum_bwd, nullptr);
  EXPECT_EQ(matmul_fwd->calls, 1u);
  EXPECT_EQ(matmul_bwd->calls, 1u);
  EXPECT_EQ(sum_fwd->calls, 1u);
  EXPECT_EQ(sum_bwd->calls, 1u);
  EXPECT_GE(summary.wall_ms, summary.busy_ms);
}

TEST(Profiler, CountsNestedScopesOnceInBusyTime) {
  profiler::Enable();
  {
    profiler::Scope outer("outer", "step", Device::CPU);
    profiler::Scope inner("inner", "step", Device::CPU);
  }
  const profiler::Summary summary = profiler::Collect();
  profiler::Disable();

  const auto *outer = Find(summary, "outer", "step");
  const auto *inner = Find(summary, "inner", "step");
  ASSERT_NE(outer, nullptr);
  ASSERT_NE(inner, nullptr);
  EXPECT_EQ(outer->nested_calls, 0u);
  EXPECT_EQ(inner->nested_calls, 1u);
  EXPECT_DOUBLE_EQ(summary.busy_ms, outer->total_ms);
}

TEST(Profiler, EnableClearsEarlierTimings) {
  profiler::Enable();
  auto a = Leaf({2}, {1, 2});
  auto b = a->mul(a);
  profiler::Enable();
  const profiler::Summary summary = profiler::Collect();
  profiler::Disable();

  EXPECT_EQ(Find(summary, "mul", "fwd"), nullptr);
}

TEST(Profiler, ReportNamesOpsAndSteps) {
  profiler::Enable();
  auto a = Leaf({2}, {1, 2});
  auto b = a->add(a);
  const std::string report = profiler::Report(1);
  profiler::Disable();

  EXPECT_NE(report.find("profile over 1 steps"), std::string::npos);
  EXPECT_NE(report.find("add"), std::string::npos);
}

TEST(Profiler, ScopeFromAnotherThreadThrowsInDebugBuilds) {
#ifdef NDEBUG
  GTEST_SKIP() << "the owner-thread check is compiled out with NDEBUG";
#else
  profiler::Enable();
  bool threw = false;
  std::thread worker([&threw] {
    try {
      const profiler::Scope scope("worker", "-", Device::CPU);
    } catch (const std::logic_error &) {
      threw = true;
    }
  });
  worker.join();
  profiler::Disable();

  EXPECT_TRUE(threw);
#endif
}

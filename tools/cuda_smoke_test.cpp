#include "cuda_smoke.h"

#include <gtest/gtest.h>

namespace micrograd::tools {
namespace {

TEST(CudaSmoke, LibrariesLink) {
  EXPECT_GE(cudnn_version(), 90000);
  EXPECT_GT(cublaslt_version(), 0);
}

TEST(CudaSmoke, KernelRuns) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "no CUDA device";
  }
  const int n = 1000;
  const std::vector<int> out = cuda_iota(n);
  ASSERT_EQ(out.size(), static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(out[static_cast<size_t>(i)], i);
  }
}

}  // namespace
}  // namespace micrograd::tools

#include "micrograd/backends/cuda/ops/Ops.h"

namespace micrograd::cuda::ops {

void RegisterCudaOps() {
#ifdef MICROGRAD_CUDA_ENABLED
  RegisterArithmeticOps();
  RegisterActivationOps();
  RegisterBroadcastOps();
  RegisterReductionOps();
  RegisterMatmulOps();
  RegisterShapeOps();
  RegisterEmbeddingOps();
  RegisterLayerNormOps();
  RegisterRmsNormOps();
  RegisterRotaryOps();
  RegisterAttentionOps();
#endif
}

}  // namespace micrograd::cuda::ops

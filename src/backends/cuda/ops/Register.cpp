#include "micrograd/backends/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::cuda::ops {

bool ForwardOverwritesOutput(OpId op) {
  switch (op) {
    case OpId::kAdd:
    case OpId::kAddBias:
    case OpId::kSub:
    case OpId::kMul:
    case OpId::kDiv:
    case OpId::kAddScalar:
    case OpId::kSubScalar:
    case OpId::kMulScalar:
    case OpId::kDivScalar:
    case OpId::kClampMin:
    case OpId::kClampMax:
    case OpId::kPow:
    case OpId::kSum:
    case OpId::kMatmul:
    case OpId::kRelu:
    case OpId::kSigmoid:
    case OpId::kTanh:
    case OpId::kGelu:
    case OpId::kExp:
    case OpId::kLog:
    case OpId::kSqrt:
    case OpId::kNeg:
    case OpId::kSoftmax:
    case OpId::kLogSoftmax:
    case OpId::kSumDim:
    case OpId::kMax:
    case OpId::kArgmax:
    case OpId::kStridedCopy:
    case OpId::kBroadcastTo:
    case OpId::kEmbeddingLookup:
    case OpId::kGatherPerRow:
    case OpId::kLayerNorm:
    case OpId::kRmsNorm:
    case OpId::kRotaryEmbedding:
    case OpId::kFlashAttention:
      return true;
    case OpId::kMean:
    case OpId::kReshape:
      return false;
  }
  return false;
}

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

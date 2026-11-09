#include "micrograd/ops/Dispatch.h"

#include <mutex>
#include <utility>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Profiler.h"
#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/backends/cuda/ops/Ops.h"

namespace micrograd {
namespace {

void RegisterBuiltinKernels() {
  ops::cpu::RegisterArithmeticOps();
  ops::cpu::RegisterMatmulOps();
  ops::cpu::RegisterReductionOps();
  ops::cpu::RegisterActivationOps();
  ops::cpu::RegisterShapeOps();
  ops::cpu::RegisterBroadcastOps();
  ops::cpu::RegisterEmbeddingOps();
  ops::cpu::RegisterLayerNormOps();
  ops::cpu::RegisterRmsNormOps();
  ops::cpu::RegisterRotaryOps();
  ops::cpu::RegisterAttentionOps();
  cuda::ops::RegisterCudaOps();
}

void EnsureRegistered() {
  static std::once_flag registered;
  std::call_once(registered, RegisterBuiltinKernels);
}

}  // namespace

const char *OpName(OpId op) {
  switch (op) {
    case OpId::kAdd:
      return "add";
    case OpId::kAddBias:
      return "add_bias";
    case OpId::kSub:
      return "sub";
    case OpId::kMul:
      return "mul";
    case OpId::kDiv:
      return "div";
    case OpId::kAddScalar:
      return "add_scalar";
    case OpId::kSubScalar:
      return "sub_scalar";
    case OpId::kMulScalar:
      return "mul_scalar";
    case OpId::kDivScalar:
      return "div_scalar";
    case OpId::kClampMin:
      return "clamp_min";
    case OpId::kClampMax:
      return "clamp_max";
    case OpId::kPow:
      return "pow";
    case OpId::kSum:
      return "sum";
    case OpId::kMatmul:
      return "matmul";
    case OpId::kRelu:
      return "relu";
    case OpId::kSigmoid:
      return "sigmoid";
    case OpId::kTanh:
      return "tanh";
    case OpId::kGelu:
      return "gelu";
    case OpId::kExp:
      return "exp";
    case OpId::kLog:
      return "log";
    case OpId::kSqrt:
      return "sqrt";
    case OpId::kNeg:
      return "neg";
    case OpId::kSoftmax:
      return "softmax";
    case OpId::kLogSoftmax:
      return "log_softmax";
    case OpId::kSumDim:
      return "sum_dim";
    case OpId::kMean:
      return "mean";
    case OpId::kMax:
      return "max";
    case OpId::kArgmax:
      return "argmax";
    case OpId::kReshape:
      return "reshape";
    case OpId::kStridedCopy:
      return "strided_copy";
    case OpId::kBroadcastTo:
      return "broadcast_to";
    case OpId::kEmbeddingLookup:
      return "embedding_lookup";
    case OpId::kGatherPerRow:
      return "gather_per_row";
    case OpId::kLayerNorm:
      return "layer_norm";
    case OpId::kRmsNorm:
      return "rms_norm";
    case OpId::kRotaryEmbedding:
      return "rotary_embedding";
    case OpId::kFlashAttention:
      return "flash_attention";
  }
  return "unknown";
}

void DispatchOp(OpId op, Device device, const OpArgs &args) {
  EnsureRegistered();
  const Tensor *typed = args.out != nullptr ? args.out : args.lhs;
  DType dtype = typed != nullptr ? typed->dtype() : DType::kFloat32;
  OpFn fn = OpRegistry::Instance().Lookup(op, device, dtype);
  profiler::Scope scope(OpName(op), "fwd", device);
  fn(args);
}

std::function<void()> MakeBackward(OpId op, Device device,
                                   const GradArgs &args) {
  EnsureRegistered();
  const Tensor *typed = args.out != nullptr ? args.out : args.lhs.get();
  DType dtype = typed != nullptr ? typed->dtype() : DType::kFloat32;
  std::function<void()> backward =
      OpRegistry::Instance().LookupBackward(op, device, dtype)(args);
  if (!profiler::Enabled()) {
    return backward;
  }
  return [backward = std::move(backward), op, device]() {
    profiler::Scope scope(OpName(op), "bwd", device);
    backward();
  };
}

std::shared_ptr<Tensor> Tensor::apply(OpId op, std::vector<size_t> out_shape,
                                      const OpInputs &inputs) {
  const bool overwrites =
      backend() == Device::CUDA && cuda::ops::ForwardOverwritesOutput(op);
  auto result = overwrites ? Tensor::empty(std::move(out_shape), backend())
                           : Tensor::zeros(std::move(out_shape), backend());
  DispatchOp(op, backend(),
             {.lhs = inputs.lhs.get(),
              .rhs = inputs.rhs.get(),
              .extra = inputs.extra.get(),
              .out = result.get(),
              .scalar = inputs.scalar,
              .dim = inputs.dim,
              .keepdim = inputs.keepdim,
              .strides = inputs.strides});

  const auto needs_grad = [](const std::shared_ptr<Tensor> &input) {
    return input && input->requires_grad_;
  };
  const bool others_need_grad =
      !inputs.grad_through_lhs_only &&
      (needs_grad(inputs.rhs) || needs_grad(inputs.extra));
  result->requires_grad_ =
      GradEnabled() && (needs_grad(inputs.lhs) || others_need_grad);
  if (!result->requires_grad_) {
    return result;
  }

  result->children_.push_back(inputs.lhs);
  if (!inputs.grad_through_lhs_only) {
    for (const auto *input : {&inputs.rhs, &inputs.extra}) {
      if (*input) {
        result->children_.push_back(*input);
      }
    }
  }
  result->backward_fn_ = MakeBackward(op, backend(),
                                      {.lhs = inputs.lhs,
                                       .rhs = inputs.rhs,
                                       .extra = inputs.extra,
                                       .out = result.get(),
                                       .scalar = inputs.scalar,
                                       .dim = inputs.dim,
                                       .strides = inputs.strides});
  return result;
}

}  // namespace micrograd

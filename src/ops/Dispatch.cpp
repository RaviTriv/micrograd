#include "micrograd/ops/Dispatch.h"

#include <mutex>
#include <utility>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/backends/cuda/ops/Ops.h"
#include "micrograd/backends/metal/ops/Ops.h"

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
  metal::ops::RegisterMetalOps();
  cuda::ops::RegisterCudaOps();
}

void EnsureRegistered() {
  static std::once_flag registered;
  std::call_once(registered, RegisterBuiltinKernels);
}

}  // namespace

void DispatchOp(OpId op, Device device, const OpArgs &args) {
  EnsureRegistered();
  const Tensor *typed = args.out != nullptr ? args.out : args.lhs;
  DType dtype = typed != nullptr ? typed->dtype() : DType::kFloat32;
  OpRegistry::Instance().Lookup(op, device, dtype)(args);
}

std::function<void()> MakeBackward(OpId op, Device device,
                                   const GradArgs &args) {
  EnsureRegistered();
  const Tensor *typed = args.out != nullptr ? args.out : args.lhs.get();
  DType dtype = typed != nullptr ? typed->dtype() : DType::kFloat32;
  return OpRegistry::Instance().LookupBackward(op, device, dtype)(args);
}

std::shared_ptr<Tensor> Tensor::apply(OpId op, std::vector<size_t> out_shape,
                                      const OpInputs &inputs) {
  auto result = Tensor::zeros(std::move(out_shape));
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
  result->requires_grad_ =
      GradEnabled() && (needs_grad(inputs.lhs) || needs_grad(inputs.rhs) ||
                        needs_grad(inputs.extra));
  if (!result->requires_grad_) {
    return result;
  }

  for (const auto *input : {&inputs.lhs, &inputs.rhs, &inputs.extra}) {
    if (*input) {
      result->children_.push_back(*input);
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

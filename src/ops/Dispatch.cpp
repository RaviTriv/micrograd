#include "micrograd/ops/Dispatch.h"

#include <mutex>

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

}  // namespace micrograd

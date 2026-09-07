#include "micrograd/ops/Dispatch.h"

#include <mutex>

#include "micrograd/metal/ops/Ops.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd {
namespace {

void RegisterBuiltinKernels() {
  ops::cpu::RegisterArithmeticOps();
  ops::cpu::RegisterMatmulOps();
  ops::cpu::RegisterReductionOps();
  ops::cpu::RegisterActivationOps();
  metal::ops::RegisterMetalOps();
}

void EnsureRegistered() {
  static std::once_flag registered;
  std::call_once(registered, RegisterBuiltinKernels);
}

}  // namespace

void DispatchOp(OpId op, Device device, const OpArgs &args) {
  EnsureRegistered();
  OpRegistry::Instance().Lookup(op, device)(args);
}

std::function<void()> MakeBackward(OpId op, Device device,
                                   const GradArgs &args) {
  EnsureRegistered();
  return OpRegistry::Instance().LookupBackward(op, device)(args);
}

}  // namespace micrograd

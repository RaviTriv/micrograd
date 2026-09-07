#include "micrograd/ops/Dispatch.h"

#include <mutex>

#include "micrograd/ops/cpu/Ops.h"

namespace micrograd {
namespace {

void RegisterBuiltinKernels() {
  ops::cpu::RegisterArithmeticOps();
  ops::cpu::RegisterMatmulOps();
  ops::cpu::RegisterReductionOps();
  ops::cpu::RegisterActivationOps();
}

}  // namespace

void DispatchOp(OpId op, Device device, const OpArgs &args) {
  static std::once_flag registered;
  std::call_once(registered, RegisterBuiltinKernels);
  OpRegistry::Instance().Lookup(op, device)(args);
}

}  // namespace micrograd

#include "micrograd/metal/ops/Ops.h"

namespace micrograd::metal::ops {

void RegisterMetalOps() {
#ifdef MICROGRAD_METAL_ENABLED
  RegisterArithmeticOps();
  RegisterMatmulOps();
  RegisterReductionOps();
  RegisterActivationOps();
#endif
}

}  // namespace micrograd::metal::ops

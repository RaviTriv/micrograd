#pragma once

#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace micrograd::cuda::ops {

struct GradTarget {
  scalar_t *ptr = nullptr;
  bool accumulate = false;
};

inline GradTarget GradForWrite(Tensor *t) {
  const bool accumulate = t->has_grad();
  return {static_cast<scalar_t *>(
              t->grad_storage_for_overwrite().device_pointer()),
          accumulate};
}

inline bool MoveGrad(Tensor *from, Tensor *to) {
  if (from == to || to->has_grad() || !from->has_grad() ||
      from->size() != to->size()) {
    return false;
  }
  to->set_grad_storage(from->take_grad_storage());
  return true;
}

}  // namespace micrograd::cuda::ops

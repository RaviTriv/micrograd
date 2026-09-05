#ifdef MICROGRAD_METAL_ENABLED

#include <algorithm>
#include <cmath>

#include "micrograd/Tensor.h"
#include "micrograd/metal/Dispatch.h"
#include "micrograd/metal/MetalContext.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::add_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "add", size())
      .buffer(gpu_data_)
      .buffer(b->gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    self_ptr->to(Backend::CPU);
    b->to(Backend::CPU);
    result->to(Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
      b->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sub_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "sub", size())
      .buffer(gpu_data_)
      .buffer(b->gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    self_ptr->to(Backend::CPU);
    b->to(Backend::CPU);
    result->to(Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
      b->grad_[i] -= result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "mul", size())
      .buffer(gpu_data_)
      .buffer(b->gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad_.data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradABuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer gradBBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "mul_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->gpu_data_)
        .buffer(b->gpu_data_)
        .buffer(gradABuf)
        .buffer(gradBBuf)
        .buffer(bufSize)
        .launch();

    auto *gradAPtr = static_cast<scalar_t *>(gradABuf.get()->contents());
    auto *gradBPtr = static_cast<scalar_t *>(gradBBuf.get()->contents());
    auto *gpuGradAPtr =
        static_cast<scalar_t *>(self_ptr->gpu_grad_->contents());
    auto *gpuGradBPtr = static_cast<scalar_t *>(b->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += gradAPtr[i];
      gpuGradAPtr[i] += gradAPtr[i];
      b->grad_[i] += gradBPtr[i];
      gpuGradBPtr[i] += gradBPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "div_op", size())
      .buffer(gpu_data_)
      .buffer(b->gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad_.data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradABuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer gradBBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "div_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->gpu_data_)
        .buffer(b->gpu_data_)
        .buffer(gradABuf)
        .buffer(gradBBuf)
        .buffer(bufSize)
        .launch();

    auto *gradAPtr = static_cast<scalar_t *>(gradABuf.get()->contents());
    auto *gradBPtr = static_cast<scalar_t *>(gradBBuf.get()->contents());
    auto *gpuGradAPtr =
        static_cast<scalar_t *>(self_ptr->gpu_grad_->contents());
    auto *gpuGradBPtr = static_cast<scalar_t *>(b->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += gradAPtr[i];
      gpuGradAPtr[i] += gradAPtr[i];
      b->grad_[i] += gradBPtr[i];
      gpuGradBPtr[i] += gradBPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::add_scalar_metal(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufScalar(ctx, sizeof(scalar_t));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufScalar.set(scalar);
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "add_scalar", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    self_ptr->to(Backend::CPU);
    result->to(Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sub_scalar_metal(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufScalar(ctx, sizeof(scalar_t));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufScalar.set(scalar);
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "sub_scalar", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    self_ptr->to(Backend::CPU);
    result->to(Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul_scalar_metal(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufScalar(ctx, sizeof(scalar_t));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufScalar.set(scalar);
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "mul_scalar", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, scalar]() {
    self_ptr->to(Backend::CPU);
    result->to(Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div_scalar_metal(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufScalar(ctx, sizeof(scalar_t));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufScalar.set(scalar);
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "div_scalar", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, scalar]() {
    self_ptr->to(Backend::CPU);
    result->to(Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] / scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::pow_metal(scalar_t exponent) {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufExp(ctx, sizeof(scalar_t));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufExp.set(exponent);
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "pow_op", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufExp)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, exponent]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad_.data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufExp(ctx, sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufExp.set(exponent);
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "pow_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->gpu_data_)
        .buffer(gradXBuf)
        .buffer(bufExp)
        .buffer(bufSize)
        .launch();

    auto *gradXPtr = static_cast<scalar_t *>(gradXBuf.get()->contents());
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += gradXPtr[i];
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::relu_metal() {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "relu", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad_.data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "relu_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->gpu_data_)
        .buffer(gradXBuf)
        .buffer(bufSize)
        .launch();

    auto *gradXPtr = static_cast<scalar_t *>(gradXBuf.get()->contents());
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += gradXPtr[i];
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid_metal() {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "sigmoid", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad_.data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer outBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->data_.data(), n,
                static_cast<scalar_t *>(outBuf.get()->contents()));

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "sigmoid_backward", n)
        .buffer(gradOutBuf)
        .buffer(outBuf)
        .buffer(gradXBuf)
        .buffer(bufSize)
        .launch();

    auto *gradXPtr = static_cast<scalar_t *>(gradXBuf.get()->contents());
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += gradXPtr[i];
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::tanh_metal() {
  auto result = std::make_shared<Tensor>(shape_);
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "tanh_op", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad_.data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer outBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->data_.data(), n,
                static_cast<scalar_t *>(outBuf.get()->contents()));

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "tanh_backward", n)
        .buffer(gradOutBuf)
        .buffer(outBuf)
        .buffer(gradXBuf)
        .buffer(bufSize)
        .launch();

    auto *gradXPtr = static_cast<scalar_t *>(gradXBuf.get()->contents());
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += gradXPtr[i];
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::matmul_metal(const std::shared_ptr<Tensor> &b) {
  size_t m = shape_[0];
  size_t k = shape_[1];
  size_t n = b->shape_[1];

  auto result = std::make_shared<Tensor>(std::vector<size_t>{m, n});
  result->to(Backend::Metal);

  auto &ctx = MetalContext::instance();

  MatmulKernelLauncher(ctx, "matmul", m, k, n)
      .A(gpu_data_)
      .B(b->gpu_data_)
      .C(result->gpu_data_)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b, m, k, n]() {
    auto &ctx = MetalContext::instance();

    result->to(Backend::CPU);

    ScopedBuffer gradCBuf(ctx, m * n * sizeof(scalar_t));
    std::copy_n(result->grad_.data(), m * n,
                static_cast<scalar_t *>(gradCBuf.get()->contents()));

    ScopedBuffer gradABuf(ctx, m * k * sizeof(scalar_t));
    MatmulKernelLauncher(ctx, "matmul_nt", m, n, k)
        .A(gradCBuf.get())
        .B(b->gpu_data_)
        .C(gradABuf.get())
        .launch();

    ScopedBuffer gradBBuf(ctx, k * n * sizeof(scalar_t));
    MatmulKernelLauncher(ctx, "matmul_tn", m, k, n, k)
        .A(self_ptr->gpu_data_)
        .B(gradCBuf.get())
        .C(gradBBuf.get())
        .launch();

    auto *gradAPtr = static_cast<scalar_t *>(gradABuf.get()->contents());
    auto *gradBPtr = static_cast<scalar_t *>(gradBBuf.get()->contents());
    auto *gpuGradAPtr =
        static_cast<scalar_t *>(self_ptr->gpu_grad_->contents());
    auto *gpuGradBPtr = static_cast<scalar_t *>(b->gpu_grad_->contents());

    for (size_t i = 0; i < m * k; i++) {
      self_ptr->grad_[i] += gradAPtr[i];
      gpuGradAPtr[i] += gradAPtr[i];
    }
    for (size_t i = 0; i < k * n; i++) {
      b->grad_[i] += gradBPtr[i];
      gpuGradBPtr[i] += gradBPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sum_metal() {
  auto result = std::make_shared<Tensor>(std::vector<size_t>{1});

  auto &ctx = MetalContext::instance();
  auto pipeline = ctx.getPipeline("sum_reduce");

  const uint32_t threadgroupSize = 256;
  auto currentSize = static_cast<uint32_t>(size());
  uint32_t numThreadgroups =
      (currentSize + threadgroupSize - 1) / threadgroupSize;

  MTL::Buffer *inputBuf = gpu_data_;
  MTL::Buffer *outputBuf = ctx.createBuffer(numThreadgroups * sizeof(scalar_t));

  while (currentSize > 1) {
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(currentSize);

    auto cmdBuf = ctx.commandQueue()->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuf, 0, 0);
    encoder->setBuffer(outputBuf, 0, 1);
    encoder->setBuffer(bufSize, 0, 2);

    MTL::Size numGroups(numThreadgroups, 1, 1);
    MTL::Size tgSize(threadgroupSize, 1, 1);
    encoder->dispatchThreadgroups(numGroups, tgSize);

    encoder->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    currentSize = numThreadgroups;
    numThreadgroups = (currentSize + threadgroupSize - 1) / threadgroupSize;

    if (currentSize > 1) {
      if (inputBuf != gpu_data_) {
        ctx.releaseBuffer(inputBuf);
      }
      inputBuf = outputBuf;
      outputBuf = ctx.createBuffer(numThreadgroups * sizeof(scalar_t));
    }
  }

  result->data()[0] = *static_cast<scalar_t *>(outputBuf->contents());

  if (inputBuf != gpu_data_) {
    ctx.releaseBuffer(inputBuf);
  }
  ctx.releaseBuffer(outputBuf);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    scalar_t gradScalar = result->grad_[0];

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufScalar(ctx, sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufScalar.set(gradScalar);
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "broadcast_scalar", n)
        .buffer(gradXBuf)
        .buffer(bufScalar)
        .buffer(bufSize)
        .launch();

    auto *gradXPtr = static_cast<scalar_t *>(gradXBuf.get()->contents());
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += gradXPtr[i];
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

}  // namespace micrograd

#endif

#ifdef MICROGRAD_METAL_ENABLED

#include "micrograd/Tensor.h"
#include "micrograd/Utils.h"
#include "micrograd/metal/Dispatch.h"
#include "micrograd/metal/MetalContext.h"
#include <cmath>
#include <sstream>

namespace {} // namespace

std::shared_ptr<Tensor> Tensor::add_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "+";
  result->to(micrograd::Backend::Metal);

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

  result->backward_fn_ = [result, self_ptr, b]() {
    self_ptr->to(micrograd::Backend::CPU);
    b->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
      b->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sub_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "-";
  result->to(micrograd::Backend::Metal);

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

  result->backward_fn_ = [result, self_ptr, b]() {
    self_ptr->to(micrograd::Backend::CPU);
    b->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
      b->grad_[i] -= result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "*";
  result->to(micrograd::Backend::Metal);

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

  result->backward_fn_ = [result, self_ptr, b]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf.get()->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    ScopedBuffer gradABuf(ctx, n * sizeof(float));
    ScopedBuffer gradBBuf(ctx, n * sizeof(float));
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

    float *gradAPtr = static_cast<float *>(gradABuf.get()->contents());
    float *gradBPtr = static_cast<float *>(gradBBuf.get()->contents());
    float *gpuGradAPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    float *gpuGradBPtr = static_cast<float *>(b->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradAPtr[i]);
      gpuGradAPtr[i] += gradAPtr[i];
      b->grad_[i] += static_cast<double>(gradBPtr[i]);
      gpuGradBPtr[i] += gradBPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "/";
  result->to(micrograd::Backend::Metal);

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

  result->backward_fn_ = [result, self_ptr, b]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf.get()->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    ScopedBuffer gradABuf(ctx, n * sizeof(float));
    ScopedBuffer gradBBuf(ctx, n * sizeof(float));
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

    float *gradAPtr = static_cast<float *>(gradABuf.get()->contents());
    float *gradBPtr = static_cast<float *>(gradBBuf.get()->contents());
    float *gpuGradAPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    float *gpuGradBPtr = static_cast<float *>(b->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradAPtr[i]);
      gpuGradAPtr[i] += gradAPtr[i];
      b->grad_[i] += static_cast<double>(gradBPtr[i]);
      gpuGradBPtr[i] += gradBPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::add_scalar_metal(double scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("+", scalar);
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufScalar(ctx, sizeof(float));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufScalar.set(static_cast<float>(scalar));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "add_scalar", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    self_ptr->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sub_scalar_metal(double scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("-", scalar);
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufScalar(ctx, sizeof(float));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufScalar.set(static_cast<float>(scalar));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "sub_scalar", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    self_ptr->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul_scalar_metal(double scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("*", scalar);
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufScalar(ctx, sizeof(float));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufScalar.set(static_cast<float>(scalar));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "mul_scalar", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr, scalar]() {
    self_ptr->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div_scalar_metal(double scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("/", scalar);
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufScalar(ctx, sizeof(float));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufScalar.set(static_cast<float>(scalar));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "div_scalar", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr, scalar]() {
    self_ptr->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] / scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::pow_metal(double exponent) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("^", exponent);
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufExp(ctx, sizeof(float));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufExp.set(static_cast<float>(exponent));
  bufSize.set(static_cast<uint32_t>(size()));

  ElementwiseKernelLauncher(ctx, "pow_op", size())
      .buffer(gpu_data_)
      .buffer(result->gpu_data_)
      .buffer(bufExp)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr, exponent]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf.get()->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    ScopedBuffer gradXBuf(ctx, n * sizeof(float));
    ScopedBuffer bufExp(ctx, sizeof(float));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufExp.set(static_cast<float>(exponent));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "pow_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->gpu_data_)
        .buffer(gradXBuf)
        .buffer(bufExp)
        .buffer(bufSize)
        .launch();

    float *gradXPtr = static_cast<float *>(gradXBuf.get()->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::relu_metal() {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "relu";
  result->to(micrograd::Backend::Metal);

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

  result->backward_fn_ = [result, self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf.get()->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    ScopedBuffer gradXBuf(ctx, n * sizeof(float));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "relu_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->gpu_data_)
        .buffer(gradXBuf)
        .buffer(bufSize)
        .launch();

    float *gradXPtr = static_cast<float *>(gradXBuf.get()->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid_metal() {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "sigmoid";
  result->to(micrograd::Backend::Metal);

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

  result->backward_fn_ = [result, self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf.get()->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    ScopedBuffer gradXBuf(ctx, n * sizeof(float));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "sigmoid_backward", n)
        .buffer(gradOutBuf)
        .buffer(result->gpu_data_)
        .buffer(gradXBuf)
        .buffer(bufSize)
        .launch();

    float *gradXPtr = static_cast<float *>(gradXBuf.get()->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::tanh_metal() {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "tanh";
  result->to(micrograd::Backend::Metal);

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

  result->backward_fn_ = [result, self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf.get()->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    ScopedBuffer gradXBuf(ctx, n * sizeof(float));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "tanh_backward", n)
        .buffer(gradOutBuf)
        .buffer(result->gpu_data_)
        .buffer(gradXBuf)
        .buffer(bufSize)
        .launch();

    float *gradXPtr = static_cast<float *>(gradXBuf.get()->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
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
  result->op_ = "@";
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();

  MatmulKernelLauncher(ctx, "matmul", m, k, n)
      .A(gpu_data_)
      .B(b->gpu_data_)
      .C(result->gpu_data_)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b, m, k, n]() {
    auto &ctx = MetalContext::instance();

    result->to(micrograd::Backend::CPU);

    ScopedBuffer gradCBuf(ctx, m * n * sizeof(float));
    float *gradCPtr = static_cast<float *>(gradCBuf.get()->contents());
    for (size_t i = 0; i < m * n; i++) {
      gradCPtr[i] = static_cast<float>(result->grad_[i]);
    }

    ScopedBuffer gradABuf(ctx, m * k * sizeof(float));
    MatmulKernelLauncher(ctx, "matmul_nt", m, n, k)
        .A(gradCBuf.get())
        .B(b->gpu_data_)
        .C(gradABuf.get())
        .launch();

    ScopedBuffer gradBBuf(ctx, k * n * sizeof(float));
    MatmulKernelLauncher(ctx, "matmul_tn", k, m, n)
        .A(self_ptr->gpu_data_)
        .B(gradCBuf.get())
        .C(gradBBuf.get())
        .launch();

    float *gradAPtr = static_cast<float *>(gradABuf.get()->contents());
    float *gradBPtr = static_cast<float *>(gradBBuf.get()->contents());
    float *gpuGradAPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    float *gpuGradBPtr = static_cast<float *>(b->gpu_grad_->contents());

    for (size_t i = 0; i < m * k; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradAPtr[i]);
      gpuGradAPtr[i] += gradAPtr[i];
    }
    for (size_t i = 0; i < k * n; i++) {
      b->grad_[i] += static_cast<double>(gradBPtr[i]);
      gpuGradBPtr[i] += gradBPtr[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sum_metal() {
  auto result = std::make_shared<Tensor>(std::vector<size_t>{1});
  result->op_ = "sum";

  auto &ctx = MetalContext::instance();
  auto pipeline = ctx.getPipeline("sum_reduce");

  const uint32_t threadgroupSize = 256;
  uint32_t currentSize = static_cast<uint32_t>(size());
  uint32_t numThreadgroups =
      (currentSize + threadgroupSize - 1) / threadgroupSize;

  MTL::Buffer *inputBuf = gpu_data_;
  MTL::Buffer *outputBuf = ctx.createBuffer(numThreadgroups * sizeof(float));

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
      outputBuf = ctx.createBuffer(numThreadgroups * sizeof(float));
    }
  }

  float finalSum = *static_cast<float *>(outputBuf->contents());
  result->data()[0] = static_cast<double>(finalSum);

  if (inputBuf != gpu_data_) {
    ctx.releaseBuffer(inputBuf);
  }
  ctx.releaseBuffer(outputBuf);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    float gradScalar = static_cast<float>(result->grad_[0]);

    ScopedBuffer gradXBuf(ctx, n * sizeof(float));
    ScopedBuffer bufScalar(ctx, sizeof(float));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufScalar.set(gradScalar);
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "broadcast_scalar", n)
        .buffer(gradXBuf)
        .buffer(bufScalar)
        .buffer(bufSize)
        .launch();

    float *gradXPtr = static_cast<float *>(gradXBuf.get()->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

#endif

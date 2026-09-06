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
      .buffer(data_.buffer())
      .buffer(b->data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    self_ptr->to(Backend::CPU);
    b->to(Backend::CPU);
    result->to(Backend::CPU);

    auto a_grad = self_ptr->grad();
    auto b_grad = b->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] += out_grad[i];
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
      .buffer(data_.buffer())
      .buffer(b->data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    self_ptr->to(Backend::CPU);
    b->to(Backend::CPU);
    result->to(Backend::CPU);

    auto a_grad = self_ptr->grad();
    auto b_grad = b->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] -= out_grad[i];
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
      .buffer(data_.buffer())
      .buffer(b->data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradABuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer gradBBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "mul_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->data_.buffer())
        .buffer(b->data_.buffer())
        .buffer(gradABuf)
        .buffer(gradBBuf)
        .buffer(bufSize)
        .launch();

    auto *gradAPtr = static_cast<scalar_t *>(gradABuf.get()->contents());
    auto *gradBPtr = static_cast<scalar_t *>(gradBBuf.get()->contents());
    auto *gpuGradAPtr = static_cast<scalar_t *>(self_ptr->grad_.host_pointer());
    auto *gpuGradBPtr = static_cast<scalar_t *>(b->grad_.host_pointer());
    for (size_t i = 0; i < n; i++) {
      gpuGradAPtr[i] += gradAPtr[i];
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
      .buffer(data_.buffer())
      .buffer(b->data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradABuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer gradBBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "div_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->data_.buffer())
        .buffer(b->data_.buffer())
        .buffer(gradABuf)
        .buffer(gradBBuf)
        .buffer(bufSize)
        .launch();

    auto *gradAPtr = static_cast<scalar_t *>(gradABuf.get()->contents());
    auto *gradBPtr = static_cast<scalar_t *>(gradBBuf.get()->contents());
    auto *gpuGradAPtr = static_cast<scalar_t *>(self_ptr->grad_.host_pointer());
    auto *gpuGradBPtr = static_cast<scalar_t *>(b->grad_.host_pointer());
    for (size_t i = 0; i < n; i++) {
      gpuGradAPtr[i] += gradAPtr[i];
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
      .buffer(data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    self_ptr->to(Backend::CPU);
    result->to(Backend::CPU);

    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
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
      .buffer(data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    self_ptr->to(Backend::CPU);
    result->to(Backend::CPU);

    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
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
      .buffer(data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, scalar]() {
    self_ptr->to(Backend::CPU);
    result->to(Backend::CPU);

    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * scalar;
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
      .buffer(data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, scalar]() {
    self_ptr->to(Backend::CPU);
    result->to(Backend::CPU);

    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] / scalar;
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
      .buffer(data_.buffer())
      .buffer(result->data_.buffer())
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
    std::copy_n(result->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufExp(ctx, sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufExp.set(exponent);
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "pow_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->data_.buffer())
        .buffer(gradXBuf)
        .buffer(bufExp)
        .buffer(bufSize)
        .launch();

    auto *gradXPtr = static_cast<scalar_t *>(gradXBuf.get()->contents());
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->grad_.host_pointer());
    for (size_t i = 0; i < n; i++) {
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
      .buffer(data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "relu_backward", n)
        .buffer(gradOutBuf)
        .buffer(self_ptr->data_.buffer())
        .buffer(gradXBuf)
        .buffer(bufSize)
        .launch();

    auto *gradXPtr = static_cast<scalar_t *>(gradXBuf.get()->contents());
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->grad_.host_pointer());
    for (size_t i = 0; i < n; i++) {
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
      .buffer(data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer outBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->data().data(), n,
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
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->grad_.host_pointer());
    for (size_t i = 0; i < n; i++) {
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
      .buffer(data_.buffer())
      .buffer(result->data_.buffer())
      .buffer(bufSize)
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer outBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(result->data().data(), n,
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
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->grad_.host_pointer());
    for (size_t i = 0; i < n; i++) {
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
      .A(data_.buffer())
      .B(b->data_.buffer())
      .C(result->data_.buffer())
      .launch();

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b, m, k, n]() {
    auto &ctx = MetalContext::instance();

    result->to(Backend::CPU);

    ScopedBuffer gradCBuf(ctx, m * n * sizeof(scalar_t));
    std::copy_n(result->grad().data(), m * n,
                static_cast<scalar_t *>(gradCBuf.get()->contents()));

    ScopedBuffer gradABuf(ctx, m * k * sizeof(scalar_t));
    MatmulKernelLauncher(ctx, "matmul_nt", m, n, k)
        .A(gradCBuf.get())
        .B(b->data_.buffer())
        .C(gradABuf.get())
        .launch();

    ScopedBuffer gradBBuf(ctx, k * n * sizeof(scalar_t));
    MatmulKernelLauncher(ctx, "matmul_tn", m, k, n, k)
        .A(self_ptr->data_.buffer())
        .B(gradCBuf.get())
        .C(gradBBuf.get())
        .launch();

    auto *gradAPtr = static_cast<scalar_t *>(gradABuf.get()->contents());
    auto *gradBPtr = static_cast<scalar_t *>(gradBBuf.get()->contents());
    auto *gpuGradAPtr = static_cast<scalar_t *>(self_ptr->grad_.host_pointer());
    auto *gpuGradBPtr = static_cast<scalar_t *>(b->grad_.host_pointer());

    for (size_t i = 0; i < m * k; i++) {
      gpuGradAPtr[i] += gradAPtr[i];
    }
    for (size_t i = 0; i < k * n; i++) {
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

  MTL::Buffer *sourceBuf = data_.buffer();
  MTL::Buffer *inputBuf = sourceBuf;
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
      if (inputBuf != sourceBuf) {
        ctx.releaseBuffer(inputBuf);
      }
      inputBuf = outputBuf;
      outputBuf = ctx.createBuffer(numThreadgroups * sizeof(scalar_t));
    }
  }

  result->data()[0] = *static_cast<scalar_t *>(outputBuf->contents());

  if (inputBuf != sourceBuf) {
    ctx.releaseBuffer(inputBuf);
  }
  ctx.releaseBuffer(outputBuf);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    scalar_t gradScalar = result->grad()[0];

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
    auto *gpuGradPtr = static_cast<scalar_t *>(self_ptr->grad_.host_pointer());
    for (size_t i = 0; i < n; i++) {
      gpuGradPtr[i] += gradXPtr[i];
    }
  };

  return result;
}

}  // namespace micrograd

#endif

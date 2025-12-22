#ifdef MICROGRAD_METAL_ENABLED

#include "micrograd/Tensor.h"
#include "micrograd/metal/MetalContext.h"
#include <cmath>
#include <iomanip>
#include <sstream>

namespace {
std::string format_scalar(const std::string &op, double scalar) {
  std::ostringstream oss;
  oss << op << " " << std::fixed << std::setprecision(2) << scalar;
  return oss.str();
}
} // namespace

std::shared_ptr<Tensor> Tensor::add_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "+";
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("add");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(b->gpu_data_, 0, 1);
  encoder->setBuffer(result->gpu_data_, 0, 2);
  encoder->setBuffer(bufSize, 0, 3);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufSize);

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
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("sub");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(b->gpu_data_, 0, 1);
  encoder->setBuffer(result->gpu_data_, 0, 2);
  encoder->setBuffer(bufSize, 0, 3);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufSize);

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
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("mul");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(b->gpu_data_, 0, 1);
  encoder->setBuffer(result->gpu_data_, 0, 2);
  encoder->setBuffer(bufSize, 0, 3);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufSize);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    auto gradOutBuf = ctx.createBuffer(n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    auto gradABuf = ctx.createBuffer(n * sizeof(float));
    auto gradBBuf = ctx.createBuffer(n * sizeof(float));

    auto bufSize = ctx.createBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(n);

    auto pipeline = ctx.getPipeline("mul_backward");
    auto cmdBuf = ctx.commandQueue()->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(gradOutBuf, 0, 0);
    encoder->setBuffer(self_ptr->gpu_data_, 0, 1);
    encoder->setBuffer(b->gpu_data_, 0, 2);
    encoder->setBuffer(gradABuf, 0, 3);
    encoder->setBuffer(gradBBuf, 0, 4);
    encoder->setBuffer(bufSize, 0, 5);

    MTL::Size gridSize(n, 1, 1);
    MTL::Size threadGroupSize(std::min(n, size_t(256)), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    float *gradAPtr = static_cast<float *>(gradABuf->contents());
    float *gradBPtr = static_cast<float *>(gradBBuf->contents());
    float *gpuGradAPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    float *gpuGradBPtr = static_cast<float *>(b->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradAPtr[i]);
      gpuGradAPtr[i] += gradAPtr[i];
      b->grad_[i] += static_cast<double>(gradBPtr[i]);
      gpuGradBPtr[i] += gradBPtr[i];
    }

    ctx.releaseBuffer(gradOutBuf);
    ctx.releaseBuffer(gradABuf);
    ctx.releaseBuffer(gradBBuf);
    ctx.releaseBuffer(bufSize);
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div_metal(const std::shared_ptr<Tensor> &b) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "/";
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("div_op");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(b->gpu_data_, 0, 1);
  encoder->setBuffer(result->gpu_data_, 0, 2);
  encoder->setBuffer(bufSize, 0, 3);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufSize);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    auto gradOutBuf = ctx.createBuffer(n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    auto gradABuf = ctx.createBuffer(n * sizeof(float));
    auto gradBBuf = ctx.createBuffer(n * sizeof(float));

    auto bufSize = ctx.createBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(n);

    auto pipeline = ctx.getPipeline("div_backward");
    auto cmdBuf = ctx.commandQueue()->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(gradOutBuf, 0, 0);
    encoder->setBuffer(self_ptr->gpu_data_, 0, 1);
    encoder->setBuffer(b->gpu_data_, 0, 2);
    encoder->setBuffer(gradABuf, 0, 3);
    encoder->setBuffer(gradBBuf, 0, 4);
    encoder->setBuffer(bufSize, 0, 5);

    MTL::Size gridSize(n, 1, 1);
    MTL::Size threadGroupSize(std::min(n, size_t(256)), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    float *gradAPtr = static_cast<float *>(gradABuf->contents());
    float *gradBPtr = static_cast<float *>(gradBBuf->contents());
    float *gpuGradAPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    float *gpuGradBPtr = static_cast<float *>(b->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradAPtr[i]);
      gpuGradAPtr[i] += gradAPtr[i];
      b->grad_[i] += static_cast<double>(gradBPtr[i]);
      gpuGradBPtr[i] += gradBPtr[i];
    }

    ctx.releaseBuffer(gradOutBuf);
    ctx.releaseBuffer(gradABuf);
    ctx.releaseBuffer(gradBBuf);
    ctx.releaseBuffer(bufSize);
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::add_scalar_metal(double scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("+", scalar);
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  auto bufScalar = ctx.createBuffer(sizeof(float));
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<float *>(bufScalar->contents()) = static_cast<float>(scalar);
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("add_scalar");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(result->gpu_data_, 0, 1);
  encoder->setBuffer(bufScalar, 0, 2);
  encoder->setBuffer(bufSize, 0, 3);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufScalar);
  ctx.releaseBuffer(bufSize);

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
  auto bufScalar = ctx.createBuffer(sizeof(float));
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<float *>(bufScalar->contents()) = static_cast<float>(scalar);
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("sub_scalar");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(result->gpu_data_, 0, 1);
  encoder->setBuffer(bufScalar, 0, 2);
  encoder->setBuffer(bufSize, 0, 3);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufScalar);
  ctx.releaseBuffer(bufSize);

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
  auto bufScalar = ctx.createBuffer(sizeof(float));
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<float *>(bufScalar->contents()) = static_cast<float>(scalar);
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("mul_scalar");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(result->gpu_data_, 0, 1);
  encoder->setBuffer(bufScalar, 0, 2);
  encoder->setBuffer(bufSize, 0, 3);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufScalar);
  ctx.releaseBuffer(bufSize);

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
  auto bufScalar = ctx.createBuffer(sizeof(float));
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<float *>(bufScalar->contents()) = static_cast<float>(scalar);
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("div_scalar");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(result->gpu_data_, 0, 1);
  encoder->setBuffer(bufScalar, 0, 2);
  encoder->setBuffer(bufSize, 0, 3);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufScalar);
  ctx.releaseBuffer(bufSize);

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
  auto bufExp = ctx.createBuffer(sizeof(float));
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<float *>(bufExp->contents()) = static_cast<float>(exponent);
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("pow_op");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(result->gpu_data_, 0, 1);
  encoder->setBuffer(bufExp, 0, 2);
  encoder->setBuffer(bufSize, 0, 3);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufExp);
  ctx.releaseBuffer(bufSize);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr, exponent]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    auto gradOutBuf = ctx.createBuffer(n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    auto gradXBuf = ctx.createBuffer(n * sizeof(float));

    auto bufExp = ctx.createBuffer(sizeof(float));
    auto bufSize = ctx.createBuffer(sizeof(uint32_t));
    *static_cast<float *>(bufExp->contents()) = static_cast<float>(exponent);
    *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(n);

    auto pipeline = ctx.getPipeline("pow_backward");
    auto cmdBuf = ctx.commandQueue()->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(gradOutBuf, 0, 0);
    encoder->setBuffer(self_ptr->gpu_data_, 0, 1);
    encoder->setBuffer(gradXBuf, 0, 2);
    encoder->setBuffer(bufExp, 0, 3);
    encoder->setBuffer(bufSize, 0, 4);

    MTL::Size gridSize(n, 1, 1);
    MTL::Size threadGroupSize(std::min(n, size_t(256)), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    float *gradXPtr = static_cast<float *>(gradXBuf->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
      gpuGradPtr[i] += gradXPtr[i];
    }

    ctx.releaseBuffer(gradOutBuf);
    ctx.releaseBuffer(gradXBuf);
    ctx.releaseBuffer(bufExp);
    ctx.releaseBuffer(bufSize);
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::relu_metal() {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "relu";
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("relu");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(result->gpu_data_, 0, 1);
  encoder->setBuffer(bufSize, 0, 2);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufSize);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    auto gradOutBuf = ctx.createBuffer(n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    auto gradXBuf = ctx.createBuffer(n * sizeof(float));

    auto bufSize = ctx.createBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(n);

    auto pipeline = ctx.getPipeline("relu_backward");
    auto cmdBuf = ctx.commandQueue()->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(gradOutBuf, 0, 0);
    encoder->setBuffer(self_ptr->gpu_data_, 0, 1);
    encoder->setBuffer(gradXBuf, 0, 2);
    encoder->setBuffer(bufSize, 0, 3);

    MTL::Size gridSize(n, 1, 1);
    MTL::Size threadGroupSize(std::min(n, size_t(256)), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    float *gradXPtr = static_cast<float *>(gradXBuf->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
      gpuGradPtr[i] += gradXPtr[i];
    }

    ctx.releaseBuffer(gradOutBuf);
    ctx.releaseBuffer(gradXBuf);
    ctx.releaseBuffer(bufSize);
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid_metal() {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "sigmoid";
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("sigmoid");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(result->gpu_data_, 0, 1);
  encoder->setBuffer(bufSize, 0, 2);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufSize);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    auto gradOutBuf = ctx.createBuffer(n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    auto gradXBuf = ctx.createBuffer(n * sizeof(float));

    auto bufSize = ctx.createBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(n);

    auto pipeline = ctx.getPipeline("sigmoid_backward");
    auto cmdBuf = ctx.commandQueue()->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(gradOutBuf, 0, 0);
    encoder->setBuffer(result->gpu_data_, 0, 1);
    encoder->setBuffer(gradXBuf, 0, 2);
    encoder->setBuffer(bufSize, 0, 3);

    MTL::Size gridSize(n, 1, 1);
    MTL::Size threadGroupSize(std::min(n, size_t(256)), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    float *gradXPtr = static_cast<float *>(gradXBuf->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
      gpuGradPtr[i] += gradXPtr[i];
    }

    ctx.releaseBuffer(gradOutBuf);
    ctx.releaseBuffer(gradXBuf);
    ctx.releaseBuffer(bufSize);
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::tanh_metal() {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "tanh";
  result->to(micrograd::Backend::Metal);

  auto &ctx = MetalContext::instance();
  auto bufSize = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(size());

  auto pipeline = ctx.getPipeline("tanh_op");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(result->gpu_data_, 0, 1);
  encoder->setBuffer(bufSize, 0, 2);

  MTL::Size gridSize(size(), 1, 1);
  MTL::Size threadGroupSize(std::min(size(), size_t(256)), 1, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufSize);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    auto &ctx = MetalContext::instance();
    size_t n = self_ptr->size();

    result->to(micrograd::Backend::CPU);
    auto gradOutBuf = ctx.createBuffer(n * sizeof(float));
    float *gradOutPtr = static_cast<float *>(gradOutBuf->contents());
    for (size_t i = 0; i < n; i++) {
      gradOutPtr[i] = static_cast<float>(result->grad_[i]);
    }

    auto gradXBuf = ctx.createBuffer(n * sizeof(float));

    auto bufSize = ctx.createBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(n);

    auto pipeline = ctx.getPipeline("tanh_backward");
    auto cmdBuf = ctx.commandQueue()->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(gradOutBuf, 0, 0);
    encoder->setBuffer(result->gpu_data_, 0, 1);
    encoder->setBuffer(gradXBuf, 0, 2);
    encoder->setBuffer(bufSize, 0, 3);

    MTL::Size gridSize(n, 1, 1);
    MTL::Size threadGroupSize(std::min(n, size_t(256)), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    float *gradXPtr = static_cast<float *>(gradXBuf->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
      gpuGradPtr[i] += gradXPtr[i];
    }

    ctx.releaseBuffer(gradOutBuf);
    ctx.releaseBuffer(gradXBuf);
    ctx.releaseBuffer(bufSize);
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

  auto bufM = ctx.createBuffer(sizeof(uint32_t));
  auto bufK = ctx.createBuffer(sizeof(uint32_t));
  auto bufN = ctx.createBuffer(sizeof(uint32_t));
  *static_cast<uint32_t *>(bufM->contents()) = static_cast<uint32_t>(m);
  *static_cast<uint32_t *>(bufK->contents()) = static_cast<uint32_t>(k);
  *static_cast<uint32_t *>(bufN->contents()) = static_cast<uint32_t>(n);

  auto pipeline = ctx.getPipeline("matmul");
  auto cmdBuf = ctx.commandQueue()->commandBuffer();
  auto encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(gpu_data_, 0, 0);
  encoder->setBuffer(b->gpu_data_, 0, 1);
  encoder->setBuffer(result->gpu_data_, 0, 2);
  encoder->setBuffer(bufM, 0, 3);
  encoder->setBuffer(bufK, 0, 4);
  encoder->setBuffer(bufN, 0, 5);

  MTL::Size gridSize(n, m, 1);
  MTL::Size threadGroupSize(std::min(n, size_t(16)), std::min(m, size_t(16)),
                            1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  ctx.releaseBuffer(bufM);
  ctx.releaseBuffer(bufK);
  ctx.releaseBuffer(bufN);

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b, m, k, n]() {
    auto &ctx = MetalContext::instance();

    result->to(micrograd::Backend::CPU);

    auto gradCBuf = ctx.createBuffer(m * n * sizeof(float));
    float *gradCPtr = static_cast<float *>(gradCBuf->contents());
    for (size_t i = 0; i < m * n; i++) {
      gradCPtr[i] = static_cast<float>(result->grad_[i]);
    }

    auto gradABuf = ctx.createBuffer(m * k * sizeof(float));
    {
      auto bufM = ctx.createBuffer(sizeof(uint32_t));
      auto bufK = ctx.createBuffer(sizeof(uint32_t));
      auto bufN = ctx.createBuffer(sizeof(uint32_t));
      *static_cast<uint32_t *>(bufM->contents()) = static_cast<uint32_t>(m);
      *static_cast<uint32_t *>(bufK->contents()) = static_cast<uint32_t>(n);
      *static_cast<uint32_t *>(bufN->contents()) = static_cast<uint32_t>(k);

      auto pipeline = ctx.getPipeline("matmul_nt");
      auto cmdBuf = ctx.commandQueue()->commandBuffer();
      auto encoder = cmdBuf->computeCommandEncoder();

      encoder->setComputePipelineState(pipeline);
      encoder->setBuffer(gradCBuf, 0, 0);  // A = dC
      encoder->setBuffer(b->gpu_data_, 0, 1);  // B = B
      encoder->setBuffer(gradABuf, 0, 2);  // C = dA
      encoder->setBuffer(bufM, 0, 3);
      encoder->setBuffer(bufK, 0, 4);
      encoder->setBuffer(bufN, 0, 5);

      MTL::Size gridSize(k, m, 1);
      MTL::Size threadGroupSize(std::min(k, size_t(16)), std::min(m, size_t(16)), 1);
      encoder->dispatchThreads(gridSize, threadGroupSize);

      encoder->endEncoding();
      cmdBuf->commit();
      cmdBuf->waitUntilCompleted();

      ctx.releaseBuffer(bufM);
      ctx.releaseBuffer(bufK);
      ctx.releaseBuffer(bufN);
    }

    auto gradBBuf = ctx.createBuffer(k * n * sizeof(float));
    {
      auto bufM = ctx.createBuffer(sizeof(uint32_t));
      auto bufK = ctx.createBuffer(sizeof(uint32_t));
      auto bufN = ctx.createBuffer(sizeof(uint32_t));
      *static_cast<uint32_t *>(bufM->contents()) = static_cast<uint32_t>(m);
      *static_cast<uint32_t *>(bufK->contents()) = static_cast<uint32_t>(k);
      *static_cast<uint32_t *>(bufN->contents()) = static_cast<uint32_t>(n);

      auto pipeline = ctx.getPipeline("matmul_tn");
      auto cmdBuf = ctx.commandQueue()->commandBuffer();
      auto encoder = cmdBuf->computeCommandEncoder();

      encoder->setComputePipelineState(pipeline);
      encoder->setBuffer(self_ptr->gpu_data_, 0, 0);  // A = self
      encoder->setBuffer(gradCBuf, 0, 1);  // B = dC
      encoder->setBuffer(gradBBuf, 0, 2);  // C = dB
      encoder->setBuffer(bufM, 0, 3);
      encoder->setBuffer(bufK, 0, 4);
      encoder->setBuffer(bufN, 0, 5);

      MTL::Size gridSize(n, k, 1);
      MTL::Size threadGroupSize(std::min(n, size_t(16)), std::min(k, size_t(16)), 1);
      encoder->dispatchThreads(gridSize, threadGroupSize);

      encoder->endEncoding();
      cmdBuf->commit();
      cmdBuf->waitUntilCompleted();

      ctx.releaseBuffer(bufM);
      ctx.releaseBuffer(bufK);
      ctx.releaseBuffer(bufN);
    }

    float *gradAPtr = static_cast<float *>(gradABuf->contents());
    float *gradBPtr = static_cast<float *>(gradBBuf->contents());
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

    ctx.releaseBuffer(gradCBuf);
    ctx.releaseBuffer(gradABuf);
    ctx.releaseBuffer(gradBBuf);
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
  uint32_t numThreadgroups = (currentSize + threadgroupSize - 1) / threadgroupSize;

  MTL::Buffer *inputBuf = gpu_data_;
  MTL::Buffer *outputBuf = ctx.createBuffer(numThreadgroups * sizeof(float));

  while (currentSize > 1) {
    auto bufSize = ctx.createBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(bufSize->contents()) = currentSize;

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

    ctx.releaseBuffer(bufSize);

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

    auto gradXBuf = ctx.createBuffer(n * sizeof(float));

    auto bufScalar = ctx.createBuffer(sizeof(float));
    auto bufSize = ctx.createBuffer(sizeof(uint32_t));
    *static_cast<float *>(bufScalar->contents()) = gradScalar;
    *static_cast<uint32_t *>(bufSize->contents()) = static_cast<uint32_t>(n);

    auto pipeline = ctx.getPipeline("broadcast_scalar");
    auto cmdBuf = ctx.commandQueue()->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(gradXBuf, 0, 0);
    encoder->setBuffer(bufScalar, 0, 1);
    encoder->setBuffer(bufSize, 0, 2);

    MTL::Size gridSize(n, 1, 1);
    MTL::Size threadGroupSize(std::min(n, size_t(256)), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    float *gradXPtr = static_cast<float *>(gradXBuf->contents());
    float *gpuGradPtr = static_cast<float *>(self_ptr->gpu_grad_->contents());
    for (size_t i = 0; i < n; i++) {
      self_ptr->grad_[i] += static_cast<double>(gradXPtr[i]);
      gpuGradPtr[i] += gradXPtr[i];
    }

    ctx.releaseBuffer(gradXBuf);
    ctx.releaseBuffer(bufScalar);
    ctx.releaseBuffer(bufSize);
  };

  return result;
}

#endif

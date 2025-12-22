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
    self_ptr->to(micrograd::Backend::CPU);
    b->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * b->data_[i];
      b->grad_[i] += result->grad_[i] * self_ptr->data_[i];
    }
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
    self_ptr->to(micrograd::Backend::CPU);
    b->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] / b->data_[i];
      b->grad_[i] -=
          result->grad_[i] * self_ptr->data_[i] / (b->data_[i] * b->data_[i]);
    }
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
    self_ptr->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * exponent *
                            std::pow(self_ptr->data_[i], exponent - 1);
    }
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
    self_ptr->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] +=
          result->grad_[i] * (self_ptr->data_[i] > 0 ? 1.0 : 0.0);
    }
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
    self_ptr->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      double sigmoid_val = result->data_[i];
      self_ptr->grad_[i] +=
          result->grad_[i] * sigmoid_val * (1.0 - sigmoid_val);
    }
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
    self_ptr->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      double tanh_val = result->data_[i];
      self_ptr->grad_[i] += result->grad_[i] * (1.0 - tanh_val * tanh_val);
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
    self_ptr->to(micrograd::Backend::CPU);
    b->to(micrograd::Backend::CPU);
    result->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        for (size_t p = 0; p < n; p++) {
          self_ptr->grad_at({i, j}) += result->grad_at({i, p}) * b->at({j, p});
        }
      }
    }
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        for (size_t p = 0; p < m; p++) {
          b->grad_at({i, j}) += self_ptr->at({p, i}) * result->grad_at({p, j});
        }
      }
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
    self_ptr->to(micrograd::Backend::CPU);

    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[0];
    }
  };

  return result;
}

#endif

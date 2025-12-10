#include <iostream>

#ifdef MICROGRAD_METAL_ENABLED
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

int main() {
#ifdef MICROGRAD_METAL_ENABLED
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  std::cout << "Accelerator: " << device->name()->utf8String() << std::endl;

  NS::Error *error = nullptr;
  NS::String *source =
      NS::String::string("#include <metal_stdlib>\n"
                         "using namespace metal;\n"
                         "\n"
                         "kernel void matmul(\n"
                         "    device const float* A [[buffer(0)]],\n"
                         "    device const float* B [[buffer(1)]],\n"
                         "    device float* C [[buffer(2)]],\n"
                         "    constant uint& M [[buffer(3)]],\n"
                         "    constant uint& K [[buffer(4)]],\n"
                         "    constant uint& N [[buffer(5)]],\n"
                         "    uint2 gid [[thread_position_in_grid]])\n"
                         "{\n"
                         "    // gid.y = row, gid.x = col\n"
                         "    if (gid.y >= M || gid.x >= N) return;\n"
                         "    float sum = 0.0f;\n"
                         "    for (uint k = 0; k < K; k++) {\n"
                         "        sum += A[gid.y * K + k] * B[k * N + gid.x];\n"
                         "    }\n"
                         "    C[gid.y * N + gid.x] = sum;\n"
                         "}\n",
                         NS::UTF8StringEncoding);

  MTL::Library *library = device->newLibrary(source, nullptr, &error);
  MTL::Function *fn = library->newFunction(
      NS::String::string("matmul", NS::UTF8StringEncoding));
  MTL::ComputePipelineState *pipeline =
      device->newComputePipelineState(fn, &error);
  fn->release();

  uint32_t M = 2, K = 2, N = 2;

  // Create buffers
  MTL::Buffer *bufA =
      device->newBuffer(M * K * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer *bufB =
      device->newBuffer(K * N * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer *bufC =
      device->newBuffer(M * N * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer *bufM =
      device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
  MTL::Buffer *bufK =
      device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
  MTL::Buffer *bufN =
      device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);

  float *A = static_cast<float *>(bufA->contents());
  float *B = static_cast<float *>(bufB->contents());
  A[0] = 1;
  A[1] = 0;
  A[2] = 2;
  A[3] = 4;

  B[0] = 6;
  B[1] = 8;
  B[2] = 4;
  B[3] = 3;

  *static_cast<uint32_t *>(bufM->contents()) = M;
  *static_cast<uint32_t *>(bufK->contents()) = K;
  *static_cast<uint32_t *>(bufN->contents()) = N;

  MTL::CommandQueue *queue = device->newCommandQueue();
  MTL::CommandBuffer *cmdBuf = queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = cmdBuf->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(bufA, 0, 0);
  encoder->setBuffer(bufB, 0, 1);
  encoder->setBuffer(bufC, 0, 2);
  encoder->setBuffer(bufM, 0, 3);
  encoder->setBuffer(bufK, 0, 4);
  encoder->setBuffer(bufN, 0, 5);

  MTL::Size gridSize(N, M, 1);
  MTL::Size threadGroupSize(N, M, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);

  encoder->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();

  float *C = static_cast<float *>(bufC->contents());
  std::cout << "  [" << C[0] << ", " << C[1] << "]" << std::endl;
  std::cout << "  [" << C[2] << ", " << C[3] << "]" << std::endl;

 
  bufA->release();
  bufB->release();
  bufC->release();
  bufM->release();
  bufK->release();
  bufN->release();
  pipeline->release();
  library->release();
  queue->release();
  device->release();
#endif
  return 0;
}

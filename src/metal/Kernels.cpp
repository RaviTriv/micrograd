#include "micrograd/metal/Kernels.h"

namespace micrograd::metal {
const char *KERNEL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= M || gid.x >= N) {
        return;
    }
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    }
    C[gid.y * N + gid.x] = sum;
}

kernel void matmul_nt(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= M || gid.x >= N) {
        return;
    }
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[gid.x * K + k];
    }
    C[gid.y * N + gid.x] = sum;
}

kernel void matmul_tn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= K || gid.x >= N) {
        return;
    }
    float sum = 0.0f;
    for (uint m = 0; m < M; m++) {
        sum += A[m * K + gid.y] * B[m * N + gid.x];
    }
    C[gid.y * N + gid.x] = sum;
}

kernel void add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = A[gid] + B[gid];
}

kernel void sub(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = A[gid] - B[gid];
}

kernel void mul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = A[gid] * B[gid];
}

kernel void div_op(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = A[gid] / B[gid];
}

kernel void add_scalar(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = A[gid] + scalar;
}

kernel void sub_scalar(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = A[gid] - scalar;
}

kernel void mul_scalar(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = A[gid] * scalar;
}

kernel void div_scalar(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = A[gid] / scalar;
}

kernel void pow_op(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant float& exponent [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = pow(A[gid], exponent);
}

kernel void relu(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = max(0.0f, A[gid]);
}

kernel void sigmoid(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = 1.0f / (1.0f + exp(-A[gid]));
}

kernel void tanh_op(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    C[gid] = tanh(A[gid]);
}

kernel void sum_reduce(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup float shared[256];

    shared[tid] = (gid < size) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[tgid] = shared[0];
    }
}


kernel void mul_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* a_data [[buffer(1)]],
    device const float* b_data [[buffer(2)]],
    device float* grad_a [[buffer(3)]],
    device float* grad_b [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    grad_a[gid] = grad_out[gid] * b_data[gid];
    grad_b[gid] = grad_out[gid] * a_data[gid];
}

kernel void div_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* a_data [[buffer(1)]],
    device const float* b_data [[buffer(2)]],
    device float* grad_a [[buffer(3)]],
    device float* grad_b [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    grad_a[gid] = grad_out[gid] / b_data[gid];
    grad_b[gid] = -grad_out[gid] * a_data[gid] / (b_data[gid] * b_data[gid]);
}

kernel void pow_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* x_data [[buffer(1)]],
    device float* grad_x [[buffer(2)]],
    constant float& exponent [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    grad_x[gid] = grad_out[gid] * exponent * pow(x_data[gid], exponent - 1.0f);
}

kernel void relu_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* x_data [[buffer(1)]],
    device float* grad_x [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    grad_x[gid] = (x_data[gid] > 0.0f) ? grad_out[gid] : 0.0f;
}

// sigmoid backward: grad = grad_out * sigmoid(x) * (1 - sigmoid(x))
// Since we already computed sigmoid(x) in forward pass, we use the output
kernel void sigmoid_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* sigmoid_out [[buffer(1)]],
    device float* grad_x [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    float s = sigmoid_out[gid];
    grad_x[gid] = grad_out[gid] * s * (1.0f - s);
}

kernel void tanh_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* tanh_out [[buffer(1)]],
    device float* grad_x [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    float t = tanh_out[gid];
    grad_x[gid] = grad_out[gid] * (1.0f - t * t);
}

kernel void broadcast_scalar(
    device float* output [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) {
        return;
    }
    output[gid] = scalar;
}
)";
}
#include "kernels.cuh"

#include <cstdio>

namespace week1 {

__global__ void vector_add(
    const float* left,
    const float* right,
    float* output,
    int size
) {
    // TODO: Use a grid-stride loop.
    // TODO: Guard every read and write with index < size.
}

__global__ void saxpy(
    float alpha,
    const float* x,
    float* y,
    int size
) {
    // TODO: y[index] = alpha * x[index] + y[index].
    // TODO: Use a grid-stride loop.
}

__global__ void gelu_elementwise(
    const float* input,
    float* output,
    int size
) {
    // TODO: Implement an approximate or exact GELU.
    // TODO: Document which GELU formula you use.
    // TODO: Compare it with the PyTorch reference.
}

void launch_vector_add(
    const float* left,
    const float* right,
    float* output,
    int size,
    cudaStream_t stream
) {
    constexpr int threads_per_block = 256;

    // TODO: Calculate the grid size safely for large arrays.
    dim3 block(threads_per_block);
    dim3 grid(1);
    vector_add<<<grid, block, 0, stream>>>(left, right, output, size);
}

void launch_saxpy(
    float alpha,
    const float* x,
    float* y,
    int size,
    cudaStream_t stream
) {
    constexpr int threads_per_block = 256;

    // TODO: Calculate the grid size safely for large arrays.
    dim3 block(threads_per_block);
    dim3 grid(1);
    saxpy<<<grid, block, 0, stream>>>(alpha, x, y, size);
}

void launch_gelu_elementwise(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    constexpr int threads_per_block = 256;

    // TODO: Calculate the grid size safely for large arrays.
    dim3 block(threads_per_block);
    dim3 grid(1);
    gelu_elementwise<<<grid, block, 0, stream>>>(input, output, size);
}

}  // namespace week1

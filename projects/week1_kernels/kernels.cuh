#pragma once

#include <cuda_runtime.h>

namespace week1 {

__global__ void vector_add(
    const float* left,
    const float* right,
    float* output,
    int size
);

__global__ void saxpy(
    float alpha,
    const float* x,
    float* y,
    int size
);

__global__ void gelu_elementwise(
    const float* input,
    float* output,
    int size
);

void launch_vector_add(
    const float* left,
    const float* right,
    float* output,
    int size,
    cudaStream_t stream
);

void launch_saxpy(
    float alpha,
    const float* x,
    float* y,
    int size,
    cudaStream_t stream
);

void launch_gelu_elementwise(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
);

}  // namespace week1

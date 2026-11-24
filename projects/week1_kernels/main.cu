#include "kernels.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

void require_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "%s failed: %s\n",
            operation,
            cudaGetErrorString(error)
        );
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    constexpr int element_count = 1 << 20;

    // TODO: Allocate host and device buffers.
    // TODO: Initialize vector-add inputs.
    // TODO: Launch and synchronize vector_add.
    // TODO: Copy output back and compare it with a CPU or PyTorch reference.

    // TODO: Repeat SAXPY and GELU in separate benchmark sections.
    // TODO: Print shape, tolerance, peak bandwidth, and elapsed time.

    std::puts("TODO: implement the Week 1 CUDA benchmark");
    return 0;
}

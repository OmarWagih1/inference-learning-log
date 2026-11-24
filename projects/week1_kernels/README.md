# Week 1 CUDA kernels

Starter CUDA kernels for the first kernel-track week:

- Vector add
- SAXPY
- Elementwise GELU

Use these blocks after completing GPU Puzzles 1-8. The goal is to get the
same result as PyTorch and explain memory throughput with Nsight Compute.

## TODOs

1. Implement each CUDA kernel body.
2. Implement grid-stride launch bounds.
3. Allocate, copy, launch, and synchronize in the host benchmark.
4. Compare output with PyTorch bit-exact or with a small numeric tolerance.
5. Capture a first Nsight Systems trace.
6. Measure DRAM throughput for vector add and SAXPY.

## Build

Use CUDA 13.x and CMake on the Linux GPU environment:

```bash
cmake -S . -B build
cmake --build build
./build/week1_kernels
```

## Acceptance

- Vector add is bit-exact versus PyTorch.
- SAXPY is bit-exact versus PyTorch.
- GELU matches PyTorch within a documented tolerance.
- Vector add exceeds 80% of peak DRAM bandwidth in Nsight Compute.

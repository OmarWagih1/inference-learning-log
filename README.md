# Inference learning log

A public record of my 36-week inference engineering plan, experiments, and
weekly progress.

- [Learning plan](https://white-glacier-08ebd6103.5.azurestaticapps.net/)
- [Week 1: GPU fundamentals](weeks/W01.md)

## Structure

- `weeks/` contains a short progress note for each week.
- `notebooks/` contains weekly notebooks and experiments.
- `projects/` is for larger, reproducible implementations.

## Week 1

The first exercises are:

- [GPU Puzzles](notebooks/W01/GPU_puzzlers.ipynb), covering NUMBA CUDA kernels.
- [P1 inference-from-scratch skeleton](projects/p1_llama_from_scratch/), where
  you implement the transformer forward pass in PyTorch.
- [Week 1 CUDA kernels](projects/week1_kernels/), with TODO blocks for vector
  add, SAXPY, and elementwise GELU.

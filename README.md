# CUDA-Matrix-Multiplication
This repository contains an implementation of matrix multiplication using CUDA to accelerate computations on the GPU.

## Technologies and Tools:
1) Programming Language: C++
2 Parallel Computing: CUDA
3) Build System: CMake
4) Operating System: Windows 11
5) GPU: NVIDIA RTX 3070 Ti

## Project Description:
The project implements high-performance matrix multiplication by utilizing CUDA parallelism. The program computes matrix multiplication by distributing calculations across multiple GPU threads, significantly improving performance compared to CPU-based methods.

Two implementations are included:

1)Basic matrix multiplication using global memory.
2)Optimized matrix multiplication with shared memory and block-wise computation, reducing memory access latency and improving efficiency.

### Requirements:
1) Installed CUDA Toolkit
2) NVCC Compiler
3) CMake for project building

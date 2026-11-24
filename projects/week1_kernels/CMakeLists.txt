cmake_minimum_required(VERSION 3.24)

project(week1_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;80;89;90")

add_executable(
    week1_kernels
    main.cu
    kernels.cu
)

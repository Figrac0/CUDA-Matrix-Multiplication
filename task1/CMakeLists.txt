cmake_minimum_required(VERSION 3.10)
project(MatrixMulCuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

add_executable(matrix_mul_original main_original.cu)
add_executable(matrix_mul_optimized main_optimized.cu)

target_link_libraries(matrix_mul_original ${CUDA_LIBRARIES})
target_link_libraries(matrix_mul_optimized ${CUDA_LIBRARIES})

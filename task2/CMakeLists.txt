cmake_minimum_required(VERSION 3.10)
project(MatrixMulCuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

# Устанавливаем параметры для компилятора CUDA
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# Добавляем исходные файлы
add_executable(matrix_mul_block main.cu)

# Линковка с CUDA библиотеками
target_link_libraries(matrix_mul_block ${CUDA_LIBRARIES})

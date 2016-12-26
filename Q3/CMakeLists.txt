project(CS4552_A1)
cmake_minimum_required(VERSION 3.5)
find_package(CUDA QUIET REQUIRED)

include_directories("${PROJECT_BINARY_DIR}")
include_directories("include")

list(APPEND CUDA_NVCC_FLAGS -x cu)
list(APPEND CUDA_NVCC_FLAGS -std=c++11)

find_package (OpenMP)
if(OPENMP_FOUND)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)

cuda_add_executable(Q3_run main.cu)
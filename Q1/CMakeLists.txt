project(CS4552_A1)
cmake_minimum_required(VERSION 3.5)
find_package(CUDA QUIET REQUIRED)

include_directories("${PROJECT_BINARY_DIR}")
include_directories("include")

include_directories("${PROJECT_BINARY_DIR}")
include_directories("include")

list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)

find_package (OpenMP)
if(OPENMP_FOUND)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp)
endif(OPENMP_FOUND)

cuda_add_executable(pi_mystery pi-mystery.cu)
cuda_add_executable(pi-curand-thrust pi-curand-thrust.cu)
cuda_add_executable(pi-myrand pi-myrand.cu)
cuda_add_executable(pi-curand pi-curand.cu)

project(CS4552_A1_Q2)
cmake_minimum_required(VERSION 3.1)
find_package(CUDA QUIET REQUIRED)

set(SOURCE_FILES main.cu)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fopenmp")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
add_definitions(-std=c++11)

set(CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

include_directories("${PROJECT_BINARY_DIR}")
#include_directories("${PROJECT_SOURCE_DIR}")
include_directories("include")

# Specify binary name and source file to build it from
cuda_add_executable(CS4552_A1_Q2 ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Util -fopenmp)
project(CS4552_A1_Q3)
cmake_minimum_required(VERSION 3.1)
find_package(CUDA QUIET REQUIRED)

set(SOURCE_FILES main.cpp)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
add_definitions(-std=c++11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fopenmp")
endif (OPENMP_FOUND)


set(CUDA_SEPARABLE_COMPILATION ON)
include_directories("${PROJECT_BINARY_DIR}")
include_directories("include")

# Specify binary name and source file to build it from
cuda_add_executable(CS4552_A1_Q3 ${SOURCE_FILES})
#add_executable(CS4552_A1_Q3 ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Util -fopenmp)
project(CS4552_A1_Q3)
cmake_minimum_required(VERSION 3.1)
find_package(CUDA QUIET REQUIRED)

set(SOURCE_FILES main.cpp)

set(GCC_FLAGS "-Wall -m64 -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

include_directories("${PROJECT_BINARY_DIR}")
include_directories("include")

# Specify binary name and source file to build it from
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CMAKE_CXX_FLAGS}")
cuda_add_executable(CS4552_A1_Q3 ${SOURCE_FILES})
#add_executable(CS4552_A1_Q3 ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Util -fopenmp)
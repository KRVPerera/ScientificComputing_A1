project(CS4552_A1_Q2)
cmake_minimum_required(VERSION 3.1)
find_package(CUDA QUIET REQUIRED)

set(GCC_FLAGS "-mcmodel=large -Wall -m64 -O3")
set(SOURCE_FILES main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}")

if (USE_DOUBLE)
    set(EPSILON 0.001)
else ()
    set(EPSILON 0.01)
endif (USE_DOUBLE)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS s)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

set(CMAKE_CXX_STANDARD 11)

cuda_add_executable(CS4552_A1_Q2 ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Util -fopenmp)
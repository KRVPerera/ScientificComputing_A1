cmake_minimum_required(VERSION 3.0)
project(CS4552_A1)

# Set version of the project
set(CS4552_A1_VERSION_MAJOR 0)
set(CS4552_A1_VERSION_MAJOR 9)

# Assignment configuration
set(MAX_THREADS 20)

# option to select float or double
option(USE_DOUBLE "Use double values in computation" ON)

configure_file("${PROJECT_SOURCE_DIR}/A1Config.h.in"
        "${PROJECT_BINARY_DIR}/A1Config.h")

# adding include path to find the generated config file
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")

set(GCC_FLAGS "-mcmodel=large -Wall -m64")
set(SOURCE_FILES main.cpp src/Util.cpp include/Util.h A1Config.h)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}")

if (USE_DOUBLE)
    set(EPSILON 0.001)
else ()
    set(EPSILON 0.01)
endif (USE_DOUBLE)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

set(CMAKE_CXX_STANDARD 11)

add_executable(CS4552_A1 ${SOURCE_FILES})
project(CS4552_A1)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)

# Set version of the project
set(CS4552_A1_VERSION_MAJOR 0)
set(CS4552_A1_VERSION_MINOR 9)


# Assignment configuration
set(MAX_THREADS 20)
option(USE_DOUBLE "Use double values in computation" ON)
configure_file("A1Config.h.in" "${PROJECT_BINARY_DIR}/A1Config.h")

# adding include path to find the generated config file
include_directories("${PROJECT_BINARY_DIR}")
include_directories("include")

# Large arrays, no need as now doing dynamic allocation
set(GCC_FLAGS "-mcmodel=large -Wall -m64 -O3")

add_subdirectory(Util)
add_subdirectory(Q2)
add_subdirectory(Q3)
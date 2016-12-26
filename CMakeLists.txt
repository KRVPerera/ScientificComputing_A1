project(CS4552_A1)
cmake_minimum_required(VERSION 3.5)
find_package(CUDA QUIET REQUIRED)
#
## Set version of the project
set(CS4552_A1_VERSION_MAJOR 0)
set(CS4552_A1_VERSION_MINOR 9)
#
#
## Assignment configuration
set(MAX_THREADS 20)
set(TRIALS_PER_THREAD 4096)
set(BLOCKS 256)
set(THREADS 256)

option(USE_DOUBLE "Use double values in computation" OFF)
configure_file("A1Config.h.in" "${PROJECT_BINARY_DIR}/A1Config.h")
#
## adding include path to find the generated config file
include_directories("${PROJECT_BINARY_DIR}")
include_directories("include")

add_subdirectory(Q1)
add_subdirectory(Q2)
add_subdirectory(Q3)

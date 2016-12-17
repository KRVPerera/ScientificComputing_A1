cmake_minimum_required(VERSION 3.0)
project(CS4552_A1)

# Set version of the project
set (CS4552_A1_VERSION_MAJOR 0)
set (CS4552_A1_VERSION_MAJOR 9)

# option to select float or double
option(USE_DOUBLE "Use double values in computation" OFF)

configure_file("${PROJECT_SOURCE_DIR}/A1Config.h.in"
        "${PROJECT_BINARY_DIR}/A1Config.h")

# adding include path to find the generated config file
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")

set(CMAKE_CXX_STANDARD 11)

#set(GCC_FLAGS "-Wall")
set(SOURCE_FILES main.cpp src/Util.cpp include/Util.h A1Config.h)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}" )

add_executable(CS4552_A1 ${SOURCE_FILES})
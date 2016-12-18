project(CS4552_A1_Q3)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)

set(GCC_FLAGS "-mcmodel=large -Wall -m64 -O3 -fstack-check")
set(SOURCE_FILES main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

add_executable(CS4552_A1_Q3 ${SOURCE_FILES} main.cpp)
target_link_libraries(${PROJECT_NAME} Util)
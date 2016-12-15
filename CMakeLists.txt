cmake_minimum_required(VERSION 3.0)
project(CS4552_A1)

set(CMAKE_CXX_STANDARD 11)

set(GCC_FLAGS "-Wall")
set(SOURCE_FILES main.cpp Util.cpp Util.h)
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}" )

add_executable(CS4552_A1 ${SOURCE_FILES})
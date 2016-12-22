cmake_minimum_required(VERSION 3.1)

#add_library(Util STATIC src/Util.cpp)
CUDA_ADD_LIBRARY(Util STATIC src/Util.cpp)
target_include_directories(Util PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
cmake_minimum_required(VERSION 3.1)

add_library(Util2 STATIC src/Util.cpp)
target_include_directories(Util2 PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
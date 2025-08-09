cmake_minimum_required(VERSION 3.15)
project(matrix_metal)

set(CMAKE_CXX_STANDARD 17)

find_library(METAL_LIB Metal)

add_library(matrix_metal SHARED matrix_metal.cpp)
target_link_libraries(matrix_metal PRIVATE ${METAL_LIB})

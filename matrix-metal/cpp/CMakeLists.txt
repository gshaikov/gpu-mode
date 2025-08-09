cmake_minimum_required(VERSION 3.15)
project(matrix_metal)

set(CMAKE_CXX_STANDARD 17)


find_library(METAL_LIB Metal)
find_library(FOUNDATION_LIB Foundation)
find_library(OBJC_LIB objc)

set(SOURCES matrix_metal.cpp matrix_metal.mm matmul.metal)
add_library(matrix_metal SHARED ${SOURCES})
target_link_libraries(matrix_metal PRIVATE ${METAL_LIB} ${FOUNDATION_LIB} ${OBJC_LIB})
set_source_files_properties(matrix_metal.mm PROPERTIES LANGUAGE CXX)

# Copy matmul.metal to the build directory after build
add_custom_command(TARGET matrix_metal POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${CMAKE_CURRENT_SOURCE_DIR}/matmul.metal
		$<TARGET_FILE_DIR:matrix_metal>/matmul.metal)

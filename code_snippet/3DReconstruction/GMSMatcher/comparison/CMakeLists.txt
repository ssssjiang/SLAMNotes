cmake_minimum_required(VERSION 3.5)
project(COMPARISON_SIFTRANSAC2SIFTGMS)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_library(siftRansac siftRansac.cpp)
add_executable(comparison comparison.cpp)

target_link_libraries(comparison siftRansac ${OpenCV_LIBS})
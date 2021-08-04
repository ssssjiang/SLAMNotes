cmake_minimum_required(VERSION 3.5)

project(SIFT+RANSAC)

find_package(OpenCV REQUIRED)
include_directories(include ${OPENCV_INCLUDE_DIRS})

add_executable(siftRansac siftRansac.cpp)
target_link_libraries(siftRansac ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.5)
project(COMPUTEORB)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})

add_executable(computeORB computeORB.cpp)
target_link_libraries(computeORB ${OpenCV_LIBRARIES})
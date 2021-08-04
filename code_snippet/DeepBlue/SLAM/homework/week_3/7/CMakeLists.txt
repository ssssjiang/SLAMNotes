cmake_minimum_required(VERSION 3.5)
project(COMPARE_TRAJECTORY)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(Sophus REQUIRED)
find_package(Pangolin 0.4 REQUIRED)
include_directories(
    ${Sophus_INCLUDE_DIRS} 
    ${Pangolin_INCLUDE_DIRS} 
    "/usr/include/eigen3")

add_executable(compare_trajectory compare_trajectory.cpp draw_trajectory.cpp)
target_link_libraries(
    compare_trajectory 
    ${Sophus_LIBRARIES}
    ${Pangolin_LIBRARIES})
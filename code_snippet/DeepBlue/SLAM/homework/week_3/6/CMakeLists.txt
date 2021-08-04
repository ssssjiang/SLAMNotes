cmake_minimum_required(VERSION 3.5)
project(DRAW_TRAJECTORY)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(Sophus REQUIRED)
find_package(Pangolin 0.4 REQUIRED)
include_directories(
    ${Sophus_INCLUDE_DIRS} 
    ${Pangolin_INCLUDE_DIRS} 
    "/usr/include/eigen3")

add_executable(draw_trajectory draw_trajectory.cpp)
target_link_libraries(
    draw_trajectory 
    ${Sophus_LIBRARIES}
    ${Pangolin_LIBRARIES})
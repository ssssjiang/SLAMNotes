cmake_minimum_required(VERSION 3.5)
project(DISPARITY)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)
find_package(Pangolin 0.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS}
                    "/usr/include/eigen3")

add_executable(disparity disparity.cpp)
target_link_libraries(disparity ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
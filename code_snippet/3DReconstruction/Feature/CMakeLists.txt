cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)
include_directories("/usr/include/eigen3" ${OpenCV_INCLUDE_DIRS})

add_executable(HarrisCornerDetector HarrisCornerDetector.cpp)
target_link_libraries(HarrisCornerDetector ${OpenCV_LIBS})
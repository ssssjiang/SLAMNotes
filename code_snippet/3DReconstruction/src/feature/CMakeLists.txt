cmake_minimum_required(VERSION 3.5)
project(COMPARISON_SIFTRANSAC2SIFTGMS)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_library(FeatureManager SHARED FeatureManager.cpp)
add_library(GMSMatcher SHARED GMSMatcher.cpp)
add_executable(te test.cpp)

target_link_libraries(te FeatureManager GMSMatcher ${OpenCV_LIBS})
cmake_minimum_required(VERSION 3.5)
project(TRACKS)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_library(FeatureManager SHARED ../feature/FeatureManager.cpp)
add_library(GMSMatcher SHARED ../feature/GMSMatcher.cpp)
add_library(ImagePair SHARED ImagePair.cpp)

add_executable(te test.cpp)

target_link_libraries(te FeatureManager GMSMatcher ImagePair ${OpenCV_LIBS})
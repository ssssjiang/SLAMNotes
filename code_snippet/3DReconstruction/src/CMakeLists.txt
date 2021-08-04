cmake_minimum_required(VERSION 3.5)
project(TRACKS)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_library(FeatureManager SHARED FeatureManager.cpp)
add_library(GMSMatcher SHARED GMSMatcher.cpp)
add_library(ImageLoader SHARED ImageLoader.cpp)
add_library(ImagePair SHARED ImagePair.cpp)


add_executable(te test.cpp)

target_link_libraries(te FeatureManager GMSMatcher ImageLoader ImagePair 
${OpenCV_LIBS})
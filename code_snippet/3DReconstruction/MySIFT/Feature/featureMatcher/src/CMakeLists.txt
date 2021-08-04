cmake_minimum_required(VERSION 3.5)

find_package( OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(FeatureMatcher SHARED FeatureMatcher.cpp)

add_executable(testFeatureMatcher testFeatureMatcher.cpp)

target_link_libraries(testFeatureMatcher FeatureMatcher ${OpenCV_LIBS})
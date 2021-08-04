cmake_minimum_required(VERSION 3.5)
project(SFM)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

# add_library(FeatureManager SHARED src/FeatureManager.cpp)
# add_library(GMSMatcher SHARED src/GMSMatcher.cpp)
# add_library(ImageLoader SHARED src/ImageLoader.cpp)
add_library(ImagePair SHARED 
src/FeatureManager.cpp 
src/ImageLoader.cpp 
src/GMSMatcher.cpp 
src/Utility.cpp
src/EGGraph.cpp
src/TrackManager.cpp)


add_executable(te src/test.cpp)

target_link_libraries(te ImagePair ${OpenCV_LIBS})
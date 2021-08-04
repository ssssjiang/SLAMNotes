set( CMAKE_CXX_FLAGS "-std=c++11" )

find_package( OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(FeatureDetector SHARED FeatureDetector.cpp)

add_executable(testFeatureDetector testFeatureDetector.cpp)

target_link_libraries( testFeatureDetector FeatureDetector ${OpenCV_LIBS} )
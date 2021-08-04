cmake_minimum_required(VERSION 3.0)

project(opemimageio_practice)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenImageIO REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11")

message(STATUS "OpenImageIO found in " ${OIIO_INCLUDE_DIR})
include_directories(${OIIO_INCLUDE_DIR})


add_subdirectory(ReadImage)
add_subdirectory(WriteImage)
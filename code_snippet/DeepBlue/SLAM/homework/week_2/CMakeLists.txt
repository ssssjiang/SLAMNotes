cmake_minimum_required(VERSION 2.8)
project(demo)

include_directories("/usr/include/eigen3")

add_executable(geometry geometry.cpp)
cmake_minimum_required(VERSION 3.0)

project(LSP)

file(GLOB sources *.h *.cpp)
add_executable(main ${sources})
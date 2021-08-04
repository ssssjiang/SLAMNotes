set(targetname triangle)

include(${PROJECT_SOURCE_DIR}/cmake/UseOpenGL.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/UseGLEW.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/UseGLFW.cmake)

set(sources triangle.cpp)

include(${PROJECT_SOURCE_DIR}/cmake/SetAppTarget.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/LinkOpenGL.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/LinkGLEW.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/LinkGLFW.cmake)
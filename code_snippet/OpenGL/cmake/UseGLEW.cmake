find_package(GLEW REQUIRED)

if (GLEW_FOUND)
    message(STATUS "GLEW found in " ${GLEW_INCLUDE_DIRS})
    include_directories(${GLEW_INCLUDE_DIRS})
else()
    message(STATUS "GLEW not found")
endif()
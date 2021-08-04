find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

if (GLFW_FOUND)
    message(STATUS "GLFW found in " ${GLFW_INCLUDE_DIRS})
    include_directories(${GLFW_INCLUDE_DIRS})
else()
    message(STATUS "GLFW not found")
endif()
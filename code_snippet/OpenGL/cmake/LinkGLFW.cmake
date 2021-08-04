if (GLFW_FOUND)
    target_link_libraries(${targetname} ${GLFW_LIBRARIES})
endif()
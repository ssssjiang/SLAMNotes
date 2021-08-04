if(NOT ${headers})
    add_executable(${targetname} ${sources})
else()
    add_executable(${targetname} ${headers} ${sources})
endif()

target_link_libraries(${targetname} ${OIIO_LIBRARIES})
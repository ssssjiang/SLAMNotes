set(LIBHELLO_SRC hello.cpp)
add_library(hello SHARED ${LIBHELLO_SRC})

add_executable(sayhello useHello.cpp)
target_link_libraries(sayhello hello)

install(TARGETS hello
LIBRARY DESTINATION lib)
install(FILES hello.h DESTINATION include)
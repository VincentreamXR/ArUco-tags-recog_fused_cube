INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)

add_executable(create_marker create_marker.cpp)
target_link_libraries(create_marker jumarker)

add_executable(jumarker_test jumarker_test.cpp)
target_link_libraries(jumarker_test jumarker)


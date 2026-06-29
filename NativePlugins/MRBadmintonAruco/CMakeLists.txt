cmake_minimum_required(VERSION 3.18)

project(MRBadmintonAruco LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_library(mrbadminton_aruco SHARED
    src/mrbadminton_aruco.cpp
)

target_link_libraries(mrbadminton_aruco PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options(mrbadminton_aruco PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

install(TARGETS mrbadminton_aruco
    LIBRARY DESTINATION .
)

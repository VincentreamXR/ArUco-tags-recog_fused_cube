# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.5)
project(jumarker LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# OpenCV
set(OPENCV_PATH /path/to/opencv CACHE PATH "Path to opencv libs and headers")
include_directories(${OPENCV_PATH}/include/opencv4)
link_directories(${OPENCV_PATH}/lib/)

# ----------------------------------------------------------------------------
#   Create Subdirectory
# ----------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(utils)

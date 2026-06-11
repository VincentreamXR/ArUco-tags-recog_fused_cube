include_directories(${PROJECT_SOURCE_DIR}/src)

# ----------------------------------------------------------------------------
#   Add files project
# ----------------------------------------------------------------------------

FILE(GLOB jumarker_src "*.cpp" )
FILE(GLOB basictypes_src "basictypes/*.cpp" )
FILE(GLOB featureextractors_src "featureextractors/*.cpp" )

# ----------------------------------------------------------------------------
#   Create jumarker library
# ----------------------------------------------------------------------------

ADD_LIBRARY(jumarker SHARED ${jumarker_src} ${basictypes_src} ${featureextractors_src})

target_link_libraries(jumarker ${OPENCV_PATH}/lib/libopencv_core.so)
target_link_libraries(jumarker ${OPENCV_PATH}/lib/libopencv_imgproc.so)
target_link_libraries(jumarker ${OPENCV_PATH}/lib/libopencv_imgcodecs.so)
target_link_libraries(jumarker ${OPENCV_PATH}/lib/libopencv_calib3d.so)
target_link_libraries(jumarker ${OPENCV_PATH}/lib/libopencv_videoio.so)
target_link_libraries(jumarker ${OPENCV_PATH}/lib/libopencv_highgui.so)
target_link_libraries(jumarker ${OPENCV_PATH}/lib/libopencv_features2d.so)
target_link_libraries(jumarker ${OPENCV_PATH}/lib/libopencv_video.so)

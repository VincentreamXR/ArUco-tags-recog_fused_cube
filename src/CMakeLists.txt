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

find_package(OpenCV REQUIRED)
target_include_directories(jumarker PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jumarker ${OpenCV_LIBS})


cmake_minimum_required(VERSION 3.10)

get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)

string(REPLACE " " "_" ProjectId ${ProjectId})

project(MOBIS)

set (CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/include/opencv4")
file(GLOB SOURCES  mobilenetv2.cpp)
 
add_executable(${PROJECT_NAME} ${SOURCES}  )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/includes)
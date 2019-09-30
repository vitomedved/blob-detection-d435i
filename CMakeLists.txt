cmake_minimum_required(VERSION 3.1.0)
project (blob-detection)

add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set (LOCAL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(REALSENSE_INCLUDE_DIR /usr/include/librealsense2)
set(OPENCV_INCLUDE_DIR /usr/local/include/opencv4)

set (INCLUDES
        ${OpenCV_INCLUDE_DIRS}
        ${REALSENSE_INCLUDE_DIR}
        ${LOCAL_INCLUDE_DIR}
)

include_directories( ${INCLUDES} )

set(SOURCES
        src/main.cpp
        src/CBlobManager.cpp
        src/CBlob.cpp
        src/BackgroundSubtraction.cpp)

add_executable( blob-detection ${SOURCES} )

target_link_libraries(blob-detection
        -lrealsense2 ${OpenCV_LIBS} -pthread
)

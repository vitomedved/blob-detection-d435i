cmake_minimum_required(VERSION 3.1.0)
project (blob-detection)

add_definitions(-std=c++17)

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#set(SOURCES 
#	./src/main.cpp
#	./src/BackgroundSubtraction.cpp
#	./src/CBlob.cpp
#	./src/CBlobManager.cpp)
#
#set(HEADERS 
#	./include/BackgroundSubtraction.hpp
#	./include/CBlob.hpp
#	./include/CBlobManager.hpp
#	./include/cv-helpers.hpp)

file(GLOB SOURCES
    include/*.hpp
    src/*.cpp
)

INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} )

#link_directories( ${CMAKE_BINARY_DIR}/bin)

add_executable( blob-detection ${SOURCES} )

target_link_libraries(blob-detection ${OpenCV_LIBS} ${realsense2_LIBRARY})
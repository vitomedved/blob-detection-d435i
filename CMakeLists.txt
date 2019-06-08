cmake_minimum_required(VERSION 3.1.0)
project ("blob-detection")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_definitions(-std=c++17)

include_directories(
	$ENV{rs2_include}
#	$ENV{opencv_include}
	"E:/opencv/build/include"
	"./include"
	)

set(SOURCES 
	"./src/main.cpp"
	"./src/BackgroundSubtraction.cpp")

message("$ENV{rs2_libx64} $ENV{rs2_libx86} $ENV{opencv_libx64}")

link_directories(
	"C:/Program Files (x86)/Intel RealSense SDK 2.0/lib/x64"
	"E:/opencv/build/x64/vc15/lib"
	)

link_libraries("opencv_world410d.lib" "opencv_world410.lib" "realsense2.lib")

add_executable(main "./src/main.cpp")
add_executable(bgsub "./src/BackgroundSubtraction.cpp")

add_subdirectory(src)
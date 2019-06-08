In this project the goal is detecting blobs of people walking underneath the camera.

Blobs should be tracked throughout the POV of the camera and its info should be written to the database.

Camera used in project is Intel RealSense D435i mounted on 3.4m height.


Phase 1: Getting everything to work

	RealSense SDK 2.0 (2.21.0) and OpenCV 4.1.0 are used as external dependencies. 
	These dependencies are linked to the VS2019 project. Currently only realsense2 is linked via CMake so remaining TODO 
	is to link OpenCV via CMake
	
	The goal here is to test out whether everything works and record our test dataset which will hold both RGB and depth data.
	The goal was achieved, dataset was recorded and saved to /recording folder inside project folder, also the same dataset was
	successfully playbacked as a normal video using OpenCV's imread() function.
	
Phase 2: Getting 4-channeled RGBD (RGB + depth) matrix frames.

	Depth data was firstly extracted to cv::Mat with values of distance in meters (16 bit data, 1 channel).
	Depth data in meters was then scaled down to 8 bit 1 channel data.
	RGB data was split into seperate channel, which gives us three 8 bit 1 channel matrixes.
	After we got all 4 channels seperated, all of these were merged in one via following order: b+g+r+depth (OpenCV prefers BGR over RGB)
	
	Data was displayed on screen along with color and depth data and there was no difference in this newly created 4-channeled 
	image andnormal BGR image.
	
Phase 3: Background subtraction

	Depth data in 8UC1 format is thresholded in regards to the first (background) image.
	On RGBD data, the following background subtraction techniques are used: KNN, MOG2.

	
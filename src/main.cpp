#include <rs.hpp>
#include <opencv2/opencv.hpp>

#include <sstream>
#include <iostream>
#include <iomanip>
#include <queue>
#include <thread>

#include "cv-helpers.hpp"

#include "BackgroundSubtraction.hpp"
#include "CBlobManager.hpp"

#include <atomic>


const auto PLAYBACK_FILEPATH = "C:/Users/vmedved/Desktop/blob-detection/recording/recording_1561980047707.bag";//"../../recording/test.bag";
//const auto RECORDING_FILEPATH = "C:/Users/vmedved/Desktop/blob-detection/recording/test2.bag";//"../../recording/test2.bag";

float CURRENT_SCALE = 3.0;

const auto timeoutThreshold = 30; // seconds of timeThreshold (max recording time)

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);

float get_depth_scale(rs2::device dev);

std::vector<cv::KeyPoint> getKeypointsOfCurrentFrame(cv::Mat& frame, cv::Ptr<cv::SimpleBlobDetector>& blobDetector);

int thr = 25;

bool doesFrameContainHuman(rs2::frameset& frameset, cv::Ptr<cv::BackgroundSubtractor> &bgSub, cv::Ptr<cv::SimpleBlobDetector> &blobDetector);

static double distanceBtwPoints(const cv::Point2f& a, const cv::Point2f& b)
{
	double xDiff = a.x - b.x;
	double yDiff = a.y - b.y;

	return std::sqrt((xDiff * xDiff) + (yDiff * yDiff));
}

/*
* Class for enqueuing and dequeuing cv::Mats efficiently
* Thanks to this awesome post by PKLab
* http://pklab.net/index.php?id=394&lang=EN
*/
class QueuedMat {
public:

	cv::Mat img; // Standard cv::Mat

	QueuedMat() {}; // Default constructor

	// Destructor (called by queue::pop)
	~QueuedMat() {
		img.release();
	};

	// Copy constructor (called by queue::push)
	QueuedMat(const QueuedMat& src) {
		src.img.copyTo(img);
	};
};

int main(int argc, char* argv[]) try
{
	// ---------------------------- RS2 ----------------------------------
	rs2::frameset frames;
	rs2::frame depth;
	rs2::frame color;
	rs2::colorizer color_map;

	// Create a shared pointer to a pipeline which is needed if I want to use both recording/playback.
	auto pipe = std::make_shared<rs2::pipeline>();

	// rs2::config which is used to choose between recording/playbacking or just a normal live stream
	rs2::config cfg; // Declare a new configuration


	std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
	std::string timestamp_string = std::to_string(ms.count());
	
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 60);
	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 90);

	rs2::pipeline_profile profile = pipe->start(cfg);
	rs2::device device = pipe->get_active_profile().get_device();

	rs2_stream align_to = find_stream_to_align(profile.get_streams());
	rs2::align align(align_to);

	float depth_scale = get_depth_scale(profile.get_device());

	// ---------------------------- time ---------------------------------------------
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	std::chrono::system_clock::time_point end;


	// --------------------------- BgSubtractors ------------------------------------------------
	BackgroundSubtraction depthSubtraction(BackgroundSubtraction::DEPTH_THRESHOLD);
	BackgroundSubtraction knnSub(BackgroundSubtraction::KNN/*, 7000, 400, false*/);
	BackgroundSubtraction mog2Sub(BackgroundSubtraction::MOG2/*, 7000, 8, false*/);


	cv::SimpleBlobDetector::Params params;
	params.filterByArea = true;
	params.minArea = 2000;
	params.maxArea = 80000;
	params.filterByCircularity = false;
	params.filterByColor = true;
	params.blobColor = 255;
	params.filterByConvexity = false;
	params.filterByInertia = false;


	cv::Ptr<cv::SimpleBlobDetector> blobDetectorForRecording = cv::SimpleBlobDetector::create(params);

	cv::SimpleBlobDetector::Params params2;
	params2.filterByArea = true;
	params2.minArea = 10000;
	params2.maxArea = 900000;
	params2.filterByCircularity = false;
	params2.filterByColor = true;
	params2.blobColor = 255;
	params2.filterByConvexity = false;
	params2.filterByInertia = false;

	cv::Ptr<cv::SimpleBlobDetector> blobDetectorForPostProcessing = cv::SimpleBlobDetector::create(params2);

	cv::Ptr<cv::BackgroundSubtractor> knn = cv::createBackgroundSubtractorKNN(200, 700, false);

	std::vector<cv::Point> realPosition, kalmanPosition;

	bool doesBlobNeedNewKalman = true;

	BlobManager blobManager;

	// Start #################################################################################
	std::mutex accessToQueueMutex;

	std::queue<QueuedMat> colorQueue;
	std::atomic<bool> isColorMatCreated(false);
	std::atomic<bool> isBlobDetected(false);
	cv::Mat colorDequeuedMat;
	
	std::thread processingThread([&]() 
	{
			const int MAX_FRAMES_IN_Q = 3;

			bool hadInvalidFrames = false;
			int MAX_BLOB_OUT_OF_FRAME_SECONDS_RECORD = 1;

			std::chrono::system_clock::time_point startTime;
			std::chrono::system_clock::time_point endTime;

			while (true)
			{
				if (!isColorMatCreated)
				{
					continue;
				}

				while (true)
				{
					accessToQueueMutex.lock();
					if (!colorQueue.empty())
					{
						colorQueue.front().img.copyTo(colorDequeuedMat);
						colorQueue.pop();
					}
					else
					{
						accessToQueueMutex.unlock();
						break;
					}
					accessToQueueMutex.unlock();

					std::vector<cv::KeyPoint> keypoints;

					cv::resize(colorDequeuedMat, colorDequeuedMat, cv::Size(224, 168));

					cv::GaussianBlur(colorDequeuedMat, colorDequeuedMat, cv::Size(5, 5), 0);
					cv::dilate(colorDequeuedMat, colorDequeuedMat, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20)));

					knn->apply(colorDequeuedMat, colorDequeuedMat);

					cv::GaussianBlur(colorDequeuedMat, colorDequeuedMat, cv::Size(5, 5), 0);

					blobDetectorForRecording->detect(colorDequeuedMat, keypoints);

					cv::drawKeypoints(colorDequeuedMat, keypoints, colorDequeuedMat, cv::Scalar(255, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

					if (keypoints.size() > 0)
					{
						hadInvalidFrames = false;
						isBlobDetected = true;
					}
					else
					{								
						if (!hadInvalidFrames)
						{
							// last frame was detected blob so this is first frame for blob to be undetected
							// start time and count 5 secs
							hadInvalidFrames = true;
							startTime = std::chrono::system_clock::now();
						}
						else
						{
							// blob was not detected in previous frame so check if time has passed threshold, if yes, stop recording
							endTime = std::chrono::system_clock::now();
							auto diff = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
							if (diff.count() > MAX_BLOB_OUT_OF_FRAME_SECONDS_RECORD)
							{
								isBlobDetected = false;
							}
						}
					}

					//cv::imshow("Processing thread", colorDequeuedMat);
					cv::waitKey(1);

					while (colorQueue.size() > MAX_FRAMES_IN_Q)
					{
						colorQueue.pop();
					}
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
			}
	});
	processingThread.detach();
	
	bool isVideoStarted = false;

	cv::VideoWriter colorVideo;
	cv::VideoWriter depthVideo;

	while (true)
	{

		rs2::frameset data = pipe->wait_for_frames(); // Wait for next set of frames from the camera
		
        data = align.process(data);

		rs2::video_frame colorFrame = data.get_color_frame();
		rs2::depth_frame depthFrame = data.get_depth_frame();

		auto color_w = colorFrame.get_width();
		auto color_h = colorFrame.get_height();

		cv::Mat colorData(cv::Size(color_w, color_h), CV_8UC3, (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat depthData = frame_to_mat(depthFrame);
		depthData.convertTo(depthData, CV_8UC1, 5120./65536.);
		

		if (!isColorMatCreated)
		{
			colorDequeuedMat = cv::Mat(color_h, color_w, CV_8UC3);
			isColorMatCreated = true;
		}

		//cv::imshow("Main thread - color", colorData);
		//cv::imshow("Main thread - depth", depthData);
		cv::waitKey(1);
	
		QueuedMat colorMat;
		colorData.copyTo(colorMat.img);

		accessToQueueMutex.lock();
		colorQueue.push(colorMat);
		accessToQueueMutex.unlock();


		if (isBlobDetected)
		{	
			if (!isVideoStarted)
			{
				std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
				std::string timestamp_string = std::to_string(ms.count());

				cv::String colorFilename = "./color_";
				colorFilename.append(timestamp_string);
				colorFilename.append(".avi");

				colorVideo = cv::VideoWriter(colorFilename, cv::VideoWriter::fourcc('F', 'F', 'V', '1'), 60, cv::Size(640, 480), true);
				
				cv::String depthFilename = "./depth_";
				depthFilename.append(timestamp_string);
				depthFilename.append(".avi");

				depthVideo = cv::VideoWriter(depthFilename, cv::VideoWriter::fourcc('H', 'F', 'Y', 'U'), 60, cv::Size(640, 480), false);
				
				isVideoStarted = true;

				std::cout << "Started recording." << std::endl;
			}

			if(colorData.empty())
			{
				printf("Empty color data\n");
			}
			if(depthData.empty())
			{
				printf("Depth data empty\n");
			}

			cv::cvtColor(colorData, colorData, cv::COLOR_BGR2RGB);
			colorVideo.write(colorData);
			depthVideo.write(depthData);
		}
		else
		{
			if (isVideoStarted)
			{
				colorVideo.release();
				depthVideo.release();
				isVideoStarted = false;
				std::cout << "Stopped recording." << std::endl;
			}
		}

		depthData.release();
		colorData.release();

		//frames = pipe->wait_for_frames();

		/*if (doesFrameContainHuman(frames, knn, blobDetectorForRecording))
		{
			//device.as<rs2::recorder>().resume();
		}*/
	}

	pipe->stop();

	return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception& e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}

float get_depth_scale(rs2::device dev)
{
	// Go over the device's sensors
	for (rs2::sensor& sensor : dev.query_sensors())
	{
		// Check if the sensor if a depth sensor
		if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
		{
			return dpt.get_depth_scale();
		}
	}
	throw std::runtime_error("Device does not have a depth sensor");
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
	//Given a vector of streams, we try to find a depth stream and another stream to align depth with.
	//We prioritize color streams to make the view look better.
	//If color is not available, we take another stream that (other than depth)
	rs2_stream align_to = RS2_STREAM_ANY;
	bool depth_stream_found = false;
	bool color_stream_found = false;
	for (rs2::stream_profile sp : streams)
	{
		rs2_stream profile_stream = sp.stream_type();
		if (profile_stream != RS2_STREAM_DEPTH)
		{
			if (!color_stream_found)         //Prefer color
				align_to = profile_stream;

			if (profile_stream == RS2_STREAM_COLOR)
			{
				color_stream_found = true;
			}
		}
		else
		{
			depth_stream_found = true;
		}
	}

	if (!depth_stream_found)
		throw std::runtime_error("No Depth stream available");

	if (align_to == RS2_STREAM_ANY)
		throw std::runtime_error("No stream found to align with Depth");

	return align_to;
}

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
	for (auto&& sp : prev)
	{
		//If previous profile is in current (maybe just added another)
		auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
		if (itr == std::end(current)) //If it previous stream wasn't found in current
		{
			return true;
		}
	}
	return false;
}


bool doesFrameContainHuman(rs2::frameset& frameset, cv::Ptr<cv::BackgroundSubtractor> &bgSub, cv::Ptr<cv::SimpleBlobDetector> &blobDetector)
{
	const int RESIZE_SCALE_FACTOR = 2;

	std::vector<cv::KeyPoint> keypoints;

	auto color = frameset.get_color_frame();
	const int color_w = color.as<rs2::video_frame>().get_width();
	const int color_h = color.as<rs2::video_frame>().get_height();

	cv::Mat colorImage(cv::Size(color_w, color_h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

	cv::resize(colorImage, colorImage, cv::Size(color_w / RESIZE_SCALE_FACTOR, color_h / RESIZE_SCALE_FACTOR));

	cv::GaussianBlur(colorImage, colorImage, cv::Size(5, 5), 0);
	cv::dilate(colorImage, colorImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20)));

	bgSub->apply(colorImage, colorImage);

	cv::GaussianBlur(colorImage, colorImage, cv::Size(5, 5), 0);

	blobDetector->detect(colorImage, keypoints);

	cv::drawKeypoints(colorImage, keypoints, colorImage, cv::Scalar(255, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cv::resize(colorImage, colorImage, cv::Size(color_w, color_h));
	// TODO: remove imShow and waitKey for release, it's only here to see when camera is recording
	//cv::imshow("Color data recording", colorImage);
	cv::waitKey(1);

	return keypoints.size() > 0;
}

std::vector<cv::KeyPoint> getKeypointsOfCurrentFrame(cv::Mat &frame, cv::Ptr<cv::SimpleBlobDetector>& blobDetector)
{
	std::vector<cv::KeyPoint> ret;

	blobDetector->detect(frame, ret);

	return ret;
}


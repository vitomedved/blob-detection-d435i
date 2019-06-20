#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <sstream>
#include <iostream>
#include <iomanip>

#include "cv-helpers.hpp"

#include "BackgroundSubtraction.hpp"
#include "CBlobManager.h"


const auto PLAYBACK_FILEPATH = "C:/Users/vmedved/Desktop/blob-detection/recording/test.bag";//"../../recording/test.bag";
const auto RECORDING_FILEPATH = "C:/Users/vmedved/Desktop/blob-detection/recording/test2.bag";//"../../recording/test2.bag";

float CURRENT_SCALE = 3.0;

const auto timeoutThreshold = 30; // seconds of timeThreshold (max recording time)

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);

float get_depth_scale(rs2::device dev);

std::vector<cv::KeyPoint> getKeypointsOfCurrentFrame(cv::Mat& frame, cv::Ptr<cv::SimpleBlobDetector>& blobDetector);

int thr = 25;

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

bool doesFrameContainHuman(rs2::frameset& frameset, cv::Ptr<cv::BackgroundSubtractor> &bgSub, cv::Ptr<cv::SimpleBlobDetector> &blobDetector);

static double distanceBtwPoints(const cv::Point2f& a, const cv::Point2f& b)
{
	double xDiff = a.x - b.x;
	double yDiff = a.y - b.y;

	return std::sqrt((xDiff * xDiff) + (yDiff * yDiff));
}

int main(int argc, char* argv[]) try
{
	// ----------------------- OPENCV ---------------------------------
	// Declare opencv window where recording/playback will be shown
	//cv::namedWindow(depthImageWindowName, cv::WINDOW_NORMAL);
	//cv::createTrackbar("Scale %", depthImageWindowName, 0, 100, onTrackbarChanged);

	//cv::namedWindow(colorImageWindowName, cv::WINDOW_NORMAL);

	//cv::namedWindow(mixedColorDepthWindowName, cv::WINDOW_NORMAL);

	// ---------------------------- RS2 ----------------------------------
	rs2::frameset frames;
	rs2::frame depth;
	rs2::frame color;
	rs2::colorizer color_map;

	// Create a shared pointer to a pipeline which is needed if I want to use both recording/playback.
	auto pipe = std::make_shared<rs2::pipeline>();

	// rs2::config which is used to choose between recording/playbacking or just a normal live stream
	rs2::config cfg; // Declare a new configuration

	//cfg.enable_record_to_file(RECORDING_FILEPATH);
	cfg.enable_device_from_file(PLAYBACK_FILEPATH);
	//cfg.enable_all_streams();

	rs2::pipeline_profile profile = pipe->start(cfg);
	rs2::device device = pipe->get_active_profile().get_device();

	//device.as<rs2::recorder>().pause();

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

	//auto KNN = cv::createBackgroundSubtractorKNN(500, 400.0, false);
	//auto MOG2 = cv::createBackgroundSubtractorMOG2(500, 16.0, false);


	cv::SimpleBlobDetector::Params params;
	params.filterByArea = true;
	params.minArea = 1000;
	params.maxArea = 20000;
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

	while(true)
	{
		// Currently, recording mode will record timeoutThrashold seconds to a .bag file
		// TODO: while time not passed threshold(get frames -> check for blobs -> resume/pause recording)
		if (device.as<rs2::recorder>())
		{
			end = std::chrono::system_clock::now();
			auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start);
			if (seconds.count() > timeoutThreshold) {
				pipe->stop();
				pipe = std::make_shared<rs2::pipeline>();
				std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
				rs2::config newCfg;
				std::string timestamp_string = std::to_string(ms.count());
				newCfg.enable_record_to_file("C:/Users/vmedved/Desktop/blob-detection/recording/recording_" + timestamp_string + ".bag");
				pipe->start(newCfg);
				device = pipe->get_active_profile().get_device();
				start = std::chrono::system_clock::now();
			}


			frames = pipe->wait_for_frames();
			if (!doesFrameContainHuman(frames, knn, blobDetectorForRecording))
			{
				/*pipe->stop();
				pipe = std::make_shared<rs2::pipeline>();
				rs2::config newCfg;
				newCfg.enable_all_streams();
				pipe->start(newCfg);
				device = pipe->get_active_profile().get_device();*/
				device.as<rs2::recorder>().pause();
			}
			else
			{
				device.as<rs2::recorder>().resume();
			}

			/*while (true)
			{
				end = std::chrono::system_clock::now();
				auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start);
				if (seconds.count() > timeoutThreshold) {
					pipe->stop();
					break;
				}

				frames = pipe->wait_for_frames();

				depth = frames.get_depth_frame().apply_filter(color_map);
				const int depth_w = depth.as<rs2::video_frame>().get_width();
				const int depth_h = depth.as<rs2::video_frame>().get_height();

				color = frames.get_color_frame();
				const int color_w = color.as<rs2::video_frame>().get_width();
				const int color_h = color.as<rs2::video_frame>().get_height();

				// Create OpenCV matrix of size (w,h) from the colorized depth data
				cv::Mat depthIimage(cv::Size(depth_w, depth_h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
				cv::Mat colorImage(cv::Size(color_w, color_h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

				auto now = std::chrono::system_clock::now();
				std::time_t end_time = std::chrono::system_clock::to_time_t(now);

				cv::imshow("Depth data recording", depthIimage);
				cv::imshow("Color data recording", colorImage);
				cv::waitKey(1);

			}*/
		}
		else if (device.as<rs2::playback>())
		{
			rs2::frameset frameset = pipe->wait_for_frames();

			if (profile_changed(pipe->get_active_profile().get_streams(), profile.get_streams()))
			{
				//If the profile was changed, update the align object, and also get the new device's depth scale
				profile = pipe->get_active_profile();
				align_to = find_stream_to_align(profile.get_streams());
				align = rs2::align(align_to);
				depth_scale = get_depth_scale(profile.get_device());
			}

			rs2::frameset alignedFrameset = align.process(frameset);

			// Trying to get both other and aligned depth frames
			rs2::video_frame colorFrame = alignedFrameset.first(align_to);
			rs2::depth_frame depthFrame = alignedFrameset.get_depth_frame();

			if (!depthFrame || !colorFrame)
			{
				continue;
			}

			const int depth_w = depthFrame.get_width();
			const int depth_h = depthFrame.get_height();

			const int color_w = colorFrame.get_width();
			const int color_h = colorFrame.get_height();


			// --------------------------------------Getting DEPTH data--------------------------------------
				cv::Mat depthData = depth_frame_to_meters(*pipe, depthFrame);
			// convert to 8 bit with scale factor 255./3 because 3m is clipping distance and there are 255 values
			depthData.convertTo(depthData, CV_8UC1, 255. / CURRENT_SCALE);

			cv::Mat thresholdedDepth;
			depthSubtraction.setDepthThreshold(thr);
			depthSubtraction.apply(depthData, thresholdedDepth);
			// -------------------------------------- Getting RGBD data --------------------------------------

			cv::Mat colorData(cv::Size(color_w, color_h), CV_8UC3, (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
			cv::cvtColor(colorData, colorData, cv::COLOR_RGB2BGR);

			// split color image (rgb channels) to seperate b, g, r
			std::vector<cv::Mat> rgbdSrc;
			cv::split(colorData, rgbdSrc);
			colorData.release();

			// add 4th, depth, channel to bgr
			rgbdSrc.push_back(thresholdedDepth);

			// merge all channels together (r + b + g + depth)
			cv::Mat rgbdData;
			cv::merge(rgbdSrc, rgbdData);

			rgbdSrc.clear();

			int SCALE_FACTOR = 6;
			cv::resize(rgbdData, rgbdData, cv::Size(color_w / SCALE_FACTOR, depth_w / SCALE_FACTOR));
			cv::resize(thresholdedDepth, thresholdedDepth, cv::Size(color_w / SCALE_FACTOR, depth_w / SCALE_FACTOR));
			
			// KNN
			cv::Mat knnResult;

			knnSub.apply(rgbdData, knnResult);

			knnResult.convertTo(knnResult, CV_8UC1);

			//cv::resize(knnResult, knnResult, cv::Size(color_w, depth_w));

			// MOG2
			cv::Mat mog2Result;

			mog2Sub.apply(rgbdData, mog2Result);

			mog2Result.convertTo(mog2Result, CV_8UC1);

			//cv::resize(mog2Result, mog2Result, cv::Size(color_w, depth_w));

			// spaja 3 maske
			cv::addWeighted(thresholdedDepth, 0.7, knnResult, 0.3, 0.0, thresholdedDepth);
			cv::addWeighted(thresholdedDepth, 0.77, mog2Result, 0.33, 0.0, thresholdedDepth);

			//cv::resize(thresholdedDepth, thresholdedDepth, cv::Size(1280, 720));

			cv::threshold(thresholdedDepth, thresholdedDepth, 185, 255, cv::THRESH_BINARY);

			cv::resize(thresholdedDepth, thresholdedDepth, cv::Size(1280, 720));

			cv::cvtColor(thresholdedDepth, thresholdedDepth, cv::COLOR_GRAY2BGR);

			auto keypoints = getKeypointsOfCurrentFrame(thresholdedDepth, blobDetectorForPostProcessing);

			if (keypoints.size() < 1)
			{
				blobManager.noBlobsDetected();
			}
			else if (keypoints.size() > 0)
			{
				auto CURRENT_FILENAME = "this/is/fake/path/to.bag";
				cv::drawKeypoints(thresholdedDepth, keypoints, thresholdedDepth, cv::Scalar(255, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
				blobManager.matchBlobs(keypoints, thresholdedDepth, device.as<rs2::playback>().get_position(), CURRENT_FILENAME);
			}
			// Draw all currently tracked blobs paths
			blobManager.drawBlobPaths(thresholdedDepth);

			cv::imshow("Masked 3 times", thresholdedDepth);
			cv::waitKey(1);

			depthData.release();
			thresholdedDepth.release();
			rgbdData.release();
			knnResult.release();
			mog2Result.release();
		}
		/*
		else if (device.as<rs2::playback>())
		{
			std::queue<QueuedMat> depthQueue;
			std::queue<QueuedMat> KNNQueue;
			std::queue<QueuedMat> MOG2Queue;

			std::atomic<bool> isDepthMatCreated = false;
			cv::Mat depthDequeuedMat;

			std::atomic<bool> isColorMatCreated = false;
			cv::Mat knnDequeuedMat;
			cv::Mat mog2DequeuedMat;

			const int MAX_FRAMES_IN_Q = 2;

			std::thread processingThread([&]() {
				while (true)
				{
					rs2::frameset frameset = pipe->wait_for_frames();

					if (profile_changed(pipe->get_active_profile().get_streams(), profile.get_streams()))
					{
						//If the profile was changed, update the align object, and also get the new device's depth scale
						profile = pipe->get_active_profile();
						align_to = find_stream_to_align(profile.get_streams());
						align = rs2::align(align_to);
						depth_scale = get_depth_scale(profile.get_device());
					}

					rs2::frameset alignedFrameset = align.process(frameset);

					// Trying to get both other and aligned depth frames
					rs2::video_frame colorFrame = alignedFrameset.first(align_to);
					rs2::depth_frame depthFrame = alignedFrameset.get_depth_frame();

					if (!depthFrame || !colorFrame)
					{
						continue;
					}

					const int depth_w = depthFrame.get_width();
					const int depth_h = depthFrame.get_height();

					if (!isDepthMatCreated)
					{
						depthDequeuedMat = cv::Mat(depth_h, depth_w, CV_8UC1);
						isDepthMatCreated = true;
					}

					const int color_w = colorFrame.get_width();
					const int color_h = colorFrame.get_height();

					if (!isColorMatCreated)
					{
						knnDequeuedMat = cv::Mat(color_h, color_w, CV_8UC1);
						mog2DequeuedMat = cv::Mat(color_h, color_w, CV_8UC1);
						isColorMatCreated = true;
					}

					// -------------------------------------- Getting DEPTH data --------------------------------------

					cv::Mat depthData = depth_frame_to_meters(*pipe, depthFrame);
					// convert to 8 bit with scale factor 255./3 because 3m is clipping distance and there are 255 values
					depthData.convertTo(depthData, CV_8UC1, 255. / CURRENT_SCALE);

					cv::Mat thresholdedDepth;
					depthSubtraction.setDepthThreshold(thr);
					depthSubtraction.apply(depthData, thresholdedDepth);
					//cv::imshow("Thresholded", thresholdedDepth);

					QueuedMat depthMat;
					thresholdedDepth.copyTo(depthMat.img);
					thresholdedDepth.release();
					depthQueue.push(depthMat);

					// -------------------------------------- Getting RGBD data --------------------------------------

					cv::Mat colorData(cv::Size(color_w, color_h), CV_8UC3, (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
					cv::cvtColor(colorData, colorData, cv::COLOR_RGB2BGR);

					// split color image (rgb channels) to seperate b, g, r
					std::vector<cv::Mat> rgbdSrc;
					cv::split(colorData, rgbdSrc);
					colorData.release();

					// add 4th, depth, channel to bgr
					rgbdSrc.push_back(depthData);

					// merge all channels together (r + b + g + depth)
					cv::Mat rgbdData;
					cv::merge(rgbdSrc, rgbdData);


					// --------------------------------- Do some filter processing on color images -----------------------------------------------

					cv::resize(colorData, colorData, cv::Size(color_w / 4, depth_w / 4));

					// KNN
					cv::Mat knnResult;

					knnSub.apply(rgbdData, knnResult);

					knnResult.convertTo(knnResult, CV_8UC1);
					cv::resize(knnResult, knnResult, cv::Size(color_w, depth_w));

					QueuedMat knnMat;
					knnResult.copyTo(knnMat.img);
					knnResult.release();
					KNNQueue.push(knnMat);

					// MOG2
					cv::Mat mog2Result;

					mog2Sub.apply(rgbdData, mog2Result);

					mog2Result.convertTo(mog2Result, CV_8UC1);
					cv::resize(mog2Result, mog2Result, cv::Size(color_w, depth_w));

					QueuedMat mog2Mat;
					mog2Result.copyTo(mog2Mat.img);
					mog2Result.release();
					MOG2Queue.push(mog2Mat);

					// -------------------------------------- Release data --------------------------------------
					depthData.release();
					rgbdData.release();


					while (depthQueue.size() > MAX_FRAMES_IN_Q ||
							KNNQueue.size() > MAX_FRAMES_IN_Q ||
							MOG2Queue.size() > MAX_FRAMES_IN_Q)
					{
						if (depthQueue.size() > MAX_FRAMES_IN_Q)
						{
							depthQueue.pop();
						}
						if (KNNQueue.size() > MAX_FRAMES_IN_Q)
						{
							KNNQueue.pop();
						}
						if (MOG2Queue.size() > MAX_FRAMES_IN_Q)
						{
							MOG2Queue.pop();
						}
					}

				}
			});

			// Main thread
			while (true) {
				while (!depthQueue.empty())
				{
					if (!isDepthMatCreated)
					{
						continue;
					}

					depthQueue.front().img.copyTo(depthDequeuedMat);
					depthQueue.pop();

					//cv::imshow("depthThresh", depthDequeuedMat);
					//cv::waitKey(1);
				}

				while (!KNNQueue.empty())
				{
					if (!isColorMatCreated)
					{
						continue;
					}

					KNNQueue.front().img.copyTo(knnDequeuedMat);
					KNNQueue.pop();

					//cv::imshow("KNN color", knnDequeuedMat);
					//cv::waitKey(1);
				}

				while (!MOG2Queue.empty())
				{
					if (!isColorMatCreated)
					{
						continue;
					}

					MOG2Queue.front().img.copyTo(mog2DequeuedMat);
					MOG2Queue.pop();

					//cv::imshow("MOG2 color", mog2DequeuedMat);
					//cv::waitKey(1);
				}

				if (depthDequeuedMat.empty() || knnDequeuedMat.empty() || mog2DequeuedMat.empty())
				{
					continue;
				}

				// If i'm here I should have at least 1 frame of each mask.
				if (depthDequeuedMat.size() != knnDequeuedMat.size() || depthDequeuedMat.channels() != knnDequeuedMat.channels())
				{
					continue;
				}
				//cv::add(depthDequeuedMat, knnDequeuedMat, result);
				cv::addWeighted(depthDequeuedMat, 0.5, knnDequeuedMat, 0.5, 0.0, depthDequeuedMat);
				if (depthDequeuedMat.size() != mog2DequeuedMat.size() || depthDequeuedMat.channels() != mog2DequeuedMat.channels())
				{
					continue;
				}
				//cv::add(result, mog2DequeuedMat, result);
				cv::addWeighted(depthDequeuedMat, 0.5, mog2DequeuedMat, 0.5, 0.0, depthDequeuedMat);

				//cv::threshold(depthDequeuedMat, depthDequeuedMat, 750, 765, cv::THRESH_TOZERO);

				cv::imshow("Masked 3 times", depthDequeuedMat);
				cv::waitKey(1);

			}
			processingThread.detach();
			
		}
		*/
		else
		{
			frames = pipe->wait_for_frames();
			if (doesFrameContainHuman(frames, knn, blobDetectorForRecording))
			{
				device.as<rs2::recorder>().resume();
			}
		}
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
	const int RESIZE_SCALE_FACTOR = 4;

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
	cv::imshow("Color data recording", colorImage);
	cv::waitKey(1);

	return keypoints.size() > 0;
}

std::vector<cv::KeyPoint> getKeypointsOfCurrentFrame(cv::Mat &frame, cv::Ptr<cv::SimpleBlobDetector>& blobDetector)
{
	std::vector<cv::KeyPoint> ret;

	blobDetector->detect(frame, ret);

	return ret;
}


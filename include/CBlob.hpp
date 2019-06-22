#pragma once

#include <opencv2/opencv.hpp>

class Blob
{
public:
	Blob(int id, cv::Point2f currentPosition, uint64_t timestamp, std::string filepath);

	void getDistanceFromPoint(cv::Point2f &dstPoint, float&outDistance);

	void addNewLocation(cv::Point2f newLocation, uint64_t timestamp);

	int getId();

	cv::Point2f getLastPoint();
	float getLastTime();

	void drawPath(cv::Mat &img);

	void calculateAngles();

private:
	int m_id;
	
	std::string m_currentFilePath;

	// Timestamp is saved in uint64_t type with value in nanoseconds
	// Point of blob has value of cv::Point2f (x, y)
	std::vector<std::pair<uint64_t, cv::Point2f>> m_path;
	std::vector<double> m_orientation;
};


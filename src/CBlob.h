#pragma once

#include <opencv2/opencv.hpp>

class Blob
{
public:
	Blob(int id, cv::Point2f currentPosition);

	void getDistanceFromPoint(cv::Point2f &dstPoint, float &outDistance);

	void addNewLocation(cv::Point2f newLocation);

	int getId();

	cv::Point2f getLastPoint();

private:
	int m_id;
	float m_orientation;

	std::vector<cv::Point2f> m_path;
};


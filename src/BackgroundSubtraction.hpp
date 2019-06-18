#pragma once

#include <opencv2/opencv.hpp>


class BackgroundSubtraction
{
public:
	
	enum SubtractorType
	{
		DEPTH_THRESHOLD,
		MOG2,
		KNN
	};
	
	//BackgroundSubtraction();

	void subtract(cv::Mat& src, cv::Mat& dst);

	void setDepthThreshold(int threshold);

	BackgroundSubtraction(BackgroundSubtraction::SubtractorType type);

	BackgroundSubtraction(BackgroundSubtraction::SubtractorType type, int history, int varThreshold, bool shadows);

	void apply(cv::Mat& src, cv::Mat &dst);

private:
	bool m_backgroundSet;

	int m_threshold;

	SubtractorType m_type;

	cv::Ptr<cv::BackgroundSubtractor> m_bgSubtractor;
	cv::Mat m_background;
};

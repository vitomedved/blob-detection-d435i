
#ifndef BACKGROUND_SUBTRACTION_HPP
#define BACKGROUND_SUBTRACTION_HPP

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

	void setDepthBackgroundRefreshThreshold(long threshold);

	BackgroundSubtraction(BackgroundSubtraction::SubtractorType type);

	BackgroundSubtraction(BackgroundSubtraction::SubtractorType type, int history, int varThreshold, bool shadows);

	void apply(cv::Mat& src, cv::Mat &dst);

private:
	bool m_backgroundSet;
	bool m_shouldSetTimeForBackgroundRefresh;

	int m_threshold;

	long m_timeCheckpoint;
	long m_backgroundRefreshTimeTreshold;


	SubtractorType m_type;

	cv::Ptr<cv::BackgroundSubtractor> m_bgSubtractor;
	cv::Mat m_background;
};

#endif // BACKGROUND_SUBTRACTION_HPP

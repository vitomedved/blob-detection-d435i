#include "BackgroundSubtraction.hpp"

void BackgroundSubtraction::setDepthThreshold(int threshold)
{
	if (DEPTH_THRESHOLD != m_type)
	{
		return;
	}
	m_threshold = threshold;
}

BackgroundSubtraction::BackgroundSubtraction(BackgroundSubtraction::SubtractorType type)
{
	m_type = type;

	switch (type)
	{
	case DEPTH_THRESHOLD:
		m_bgSubtractor = nullptr;
		m_backgroundSet = false;
		m_threshold = 3;
		break;
	case MOG2:
		m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
		break;
	case KNN:
		m_bgSubtractor = cv::createBackgroundSubtractorKNN(500, 400.0, false);
		break;
	}
}

BackgroundSubtraction::BackgroundSubtraction(BackgroundSubtraction::SubtractorType type, int history, int varThreshold, bool shadows)
{
	m_type = type;

	switch (type)
	{
	case DEPTH_THRESHOLD:
		m_bgSubtractor = nullptr;
		m_backgroundSet = false;
		m_threshold = 3;
		break;
	case MOG2:
		m_bgSubtractor = cv::createBackgroundSubtractorMOG2(history, varThreshold, shadows);
		break;
	case KNN:
		m_bgSubtractor = cv::createBackgroundSubtractorKNN(history, varThreshold, shadows);
		break;
	}
}

void BackgroundSubtraction::apply(cv::Mat& src, cv::Mat& dst)
{
	if (DEPTH_THRESHOLD == m_type)
	{
		subtract(src, dst);
	}
	else
	{
		cv::Mat fgFrame;
		m_bgSubtractor->apply(src, fgFrame, 0.25);


		// Blur the foreground mask to reduce the effect of noise and false positives
		cv::blur(fgFrame, fgFrame, cv::Size(15, 15), cv::Point(-1, -1));

		// Remove the shadow parts and the noise
		cv::threshold(fgFrame, fgFrame, 128, 255, cv::THRESH_BINARY);

		fgFrame.convertTo(fgFrame, CV_8U);

		fgFrame.copyTo(dst);
	}
}

void BackgroundSubtraction::subtract(cv::Mat& src, cv::Mat& dst)
{
	if (DEPTH_THRESHOLD != m_type)
	{
		return;
	}

	if (m_backgroundSet)
	{
		int rows = src.rows;
		int cols = src.cols;

		dst.create(rows, cols, CV_8U);

		for (int i = 0; i < rows; i++)
		{
			uchar* srcRow = src.ptr<uchar>(i);
			uchar* dstRow = dst.ptr<uchar>(i);
			uchar* bgRow = m_background.ptr<uchar>(i);

			for (int j = 0; j < cols; j++)
			{
				auto temp = abs(srcRow[j] - bgRow[j]);
				if (temp < m_threshold)
				{
					dstRow[j] = 0;
				}
				else
				{
					dstRow[j] = 255;
				}
			}
		}
		
	}
	else
	{
		src.copyTo(m_background);

		dst.create(src.rows, src.cols, CV_8U);
		dst.setTo(cv::Scalar(0));

		m_backgroundSet = true;
	}
}

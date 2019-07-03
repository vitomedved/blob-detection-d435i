#pragma once

#include <opencv2/opencv.hpp>

#include "CBlob.h"

class BlobManager
{
public:
	BlobManager();
	//void updateBlobs(std::vector<cv::KeyPoint> blobKeypoints, cv::Mat paintMeLikeOneOfYourFrenchGirlsFrame);
	void matchBlobs(std::vector<cv::KeyPoint> blobKeypoints, cv::Mat &paintMeLikeOneOfYourFrenchGirlsFrame, uint64_t currentFrameTime, std::string currentFilename);

	void noBlobsDetected();

	void drawBlobPaths(cv::Mat &img);

private:

	bool m_areAllBlobsAssigned;

	int m_currentBlobId;

	std::vector<Blob> m_blobs;
	std::vector<Blob> m_currentlyTrackedBlobs;

	void registerNewBlobs(std::vector<cv::KeyPoint> blobKeypoints, uint64_t currentFrameTime, std::string currentFilename);

};



#include "CBlobManager.h"

BlobManager::BlobManager():
	m_areAllBlobsAssigned(true),
	m_currentBlobId(0)
{

}

void BlobManager::registerNewBlobs(std::vector<cv::KeyPoint> blobKeypoints)
{
	for (auto blobKeypoint : blobKeypoints)
	{
		Blob blob(m_currentBlobId++, blobKeypoint.pt);
		m_currentlyTrackedBlobs.push_back(blob);
	}
}

// If: there are no new blobs, return from function
// Else: if there are no currently tracked blobs, add all new blobs to currently tracked
// Else: Create matrix with size of (numberOfNewBlobs, numberOfCurrentlyTrackedBlobs) and iterate through it and set distances of all newBlob-trackedBlob pairs
//		 While number of assigned new blobs is lower than number of new blobs (which means I did not assign some of the new blobs)
//				or number of assigned currently tracked blobs is lower than number of currently tracked blobs (which means some of the currently tracked blobs have no pair)
//			iterate through matrix with pairs and try to find a match (find minimum distance and assign newBlob-existingBlob pair)
//		After this iteration, check whether no blob pairs are assigned and there are currently tracked blobs left with no pair -> move them to m_blobs
//		Also check if there are no blob pairs assigned and there are currently new blobs left with no pair -> add them to currently tracked blobs
//		ALSO if there are blob pairs assigned, mark row and column of those blobs and "match them" (update current blobs path)

void BlobManager::matchBlobs(std::vector<cv::KeyPoint> blobKeypoints, cv::Mat &paintMeLikeOneOfYourFrenchGirlsFrame)
{

	size_t numberOfNewBlobs = blobKeypoints.size();
	size_t numberOfCurrentlyTrackedBlobs = m_currentlyTrackedBlobs.size();

	if (0 == numberOfNewBlobs)
	{
		return;
	}

	if (0 == numberOfCurrentlyTrackedBlobs)
	{
		registerNewBlobs(blobKeypoints);
		return;
	}

	cv::Mat matchingMatrix(numberOfNewBlobs, numberOfCurrentlyTrackedBlobs, CV_32F);

	for (int i = 0; i < numberOfNewBlobs; i++)
	{
		for (int j = 0; j < numberOfCurrentlyTrackedBlobs; j++)
		{
			// TODO: if this doesnt work, change return type to float and assign value to matchingMatrix.at<...
			 m_currentlyTrackedBlobs[j].getDistanceFromPoint(blobKeypoints[i].pt, matchingMatrix.at<float>(i, j));
		}
	}

	int numberOfCurrentlyAssignedNewBlobs = 0;
	int numberOfCurrentlyAssignedExistingBlobs = 0;
	std::vector<int> assignedRows, assignedCols;

	while (numberOfCurrentlyAssignedNewBlobs < numberOfNewBlobs || numberOfCurrentlyAssignedExistingBlobs < numberOfCurrentlyTrackedBlobs)
	{
		// If here, at least one row and one column exist
		float currentMinDistance = matchingMatrix.at<float>(0, 0);
		int newBlobCurrentMatch = 0;
		int trackedBlobCurrentMatch = 0;
		bool blobAssigned = false;
		for (int i = 0; i < numberOfNewBlobs; i++)
		{
			if (std::find(assignedRows.begin(), assignedRows.end(), i) != assignedRows.end())
			{
				continue;
			}

			for (int j = 0; j < numberOfCurrentlyTrackedBlobs; j++)
			{
				if (std::find(assignedCols.begin(), assignedCols.end(), j) != assignedCols.end())
				{
					continue;
				}

				float currentDistance = matchingMatrix.at<float>(i, j);
				if (currentDistance <= currentMinDistance)
				{
					currentMinDistance = currentDistance;
					newBlobCurrentMatch = i;
					trackedBlobCurrentMatch = j;
					blobAssigned = true;
				}
			}

			// ako sam out of range, tj ako novi blob nema par
		}

		// If I enter this if, I know that there are unpaired currently selected blobs with new blobs.
		// Because there are more currentlyTrackedBlobs I should move those to already tracked blobs @ref m_blobs
		if (numberOfCurrentlyAssignedExistingBlobs < numberOfCurrentlyTrackedBlobs && !blobAssigned)
		{
			// For all blobs check if index is not in tracked, move to already tracked because this means that currentlyTracked blob has no new pair (went out of frame I guess)
			for (int x = 0; x < numberOfCurrentlyTrackedBlobs; x++)
			{
				if (std::find(assignedCols.begin(), assignedCols.end(), x) == assignedCols.end())
				{
					// blob on x position has no pair therefore move him to m_blobs
					m_blobs.push_back(m_currentlyTrackedBlobs[x]);
					// erase blob from currently tracked blobs
					m_currentlyTrackedBlobs.erase(m_currentlyTrackedBlobs.begin() + x);

					numberOfCurrentlyAssignedExistingBlobs++;
				}
			}
		}

		// Logic for pairing new blobs with currently tracked ones
		if (numberOfCurrentlyAssignedNewBlobs < numberOfNewBlobs)
		{
			if (blobAssigned)
			{
				// If you are here, you know that blob has found a match
				// Which means that I know index of currently paired blobs
				// Because I know that, I will remove new blob keypoint from given new blobKeypoints
				// This will help me with adding all unpaired new blobs to currentlyTrackedBlobs.
				m_currentlyTrackedBlobs[trackedBlobCurrentMatch].addNewLocation(blobKeypoints[newBlobCurrentMatch].pt);
				assignedRows.push_back(newBlobCurrentMatch);
				assignedCols.push_back(trackedBlobCurrentMatch);

				blobKeypoints.erase(blobKeypoints.begin() + newBlobCurrentMatch);

				numberOfCurrentlyAssignedExistingBlobs++;
				numberOfCurrentlyAssignedNewBlobs++;
			}
			else
			{
				// Because I know that no new blobs are paired, there are only unpaired new blobs left
				// which means I will add 1 by 1 new blob to currently tracked blobs.
				Blob newBlob(m_currentBlobId++, blobKeypoints[0].pt);
				m_currentlyTrackedBlobs.push_back(newBlob);

				numberOfCurrentlyAssignedNewBlobs++;
			}
		}
	}

	// TODO: move to main
	for (auto blob : m_currentlyTrackedBlobs)
	{
		cv::putText(paintMeLikeOneOfYourFrenchGirlsFrame, std::to_string(blob.getId()), blob.getLastPoint(), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
	}
}

void BlobManager::noBlobsDetected()
{
	while (m_currentlyTrackedBlobs.size() > 0)
	{
		m_blobs.push_back(m_currentlyTrackedBlobs[0]);
		m_currentlyTrackedBlobs.erase(m_currentlyTrackedBlobs.begin() + 0);
	}
}

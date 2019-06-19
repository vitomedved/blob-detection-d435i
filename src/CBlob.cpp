
#include "CBlob.h"


Blob::Blob(int id, cv::Point2f currentPosition):
	m_orientation(0.0)
{
	m_id = id;
	m_path.push_back(currentPosition);
}

void Blob::getDistanceFromPoint(cv::Point2f& dstPoint, float& outDistance)
{
	cv::Point2f diff = dstPoint - m_path[m_path.size() - 1];

	outDistance = cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

void Blob::addNewLocation(cv::Point2f newLocation)
{
	m_path.push_back(newLocation);
}

int Blob::getId()
{
	return m_id;
}

cv::Point2f Blob::getLastPoint()
{
	return m_path[m_path.size() - 1];
}

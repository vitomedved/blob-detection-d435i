
#include "CBlob.hpp"


Blob::Blob(int id, cv::Point2f currentPosition, uint64_t timestamp, std::string filepath)
{
	m_currentFilePath = filepath;
	m_id = id;
	addNewLocation(currentPosition, timestamp);
}

void Blob::getDistanceFromPoint(cv::Point2f& dstPoint, float& outDistance)
{
	cv::Point2f diff = dstPoint - m_path[m_path.size() - 1].second;

	outDistance = cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

void Blob::addNewLocation(cv::Point2f newLocation, uint64_t timestamp)
{
	std::pair<uint64_t, cv::Point2f> position;
	
	position.first = timestamp;
	position.second = newLocation;

	m_path.push_back(position);
}

int Blob::getId()
{
	return m_id;
}

cv::Point2f Blob::getLastPoint()
{
	return m_path[m_path.size() - 1].second;
}

float Blob::getLastTime()
{
	auto time = m_path[m_path.size() - 1].first;
	float ret = time * 0.000000001;
	return ret;
}

void Blob::drawPath(cv::Mat &img)
{
	for (int i = 1; i < m_path.size(); i++)
	{
		cv::line(img, m_path[i - 1].second, m_path[i].second, cv::Scalar(0, 255, 0), 2);
	}
}

#define PI 3.14

#define CURRENT_X m_path[i].second.x
#define CURRENT_Y m_path[i].second.y
#define NEXT_X m_path[i + 1].second.x
#define NEXT_Y m_path[i + 1].second.y

void Blob::calculateAngles()
{
	for (int i = 0; i < m_path.size() - 1; i++)
	{
		float xDiff = NEXT_X - CURRENT_X;
		float yDiff = NEXT_Y - CURRENT_Y;
		float angle = std::atan2(yDiff, xDiff) * 180.0 / PI;
		
		m_orientation.push_back(angle);
	}
	if (m_orientation.size() > 1)
	{
		m_orientation.push_back(m_orientation[m_orientation.size() - 1]);
	}
}


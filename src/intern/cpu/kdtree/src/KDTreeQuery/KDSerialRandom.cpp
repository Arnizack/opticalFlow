#include "KDSerialRandom.hpp"
#include<math.h>
#include<random>
namespace kdtree
{
	KDSerialRandom::KDSerialRandom(uint32_t weight, uint32_t height, uint8_t sampleCount)
	{
		Weight = weight;
		Height = height;
		uint32_t count = calculateRandomNumberCount(weight, height, sampleCount);
		createRandomNumbers(count);

	}
	float KDSerialRandom::Urand(uint32_t nodeIdx, uint32_t pointX, uint32_t pointY)
	{
		
		uint32_t inputNodeIdx = pointX * Weight + pointY;
		uint32_t inputNodeCount = RandomNumbers.size();
		uint32_t inputNodeSeed = inputNodeCount * RandomNumbers[inputNodeIdx % inputNodeCount];
		return RandomNumbers[(inputNodeSeed + nodeIdx) % inputNodeCount];
		//return 0.5f;
		//return (float)rand() / RAND_MAX;

	}
	void KDSerialRandom::createRandomNumbers(int count)
	{
		RandomNumbers = std::vector<float>(count);
		for (int i = 0; i < count; i++)
		{
			RandomNumbers[i] = (float)rand()/RAND_MAX;

		}
	}
	uint32_t KDSerialRandom::calculateRandomNumberCount(uint32_t weight, uint32_t height, uint8_t sampleCount)
	{
		double countd = 63 + (log2(weight * height) - log2(sampleCount)) * 32;
		countd *= 6;
		//make count power of 2
		uint32_t count = static_cast<uint32_t>(pow(2,ceil(log2(countd))));

		return count;
	}
}

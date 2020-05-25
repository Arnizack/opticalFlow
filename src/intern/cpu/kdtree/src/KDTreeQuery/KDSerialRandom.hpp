#pragma once
#include"IKDRandom.hpp"
#include <stdint.h>
#include <vector>
namespace kdtree
{
	class KDSerialRandom : public IKDRandom
	{

	public:
		KDSerialRandom(uint32_t weight, uint32_t height, uint8_t sampleCount);
		//a uniform random Variable
		//pointX is the X Coordinate of the input in the KD Query
		//pointY is the Y Coordinate of the input in the KD Query
		float Urand(uint32_t nodeIdx, uint32_t pointX, uint32_t pointY) override;
	private:
		uint32_t Weight;
		uint32_t Height;

		std::vector<float> RandomNumbers;

		void createRandomNumbers(int count);

		uint32_t calculateRandomNumberCount(uint32_t weight, uint32_t height, uint8_t sampleCount);
		

	};

}
                
#include<gtest/gtest.h>
#include"KDTreeQuery/KDTreeMath.hpp"

TEST(KDTree, CDF1)
{

	std::vector<std::pair<float, float>> InputExpected
		= { {0,0.5},{-3,0} ,{3,1} };

	for (const auto& pair : InputExpected)
	{
		float actual = kdtree::CdfApprox(pair.first);
		float expected = pair.second;
		EXPECT_NEAR(expected, actual, 0.01);
	}
}

TEST(KDTree, Distance1)
{
	kdtree::KdTreeVal valA(3, 7, -4, 2, 0);
	kdtree::KdTreeVal valB(1, 5, 3, 7, 5);

	float expected = 4+4+49+25+25;
	float actual = kdtree::DistanceSquared(valA, valB);
	EXPECT_NEAR(expected, actual,0.01);
}

TEST(KDTree, Floor1)
{
	uint8_t expected = 2;
	uint8_t actual = kdtree::Floor(2.9);
	EXPECT_NEAR(expected, actual, 0.01);

}

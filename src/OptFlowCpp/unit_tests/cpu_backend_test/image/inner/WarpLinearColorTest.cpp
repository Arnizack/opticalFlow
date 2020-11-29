#include"cpu_backend/image/inner/WarpLinearColorImage.h"
#include"gtest/gtest.h"

TEST(WarpLinearColorImageTest, test1)
{
	float image[12]
	{
		1 , 2,
		3, 4,

		5, 6,
		7, 8,

		9, 10,
		11, 12
	};

	double flow[8]
	{
		0.5 , 0.5,
		-0.5 , -0.5,

		0.5 , -0.5,
		0.5 , -0.5
	};

	float actual_result[12] = { 0 };

	cpu_backend::WarpLinearColorImage(actual_result, image, flow, 2, 2,3);

	float expected_result[12] = {
	2.5,2.5,2.5,2.5,
	6.5,6.5,6.5,6.5,
	10.5,10.5,10.5,10.5
	};
	for (int i = 0; i < 12; i++)
		EXPECT_NEAR(expected_result[i], actual_result[i], 0.00001);
}
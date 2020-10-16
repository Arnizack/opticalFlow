#include"gtest/gtest.h"
#include"cpu_backend/image/inner/convolution2D.h"

namespace cpu_backend
{
	namespace testing
	{
		TEST(Convolute2D, Convolute2DAtTest)
		{
			double grid[12] = { 
				0,1,2,
				3,4,5,
				6,7,8,
				9,10,11 
			};
			double kernel[15] = {
				0,1,2,
				3,4,5,
				6,7,8,
				9,10,11,
				12,13,14 
			};
			const int kernel_width = 3;
			const int kernel_height = 5;
			const int width = 3;
			const int height = 4;

			double actual = Convolute2DAt<double, kernel_width, kernel_height>(
				1,2,grid,width,height,kernel
				);
			
			double expected = 506;

			EXPECT_NEAR(actual, expected, 0.002);

		}
	}
}
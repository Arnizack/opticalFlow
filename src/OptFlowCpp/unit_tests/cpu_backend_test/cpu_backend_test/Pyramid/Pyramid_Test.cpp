#include "cpu_backend/pyramid/Pyramid.h"

#include "gtest/gtest.h"

namespace cpu_backend
{
	namespace testing
	{
		template<typename T>
		void TestForType()
		{
			Pyramid<T> test(0);

			for (int i = 0; i < 10; i++)
			{
				test.AddLevel(i);
			}

			for (int i = 0; i < 10; i++)
			{
				if (i == 9)
				{
					EXPECT_EQ(test.IsEndLevel(), true);
				}
				else
				{
					EXPECT_EQ(test.IsEndLevel(), false);
				}

				EXPECT_EQ(test.NextLevel(), i);
			}
		}

		TEST(PyramidTest, Int)
		{
			//TestForType<int>();
		}

		TEST(PyramidTest, Float)
		{
			//TestForType<float>();
		}

		TEST(PyramidTest, Double)
		{
			//TestForType<double>();
		}
	}
}
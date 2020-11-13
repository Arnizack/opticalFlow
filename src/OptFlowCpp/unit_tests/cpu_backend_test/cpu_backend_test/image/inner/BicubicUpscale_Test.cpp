#include <gtest/gtest.h>

#include "cpu_backend/image/inner/BicubicScale.h"
#include <vector>

namespace cpu_backend
{
	namespace testing
	{
		template<typename T>
		void Test2DForTyp()
		{
			const size_t width = 2;
			const size_t height = 2;

			const size_t size = width * height;

			T in[size];

			for (size_t i = 0; i < size; i++)
			{
				in[i] = (i + 1) * 10;
			}

			const size_t dst_width = 4;
			const size_t dst_height = 4;

			const size_t dst_size = dst_width * dst_height;

			T out[dst_size];
			cpu_backend::_inner::Bicubic2DScale<T>(in, out, width, height, dst_width, dst_height);
			

			T comp[dst_size] = { 3.53485, 8.44025, 13.2138, 11.1484,
								11.6675, 21.9659, 29.0057, 22.8955,
								21.2146, 36.0455, 43.0853, 32.4426,
								18.7619, 30.8963, 35.6699, 26.3754 };

			T temp;

			for (size_t i = 0; i < dst_width * dst_height; i++)
			{
				temp = comp[i] - out[i];
				if (temp < 0)
					temp *= -1;
				EXPECT_LE(temp, 1);
				//ASSERT_DOUBLE_EQ(comp[i], (*out)[i]);
			}
		}

		template<typename T>
		void Test3DForTyp()
		{
			const size_t width = 2;
			const size_t height = 2;
			const size_t depth = 3;
			const size_t wh = width * height;
			const size_t size = wh * depth;

			T in[size];

			for (size_t z = 0; z < depth; z++)
			{
				for (size_t i = 0; i < wh; i++)
				{
					in[i + z * wh] = (i + 1) * 10;
				}
			}

			const size_t dst_width = 4;
			const size_t dst_height = 4;
			const size_t dst_size = dst_width * dst_height * depth;

			T out[dst_size];

			cpu_backend::_inner::BicubicFlowScale<T>(in, out, width, height, dst_width, dst_height);

			T comp[dst_width * dst_height] = { 3.53485, 8.44025, 13.2138, 11.1484,
												11.6675, 21.9659, 29.0057, 22.8955,
												21.2146, 36.0455, 43.0853, 32.4426,
												18.7619, 30.8963, 35.6699, 26.3754 };

			T temp;
			const size_t dst_wh = dst_height * dst_width;

			for (size_t z = 0; z < depth; z++)
			{
				for (size_t i = 0; i < dst_wh; i++)
				{
					temp = comp[i] - out[i + z * dst_wh];
					if (temp < 0)
						temp *= -1;
					EXPECT_LE(temp, 1);
					//EXPECT_EQ((*out)[i + (z * dst_wh)], comp[i]);
				}
			}
		}

		TEST(BicubicScalerTest, Int2D)
		{
			Test2DForTyp<int>();
		}

		TEST(BicubicScalerTest, Float2D)
		{
			Test2DForTyp<float>();
		}

		TEST(BicubicScalerTest, double2D)
		{
			Test2DForTyp<double>();
		}

		TEST(BicubicScalerTest, Int3D)
		{
			Test3DForTyp<int>();
		}

		TEST(BicubicScalerTest, Float3D)
		{
			Test3DForTyp<float>();
		}

		TEST(BicubicScalerTest, double3D)
		{
			Test3DForTyp<double>();
		}
	}
}
#include <gtest/gtest.h>

#include "cpu_backend/ArrayScaler.h"
#include "cpu_backend/ArrayFactory.h"

namespace cpu_backend
{
	namespace testing
	{
		template<typename T>
		void Test2DForTyp()
		{
			std::shared_ptr<ArrayFactory<T, 2>> arr_factory = std::make_shared<ArrayFactory<T, 2>>(ArrayFactory<T, 2>());

			ArrayScaler<T, 2> scaler(arr_factory);

			const size_t width = 2;
			const size_t height = 2;

			auto in = std::dynamic_pointer_cast<Array<T,2>>(arr_factory->Zeros({ width, height}));

			for (size_t i = 0; i < width * height; i++)
			{
				(*in)[i] = (i + 1) * 10;
			}

			const size_t dst_width = 4;
			const size_t dst_height = 4;

			auto out = std::dynamic_pointer_cast<Array<T, 2>>( scaler.Scale(in, dst_width, dst_height) );

			EXPECT_EQ(dst_width, out->Shape[0]);
			EXPECT_EQ(dst_height, out->Shape[1]);

			T comp[dst_width * dst_height] = { 3.53485, 8.44025, 13.2138, 11.1484,
												11.6675, 21.9659, 29.0057, 22.8955,
												21.2146, 36.0455, 43.0853, 32.4426,
												18.7619, 30.8963, 35.6699, 26.3754 };

			T temp;

			for (size_t i = 0; i < dst_width * dst_height; i++)
			{
				temp = comp[i] - (*out)[i];
				if (temp < 0)
					temp *= -1;
				EXPECT_LE(temp, 1);
				//ASSERT_DOUBLE_EQ(comp[i], (*out)[i]);
			}
		}

		template<typename T>
		void Test3DForTyp()
		{
			std::shared_ptr<ArrayFactory<T, 3>> arr_factory = std::make_shared<ArrayFactory<T, 3>>(ArrayFactory<T, 3>());

			ArrayScaler<T, 3> scaler(arr_factory);

			const size_t width = 2;
			const size_t height = 2;
			const size_t depth = 3;

			auto in = std::dynamic_pointer_cast<Array<T, 3>>(arr_factory->Zeros({ width, height, depth }));

			const size_t wh = width * height;

			for (size_t z = 0; z < depth; z++)
			{
				for (size_t i = 0; i < wh; i++)
				{
					(*in)[i + z * wh] = (i + 1) * 10;
				}
			}

			const size_t dst_width = 4;
			const size_t dst_height = 4;

			auto out = std::dynamic_pointer_cast<Array<T, 3>>(scaler.Scale(in, dst_width, dst_height));

			EXPECT_EQ(dst_width, out->Shape[0]);
			EXPECT_EQ(dst_height, out->Shape[1]);
			EXPECT_EQ(depth, out->Shape[2]);

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
					temp = comp[i] - (*out)[i + z * dst_wh];
					if (temp < 0)
						temp *= -1;
					EXPECT_LE(temp, 1);
					//EXPECT_EQ((*out)[i + (z * dst_wh)], comp[i]);
				}
			}
		}

		TEST(ScalerTest, Int2D)
		{
			Test2DForTyp<int>();
		}

		TEST(ScalerTest, Float2D)
		{
			Test2DForTyp<float>();
		}

		TEST(ScalerTest, double2D)
		{
			Test2DForTyp<double>();
		}

		TEST(ScalerTest, Int3D)
		{
			Test3DForTyp<int>();
		}

		TEST(ScalerTest, Float3D)
		{
			Test3DForTyp<float>();
		}

		TEST(ScalerTest, double3D)
		{
			Test3DForTyp<double>();
		}
	}
}
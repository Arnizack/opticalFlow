#include "gtest/gtest.h"

#include "cpu_backend/flow/FlowFieldScaler.h"
#include "cpu_backend/ArrayFactory.h"
#include "cpu_backend/image/inner/BicubicScale.h"
#include "cpu_backend/image/inner/DownScaleGaussianGrayScale.h"

namespace cpu_backend
{
	namespace testing
	{
		TEST(FlowFieldScaler, Test)
		{
			const size_t width = 5;
			const size_t height = 5;
			const size_t depth = 2;
			const size_t size = width * height * depth;

			double arr_in[size];
			

			for (int i = 0; i < size; i++)
			{
				arr_in[i] = i;
			}

			ArrayFactory<double, 3> arr_factory;

			auto input = std::dynamic_pointer_cast<Array<double, 3>>(arr_factory.CreateFromSource(arr_in, { 5,5,2 }) );

			FlowFieldScaler scaler(std::make_shared<ArrayFactory<double, 3>>(arr_factory));

			int new_width = 10;
			int new_height = 10;

			auto out = std::dynamic_pointer_cast<Array<double, 3>>( scaler.Scale(input, new_width, new_height) );

			size_t new_size = out->Shape[0] * out->Shape[1] * out->Shape[2];

			EXPECT_EQ(out->Shape[2], new_width);
			EXPECT_EQ(out->Shape[1], new_height);

			double arr_out1[200];
			double arr_temp[size];

			for (int i = 0; i < size; i++)
			{
				arr_temp[i] = i * 2.0;

				EXPECT_EQ((*input)[i], i);
			}

			_inner::BicubicGrayScale<double>(arr_temp, arr_out1, width, height, new_width, new_height);
			_inner::BicubicGrayScale<double>(arr_temp + 25, arr_out1 + 100, width, height, new_width, new_height);

			for (int i = 0; i < new_size; i++)
			{
				EXPECT_EQ((*out)[i], arr_out1[i]);
			}

			new_width = 2;
			new_height = 2;

			out = std::dynamic_pointer_cast<Array<double, 3>>( scaler.Scale(input, new_width, new_height) );

			new_size = out->Shape[0] * out->Shape[1] * out->Shape[2];

			EXPECT_EQ(out->Shape[2], new_width);
			EXPECT_EQ(out->Shape[1], new_height);

			double arr_out2[8];

			for (int i = 0; i < size; i++)
			{
				arr_temp[i] = i * 0.4;
			}

			_inner::DownScaleGaussianGrayScale<double>(arr_temp, width, height, new_width, new_height, arr_out2);
			_inner::DownScaleGaussianGrayScale<double>(arr_temp + 25, width, height, new_width, new_height, arr_out2 + 4);

			for (int i = 0; i < new_size; i++)
			{
				EXPECT_EQ((*out)[i], arr_out2[i]);
			}
		}
	}
}
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

			auto input = arr_factory.CreateFromSource(arr_in, { 5,5,2 });

			FlowFieldScaler scaler(std::make_shared<ArrayFactory<double, 3>>(arr_factory));

			auto out = std::dynamic_pointer_cast<Array<double, 3>>( scaler.Scale(input, 10, 10) );

			size_t new_size = out->Shape[0] * out->Shape[1] * out->Shape[2];

			EXPECT_EQ(out->Shape[0], 10);
			EXPECT_EQ(out->Shape[1], 10);

			double arr_out1[200];
			double arr_temp[size];

			for (int i = 0; i < size; i++)
			{
				arr_temp[i] = i * 0.5;
			}

			_inner::BicubicGrayScale<double>(arr_temp, arr_out1, 5, 5, 10, 10);
			_inner::BicubicGrayScale<double>(arr_temp + 25, arr_out1 + 100, 5, 5, 10, 10);

			for (int i = 0; i < new_size; i++)
			{
				EXPECT_EQ((*out)[i], arr_out1[i]);
			}

			out = std::dynamic_pointer_cast<Array<double, 3>>( scaler.Scale(input, 2, 2) );

			new_size = out->Shape[0] * out->Shape[1] * out->Shape[2];

			EXPECT_EQ(out->Shape[0], 2);
			EXPECT_EQ(out->Shape[1], 2);

			double arr_out2[8];

			for (int i = 0; i < size; i++)
			{
				arr_temp[i] = i * 2.0;
			}

			_inner::DownScaleGaussianGrayScale<double>(arr_temp, 5, 5, 2, 2, arr_out2);
			_inner::DownScaleGaussianGrayScale<double>(arr_temp + 25, 5, 5, 2, 2, arr_out2 + 4);

			for (int i = 0; i < new_size; i++)
			{
				EXPECT_EQ((*out)[i], arr_out2[i]);
			}
		}
	}
}
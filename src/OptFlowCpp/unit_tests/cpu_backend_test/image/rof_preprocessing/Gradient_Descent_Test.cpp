#include"gtest/gtest.h"

#include <iostream>
#include "cpu_backend/image/ROFPreProcessing/ROFPreProcessing.h"

namespace cpu_backend
{
	namespace testing
	{
		TEST(ROFPreProcessing, Gradient_Descent_Test)
		{
			auto settings = std::make_shared<ROFPreProcessingSettings>();

			auto arr_factory2D = std::make_shared<ArrayFactory<float, 2>>(ArrayFactory<float, 2>());
			auto arr_factory3D = std::make_shared<ArrayFactory<float, 3>>(ArrayFactory<float, 3>());

			auto statistics = std::make_shared<Statistics<float>>(Statistics<float>());

			auto arith_vector = std::make_shared<ArithmeticVector<float, 2>>(ArithmeticVector<float, 2>(arr_factory2D));

			auto rof_preprocessing = ROFPreProcessing(settings, arr_factory2D, arr_factory3D, statistics, arith_vector);

			const size_t size = 25;
			float arr[size];

			for (int i = 0; i < 25; i++)
			{
				arr[i] = i + 1;
			}

			float out_arr[size];

			rof_preprocessing.pi_gradient_descend(arr, 1, 5, 5, out_arr);

			for (int i = 0; i < 25; i++)
			{
				if (i % 5 == 0)
					std::cout << '\n';

				std::cout << arr[i] << ' ';
			}
		}
	}
}
#pragma once

#include "cpu_backend/Statistics.h"
#include "cpu_backend/Array.h"

#include "gtest/gtest.h"

namespace cpu_backend
{
	namespace testing
	{
		template<class InnerTyp>
		void StatisticsTestForTyp()
		{
			const int size = 4;
			const int dim = 3;
			std::array<const size_t, dim> shape = { 2,2,1 };
			InnerTyp arr[size];
			for (auto i = 0; i < size; i++)
			{
				arr[i] = i + 1;
			}

			Array<InnerTyp, dim> obj(shape, arr);

			auto ptr_array = std::make_shared<Array<InnerTyp, dim>>(obj);

			auto ptr_cont = std::static_pointer_cast<Container<InnerTyp>>(ptr_array);

			Statistics<InnerTyp> stats;

			double sum = stats.Sum(ptr_cont);
			double sum_compare = 0;

			for (int i = 0; i < size; i++)
			{
				sum_compare += arr[i];
			}

			EXPECT_EQ(sum, sum_compare);

			double mean = stats.Mean(ptr_cont);

			double mean_comp = sum_compare / size;

			mean_comp = 2.5;

			EXPECT_EQ(mean, mean_comp);

			double std_dev = stats.StandardDeviation(ptr_cont);

			double sum_std_dev = 0;

			for (int i = 0; i < size; i++)
			{
				sum_std_dev += (arr[i] - mean) * (arr[i] - mean);
			}

			double std_dev_comp_temp = sum_std_dev / size;

			double std_dev_comp = sqrt(std_dev_comp_temp);

			EXPECT_EQ(std_dev_comp_temp, 1.25);

			EXPECT_EQ(std_dev, std_dev_comp);
		}

		TEST(StatisticsTest, Int)
		{
			StatisticsTestForTyp<int>();
		}

		TEST(StatisticsTest, Double)
		{
			StatisticsTestForTyp<double>();
		}

		TEST(StatisticsTest, Float)
		{
			StatisticsTestForTyp<float>();
		}
	}
}
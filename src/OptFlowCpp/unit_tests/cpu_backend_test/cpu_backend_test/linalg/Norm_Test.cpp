#include "cpu_backend/linalg/Norm.h"

#include <array>
#include <gtest/gtest.h>

namespace cpu_backend
{
	template<typename T>
	T compareNorm(const T* const arr, const size_t size)
	{
		T norm_value = 0;

		for (size_t i = 0; i < size; i++)
		{
			norm_value += arr[i] * arr[i];
		}

		return (T)sqrt((double)norm_value);
	}

	template<typename T>
	void TestNormForTyp()
	{
		const int dim = 2;
		const int size = 10;

		std::array<const size_t, dim> shape = { 5,2 };

		T arr[size];

		for (int i = 0; i < size; i++)
		{
			arr[i] = i;
		}

		Array<T, dim> in_obj(shape, arr);

		for (int i = 0; i < size; i++)
		{
			EXPECT_EQ(arr[i], in_obj[i]);
		}

		std::shared_ptr<Array<T, dim>> in = std::make_shared<Array<T, dim>>(in_obj);

		Norm<T, dim> norm;

		std::shared_ptr<T> out = norm.Apply(in);

		T norm_value = compareNorm(arr, size);

		T out_value = *out;

		EXPECT_EQ(norm_value, out_value);
	}

	

	TEST(NormTest, Double)
	{
		TestNormForTyp<double>();
	}

	TEST(NormTest, Int)
	{
		TestNormForTyp<int>();
	}
}
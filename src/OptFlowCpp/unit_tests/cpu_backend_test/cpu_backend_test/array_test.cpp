#include "cpu_backend\Array.h"
#include "cpu_backend/ArrayFactory.h"

#include <array>
#include"gtest/gtest.h"

namespace cpu_backend
{
	namespace testing
	{

		TEST(CpuArrayTest, ShapeSizeSrcConstructor)
		{
			const int size = 4;
			const int dim = 3;
			std::array<const size_t, dim> shape = { 2,2,1 };
			int arr[size];
			for (auto i = 0; i < size; i++)
			{
				arr[i] = i + 1;
			}

			Array<int, dim> obj(shape, size, arr);

			std::vector<int> dst(obj.Size());

			std::vector<int> ptr_arr(size);
			obj.CopyDataTo(ptr_arr.data());

			std::array<const size_t, dim> shape_copy = obj.Shape;

			EXPECT_EQ(obj.Size(), size);

			for (auto i = 0; i < dim; i++)
			{
				EXPECT_EQ(shape[i], shape_copy[i]);
			}

			for (auto i = 0; i < size; i++)
			{
				EXPECT_EQ(arr[i], ptr_arr[i]);
				EXPECT_EQ(arr[i], obj[i]);
			}
		}

		TEST(CpuArrayTest, ShapeSrcConstructor)
		{
			const int size = 4;
			const int dim = 3;
			std::array<const size_t, dim> shape = { 2,2,1 };
			int arr[size];
			for (auto i = 0; i < size; i++)
			{
				arr[i] = i + 1;
			}

			Array<int, dim> obj(shape, arr);

			for (auto i = 0; i < size; i++)
			{
				EXPECT_EQ(arr[i], obj[i]);
			}
		}

		TEST(CpuArrayFactoryTest, Full)
		{
			const int value = 12;
			const int size = 4;
			const int dim = 3;
			std::array<const size_t, dim> shape = { 2,2,1 };

			cpu_backend::ArrayFactory<int, dim> factory;

			auto test_obj = factory.Full(value, shape);

			EXPECT_EQ(size, test_obj.get()->Size());

			auto copied_shape = test_obj.get()->Shape;
			for (int i = 0; i < dim; i++)
			{
				EXPECT_EQ(shape[i], copied_shape[i]);
			}

			int copied_data[size];
			test_obj.get()->CopyDataTo(copied_data);
			for (int i = 0; i < size; i++)
			{
				EXPECT_EQ(value, copied_data[i]);
			}
		}

		TEST(CpuArrayFactoryTest, Zero)
		{
			const int size = 4;
			const int dim = 3;
			std::array<const size_t, dim> shape = { 2,2,1 };

			cpu_backend::ArrayFactory<int, dim> factory;

			auto test_obj = factory.Zeros(shape);

			EXPECT_EQ(size, test_obj.get()->Size());

			auto copied_shape = test_obj.get()->Shape;
			for (int i = 0; i < dim; i++)
			{
				EXPECT_EQ(shape[i], copied_shape[i]);
			}

			int copied_data[size];
			test_obj.get()->CopyDataTo(copied_data);
			for (int i = 0; i < size; i++)
			{
				EXPECT_EQ(0, copied_data[i]);
			}
		}
	}
}
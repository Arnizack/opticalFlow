#include "cpu_backend\Array.h"
#include "core\IContainer.h"
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

			int* ptr_arr = dst.data();
			obj.CopyDataTo(ptr_arr);

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
	}
}
#include "pch.h"
#include "..\cpu_backend\Array.h"
#include "..\core\IContainer.h"
#include "..\cpu_backend\ArrayFactory.h"
#include <array>


namespace cpu_utext
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

		cpu::Array<int, dim> obj(shape, size, arr);

		int* ptr_arr;
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

		cpu::Array<int, dim> obj(shape, arr);

		for (auto i = 0; i < size; i++)
		{
			EXPECT_EQ(arr[i], obj[i]);
		}
	}

	TEST(CpuArrayFactoryTest, Zeros)
	{
		const int dim = 2;
		const int size = 4;
		std::array<const size_t, dim> shape = { 2,2 };
		cpu::ArrayFactory<int, dim> factory;
		std::shared_ptr<core::IArray<int, dim>> test_obj = factory.Zeros(shape);

		auto shape_copied = test_obj.get()->Shape;
		for (auto i = 0; i < dim; i++)
		{
			EXPECT_EQ(shape[i], shape_copied[i]);
		}

		int* ptr;
		test_obj.get()->CopyDataTo(ptr);

		for (auto i = 0; i < size; i++)
		{
			EXPECT_EQ(ptr[i], 0);
		}
	}

	TEST(CpuArrayFactoryTest, Full)
	{
		const int fill_value = 22;
		const int dim = 2;
		const int size = 4;
		std::array<const size_t, dim> shape = { 2,2 };
		cpu::ArrayFactory<int, dim> factory;
		std::shared_ptr<core::IArray<int, dim>> test_obj = factory.Full(fill_value, shape);

		auto shape_copied = test_obj.get()->Shape;
		for (auto i = 0; i < dim; i++)
		{
			EXPECT_EQ(shape[i], shape_copied[i]);
		}

		int* ptr;
		test_obj.get()->CopyDataTo(ptr);

		for (auto i = 0; i < size; i++)
		{
			EXPECT_EQ(ptr[i], fill_value);
		}
	}
}
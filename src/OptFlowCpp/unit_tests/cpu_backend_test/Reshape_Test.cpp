#pragma once

#include "cpu_backend/Reshaper.h"
#include "cpu_backend/Container.h"

#include "gtest/gtest.h"

namespace cpu_backend
{
	namespace testing
	{
		template<typename T, size_t DimCount>
		std::shared_ptr<Array<T, 2>> setup()
		{
			const int size = 10;
			const int dim = 2;
			std::array<const size_t, dim> shape_in = { 5,2 };

			T arr[size] = { 0,1,2,3,4,5,6,7,8,9 };
			Array<T, dim> in_obj(shape_in, arr);
			return std::make_shared<Array<T, dim>>(in_obj);
		}

		template<typename T, size_t DimCount>
		void control(std::shared_ptr<core::IArray<T, DimCount>> obj, std::array<const size_t, DimCount> shape_change)
		{
			const int size = 10;
			std::shared_ptr<T[]> ptr = std::make_unique<T[]>(size);
			
			auto shape_out = obj.get()->Shape;
			obj.get()->CopyDataTo(ptr.get());

			EXPECT_EQ(shape_out.size(), DimCount);
			for (auto i = 0; i < DimCount; i++)
			{
				EXPECT_EQ(shape_change[i], shape_out[i]);
			}

			for (int i = 0; i < size; i++)
			{
				EXPECT_EQ(i, ptr[i]);
			}
		}

		TEST(ReshaperTest, Dim1)
		{
			const int dim = 1;
			auto in = setup<int, dim>();

			Reshaper<int> reshaper;

			auto temp = std::dynamic_pointer_cast<Container<int>>(in);

			std::array<const size_t, dim> shape_change = { 10 };
			auto out = reshaper.Reshape1D(temp);

			control<int, dim>(out, shape_change);
		}

		TEST(ReshaperTest, Dim2)
		{
			const int dim = 2;
			auto in = setup<int, dim>();

			Reshaper<int> reshaper;

			auto temp = std::dynamic_pointer_cast<Container<int>>(in);


			std::array<const size_t, dim> shape_change = { 1, 10 };
			auto out = reshaper.Reshape2D(temp, shape_change);

			control<int, dim>(out, shape_change);
		}

		TEST(ReshaperTest, Dim3)
		{
			const int dim = 3;
			auto in = setup<int, dim>();

			Reshaper<int> reshaper;

			auto temp = std::dynamic_pointer_cast<Container<int>>(in);

			std::array<const size_t, dim> shape_change = { 1, 5 ,2 };
			auto out = reshaper.Reshape3D(temp, shape_change);

			control<int, dim>(out, shape_change);
		}
	}
}
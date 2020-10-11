#include "cpu_backend/linalg/ArithmeticChained.h"

#include <gtest/gtest.h>

namespace cpu_backend
{
	namespace testing
	{
		template<typename T>
		void control_MulAdd(const T* const ptr, const T* const arr_a, const T* const arr_b, const T* const arr_c, const size_t& size, const size_t& control_size)
		{
			EXPECT_EQ(size, control_size);
			for (auto i = 0; i < size; i++)
			{
				EXPECT_EQ((arr_a[i] * arr_b[i]) + arr_c[i], ptr[i]);
			}
		}
		
		template<typename T>
		void TestForType()
		{
			const int size = 4;

			const int dim = 1;
			std::array<const size_t, dim> shape = { 4 };

			T arr_a[size];
			T arr_b[size];
			T arr_c[size];
			T arr_d[size];
			for (auto i = 0; i < size; i++)
			{
				arr_a[i] = i + 1;
				arr_b[i] = i * 2;
				arr_c[i] = (i + 1) * 2;
				arr_d[i] = i * 3;
			}

			Array<T, dim> obj_a(shape, arr_a);
			Array<T, dim> obj_b(shape, arr_b);
			Array<T, dim> obj_c(shape, arr_c);
			Array<T, dim> obj_d(shape, arr_d);

			std::shared_ptr<Array<T, dim>> in_a =
				std::make_shared<Array<T, dim>>(obj_a);

			std::shared_ptr<Array<T, dim>> in_b =
				std::make_shared<Array<T, dim>>(obj_b);

			std::shared_ptr<Array<T, dim>> in_c =
				std::make_shared<Array<T, dim>>(obj_c);

			std::shared_ptr<Array<T, dim>> in_d =
				std::make_shared<Array<T, dim>>(obj_d);

			T ptr[size];

			ArithmeticChained<T, dim> arith_chain;
			std::shared_ptr<core::IArray<T, dim>> test_obj;

			test_obj = arith_chain.MulAdd(in_a, in_b, in_c);
			test_obj.get()->CopyDataTo(ptr);
			control_MulAdd(ptr, arr_a, arr_b, arr_c, size, test_obj.get()->Size());
		}
		

		TEST(CpuArithmeticTest, ChainedDouble)
		{
			TestForType<double>();
		}

		TEST(CpuArithmeticTest, ChainedFloat)
		{
			TestForType<float>();
		}

		TEST(CpuArithmeticTest, ChainedInt)
		{
			TestForType<int>();
		}
	}
}
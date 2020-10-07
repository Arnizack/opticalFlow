#include "cpu_backend\linalg\ArithmeticBasic.h"
#include "cpu_backend\linalg\ArithmeticChained.h"
#include "cpu_backend\Array.h"

#include"gtest/gtest.h"
#include"gmock/gmock.h"

namespace cpu_backend
{
	namespace testing
	{

		template<typename T>
		void control_add_sub(const T* ptr, const T* arr_a, const T* arr_b, const double& alpha, const size_t& size, const size_t& control_size)
		{
			EXPECT_EQ(size, control_size);
			for (auto i = 0; i < size; i++)
			{
				EXPECT_EQ(arr_a[i] + alpha * arr_b[i], ptr[i]);
			}
		}

		template<typename T>
		void control_mul(const T* ptr, const T* arr_a, const T* arr_b, const size_t& size, const size_t& control_size)
		{
			EXPECT_EQ(size, control_size);
			for (auto i = 0; i < size; i++)
			{
				EXPECT_EQ(arr_a[i] * arr_b[i], ptr[i]);
			}
		}

		template<typename T>
		void control_div(const T* ptr, const T* arr_a, const T* arr_b, const size_t& size, const size_t& control_size)
		{
			EXPECT_EQ(size, control_size);
			for (auto i = 0; i < size; i++)
			{
				if (arr_b[i] == 0)
				{
					EXPECT_EQ((T)INFINITY, ptr[i]);
				}
				else {
					EXPECT_EQ((T)(arr_a[i] / arr_b[i]), ptr[i]);
				}
			}
		}
		

		template<typename T>
		void control_pow(const T* ptr, const T* arr_a, const T* arr_b, const size_t& size, const size_t& control_size)
		{
			EXPECT_EQ(size, control_size);
			for (auto i = 0; i < size; i++)
			{
				EXPECT_EQ((T)pow((double)arr_a[i], (double)arr_b[i]), ptr[i]);
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
			for (auto i = 0; i < size; i++)
			{
				arr_a[i] = i + 1;
				arr_b[i] = i * 2;
			}

			Array<T, dim> obj_a(shape, size, arr_a);
			Array<T, dim> obj_b(shape, size, arr_b);

			std::shared_ptr<Array<T, dim>> in_a =
				std::make_shared<Array<T, dim>>(obj_a);

			std::shared_ptr<Array<T, dim>> in_b =
				std::make_shared<Array<T, dim>>(obj_b);

			T ptr[size];

			ArithmeticBasic<T, dim> arith_base;
			std::shared_ptr<core::IArray<T, dim>> test_obj;

			//Add
			test_obj = arith_base.Add(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, arr_a, arr_b, 1, size, test_obj.get()->Size());

			//AddTo
			arith_base.AddTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, arr_a, arr_b, 1, size, test_obj.get()->Size());
			
			//Sub
			test_obj = arith_base.Sub(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, arr_a, arr_b, -1, size, test_obj.get()->Size());
			
			//SubTo
			arith_base.SubTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, arr_a, arr_b, -1, size, test_obj.get()->Size());
			
			//Mul
			test_obj = arith_base.Mul(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_mul<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());

			//MulTo
			arith_base.MulTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_mul<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());
			
			//Div
			test_obj = arith_base.Div(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_div<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());
			
			//DivTo
			arith_base.DivTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_div<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());
			
			//Pow
			test_obj = arith_base.Pow(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_pow<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());

			//Pow
			arith_base.PowTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_pow<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());
		}

		TEST(CpuArithmeticTest, BasicDouble)
		{
			TestForType<double>();
		}

		TEST(CpuArithmeticTest, BasicFloat)
		{
			TestForType<float>();
		}
		
		TEST(CpuArithmeticTest, BasicInt)
		{
			TestForType<int>();
		}
	}
}
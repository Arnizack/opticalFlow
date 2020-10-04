#include "..\cpu_backend\linalg\ArithmeticBasic.h"
#include "..\core\linalg\IArithmeticBasic.h"
#include "..\cpu_backend\Array.h"
#include "..\core\IContainer.h"
#include <array>
#include <memory>
#include"gtest/gtest.h"
#include"gmock/gmock.h"

namespace cpu_utext
{
	namespace linalg_test
	{

		template<typename T>
		void TestForTypeAdd()
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

			T* ptr;

			cpu::Array<T, dim> obj_a(shape, size, arr_a);
			cpu::Array<T, dim> obj_b(shape, size, arr_b);
			std::shared_ptr<cpu::Array<T, dim>> in_a = std::make_shared<cpu::Array<T, dim>>(obj_a);
			std::shared_ptr<cpu::Array<T, dim>> in_b = std::make_shared<cpu::Array<T, dim>>(obj_b);

			std::shared_ptr<cpu::Array<T, dim>> test_obj;
			
			//Add
			test_obj = cpu::linalg::ArithmeticBasic<T, dim>::Add(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, arr_a, arr_b, 1, size, test_obj.get()->Size());
			//AddTo
			cpu::linalg::ArithmeticBasic<T, dim>::AddTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, arr_a, arr_b, 1, size, test_obj.get()->Size());
		}

		template<typename T>
		void TestForTypeSub()
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

			T* ptr;

			cpu::Array<T, dim> obj_a(shape, size, arr_a);
			cpu::Array<T, dim> obj_b(shape, size, arr_b);
			std::shared_ptr<cpu::Array<T, dim>> in_a = std::make_shared<cpu::Array<T, dim>>(obj_a);
			std::shared_ptr<cpu::Array<T, dim>> in_b = std::make_shared<cpu::Array<T, dim>>(obj_b);

			std::shared_ptr<cpu::Array<T, dim>> test_obj;
			
			//Sub
			test_obj = cpu::linalg::ArithmeticBasic<T, dim>::Sub(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, arr_a, arr_b, -1, size, test_obj.get()->Size());
			//SubTo
			cpu::linalg::ArithmeticBasic<T, dim>::SubTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, arr_a, arr_b, -1, size, test_obj.get()->Size());
		}

		template<typename T>
		void TestForTypeMul()
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

			T* ptr;

			cpu::Array<T, dim> obj_a(shape, size, arr_a);
			cpu::Array<T, dim> obj_b(shape, size, arr_b);
			std::shared_ptr<cpu::Array<T, dim>> in_a = std::make_shared<cpu::Array<T, dim>>(obj_a);
			std::shared_ptr<cpu::Array<T, dim>> in_b = std::make_shared<cpu::Array<T, dim>>(obj_b);

			std::shared_ptr<cpu::Array<T, dim>> test_obj;

			//Sub
			test_obj = cpu::linalg::ArithmeticBasic<T, dim>::Mul(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_mul<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());
			//SubTo
			cpu::linalg::ArithmeticBasic<T, dim>::MulTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_mul<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());
		}

		TEST(CpuArithmeticTest, DoubleAdd)
		{
			TestForTypeAdd<double>();
		}
		TEST(CpuArithmeticTest, DoubleSub)
		{
			TestForTypeSub<double>();
		}

		TEST(CpuArithmeticTest, Float)
		{
			TestForTypeAdd<float>();
		}

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
				EXPECT_EQ(arr_a[i] / arr_b[i], ptr[i]);
			}
		}
	}
}
#include "pch.h"
#include "..\cpu_backend\linalg\ArithmeticBasic.h"
#include "..\core\linalg\IArithmeticBasic.h"
#include "..\cpu_backend\Array.h"
#include "..\core\IContainer.h"
#include <array>
#include <memory>

namespace cpu_utext
{
	namespace linalg_test
	{
		class ArithmeticBasicTest : public ::testing::Test
		{
		protected:
			void SetUp() override
			{
				_size = 4;
				_dim = 1;

				std::array<const size_t, dim> shape = { 4 };

				arr_a = new double[_size];
				arr_b[size];
				for (auto i = 0; i < size; i++)
				{
					arr_a[i] = i + 1;
					arr_b[i] = i * 2;
				}

				_in_a = std::make_shared<cpu::Array<double, dim>>(cpu::Array<double, dim>(shape, size, arr_a));
				_in_a = std::make_shared<cpu::Array<double, dim>>(cpu::Array<double, dim>(shape, size, arr_b));
			}
			std::shared_ptr<cpu::Array<double, 1>> _in_a;
			std::shared_ptr<cpu::Array<double, 1>> _in_b;
			std::shared_ptr<cpu::Array<double, 1>> _test_obj;
			const int _size;
			const int _dim;
			double _arr_a[];
			double _arr_b[];
		};

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

					

			//Sub
			test_obj = cpu::linalg::ArithmeticBasic<T, dim>::Sub(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, arr_a, arr_b, -1, size, test_obj.get()->Size());
			//SubTo
			cpu::linalg::ArithmeticBasic<T, dim>::SubTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			//control_add_sub<T>(ptr, arr_a, arr_b, -1, size, test_obj.get()->Size());
			/*
			//Mul
			test_obj = cpu::linalg::ArithmeticBasic<T, dim>::Mul(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_mul<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());
			//MulTo
			cpu::linalg::ArithmeticBasic<T, dim>::MulTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_mul<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());

			//Div
			test_obj = cpu::linalg::ArithmeticBasic<T, dim>::Div(in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_div<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());
			//DivTo
			cpu::linalg::ArithmeticBasic<T, dim>::DivTo(test_obj, in_a, in_b);
			test_obj.get()->CopyDataTo(ptr);
			control_div<T>(ptr, arr_a, arr_b, size, test_obj.get()->Size());
			*/
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

		TEST_F(CpuArithmeticTest, Double)
		{
			_test_obj = cpu::linalg::ArithmeticBasic<double, 1>::Add(_in_a, _in_b);
			double* ptr;
			test_obj.get()->CopyDataTo(ptr);
			control_add_sub<T>(ptr, _arr_a, _arr_b, 1, _size, test_obj.get()->Size());
		}

		TEST(CpuArithmeticTest, Float)
		{
			const int size = 4;
			const int dim = 1;
			std::array<const size_t, dim> shape = { 4 };
			float arr[size];
			for (auto i = 0; i < size; i++)
			{
				arr[i] = i + 1;
			}

			cpu::Array<float, dim> obj(shape, size, arr);

			std::shared_ptr<cpu::Array<float, dim>> test_obj = cpu::linalg::ArithmeticBasic<float, dim>::Add(std::make_shared<cpu::Array<float, dim>>(obj), std::make_shared<cpu::Array<float, dim>>(obj));

			float* ptr;
			test_obj.get()->CopyDataTo(ptr);
			EXPECT_EQ(size, test_obj.get()->Size());
			EXPECT_EQ(arr[3] * 2, ptr[3]);
		}
	}
}
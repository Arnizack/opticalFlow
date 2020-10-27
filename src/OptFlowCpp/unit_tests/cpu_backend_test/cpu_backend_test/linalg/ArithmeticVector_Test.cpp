#include "cpu_backend\linalg\ArithmeticBasic.h"
#include "cpu_backend\linalg\ArithmeticVector.h"
#include "cpu_backend\Array.h"

#include"gtest/gtest.h"
#include"gmock/gmock.h"

namespace cpu_backend
{
	namespace testing
	{
		template<typename T>
		double NormEuclidCompare(const T* const arr, const size_t& size)
		{
			double norm = 0;

			for (size_t i = 0; i < size; i++)
			{
				norm += arr[i] * arr[i];
			}

			return sqrt(norm);
		}

		template<typename T>
		double ScalarCompare(const T* const arr_a, const T* const arr_b, const size_t& size)
		{
			double out = 0;

			for (size_t i = 0; i < size; i++)
			{
				out += arr_a[i] * arr_b[i];
			}

			return out;
		}

		template<typename T, size_t dim>
		void ScaleCompare(const double& fac, const T* const arr_a, const size_t& size, std::shared_ptr<Array<T, dim>> result)
		{

			for (size_t i = 0; i < size; i++)
			{
				EXPECT_EQ(fac * arr_a[i], (*result)[i]);
			}

			return;
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

			Array<T, dim> obj_a(shape, arr_a);
			Array<T, dim> obj_b(shape, arr_b);

			std::shared_ptr<Array<T, dim>> in_a =
				std::make_shared<Array<T, dim>>(obj_a);

			std::shared_ptr<Array<T, dim>> in_b =
				std::make_shared<Array<T, dim>>(obj_b);

			ArrayFactory<T, dim> factory;

			cpu_backend::ArithmeticVector<T, dim> arith_vec(std::make_shared<ArrayFactory<T, dim>>(factory));

			double out_norm = arith_vec.NormEuclidean(in_a);
			double norm_control = NormEuclidCompare<T>(arr_a, size);
			EXPECT_EQ(out_norm, norm_control);

			double out_scalar = arith_vec.ScalarProduct(in_a, in_b);
			double scalar_compare = ScalarCompare<T>(arr_a, arr_b, size);
			EXPECT_EQ(out_scalar, scalar_compare);

			double fac = 2;
			std::shared_ptr<core::IArray<T, dim>> out_scale_temp = arith_vec.Scale(fac, in_a);
			std::shared_ptr<Array<T, dim>> out_scale = std::dynamic_pointer_cast<Array<T, dim>>(out_scale_temp);
			ScaleCompare<T, dim>(fac, arr_a, size, out_scale);

			fac = 0.5;
			arith_vec.ScaleTo(fac, in_a);
			out_scale = std::dynamic_pointer_cast<Array<T, dim>>(in_a);
			ScaleCompare<T, dim>(fac, arr_a, size, out_scale);
			return;
		}

		TEST(CpuArithmeticTest, VectorDouble)
		{
			TestForType<double>();
		}

		TEST(CpuArithmeticTest, VectorFloat)
		{
			TestForType<float>();
		}

		TEST(CpuArithmeticTest, VectorInt)
		{
			TestForType<int>();
		}
	}
}
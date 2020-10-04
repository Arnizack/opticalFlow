#pragma once
#include "..\core\linalg\IArithmeticBasic.h"
#include "..\Array.h"

#include <math.h>
#include <cblas.h>

namespace cpu
{
	namespace linalg
	{
		/*
		* OTHER
		*/
		template<class InnerTyp, size_t DimCount>
		class ArithmeticBasic : public core::linalg::IArithmeticBasic<InnerTyp>
		{
			using PtrVector = std::shared_ptr<core::IContainer<InnerTyp>>;

		public:

			//a+b
			static PtrVector Add(const PtrVector a, const PtrVector b)
			{
				//a+b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] + ptr_b[i];
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}
			
			//x = a+b
			static void AddTo(PtrVector x, const PtrVector a, const PtrVector b)
			{
				//x = a+b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] * ptr_b[i];
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

			//a-b
			static PtrVector Sub(const PtrVector a, const PtrVector b)
			{
				//a-b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] - ptr_b[i];
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}

			//x = a-b
			static void SubTo(PtrVector x, const PtrVector a, const PtrVector& b)
			{
				//x = a+b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] - ptr_b[i];
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

			//a*b
			static PtrVector Mul(const PtrVector a, const PtrVector b)
			{
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] * ptr_b[i];
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}

			//x = a*b
			static void MulTo(PtrVector& x, const PtrVector a, const PtrVector b)
			{
				//x = a*b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] * ptr_b[i];
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

			//a/b
			static PtrVector Div(const PtrVector a, const PtrVector b)
			{
				//a/b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] / ptr_b[i];
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}

			//x = a/b
			static void DivTo(PtrVector x, const PtrVector a, const PtrVector b)
			{
				//x = a/b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] / ptr_b[i];
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

			//a**b
			static PtrVector Pow(const PtrVector a, const PtrVector b)
			{
				//a^b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = pow(ptr_a[i], ptr_b[i]);
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}

			//x = a**b
			static void PowTo(PtrVector x, const PtrVector a, const PtrVector b)
			{
				//a^b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				InnerTyp* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = pow(ptr_a[i], ptr_b[i]);
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

			//a**b
			static PtrVector Pow(const PtrVector a, const InnerTyp& b)
			{
				//a^b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = pow(ptr_a[i], b);
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}

			//x = a**b
			static void PowTo(PtrVector x, const PtrVector a, const InnerTyp& b)
			{
				//a^b
				InnerTyp* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				const size_t size = a.get()->Size();

				InnerTyp ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = pow(ptr_a[i], b);
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

		private:
			static PtrVector return_data(std::array<const size_t, DimCount> shape, const size_t& size, const InnerTyp* const data)
			{
				cpu::Array<InnerTyp, DimCount> out_temp(shape, size, data);
				return std::make_shared<cpu::Array<InnerTyp, 1>>(out_temp);
			}
		};

		/*
		* DOUBLE
		*/
		template<size_t DimCount>
		class ArithmeticBasic<double, DimCount> : public core::linalg::IArithmeticBasic<double>
		{
			using PtrVector = std::shared_ptr<core::IContainer<double>>;

		public:

			//a+b
			static PtrVector Add(const PtrVector a, const PtrVector b)
			{
				//a+b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				cblas_daxpby(size, 1, ptr_a, 1, 1, ptr_b, 1);

				return return_data(a.get()->Shape, size, ptr_b);
			}

			//x = a+b
			static void AddTo(PtrVector x, const PtrVector a, const PtrVector b)
			{
				//x = a+b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				cblas_daxpby(size, 1, ptr_a, 1, 1, ptr_b, 1);

				x = return_data(a.get()->Shape, size, ptr_b);
				return;
			}

			//a-b
			static PtrVector Sub(const PtrVector a, const PtrVector b)
			{
				//a-b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				cblas_daxpby(size, 1, ptr_a, 1, -1, ptr_b, 1);

				return return_data(a.get()->Shape, size, ptr_b);
			}

			//x = a-b
			static void SubTo(PtrVector x, const PtrVector a, const PtrVector& b)
			{
				//x = a-b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				cblas_daxpby(size, 1, ptr_a, 1, -1, ptr_b, 1);

				x = return_data(a.get()->Shape, size, ptr_b);
				return;
			}

			//a*b
			static PtrVector Mul(const PtrVector a, const PtrVector b)
			{
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				for (int i = 0; i < size; i++)
				{
					ptr_b[i] = ptr_a[i] * ptr_b[i];
				}

				return return_data(a.get()->Shape, size, ptr_b);
			}

			//x = a*b
			static void MulTo(PtrVector& x, const PtrVector a, const PtrVector b)
			{
				//x = a*b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				for (int i = 0; i < size; i++)
				{
					ptr_b[i] = ptr_a[i] * ptr_b[i];
				}

				x = return_data(a.get()->Shape, size, ptr_b);
				return;
			}

			//a/b
			static PtrVector Div(const PtrVector a, const PtrVector b)
			{
				//a/b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				for (int i = 0; i < size; i++)
				{
					ptr_b[i] = ptr_a[i] / ptr_b[i];
				}

				return return_data(a.get()->Shape, size, ptr_b);
			}

			//x = a/b
			static void DivTo(PtrVector x, const PtrVector a, const PtrVector b)
			{
				//x = a/b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				//double* ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_b[i] = ptr_a[i] / ptr_b[i];
				}

				x = return_data(a.get()->Shape, size, ptr_b);
				return;
			}

			//a**b
			static PtrVector Pow(const PtrVector a, const PtrVector b)
			{
				//a^b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				//double ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_b[i] = pow(ptr_a[i], ptr_b[i]);
				}

				return return_data(a.get()->Shape, size, ptr_b);
			}

			//x = a**b
			static void PowTo(PtrVector x, const PtrVector a, const PtrVector b)
			{
				//a^b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				double* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				//double ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_b[i] = pow(ptr_a[i], ptr_b[i]);
				}

				x = return_data(a.get()->Shape, size, ptr_b);
				return;
			}

			//a**b
			static PtrVector Pow(const PtrVector a, const double& b)
			{
				//a^b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				const size_t size = a.get()->Size();

				//double ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_a[i] = pow(ptr_a[i], b);
				}

				return return_data(a.get()->Shape, size, ptr_a);
			}

			//x = a**b
			static void PowTo(PtrVector x, const PtrVector a, const double& b)
			{
				//a^b
				double* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				const size_t size = a.get()->Size();

				//double ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_a[i] = pow(ptr_a[i], b);
				}

				x = return_data(a.get()->Shape, size, ptr_a);
				return;
			}

		private:
			static PtrVector return_data(std::array<const size_t, DimCount> shape, const size_t& size, const double* const data)
			{
				cpu::Array<double, DimCount> out_temp(shape, size, data);
				return std::make_shared<cpu::Array<double, 1>>(out_temp);
			}
		};

		/*
		* FLOAT
		*/
		template<size_t DimCount>
		class ArithmeticBasic<float, DimCount> : public core::linalg::IArithmeticBasic<float>
		{
			using PtrVector = std::shared_ptr<core::IContainer<float>>;

		public:

			//a+b
			static PtrVector Add(const PtrVector a, const PtrVector b)
			{
				//a+b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				cblas_saxpby(size, 1, ptr_a, 1, 1, ptr_b, 1);

				return return_data(a.get()->Shape, size, ptr_b);
			}

			//x = a+b
			static void AddTo(PtrVector x, const PtrVector a, const PtrVector b)
			{
				//x = a+b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				cblas_saxpby(size, 1, ptr_a, 1, 1, ptr_b, 1);

				x = return_data(a.get()->Shape, size, ptr_b);
				return;
			}

			//a-b
			static PtrVector Sub(const PtrVector a, const PtrVector b)
			{
				//a-b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				cblas_saxpby(size, 1, ptr_a, 1, -1, ptr_b, 1);

				return return_data(a.get()->Shape, size, ptr_b);
			}

			//x = a-b
			static void SubTo(PtrVector x, const PtrVector a, const PtrVector& b)
			{
				//x = a-b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				cblas_saxpby(size, 1, ptr_a, 1, -1, ptr_b, 1);

				x = return_data(a.get()->Shape, size, ptr_b);
				return;
			}

			//a*b
			static PtrVector Mul(const PtrVector a, const PtrVector b)
			{
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				float ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] * ptr_b[i];
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}

			//x = a*b
			static void MulTo(PtrVector& x, const PtrVector a, const PtrVector b)
			{
				//x = a*b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				float ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] * ptr_b[i];
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

			//a/b
			static PtrVector Div(const PtrVector a, const PtrVector b)
			{
				//a/b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				float ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] / ptr_b[i];
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}

			//x = a/b
			static void DivTo(PtrVector x, const PtrVector a, const PtrVector b)
			{
				//x = a/b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				float ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = ptr_a[i] / ptr_b[i];
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

			//a**b
			static PtrVector Pow(const PtrVector a, const PtrVector b)
			{
				//a^b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				float ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = pow(ptr_a[i], ptr_b[i]);
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}

			//x = a**b
			static void PowTo(PtrVector x, const PtrVector a, const PtrVector b)
			{
				//a^b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				float* ptr_b;
				b.get()->CopyDataTo(ptr_b);

				const size_t size = a.get()->Size();

				float ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = pow(ptr_a[i], ptr_b[i]);
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

			//a**b
			static PtrVector Pow(const PtrVector a, const float& b)
			{
				//a^b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				const size_t size = a.get()->Size();

				float ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = pow(ptr_a[i], b);
				}

				return return_data(a.get()->Shape, size, ptr_c);
			}

			//x = a**b
			static void PowTo(PtrVector x, const PtrVector a, const float& b)
			{
				//a^b
				float* ptr_a;
				a.get()->CopyDataTo(ptr_a);

				const size_t size = a.get()->Size();

				float ptr_c[size];

				for (int i = 0; i < size; i++)
				{
					ptr_c[i] = pow(ptr_a[i], b);
				}

				x = return_data(a.get()->Shape, size, ptr_c);
				return;
			}

		private:
			static PtrVector return_data(std::array<const size_t, DimCount> shape, const size_t& size, const float* const data)
			{
				cpu::Array<float, 1> out_temp(shape, size, data);
				return std::make_shared<cpu::Array<float, 1>>(out_temp);
			}
		};
	}
}
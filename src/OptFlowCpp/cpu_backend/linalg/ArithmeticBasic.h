#pragma once
#include "core\linalg\IArithmeticBasic.h"

#include "../Array.h"

#include <math.h>
#include <cblas.h>

namespace cpu_backend
{

	/*
	* All Types
	*/
	template<class InnerTyp, size_t DimCount>
	class ArithmeticBasic : public core::IArithmeticBasic<InnerTyp, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<InnerTyp, DimCount>>;

	public:

		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) override
		{
			//a+b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] + ptr_b[i];
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a+b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] * ptr_b[i];
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
		{
			//a-b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] - ptr_b[i];
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector& b) override
		{
			//x = a+b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] - ptr_b[i];
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
		{
			//a*b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] * ptr_b[i];
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a*b
		virtual void MulTo(PtrVector& x, const PtrVector a, const PtrVector b) override
		{
			//x = a*b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] * ptr_b[i];
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b) override
		{
			//a/b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				if (ptr_b[i] == 0)
				{
					ptr_c[i] = (InnerTyp)INFINITY;
				}
				else
				{
					ptr_c[i] = (InnerTyp)(ptr_a[i] / ptr_b[i]);
				}
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a/b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				if (ptr_b[i] == 0)
				{
					ptr_c[i] = (InnerTyp)INFINITY;
				}
				else
				{
					ptr_c[i] = (InnerTyp)(ptr_a[i] / ptr_b[i]);
				}
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}
		
		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = (InnerTyp) pow((double)ptr_a[i], (double)ptr_b[i]);
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<InnerTyp[]> ptr_b(new InnerTyp[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = (InnerTyp)pow((double)ptr_a[i], (double)ptr_b[i]);
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			double b_conv = (double)b;

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = (InnerTyp)pow((double)ptr_a[i], b_conv);
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr_a(new InnerTyp[size]);
			a.get()->CopyDataTo(ptr_a.get());

			double b_conv = (double)b;

			std::unique_ptr<InnerTyp[]> ptr_c(new InnerTyp[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = (InnerTyp)pow((double)ptr_a[i], b_conv);
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}
		
	private:
		PtrVector return_data(std::array<const size_t, DimCount> shape, const size_t& size, const InnerTyp* const data)
		{
			cpu_backend::Array<InnerTyp, DimCount> out_temp(shape, size, data);
			return std::make_shared<cpu_backend::Array<InnerTyp, DimCount>>(out_temp);
		}
	};

	/*
	* DOUBLE
	*/
	template<size_t DimCount>
	class ArithmeticBasic<double, DimCount> : public core::IArithmeticBasic<double, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<double, DimCount>>;

	public:

		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) override
		{
			//a+b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			cblas_daxpby(size, 1, ptr_a.get(), 1, 1, ptr_b.get(), 1);

			PtrVector out = return_data(a.get()->Shape, size, ptr_b.get());

			return out;
		}

		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a+b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			cblas_daxpby(size, 1, ptr_a.get(), 1, 1, ptr_b.get(), 1);

			x = return_data(a.get()->Shape, size, ptr_b.get());

			return;
		}

		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
		{
			//a-b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			cblas_daxpby(size, 1, ptr_a.get(), 1, -1, ptr_b.get(), 1);

			PtrVector out = return_data(a.get()->Shape, size, ptr_b.get());

			return out;
		}

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector& b) override
		{
			//x = a-b

			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			cblas_daxpby(size, 1, ptr_a.get(), 1, -1, ptr_b.get(), 1);

			x = return_data(a.get()->Shape, size, ptr_b.get());

			return;
		}

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
		{
			//a*b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<double[]> ptr_c(new double[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] * ptr_b[i];
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a*b
		virtual void MulTo(PtrVector& x, const PtrVector a, const PtrVector b) override
		{
			//x = a*b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<double[]> ptr_c(new double[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] * ptr_b[i];
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b) override
		{
			//a/b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<double[]> ptr_c(new double[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] / ptr_b[i];
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a/b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<double[]> ptr_c(new double[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] / ptr_b[i];
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<double[]> ptr_c(new double[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = pow(ptr_a[i], ptr_b[i]);
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_b(new double[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<double[]> ptr_c(new double[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = pow(ptr_a[i], ptr_b[i]);
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_c(new double[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = pow(ptr_a[i], b);
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<double[]> ptr_a(new double[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<double[]> ptr_c(new double[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = pow(ptr_a[i], b);
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

	private:
		PtrVector return_data(std::array<const size_t, DimCount> shape, const size_t& size, const double* const data)
		{
			cpu_backend::Array<double, DimCount> out_temp(shape, size, data);
			return std::make_shared<Array<double, DimCount>>(out_temp);
		}
	};

	/*
	* FLOAT
	*/
	template<size_t DimCount>
	class ArithmeticBasic<float, DimCount> : public core::IArithmeticBasic<float, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<float, DimCount>>;

	public:

		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) override
		{
			//a+b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			cblas_saxpby(size, 1, ptr_a.get(), 1, 1, ptr_b.get(), 1);

			PtrVector out = return_data(a.get()->Shape, size, ptr_b.get());

			return out;
		}

		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a+b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			cblas_saxpby(size, 1, ptr_a.get(), 1, 1, ptr_b.get(), 1);

			x = return_data(a.get()->Shape, size, ptr_b.get());

			return;
		}

		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
		{
			//a-b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			cblas_saxpby(size, 1, ptr_a.get(), 1, -1, ptr_b.get(), 1);

			PtrVector out = return_data(a.get()->Shape, size, ptr_b.get());

			return out;
		}

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector& b) override
		{
			//x = a-b

			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			cblas_saxpby(size, 1, ptr_a.get(), 1, -1, ptr_b.get(), 1);

			x = return_data(a.get()->Shape, size, ptr_b.get());

			return;
		}

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
		{
			//a*b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<float[]> ptr_c(new float[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] * ptr_b[i];
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a*b
		virtual void MulTo(PtrVector& x, const PtrVector a, const PtrVector b) override
		{
			//x = a*b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<float[]> ptr_c(new float[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] * ptr_b[i];
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b) override
		{
			//a/b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<float[]> ptr_c(new float[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] / ptr_b[i];
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a/b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<float[]> ptr_c(new float[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = ptr_a[i] / ptr_b[i];
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<float[]> ptr_c(new float[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = pow(ptr_a[i], ptr_b[i]);
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_b(new float[b.get()->Size()]);
			b.get()->CopyDataTo(ptr_b.get());

			std::unique_ptr<float[]> ptr_c(new float[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = pow(ptr_a[i], ptr_b[i]);
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_c(new float[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = pow(ptr_a[i], b);
			}

			PtrVector out = return_data(a.get()->Shape, size, ptr_c.get());

			return out;
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = a.get()->Size();

			std::unique_ptr<float[]> ptr_a(new float[size]);
			a.get()->CopyDataTo(ptr_a.get());

			std::unique_ptr<float[]> ptr_c(new float[size]);

			for (int i = 0; i < size; i++)
			{
				ptr_c[i] = pow(ptr_a[i], b);
			}

			x = return_data(a.get()->Shape, size, ptr_c.get());

			return;
		}

	private:
		PtrVector return_data(std::array<const size_t, DimCount> shape, const size_t& size, const float* const data)
		{
			cpu_backend::Array<float, DimCount> out_temp(shape, size, data);
			return std::make_shared<Array<float, DimCount>>(out_temp);
		}
	};

}
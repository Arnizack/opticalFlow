#pragma once
#include "core/linalg/IArithmeticChained.h"
#include "ArithmeticBasic.h"
#include "../Array.h"
#include <omp.h>

namespace cpu_backend
{
	template<class InnerTyp, size_t DimCount>
	class ArithmeticChained 
		: public core::IArithmeticChained<InnerTyp, DimCount>, 
		public ArithmeticBasic<InnerTyp, DimCount>,
		public core::IArithmeticBasic<InnerTyp, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<InnerTyp, DimCount>>;
		using PtrArrayFactory = std::shared_ptr<core::IArrayFactory<InnerTyp, DimCount>>;
		using PtrArithmeticBase = std::shared_ptr<core::IArithmeticBasic<InnerTyp, DimCount>>;

	public:
		ArithmeticChained(const PtrArrayFactory factory)
			: ArithmeticBasic<InnerTyp, DimCount>(factory), _factory(std::dynamic_pointer_cast<ArrayFactory<InnerTyp, DimCount>>(factory))
		{}

		//a*b+c
		virtual PtrVector MulAdd(const PtrVector a, const PtrVector b, const PtrVector c) override
		{
			const size_t size = a->Size();
			auto out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>( _factory->Zeros(a->Shape) );

			auto in_a = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			auto in_b = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);
			auto in_c = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(c);

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*in_a)[i] * (*in_b)[i] + (*in_c)[i];
			}

			return out;
		}

		//a*b+c*d
		virtual PtrVector MulAddMul(const PtrVector a, const PtrVector b, const PtrVector c, const PtrVector d) override
		{
			const size_t size = a->Size();
			auto out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>( _factory->Zeros(a->Shape) );

			auto in_a = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			auto in_b = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);
			auto in_c = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(c);
			auto in_d = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(d);

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*in_a)[i] * (*in_b)[i] + (*in_c)[i] * (*in_d)[i];
			}

			return out;
		}

		//x = alpha*a + b
		virtual void ScaleAddTo(const PtrVector x, const double& alpha, const PtrVector a, const PtrVector b) override
		{
			const size_t size = a->Size();
			auto out = std::static_pointer_cast<Array<InnerTyp, DimCount>>(x)->Data();

			auto in_a = std::static_pointer_cast<Array<InnerTyp, DimCount>>(a)->Data();
			auto in_b = std::static_pointer_cast<Array<InnerTyp, DimCount>>(b)->Data();

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				out[i] = alpha * in_a[i] + in_b[i];
			}

			return;
		}

		//Methods of IArithmeticBasic
		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) override
		{
			return ArithmeticBasic<InnerTyp, DimCount>::Add(a,b);
		}

		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			ArithmeticBasic<InnerTyp, DimCount>::AddTo(x, a, b);
			return;
		}

		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
		{
			return ArithmeticBasic<InnerTyp, DimCount>::Sub(a, b);
		}

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			ArithmeticBasic<InnerTyp, DimCount>::SubTo(x, a, b);
			return;
		}

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
		{
			return ArithmeticBasic<InnerTyp, DimCount>::Mul(a, b);
		}

		//x = a*b
		virtual void MulTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			ArithmeticBasic<InnerTyp, DimCount>::MulTo(x, a, b);
			return;
		}

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b) override
		{
			return ArithmeticBasic<InnerTyp, DimCount>::Div(a, b);
		}

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			ArithmeticBasic<InnerTyp, DimCount>::DivTo(x, a, b);
			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
		{
			return ArithmeticBasic<InnerTyp, DimCount>::Pow(a, b);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			ArithmeticBasic<InnerTyp, DimCount>::PowTo(x, a, b);
			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b) override
		{
			return ArithmeticBasic<InnerTyp, DimCount>::Pow(a, b);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
		{
			ArithmeticBasic<InnerTyp, DimCount>::PowTo(x, a, b);
			return;
		}

		private:
			std::shared_ptr< ArrayFactory<InnerTyp, DimCount>> _factory;
	};
}
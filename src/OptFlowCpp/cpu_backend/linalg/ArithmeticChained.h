#pragma once
#include "core/linalg/IArithmeticChained.h"
#include "ArithmeticBasic.h"
#include "../Array.h"

namespace cpu_backend
{
	template<class InnerTyp, size_t DimCount>
	class ArithmeticChained : public core::IArithmeticChained<InnerTyp, DimCount>, public ArithmeticBasic<InnerTyp, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<InnerTyp, DimCount>>;
	public:
		//a*b+c
		virtual PtrVector MulAdd(const PtrVector a, const PtrVector b, const PtrVector c) override
		{
			return _arithmetic_base.Add(_arithmetic_base.Mul(a, b), c);
		}

		//a*b+c*d
		virtual PtrVector MulAddMul(const PtrVector a, const PtrVector b, const PtrVector c, const PtrVector d) override
		{
			return _arithmetic_base.Add(_arithmetic_base.Mul(a, b), _arithmetic_base.Mul(a, b));
		}



		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) 
		{
			return _arithmetic_base.Add(a, b);
		}

		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) 
		{
			_arithmetic_base.AddTo(x, a, b);
			return;
		}

		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) 
		{
			return _arithmetic_base.Sub(a, b);
		}

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector b) 
		{
			_arithmetic_base.SubTo(x, a, b);
			return;
		}

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b)
		{
			return _arithmetic_base.Mul(a, b);
		}

		//x = a*b
		virtual void MulTo(PtrVector x, const PtrVector a, const PtrVector b)
		{
			_arithmetic_base.MulTo(x, a, b);
			return;
		}

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b)
		{
			return _arithmetic_base.Div(a, b);
		}

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b)
		{
			_arithmetic_base.DivTo(x, a, b);
			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b)
		{
			return _arithmetic_base.Pow(a, b);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b)
		{
			_arithmetic_base.PowTo(x, a, b);
			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b)
		{
			return _arithmetic_base.Pow(a, b);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b)
		{
			_arithmetic_base.PowTo(x, a, b);
			return;
		}

		private:
			cpu_backend::ArithmeticBasic<InnerTyp, DimCount> _arithmetic_base;
	};
}
#pragma once
#include"../IArray.h"
#include<memory>
namespace core
{

	template<class InnerTyp, size_t DimCount>
	class IArithmeticBasic
	{
		using PtrVector = std::shared_ptr<IArray<InnerTyp, DimCount>>;
	public:
		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) = 0;
			
		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) = 0;
			
		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) = 0;

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector& b) = 0;

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b) = 0;

		//x = a*b
		virtual void MulTo(PtrVector& x, const PtrVector a, const PtrVector b) = 0;

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b) = 0;

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) = 0;
		
		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b) = 0;

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) = 0;

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b) = 0;

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b) = 0;
	};
	
}
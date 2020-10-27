#pragma once
#include"IArithmeticBasic.h"
#include<memory>

namespace core
{

	template<class InnerTyp, size_t DimCount>
	class IArithmeticChained : public IArithmeticBasic<InnerTyp, DimCount>
	{
		using PtrVector = std::shared_ptr<IArray<InnerTyp, DimCount>>;
	public:
		//a*b+c
		virtual PtrVector MulAdd(const PtrVector a, const PtrVector b,const PtrVector c) = 0;

		//a*b+c*d
		virtual PtrVector MulAddMul(const PtrVector a,const PtrVector b,const PtrVector c,
			const PtrVector d) = 0;
		
		//x = alpha*a + b
		virtual void ScaleAddTo(const PtrVector x, const double& alpha, const PtrVector a,
			const PtrVector b) = 0;

	};
	
}
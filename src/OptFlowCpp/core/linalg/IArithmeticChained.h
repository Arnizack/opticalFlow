#pragma once
#include"../pch.h"
#include"IArithmeticBasic.h"
#include"../IContainer.h"

namespace core
{
	namespace linalg
	{
		template<class InnerTyp>
		class IArithmeticChained : public IArithmeticBasic<InnerTyp>
		{
			using PtrVector = std::shared_ptr<IContainer<InnerTyp>>;
		public:
			//a*b+c
			virtual PtrVector MulAdd(const PtrVector a, const PtrVector b,const PtrVector c) = 0;

			//a*b+c*d
			virtual PtrVector MulAddMul(const PtrVector a,const PtrVector b,const PtrVector c,
				const PtrVector d) = 0;
			


		};
	}
}
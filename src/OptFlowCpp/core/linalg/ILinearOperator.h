#pragma once
#include"../IContainer.h"
#include"../IArray.h"
#include<memory>

namespace core
{
	namespace linalg
	{

		

		template<class InputTyp, class OutputTyp>
		class ILinearOperator
		{
		public:
			virtual OutputTyp MultVec(const InputTyp vec) = 0;
			virtual void MultVecTo(OutputTyp dst,const InputTyp vec) = 0;
			virtual ILinearOperator<OutputTyp,InputTyp> Transpose() = 0;
			virtual bool IsSymetric() = 0;
		};

		
	}
}
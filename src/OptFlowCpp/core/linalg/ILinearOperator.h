#pragma once
#include"../IContainer.h"
#include"../IArray.h"
#include"../pch.h"
namespace core
{
	namespace linalg
	{

		

		template<class InputTyp, class OutputTyp, class TransposeTyp>
		class ILinearOperator
		{
		public:
			virtual OutputTyp MultVec(const InputTyp vec) = 0;
			virtual void MultVecTo(OutputTyp dst,const InputTyp vec) = 0;
			virtual TransposeTyp Transpose() = 0;
			virtual bool IsSymetric() = 0;
		};

		
	}
}
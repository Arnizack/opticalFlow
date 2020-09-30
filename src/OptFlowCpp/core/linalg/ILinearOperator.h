#pragma once
#include"IOperator.h"

namespace core
{
	namespace linalg
	{

		

		template<class InputTyp, class OutputTyp>
		class ILinearOperator : public IOperator<InputTyp, OutputTyp>
		{
		public:
			virtual ILinearOperator<OutputTyp,InputTyp> Transpose() = 0;
			virtual bool IsSymetric() = 0;
		};

		
	}
}
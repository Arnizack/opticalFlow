#pragma once
#include"IOperator.h"
#include<memory>

namespace core
{


	template<class InputTyp, class OutputTyp>
	class ILinearOperator : public IOperator<InputTyp, OutputTyp>
	{
	public:
		virtual std::shared_ptr<ILinearOperator<OutputTyp, InputTyp>> Transpose() = 0;
		virtual bool IsSymetric() = 0;
	};

		
	
}
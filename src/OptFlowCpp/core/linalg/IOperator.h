#pragma once
namespace core
{

	template<class InputTyp, class OutputTyp>
	class IOperator
	{
	public:
		virtual OutputTyp Apply(const InputTyp vec) = 0;
		virtual void ApplyTo(OutputTyp dst, const InputTyp vec) = 0;
	};

}
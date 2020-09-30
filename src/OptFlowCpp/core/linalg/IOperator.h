#pragma once
namespace core
{
	namespace linalg
	{
		template<class InputTyp, class OutputTyp>
		class IOperator
		{
		public:
			virtual OutputTyp Apply(const InputTyp vec) = 0;
			virtual void ApplyTo(OutputTyp dst, const InputTyp vec) = 0;
		};
	}
}
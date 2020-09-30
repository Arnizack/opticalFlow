#pragma once
#include"../IContainer.h"
#include<memory>
namespace core
{
	namespace linalg
	{
		template<class InnerTyp>
		class IArithmeticBasic
		{
			using PtrVector = std::shared_ptr<IContainer<InnerTyp>>;
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
			virtual void MulTo(T& x, const PtrVector a, const PtrVector b) = 0;

			//a/b
			virtual PtrVector Div(const PtrVector a, const PtrVector b) = 0;

			//x = a/b
			virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) = 0;

			//a**b
			virtual PtrVector Pow(const PtrVector a, const PtrVector b);

			//x = a**b
			virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b);

			//a**b
			virtual PtrVector Pow(const PtrVector a, const double& b);

			//x = a**b
			virtual void PowTo(PtrVector x, const PtrVector a, const double& b);
		};
	}
}
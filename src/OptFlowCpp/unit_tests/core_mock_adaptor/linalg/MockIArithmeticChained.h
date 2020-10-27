#pragma once
#include"core/linalg/IArithmeticChained.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{
		class MockIArithmeticChained1D : public IArithmeticChained<double, 1>
		{
			using PtrVector = std::shared_ptr<IArray<double, 1>>;
		public:
			//Inherited via IArithmeticChained
			MOCK_METHOD3(MulAdd, PtrVector(const PtrVector a, const PtrVector b, const PtrVector c));
			MOCK_METHOD4(MulAddMul, PtrVector(const PtrVector a, const PtrVector b, const PtrVector c, const PtrVector d));
			MOCK_METHOD4(ScaleAddTo, void(const PtrVector x, const double& alpha, const PtrVector a, const PtrVector b));

			// Inherited via IArithmeticBasic
			MOCK_METHOD2(Add, PtrVector(const PtrVector a, const PtrVector b));
			MOCK_METHOD3(AddTo, void(PtrVector x, const PtrVector a, const PtrVector b));

			MOCK_METHOD2(Sub, PtrVector(const PtrVector a, const PtrVector b));
			MOCK_METHOD3(SubTo, void(PtrVector x, const PtrVector a, const PtrVector b));
			MOCK_METHOD2(Mul, PtrVector(const PtrVector a, const PtrVector b));
			MOCK_METHOD3(MulTo, void(PtrVector x, const PtrVector a, const PtrVector b));

			MOCK_METHOD2(Div, PtrVector(const PtrVector a, const PtrVector b));
			MOCK_METHOD3(DivTo, void(PtrVector x, const PtrVector a, const PtrVector b));
			MOCK_METHOD2(Pow, PtrVector(const PtrVector a, const PtrVector b));
			MOCK_METHOD3(PowTo, void(PtrVector x, const PtrVector a, const PtrVector b));
			MOCK_METHOD2(Pow, PtrVector(const PtrVector a, const double& b));
			MOCK_METHOD3(PowTo, void(PtrVector x, const PtrVector a, const double& b));
		};
	}
}
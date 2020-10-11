#pragma once
#include"core/linalg/IArithmeticBasic.h"
#include"gmock/gmock.h"
namespace core
{
	namespace testing
	{
		class MockIArithmeticBasic3D : public IArithmeticBasic<double,3>
		{
			using PtrVector = std::shared_ptr<IArray<double, 3>>;
		public:
			// Inherited via IArithmeticBasic
			MOCK_METHOD2( Add, PtrVector(const PtrVector a, const PtrVector b) );
			MOCK_METHOD3( AddTo, void(PtrVector x, const PtrVector a, const PtrVector b) );

			MOCK_METHOD2(Sub, PtrVector(const PtrVector a, const PtrVector b) );
			MOCK_METHOD3(SubTo, void(PtrVector x, const PtrVector a, const PtrVector b) );
			MOCK_METHOD2(Mul, PtrVector(const PtrVector a, const PtrVector b) );
			MOCK_METHOD3(MulTo, void(PtrVector x, const PtrVector a, const PtrVector b) );
			
			MOCK_METHOD2(Div, PtrVector(const PtrVector a, const PtrVector b) );
			MOCK_METHOD3 (DivTo, void(PtrVector x, const PtrVector a, const PtrVector b) );
			MOCK_METHOD2 (Pow, PtrVector(const PtrVector a, const PtrVector b));
			MOCK_METHOD3(PowTo, void(PtrVector x, const PtrVector a, const PtrVector b) );
			MOCK_METHOD2(Pow, PtrVector(const PtrVector a, const double& b) );
			MOCK_METHOD3(PowTo, void(PtrVector x, const PtrVector a, const double& b) );


		};

	}
}
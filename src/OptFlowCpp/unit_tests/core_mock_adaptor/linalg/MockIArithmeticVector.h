#pragma once
#include"core/linalg/IArithmeticVector.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{
		class MockIArithmeticVector1D : public IArithmeticVector<double, 1>
		{
			using PtrVector = std::shared_ptr<IArray<double, 1>>;
			using PtrMatrix = std::shared_ptr<IArray<double, 1>>;

		public:
			MOCK_METHOD1(NormEuclidean, double(const PtrVector vec));
			
			MOCK_METHOD2(ScalarProduct, double(const PtrVector a, const PtrVector b));

			MOCK_METHOD2(Scale, PtrMatrix(const double& fac, const PtrMatrix a));
			MOCK_METHOD2(ScaleTo, void(const double& fac, const PtrMatrix a));

			MOCK_METHOD4(ScalarDivScalar, double(const PtrVector a, const PtrVector b, const PtrVector c, const PtrVector d));
		};
	}
}
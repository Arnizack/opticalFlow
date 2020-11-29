#pragma once
//#include "IArithmeticBasic.h"
#include"../IArray.h"
#include <memory>

namespace core
{
	template<class InnerTyp, size_t DimCount>
	class IArithmeticVector
	{
		using PtrVector = std::shared_ptr<IArray<InnerTyp, 1>>;
		using PtrMatrix = std::shared_ptr<IArray<InnerTyp, DimCount>>;

	public:
		// ||vec|| / norm(vec)
		virtual double NormEuclidean(const PtrVector vec) = 0;

		// <a, b>
		virtual double ScalarProduct(const PtrVector a, const PtrVector b) = 0;

		// x = fac * A
		virtual PtrMatrix Scale(const double& fac, const PtrMatrix a) = 0;

		// A = fac * A
		virtual void ScaleTo(const double& fac, const PtrMatrix a) = 0;

		//<a,b> / <c,d>
		virtual double ScalarDivScalar(const PtrVector a, const PtrVector b, const PtrVector c,
			const PtrVector d) = 0;
	};
}
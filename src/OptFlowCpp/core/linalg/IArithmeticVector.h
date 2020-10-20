#pragma once
#include "IArithmeticBasic.h"

namespace core
{
	template<class InnerTyp, size_t DimCount>
	class IArithmeticVector : public IArithmeticBasic<InnerTyp, DimCount>
	{
		using PtrVector = std::shared_ptr<IArray<InnerTyp, 1>>;
		using PtrMatrix = std::shared_ptr<IArray<InnerTyp, DimCount>>;

	public:
		// ||vec|| / norm(vec)
		virtual double NormEuclidean(const PtrVector vec) = 0;

		// <a, b>
		virtual double ScalarProduct(const PtrVector a, const PtrVector b) = 0;
	};
}
#pragma once
#include "..\IArray.h"
#include <memory>
namespace core
{
	namespace solver
	{
		template<class InnerTyp, size_t DimCount, class SettingsTyp>
		class IImageSolver
		{
		public:
			using PtrImage = std::shared_ptr<IArray<InnerTyp, DimCount>>;

			virtual PtrImage Solve(const PtrImage image) = 0;
		};
	}
}
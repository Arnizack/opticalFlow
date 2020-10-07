#pragma once
#include "IArray.h"
#include <memory>
namespace core
{

	template<class InnerTyp, size_t DimCount>
	class IScaler
	{
	public:
		using PtrArray = std::shared_ptr<IArray<InnerTyp, DimCount>>;

		virtual PtrArray Scale(const PtrArray input, const size_t& dst_width, 
			const size_t& dst_height) = 0;
	};
	
}
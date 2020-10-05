#pragma once
#include "..\IArray.h"
#include <memory>
namespace core
{

	template<class InnerTyp, size_t DimCount>
	class IScalable
	{
		using PtrArray = std::shared_ptr<IArray<InnerTyp, DimCount>>;

		virtual PtrArray Scale(const PtrArray input, const int& dst_width, 
			const int& dst_height) = 0;
	};
	
}
#pragma once
#include"IContainer.h"
#include<array>

namespace core
{

	template<class InnerTyp, size_t DimCount>
	class IArray : public IContainer<InnerTyp>
	{
	public:
		IArray(std::array<const size_t, DimCount> shape);
		std::array<const size_t, DimCount> Shape;

	};

	template<class InnerTyp, size_t DimCount>
	inline IArray<InnerTyp, DimCount>::IArray(std::array<const size_t, DimCount> shape)
		: Shape(shape)
	{
	}
}
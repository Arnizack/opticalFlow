#pragma once
#include"IArray.h"
#include<memory>
#include<array>

namespace core
{
	template<class InnerTyp, size_t DimCount>
	class IArrayFactory
	{
	public:
		using PtrArray = std::shared_ptr < IArray<InnerTyp, DimCount>>;

		virtual PtrArray Zeros(std::array<const size_t, DimCount> shape) = 0;
		virtual PtrArray Full(const InnerTyp& fill_value, std::array<const size_t, DimCount> shape) = 0;
		virtual PtrArray CreateFromSource(const InnerTyp* source, std::array<const size_t, DimCount> shape) = 0;
	};

}
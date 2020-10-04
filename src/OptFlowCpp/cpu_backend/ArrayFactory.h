#pragma once
#include "..\core\IArrayFactory.h"
#include "Array.h"

namespace cpu
{
	template<class InnerTyp, size_t DimCount>
	class ArrayFactory : public core::IArrayFactory<InnerTyp, DimCount>
	{
		using PtrArray = std::shared_ptr<core::IArray<InnerTyp, DimCount>>;
	
	public:
		virtual PtrArray Full(const InnerTyp& fill_value, std::array<const size_t, DimCount> shape) override
		{
			int size = 1;
			for (size_t i = 0; i < DimCount; i++)
			{
				size *= shape[i];
			}

			return std::make_shared<cpu::Array<InnerTyp, DimCount>>(cpu::Array<InnerTyp, DimCount>(shape, size, fill_value));
		}

		virtual PtrArray Zeros(std::array<const size_t, DimCount> shape) override
		{
			return Full(0, shape);
		}
	};
}
#pragma once
#include "core\IArrayFactory.h"
#include "Array.h"

#include <memory>

namespace cpu_backend
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

			return std::make_shared<cpu_backend::Array<InnerTyp, DimCount>>(cpu_backend::Array<InnerTyp, DimCount>(shape, fill_value));
		}

		virtual PtrArray Zeros(std::array<const size_t, DimCount> shape) override
		{
			return Full(0, shape);
		}

		virtual PtrArray CreateFromSource(const InnerTyp* source, std::array<const size_t, DimCount> shape) override
		{
			return std::make_shared<cpu_backend::Array<InnerTyp, DimCount>>(shape, source);
		}
	};
}
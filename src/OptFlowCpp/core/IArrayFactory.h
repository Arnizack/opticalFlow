#pragma once
#include"IArray.h"
#include"pch.h"
namespace core
{
	template<class InnerTyp,size_t DimCount>
	class IArrayFactory
	{
	public:
		using PtrArray = std::shared_ptr < IArray<InnerTyp, DimCount>>;
		virtual PtrArray Zeros(const size_t (&shape)[DimCount] )=0;
		virtual PtrArray Full(const InnerTyp& fill_value, const size_t(&shape)[DimCount]) = 0;
	};
	
}
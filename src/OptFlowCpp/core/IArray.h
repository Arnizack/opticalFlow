#pragma once
#include"IContainer.h"
namespace core
{

	template<class InnerTyp,size_t DimCount>
	class IArray : public IContainer<InnerTyp>
	{
	public:
		IArray(size_t(&shape)[DimCount]);
		const size_t Shape[DimCount];
	};
	
	template<class InnerTyp, size_t DimCount>
	inline IArray<InnerTyp, DimCount>::IArray(size_t(&shape)[DimCount]) : Shape(shape)
	{
	}
}
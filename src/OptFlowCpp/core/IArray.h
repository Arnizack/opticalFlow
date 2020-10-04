#pragma once
#include"IContainer.h"
#include<memory>

namespace core
{

	template<class InnerTyp,size_t DimCount>
	class IArray : public IContainer<InnerTyp>
	{
	public:
		virtual size_t[DimCount] Shape() = 0;

	};
	
}
#pragma once
#include"../linalg/ILinearOperator.h"
#include"../IArray.h"
#include<memory>
namespace core
{
	
	template<class InnerTyp, size_t DimCount>

	class IColorConvolution :
		public ILinearOperator< std::shared_ptr < IArray< InnerTyp, 3>>, 
		std::shared_ptr<IArray< InnerTyp, 3>>>
	{
	};
	
}
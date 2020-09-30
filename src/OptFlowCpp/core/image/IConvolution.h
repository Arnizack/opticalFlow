#pragma once
#include"../linalg/ILinearOperator.h"
#include"../IArray.h"
#include<memory>
namespace core
{
	namespace image
	{
		template<class InnerTyp, size_t DimCount>
		
		class IConvolution : 
			linalg::ILinearOperator< std::shared_ptr < IArray< InnerTyp, DimCount>>,
			std::shared_ptr<IArray< InnerTyp, DimCount>>, IConvolution<InnerTyp,DimCount> >
		{
		};
	}
}
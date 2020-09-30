#pragma once
#include"../linalg/ILinearOperator.h"
#include"../IArray.h"
#include<memory>
namespace core
{
	namespace image
	{
		template<class InnerTyp, size_t DimCount>

		class IColorConvolution :
			public linalg::ILinearOperator< std::shared_ptr < IArray< InnerTyp, 3>>, std::shared_ptr<IArray< InnerTyp, 3>>>
		{
		};
	}
}
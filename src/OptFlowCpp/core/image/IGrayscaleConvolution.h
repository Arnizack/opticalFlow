#pragma once
#include"../linalg/ILinearOperator.h"
#include"../IArray.h"
#include<memory>
namespace core
{

	template<class InnerTyp>
		
	class IGrayscaleConvolution :
			public ILinearOperator< 
		std::shared_ptr<IArray< InnerTyp, 2>>, 
		std::shared_ptr<IArray< InnerTyp, 2>>>
	{
	};
	
}
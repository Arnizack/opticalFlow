#pragma once
#include "..\linalg\IOperator.h"
#include "..\IArray.h"
#include <memory>

namespace core
{

	class IGrayscaleFilter : 
		public IOperator<std::shared_ptr<IArray<float, 2>>, 
		std::shared_ptr<IArray<float, 2>>>
	{
	};
	
}
#pragma once
#include "..\linalg\IOperator.h"
#include "..\IArray.h"
#include <memory>

namespace core
{

	class IColorFilter : 
		public IOperator<std::shared_ptr<IArray<float, 3>>, 
		std::shared_ptr<IArray<float, 3>>>
	{
	};
	
}
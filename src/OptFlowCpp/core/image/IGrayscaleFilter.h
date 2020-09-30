#pragma once
#include "..\linalg\IOperator.h"
#include "..\IArray.h"
#include <memory>

namespace core
{
	namespace image
	{
		class IGrayscaleFilter : 
			public linalg::IOperator<std::shared_ptr<IArray<float, 2>>, std::shared_ptr<IArray<float, 2>>>
		{
		};
	}
}
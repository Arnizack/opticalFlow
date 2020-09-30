#pragma once
#include "..\linalg\IOperator.h"
#include "..\IArray.h"
#include <memory>

namespace core
{
	namespace image
	{
		class IColorFilter : 
			public linalg::IOperator<std::shared_ptr<IArray<float, 3>>, std::shared_ptr<IArray<float, 3>>>
		{
		};
	}
}
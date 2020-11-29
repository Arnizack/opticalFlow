#pragma once
#include "../IScaler.h"
#include "../IArray.h"

namespace core
{

	class IGrayscaleScaler : public IScaler<IArray<float, 2>>
	{};
	
}
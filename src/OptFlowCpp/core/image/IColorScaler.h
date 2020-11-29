#pragma once
#include "../IScaler.h"
#include "../IArray.h"

namespace core
{

	class IColorScalable : public IScaler<IArray<float, 3>>
	{};
	
}
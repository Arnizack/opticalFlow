#pragma once
#include "IScalable.h"

namespace core
{
	namespace image
	{
		class IColorScalable : public IScalable<float, 3>
		{};
	}
}
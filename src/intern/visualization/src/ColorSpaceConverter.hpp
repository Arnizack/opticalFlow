#pragma once
#include"HSV.hpp"
#include"Color.h"
namespace visualization
{
	class ColorSpaceConverter
	{
	public:
		core::Color HSVToRGB(const HSV& hsv) const;
	};

}
                
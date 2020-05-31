#include "FlowFieldVisualizer.h"
#include"OpticalFlowMath.h"
#include"ColorSpaceConverter.hpp"
#include<cmath>
#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

namespace visualization
{
	FlowFieldVisualizer::FlowFieldVisualizer(float hueShift) : HueShift(hueShift)
	{
		
	}
	core::ImageRGB FlowFieldVisualizer::visualize(const core::FlowField& flow, SCALE scale)
	{
		uint32_t width = flow.GetWidth();
		uint32_t heigth = flow.GetHeight();
		core::ImageRGB result(width, heigth);
		// find Max Value
		float maxLength = 0;
		for (uint32_t x = 0; x < width; x++)
		{
			for (uint32_t y = 0; y < heigth; y++)
			{
				maxLength = std::max(core::Length(flow.GetVector(x, y)), maxLength);
			}
		}
		for (uint32_t x = 0; x < width; x++)
		{
			for (uint32_t y = 0; y < heigth; y++)
			{
				auto color = fieldVecToColor(flow.GetVector(x, y), maxLength, scale);
				result.SetPixel(x, y, color);
			}
		}
		return result;


	}
	core::Color FlowFieldVisualizer::fieldVecToColor(const core::FlowVector& vec, float maxLength, SCALE scale)
	{
		ColorSpaceConverter Converter = ColorSpaceConverter();
		float x = vec.vector_X;
		float y = vec.vector_Y;
		float length = core::Length(vec);
		length /= maxLength;

		float theta = atan(y / x);
		const float radiantToEuler = static_cast<float>( (180 / M_PI));
		float angle = theta * radiantToEuler+ HueShift;
		if (angle >= 360)
			angle -= 360;
		switch (scale)
		{
		case visualization::SCALE::LOG:
			length = log(length);
			break;
		case visualization::SCALE::LINEAR:
			break;
		case visualization::SCALE::SQRT:
			length = sqrt(length);
			break;
		case visualization::SCALE::QUARDATIC:
			length = length * length;
			break;
		default:
			break;
		}
		return Converter.HSVToRGB(HSV(angle, length, 1));

	}
}

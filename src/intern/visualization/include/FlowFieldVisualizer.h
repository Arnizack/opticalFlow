#pragma once
#include"FlowField.h"
#include"ImageRGB.h"
#include"FlowVector.h"
namespace visualization
{
	enum class SCALE
	{
		LOG, LINEAR, SQRT,QUARDATIC
	};

	class FlowFieldVisualizer
	{
	public:
		FlowFieldVisualizer(float hueShift=0);

		//Scales all the Flow Vector length, to be between 0 and 1. Then scales the Length with the selected Sacle;
		//Finaly it maps the Vector on an Color Circle
		core::ImageRGB visualize(const core::FlowField& flow, SCALE scale = SCALE::LINEAR);
		core::Color fieldVecToColor(const core::FlowVector& vec, float maxLength, SCALE scale = SCALE::LOG);
	private:
		
		
		float HueShift;
	};

}
                
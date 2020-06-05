#include"NaiveBilateralFlowFilter.hpp"
#include"OpticalFlowMath.h"

#include <windows.h>
#include <ppl.h>
#include<algorithm>

namespace cpu::bilateralfilter
{
	core::FlowField NaiveBilateralFilter::filter(cpu::bilateralfilter::ReliabilityMap& map, 
	core::FlowField& flow, core::ImageRGB& templateImg,float delta_d, float delta_c)
{
	uint32_t width = flow.GetWidth();
	uint32_t heigth = flow.GetHeight();

	//considered Image Region
	int consideredImagRegHalf = delta_d;

	core::FlowField resultFlow(width, heigth);

	concurrency::parallel_for(0, static_cast<int>(width), [&](int x_0) 

	//for (int x_0 = 0; x_0 < width; x_0++)
	{
		for (int y_0 = 0; y_0 < heigth; y_0++)
		{
			uint32_t x_min = max(x_0 - consideredImagRegHalf, 0);
			uint32_t x_max = min(x_0 + consideredImagRegHalf, (int)width);

			uint32_t y_min = max(y_0 - consideredImagRegHalf, 0);
			uint32_t y_max = min(y_0 + consideredImagRegHalf, (int)heigth);

			auto color_0 = templateImg.GetPixel(x_0, y_0);

			core::FlowVector vec_result;

			float divider = 0;

			float flow_x = 0;
			float flow_y = 0;

			for (uint32_t x = x_min; x < x_max; x++)
			{
				for (uint32_t y = y_min; y < y_max; y++)
				{
					auto color = templateImg.GetPixel(x, y);
					float distance_d = core::Distance(x, y, x_0, y_0);
					float distanceSqr_c = core::ColorSqueredDistance(color, color_0);
					float distanceSqr_d = distance_d * distance_d;
					float exponent_d = -distanceSqr_d / (2 * delta_d);
					float exponent_c = -distanceSqr_c / (2 * delta_c);
					float weigth = exp(exponent_c + exponent_d);

					weigth *= map.GetReliability(x, y);

					auto vec = flow.GetVector(x, y);

					flow_x += (float)vec.vector_X* weigth;
					flow_y += (float)vec.vector_Y * weigth;
					divider += weigth;
					
				}
			}

			flow_x /= divider;
			flow_y /= divider;

			resultFlow.SetVector(x_0, y_0, core::FlowVector(flow_x,flow_y));
		}
	}
	);
	return resultFlow;
}
}

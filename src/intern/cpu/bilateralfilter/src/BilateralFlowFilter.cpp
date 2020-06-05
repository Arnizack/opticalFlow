#include "BilateralFlowFilter.h"

#include <windows.h>
#include <ppl.h>
#include<algorithm>

namespace  cpu::bilateralfilter
{
	core::FlowField BilateralFlowFilter::filter(const core::FlowField & flow, const ReliabilityMap & map, kdtree::KDTreeData& tree, const core::ImageRGB templateImage)
	{
		uint32_t width = flow.GetWidth();
		uint32_t heigth = flow.GetHeight();
		core::FlowField resultFlow(width,heigth);

		concurrency::parallel_for(0, static_cast<int>(width), [&](int x)
			//for (uint32_t x = 0; x < width; x++)
		{
			for (uint32_t y = 0; y < heigth; y++)
			{
				auto color = templateImage.GetPixel(x, y);
				auto resultsQuery = kdtree::queryKDTree(tree, x, y, color);
				float divider = 0;

				float resultX = 0;
				float resultY = 0;

				for (const auto& result : resultsQuery)
				{
					float weigth = result.Weight * map.GetReliability(result.X, result.Y);
					core::FlowVector vec = flow.GetVector(result.X, result.Y);
					resultX += vec.vector_X * weigth;
					resultY += vec.vector_Y * weigth;
					divider += weigth;
				}

				core::FlowVector vec2(resultX / divider, resultY / divider);
				resultFlow.SetVector(x, y, vec2);

			}
		});
		return resultFlow;
	}
}
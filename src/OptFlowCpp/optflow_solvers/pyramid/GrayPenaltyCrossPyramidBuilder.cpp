#pragma once
#include"pch.h"
#include"GrayPenaltyCrossPyramidBuilder.h"
#include"Pyramid.h"

namespace optflow_solvers
{
    std::shared_ptr<core::IPyramid<core::IGrayPenaltyCrossProblem>> GrayPenaltyCrossPyramidBuilder::Create(std::shared_ptr<core::IGrayPenaltyCrossProblem> last_level)
    {
		size_t width = last_level->FirstFrame->Shape[2];
		size_t height = last_level->SecondFrame->Shape[1];
		switch (_resolution_definition)
		{
		case _inner::ResolutionDefinition::FACTORS:
			ComputeResolutionFromFactors(width, height);
			break;
		case _inner::ResolutionDefinition::FACTORS_MINRES:
			ComputeResolutionFromFactorMinRes(width, height);
		default:
			break;
		}

		std::vector<std::shared_ptr< core::IGrayPenaltyCrossProblem>> levels = CreateLayers(last_level);

		return std::make_shared<Pyramid<core::IGrayPenaltyCrossProblem>>(levels);

    }
}
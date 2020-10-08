#pragma once
#include<memory>
#include"../IArray.h"

namespace core
{
	class IGrayWarper
	{
	public:
		using PtrGrayImg = std::shared_ptr<IArray<float, 2>>;
		using PtrFlowField = std::shared_ptr<IArray<double, 3>>;

		virtual PtrGrayImg Warp(PtrGrayImg Image, PtrFlowField Flow) = 0;

	};
}
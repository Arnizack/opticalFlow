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

		virtual void SetImage(PtrGrayImg Image) = 0;

		virtual PtrGrayImg Warp(PtrFlowField Flow) = 0;

	};
}
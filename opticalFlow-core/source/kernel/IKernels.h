#pragma once
#include"datastructures/DeviceData.h"
#include<memory>

namespace kernel
{
	enum class DIRECTION
	{
		X, Y
	};
	class IKernels
	{
	public:
		virtual bool multArray(int size, std::shared_ptr< datastructures::IDeviceArray<float>> src, float scalar, std::shared_ptr<datastructures::IDeviceArray<float>> dst) = 0;
	};
}
#pragma once
#include"datastructurs/DeviceData.h"
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
		virtual bool multArray(int size, std::shared_ptr< datastructurs::IDeviceArray<float>> src, float scalar, std::shared_ptr<datastructurs::IDeviceArray<float>> dst) = 0;
	};
}
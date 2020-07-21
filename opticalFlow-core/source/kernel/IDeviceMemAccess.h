#pragma once
#include"datastructurs/DeviceData.h"
namespace kernel
{
	class IDeviceMemAccess
	{
	public:
		
		void load(float* dst, datastructures::IDeviceArray<float> data);
		void load(int* dst, datastructures::IDeviceArray<int> data);

	};



}
                
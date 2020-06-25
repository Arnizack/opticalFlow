#pragma once
#include"datastructurs/DeviceData.h"
namespace kernel
{
	class IDeviceMemAccess
	{
	public:
		
		void load(float* dst, datastructurs::IDeviceArray<float> data);
		void load(int* dst, datastructurs::IDeviceArray<int> data);

	};



}
                
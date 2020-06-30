#pragma once
#include"datastructurs/DeviceData.h"
namespace kernel
{
	class IDeviceMemAccess
	{
	public:
		
		virtual void load(float* dst,const datastructurs::IDeviceArray<float>& data) = 0;
		virtual void load(int* dst, const datastructurs::IDeviceArray<int>& data) = 0;
		virtual void load(float* dst, const datastructurs::IDevice2DMatrix<float,1>& data) = 0;
		virtual void load(float* dst, const datastructurs::IDevice2DMatrix<float, 2>& data) = 0;

		virtual void load(float* dst, const datastructurs::IDevice2DMatrix<float, 4>& data) = 0;


	};



}
                
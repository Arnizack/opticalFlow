#pragma once
#include"datastructures/DeviceData.h"
namespace kernel
{
	class IDeviceMemAccess
	{
	public:
		
		virtual void load(float* dst,const datastructures::IDeviceArray<float>& data) = 0;
		virtual void load(int* dst, const datastructures::IDeviceArray<int>& data) = 0;
		virtual void load(float* dst, const datastructures::IDevice2DMatrix<float,1>& data) = 0;
		virtual void load(float* dst, const datastructures::IDevice2DMatrix<float, 2>& data) = 0;

		virtual void load(float* dst, const datastructures::IDevice2DMatrix<float, 4>& data) = 0;


	};



}
                
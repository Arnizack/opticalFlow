#pragma once
#include"datastructurs/DeviceData.h"

namespace kernel
{
	enum class DIRECTION
	{
		X, Y
	};
	class IKernels
	{
	public:
		virtual bool convolution1d(const datastructures::IDevice2DMatrix<float, 4>&  source, datastructures::IDevice2DMatrix<float, 4>& destination,const datastructures::IDeviceArray<float>& kernel,
			DIRECTION dir) = 0;
	};
}
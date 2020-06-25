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
		virtual bool convolution1d(const datastructurs::IDevice2DMatrix<float, 4>&  source, datastructurs::IDevice2DMatrix<float, 4>& destination,const datastructurs::IDeviceArray<float>& kernel,
			DIRECTION dir) = 0;
	};
}
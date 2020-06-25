#pragma once
#include"kernel/IKernels.h"


namespace kernel
{


	class CPUKernels : public IKernels
	{
	public:
		// Inherited via IKernels
		

		// Inherited via IKernels
		virtual bool convolution1d(const datastructurs::IDevice2DMatrix<float, 4>& source, datastructurs::IDevice2DMatrix<float, 4>& destination, const datastructurs::IDeviceArray<float>& kernel, DIRECTION dir) override;

	};

}
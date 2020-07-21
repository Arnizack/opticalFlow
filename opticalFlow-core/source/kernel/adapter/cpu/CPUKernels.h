#pragma once
#include"kernel/IKernels.h"


namespace kernel
{


	class CPUKernels : public IKernels
	{
	public:
		// Inherited via IKernels
		

		// Inherited via IKernels
		virtual bool convolution1d(const datastructures::IDevice2DMatrix<float, 4>& source, datastructures::IDevice2DMatrix<float, 4>& destination, const datastructures::IDeviceArray<float>& kernel, DIRECTION dir) override;

	};

}
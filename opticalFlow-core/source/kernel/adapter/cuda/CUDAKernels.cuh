#pragma once
#pragma once
#include"kernel/IKernels.h"


namespace kernel
{


	class CUDAKernels : public IKernels
	{
	public:

		// Inherited via IKernels
		virtual bool multArray(int size, std::shared_ptr<datastructures::IDeviceArray<float>> src, float scalar, std::shared_ptr<datastructures::IDeviceArray<float>> dst) override;
	};

}
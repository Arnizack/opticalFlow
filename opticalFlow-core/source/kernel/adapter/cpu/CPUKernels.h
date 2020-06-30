#pragma once
#include"kernel/IKernels.h"


namespace kernel
{


	class CPUKernels : public IKernels
	{
	public:

		// Inherited via IKernels
		virtual bool multArray(int size, std::shared_ptr<datastructurs::IDeviceArray<float>> src, float scalar, std::shared_ptr<datastructurs::IDeviceArray<float>> dst) override;
	};

}
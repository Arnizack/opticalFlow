#include "CPUKernels.h"

#include"platforms/cpu/CPUMacros.h"
#include"kernel/kernels/multArray.h"
#include"platforms/cpu/launchers/serializeLaunch.h"
#include"platforms/cpu/datastructures/HostArray.h"
#include"platforms/cpu/CPUBackend.h"


namespace kernel
{


	



	bool CPUKernels::multArray(int size, std::shared_ptr<datastructures::IDeviceArray<float>> src, float scalar, std::shared_ptr<datastructures::IDeviceArray<float>> dst)
	{
		/*
		auto srcH = std::static_pointer_cast<cpu::Host_Array<float>>(src);
		auto dstH = std::static_pointer_cast<cpu::Host_Array<float>>(dst);

		cpu::BackendCPU::ds::ArrayMetaData<float> srcMD = srcH->getKernelData();
		cpu::BackendCPU::ds::ArrayMetaData<float> dstMD = dstH->getKernelData();
		
		

		cpu::launchSerial(kernels::multKernel<cpu::BackendCPU>::multArray,size, srcMD, scalar, dstMD);
		*/
		return true;
	}

}

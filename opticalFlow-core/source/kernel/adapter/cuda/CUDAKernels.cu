#pragma once
#include "CUDAKernels.cuh"
#include"platforms/cuda/CUDAMacros.h"
#include"platforms/cuda/datastructures/HostDeviceArray.h"
#include"platforms/cuda/CUDABackend.cuh"
#include"platforms/cuda/kernelLauncher.cuh"
#include"kernel/kernels/multArray.h"

using backend = cuda::BackendCUDA;

bool kernel::CUDAKernels::multArray(int size, std::shared_ptr<datastructures::IDeviceArray<float>> src, float scalar, std::shared_ptr<datastructures::IDeviceArray<float>> dst)
{
	using d_array = backend::ds::Array<float>;
	
	auto srcH = std::static_pointer_cast<datastructures::HostDeviceArray<float>>(src);
	auto dstH = std::static_pointer_cast<datastructures::HostDeviceArray<float>>(dst);

	d_array srcD = srcH->getCudaArray();
	d_array dstD = dstH->getCudaArray();
	KernelLauncher launcher(0,32);
	launcher.launchWithParameters<kernels::multKernel<backend>,int,d_array,float,d_array>(2,2,12,srcH->size(),srcD,scalar,dstD);
	launcher.wait();

	return true;
}

#include "CUDADeviceMemAccess.hpp"
#include "platforms/cuda/datastructures/HostDevice2DMatrix.h"
#include "platforms/cuda/datastructures/HostDeviceArray.h"

void cuda::CUDADeviceMemAccess::load(float* dst, const datastructures::IDeviceArray<float>& data)
{
	const datastructures::HostDeviceArray<float>& cuda_data = static_cast<const datastructures::HostDeviceArray<float>&>(data);
	cuda_data.to_host(dst);
}

void cuda::CUDADeviceMemAccess::load(int* dst, const datastructures::IDeviceArray<int>& data)
{
	const datastructures::HostDeviceArray<int>& cuda_data = static_cast<const datastructures::HostDeviceArray<int>&>(data);
	cuda_data.to_host(dst);
}

void cuda::CUDADeviceMemAccess::load(float* dst, const datastructures::IDevice2DMatrix<float, 1>& data)
{
	const datastructures::HostDevice2DMatrix<float,1>& cuda_data = static_cast<const datastructures::HostDevice2DMatrix<float,1>&>(data);
	cuda_data.to_host(dst);
}

void cuda::CUDADeviceMemAccess::load(float* dst, const datastructures::IDevice2DMatrix<float, 2>& data)
{
	const datastructures::HostDevice2DMatrix<float, 2>& cuda_data = static_cast<const datastructures::HostDevice2DMatrix<float, 2>&>(data);
	cuda_data.to_host(dst);
}

void cuda::CUDADeviceMemAccess::load(float* dst, const datastructures::IDevice2DMatrix<float, 4>& data)
{
	const datastructures::HostDevice2DMatrix<float, 4>& cuda_data = static_cast<const datastructures::HostDevice2DMatrix<float, 4>&>(data);
	cuda_data.to_host(dst);
}

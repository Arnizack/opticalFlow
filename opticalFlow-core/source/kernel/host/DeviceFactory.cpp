#include "DeviceFactory.h"
#include"kernel/adapter/cpu/CPUDeviceBuilder.h"
#include"kernel/adapter/cuda/CUDADeviceBuilder.h"

namespace kernel
{

	std::shared_ptr<Device> kernel::DeviceFactory::createCPUDevice()
	{
		CPUDeviceBuilder builder;
		
		if (builder.isAvailable())
		{
			auto dataFactory = builder.createDataStrucFactory();
			auto memAccess = builder.createMemAccesser();
			auto kernels = builder.createKernels();
			auto result = std::make_shared<Device>(
				std::move(dataFactory),
				std::move(memAccess),
				std::move(kernels)
				);
			return result;
		}
		
		return nullptr;
	}

	std::shared_ptr<Device> DeviceFactory::createCUDADevice()
	{
		CUDADeviceBuilder builder;
		if (builder.isAvailable())
		{
			auto dataFactory = builder.createDataStrucFactory();
			auto memAccess = builder.createMemAccesser();
			auto kernels = builder.createKernels();
			auto result = std::make_shared<Device>(
				std::move(dataFactory),
				std::move(memAccess),
				std::move(kernels)
				);
			return result;
		}

		return nullptr;
	}
	


}
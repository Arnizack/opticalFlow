#include "DeviceFactory.h"
#include"kernel/adapter/cpu/CPUDeviceBuilder.h"

namespace kernel
{

	std::shared_ptr<Device> kernel::DeviceFactory::createCPUDevice()
	{
		CPUDeviceBuilder builder;
		if (builder.isAvailable())
		{
			auto result = std::make_shared<Device>(
				builder.createDataStrucFactory(),
				builder.createMemAccesser(),
				builder.createKernels()
				);
			return result;
		}
		return nullptr;
	}
}
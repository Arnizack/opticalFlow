#pragma once
#include<memory>
#include"kernel/IDataStructuresFactory.h"
#include"kernel/IDeviceMemAccess.h"
#include"kernel/IKernels.h"


namespace kernel
{
	struct Device
	{
	public:
		Device(std::unique_ptr<IDataStructuresFactory>&& dsFactory,
			std::unique_ptr<IDeviceMemAccess>&& dmAccess,
			std::unique_ptr<IKernels>&& kernels);

		std::unique_ptr<IDataStructuresFactory> DataStructureFactory;
		std::unique_ptr<IDeviceMemAccess> DeviceMemoryAccess;
		std::unique_ptr<IKernels> Kernels;

	};
}
#pragma once
#include"kernel/IDataStructuresFactory.h"
#include"kernel/IDeviceMemAccess.h"
#include"kernel/IKernels.h"

namespace kernel
{
	class CPUDeviceBuilder
	{
	public:
		bool isAvailable();
		std::unique_ptr<IKernels> createKernels();
		std::unique_ptr<IDataStructuresFactory> createDataStrucFactory();
		std::unique_ptr<IDeviceMemAccess> createMemAccesser();


	};
}
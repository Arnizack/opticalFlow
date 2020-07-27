#include"CPUDeviceBuilder.h"
#include"CPUKernels.h"
#include"platforms/cpu/CPUDataStructsFactory.hpp"
#include"platforms/cpu/DeviceMemAccess.hpp"
#include"kernel/adapter/cpu/CPUKernels.h"

namespace kernel
{
	bool CPUDeviceBuilder::isAvailable()
	{
		return true;
	}
	std::unique_ptr<IKernels> CPUDeviceBuilder::createKernels()
	{
		return std::make_unique<kernel::CPUKernels>();
	}
	std::unique_ptr<IDataStructuresFactory> CPUDeviceBuilder::createDataStrucFactory()
	{
		auto result = std::make_unique<cpu::CPUDataStructsFactory>();
		return result;
	}
	std::unique_ptr<IDeviceMemAccess> CPUDeviceBuilder::createMemAccesser()
	{
		auto result = std::make_unique<cpu::DeviceMemAccess>();
		return result;
	}
}
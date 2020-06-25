#include"CPUDeviceBuilder.h"
#include"CPUKernels.h"

namespace kernel
{
	bool CPUDeviceBuilder::isAvailable()
	{
		return true;
	}
	std::unique_ptr<IKernels> CPUDeviceBuilder::createKernels()
	{
		return std::make_unique<CPUKernels>();
	}
}
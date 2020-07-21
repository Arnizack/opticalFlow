#include"Device.h"

namespace kernels
{

}



kernel::Device::Device(std::unique_ptr<kernel::IDataStructuresFactory>&& dsFactory, std::unique_ptr<IDeviceMemAccess>&& dmAccess, std::unique_ptr<IKernels>&& kernels)

{
	DataStructureFactory =std::move(dsFactory);
	DeviceMemoryAccess = std::move(dmAccess);
	Kernels = std::move(kernels);
}

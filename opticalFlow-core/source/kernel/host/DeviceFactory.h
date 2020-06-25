#pragma once
#include"Device.h"
#include<memory>


namespace kernel
{
	class DeviceFactory
	{

		public:
			std::shared_ptr<Device> createCPUDevice();
			std::shared_ptr<Device> createCUDADevice();
	};
}
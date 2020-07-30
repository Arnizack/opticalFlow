#include<gtest/gtest.h>
#include"kernel/host/DeviceFactory.h"
#include"datastructures/DeviceData.h"
#include"utilities/Timer.h"
#include<array>

template<class T>
class TD;

namespace kernls
{
	void unitTestMultArray(std::shared_ptr<kernel::Device> dev)
	{

		constexpr int size = 10000;



		auto src = std::make_unique<std::array<float, size>>();
		auto dst = std::make_unique<std::array<float, size>>();


		auto srcB = dev->DataStructureFactory->createArray(src->data(), size);
		auto dstB = dev->DataStructureFactory->createArray(dst->data(), size);

		utilities::Timer t1("Kernel");

		dev->Kernels->multArray(size, srcB, 2, dstB);

		t1.Stop();

		dev->DeviceMemoryAccess->load(dst->data(), *dstB);


		auto src2 = std::make_unique<std::array<float, size>>();
		auto dst2 = std::make_unique<std::array<float, size>>();


		utilities::Timer t2("Naive");

		for (int i = 0; i < size; i++)
		{
			(*dst2)[i] = (*src2)[i] * 2;
		}
		t2.Stop();

		for (int i = 0; i < size; i++)
		{
			EXPECT_EQ((*dst)[i], (*src)[i] * 2);
		}
	}

	TEST(kernels, multArray)
	{
		
		auto devFactory = kernel::DeviceFactory();
		
		
		std::shared_ptr<kernel::Device> cpu_dev = devFactory.createCPUDevice();
		unitTestMultArray(cpu_dev);
		
		std::shared_ptr<kernel::Device> cuda_dev = devFactory.createCUDADevice();
		unitTestMultArray(cuda_dev);
	}

}
#include<gtest/gtest.h>
#include"kernel/host/DeviceFactory.h"
#include"datastructurs/DeviceData.h"
#include"utilities/Timer.h"
#include<array>

template<class T>
class TD;

namespace kernls
{
	TEST(kernels, multArray)
	{
		auto devFactory = kernel::DeviceFactory();
		
		
		std::shared_ptr<kernel::Device> dev = devFactory.createCPUDevice();
		
		constexpr int size = 10000;

		

		std::array<float, size> src = { 1 };
		std::array<float, size> dst = { 1 };

		
		auto srcB = dev->DataStructureFactory->createArray(src.data(), size);
		auto dstB = dev->DataStructureFactory->createArray(dst.data(), size);
		
		utilities::Timer t1("Kernel");
		
		dev->Kernels->multArray(size, srcB, 2, dstB);
		
		t1.Stop();

		dev->DeviceMemoryAccess->load(dst.data(),*dstB);
		
		std::array<float, size> src2 = { 1 };
		std::array<float, size> dst2 = { 1 };



		utilities::Timer t2("Naive");

		for (int i = 0; i < size; i++)
		{
			dst2[i] = src2[i] * 2;
		}
		t2.Stop();

		for (int i = 0; i < size; i++)
		{
			EXPECT_EQ(dst[i], src[i] * 2);
		}


	}

}
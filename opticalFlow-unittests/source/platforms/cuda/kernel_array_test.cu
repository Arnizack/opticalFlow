
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "platforms/cuda/datastructures/HostDeviceArray.h"
#include<gtest/gtest.h>

__global__ void testArray(int* array, int* dst, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		sum += array[i];
		printf("sum: %d\n", sum);
	}
	*dst = sum;
}

TEST(cuda, kernel_array_test)
{
	std::vector<int> src_array(200);
	int counter = 0;
	for (int& item : src_array)
		item = counter++;
	
	int* d_dst;
	cudaMalloc(&d_dst, sizeof(int));
	datastructures::HostDeviceArray<int> h_array(src_array.data(), src_array.size());
	int* d_array = h_array.getCudaArray();
	testArray<<<1,1>>>(d_array, d_dst, src_array.size());

	int actual_sum;

	cudaMemcpy(&actual_sum, d_dst, sizeof(int), cudaMemcpyDeviceToHost);

	int expected_sum = 0;
	
	for (const int& item : src_array)
		expected_sum += item;

	EXPECT_EQ(actual_sum, expected_sum);

}
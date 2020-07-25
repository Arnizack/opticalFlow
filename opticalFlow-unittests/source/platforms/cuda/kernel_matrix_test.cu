
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "platforms/cuda/datastructures/HostDevice2DMatrix.h"
#include<gtest/gtest.h>


__global__ void testMatrix(datastructures::ThreadDevice2DMatrix<int4> mat, int width, int heigth,int* dst)
{
	int sum = 0;
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < heigth; y++)
		{
			int2 idx = make_int2(x, y);
			sum += mat[idx].x + mat[idx].y + mat[idx].z + mat[idx].w;
		}
	}
	*dst = sum;
}

template<typename T>
class TD;

TEST(cuda, kernel_matrix_test)
{
	int* d_dst;
	cudaMalloc(&d_dst, sizeof(int));
	
	std::vector<int> data(4*4);
	int count = 0;
	for (int& item : data)
		item = count++;

	datastructures::HostDevice2DMatrix<int, 4> mat(data.data(), 2, 2);
	datastructures::ThreadDevice2DMatrix<int4> cuda_mat = mat.getCuda2DMatrix();
	


	testMatrix<<<1,1>>>(cuda_mat,2,2,d_dst);
	cudaDeviceSynchronize();
	
	int actual_sum;

	cudaMemcpy(&actual_sum, d_dst, sizeof(int), cudaMemcpyDeviceToHost);

	int expected_sum = 0;

	for (const int& number : data)
		expected_sum += number;

	EXPECT_EQ(expected_sum, actual_sum);

}
#include "kernelLauncher.hpp"
#include<algorithm>

KernelLauncher::KernelLauncher(const int& sharedMemoryCount, const int& blockSize)
	: SharedMemoryCount(sharedMemoryCount), blockCount(blockSize)
{
}

bool KernelLauncher::considerTilesBuffer(const int& dataWidth, const int& dataHeigth, const int& tilesWidth,
	const int& tilesHeigth, const int& paddingXRadius, const int& paddingYRadius, const int& typeSize)
{
	if (dataWidth >= 1 && dataHeigth >= 1 && tilesHeigth >= 1 && tilesWidth >= 1
		&& paddingXRadius >= 0 && paddingYRadius >= 0)
	{
		b_0 += dataWidth * dataHeigth * typeSize;
		b_1 += dataHeigth * (tilesWidth + 2 * paddingXRadius) * typeSize;
		b_2 += dataWidth * (tilesHeigth + 2 * paddingYRadius) * typeSize;
		b_3 += (tilesWidth + 2 * paddingXRadius) * (tilesHeigth + 2 * paddingYRadius) * typeSize;
		return true;
	}
	return false;
}

void KernelLauncher::wait()
{
	cudaDeviceSynchronize();
}

void KernelLauncher::calcParameters()
{
	int S = SharedMemoryCount;
	double insqrt = b_1 * b_2 * (b_0 * (S - b_3) + b_1 * b_2);
	float xgrid = b_1 * b_2 + sqrt(insqrt);// std::max(b_1 * b_2 + sqrt(insqrt), b_1 * b_2 - sqrt(insqrt));
	float ygrid = xgrid;
	xgrid /= b_1 * (S - b_3);
	ygrid /= b_2 * (S - b_3);

	printf("gridDim float: %f,%f\n", xgrid, ygrid);

	gridDimX = ceil(xgrid);
	gridDimY = ceil(ygrid);



	printf("gridDim: %d,%d\n", gridDimX, gridDimY);
}

#pragma once
struct KernelInfo
{
	char* SharedMemStart;

	inline __device__ KernelInfo(void* sharedMemStart) : SharedMemStart(static_cast<char*>(sharedMemStart))
	{
	}
};
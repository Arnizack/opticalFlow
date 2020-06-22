#pragma once
#include<cuda_runtime.h>

template<typename T>
class device_ptr
{
private:
	T* rawpointer;
	
public:
	__host__ device_ptr(T* ptr)
	{

	}

	


};
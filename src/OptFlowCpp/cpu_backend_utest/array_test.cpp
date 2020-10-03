#include "pch.h"
#include "..\cpu_backend\CpuArray.h"
#include "..\core\IContainer.h"


namespace cpu_utext
{
	TEST(CpuArrayTest, InitializerListConstructor)
	{
		const int size = 4;
		size_t shape[3] = { 2,2,1 };
		const int arr[4] = { 1,2,3,4 };
		cpu::Array<int, 3> obj(shape, arr);
	}
}
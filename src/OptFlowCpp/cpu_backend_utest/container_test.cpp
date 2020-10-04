#include "pch.h"
#include "..\cpu_backend\CpuContainer.h"
#include "..\core\IContainer.h"


namespace cpu_utext
{
	TEST(CpuContainerTest, SizePtrConstructor)
	{
		const int size_given = 10;
		int arr[size_given] = { 1,2,3,4,5,6,7,8,9,10 };

		cpu::Container<int> obj(size_given, arr);

		int size_set = obj.Size();
		int* ptr;
		obj.CopyDataTo(ptr);

		const int test = obj[2];

		EXPECT_EQ(size_set, size_given);
		for (int i = 0; i < size_given; i++)
		{
			EXPECT_EQ(obj[i], arr[i]);
			EXPECT_EQ(ptr[i], arr[i]);
		}
	}

	TEST(CpuContainerTest, InitializerListConstructor)
	{
		const int size_given = 10;
		int arr[size_given] = { 1,2,3,4,5,6,7,8,9,10 };
		cpu::Container<int> obj = { 1,2,3,4,5,6,7,8,9,10 };
		int size_set = obj.Size();
		int* ptr;
		obj.CopyDataTo(ptr);

		EXPECT_EQ(size_set, size_given);
		for (int i = 0; i < size_given; i++)
		{
			EXPECT_EQ(obj[i], arr[i]);
			EXPECT_EQ(ptr[i], arr[i]);
		}
	}

	TEST(CpuContainerTest, DefaultConstructor)
	{
		cpu::Container<int> obj();
		int size = 0;
		EXPECT_EQ(size, 0);
	}
}
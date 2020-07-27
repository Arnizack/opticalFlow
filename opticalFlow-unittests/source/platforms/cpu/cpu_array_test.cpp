#include <gtest/gtest.h>

#include "platforms/cpu/datastructures/HostArray.h"
#include <vector>

TEST(cpu, host_array_test)
{
	std::vector<int> control(200);

	for (int i = 0; i < 200; i++)
	{
		control[i] = i;
	}

	cpu::HostArray<int> test(control.data(), 200);

	const int *const temp = test.getKernelData();

	std::vector<int> testVector = test.getVector();

	EXPECT_EQ(control[121], testVector[121]);

	int value;

	for (int i = 0; i < 200; i++)
	{
		value = temp[i];
		EXPECT_EQ(control[i], value);
	}
}
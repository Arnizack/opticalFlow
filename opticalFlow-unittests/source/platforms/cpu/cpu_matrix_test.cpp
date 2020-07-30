#include <gtest/gtest.h>

#include <vector>

#include "platforms/cpu/datastructures/Host2DMatrix.h"
#include"platforms/cpu/CPUBackend.h"

TEST(cpu, host_matrix_test)
{
	cpu::Vec<int, 2> data[8];

	for (int i = 0; i < 8; i++)
	{
		data[i] = cpu::Vec<int, 2> {i, i};
	}

	cpu::Host2DMatrix<int, 2> test(data, 4, 2);
	cpu::Matrix2D<int, 2> temp = test.get2DMatrix();

	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(data[i][0], temp[cpu::dataTypesCPU::int2(i, 0)].x);
		EXPECT_EQ(data[i][1], temp[cpu::dataTypesCPU::int2(i, 0)].y);
		//EXPECT_EQ(data[i][0], temp[{i,0}].x);
		//EXPECT_EQ(data[i][1], temp[{i, 0}].y);
	}
}
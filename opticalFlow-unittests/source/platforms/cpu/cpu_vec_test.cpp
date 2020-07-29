#include "platforms/cpu/datastructures/Vec2D.h"
#include "platforms/cpu/datastructures/Vec3D.h"
#include "platforms/cpu/datastructures/Vec4D.h"
#include "platforms/cpu/datastructures/Vec.h"

#include <gtest/gtest.h>

TEST(cpu, Vec_nDim_Test)
{
	cpu::Vec<int, 7> test ( 1,2,3,4,5,6,7 );
	static const cpu::Vec<int, 7> temp = { 8,9,10,11,12,13,14 };

	int count;

	for (int i = 0; i < 7; i++)
	{
		count = i + 1;
		EXPECT_EQ(test[i], count);
	}

	test = temp;

	for (int i = 0; i < 7; i++)
	{
		count = i + 8;
		EXPECT_EQ(test[i], count);
	}

	test += temp;

	for (int i = 0; i < 7; i++)
	{
		count = (i + 8) * 2;
		EXPECT_EQ(test[i], count);
	}

	test -= temp;

	for (int i = 0; i < 7; i++)
	{
		count = (i + 8);
		EXPECT_EQ(test[i], count);
	}

	test *= 2;

	for (int i = 0; i < 7; i++)
	{
		count = (i + 8) * 2;
		EXPECT_EQ(test[i], count);
	}

	test /= 2;

	for (int i = 0; i < 7; i++)
	{
		count = (i + 8);
		EXPECT_EQ(test[i], count);
	}
}

TEST(cpu, Vec_2D_Test)
{
	cpu::Vec2D<int> test { 1,2 };
	static const cpu::Vec2D<int> temp ( 3,4 );

	static const cpu::Vec<int, 2> constructorTestVec { 10,11 };
	static const cpu::Vec2D<int> constructorTestVec2D (constructorTestVec);

	EXPECT_EQ(test.x, 1);
	EXPECT_EQ(test.y, 2);

	test = temp;

	EXPECT_EQ(test.x, 3);
	EXPECT_EQ(test.y, 4);

	test += temp;

	EXPECT_EQ(test.x, 6);
	EXPECT_EQ(test.y, 8);

	test -= temp;

	EXPECT_EQ(test.x, 3);
	EXPECT_EQ(test.y, 4);

	test *= 2;

	EXPECT_EQ(test.x, 6);
	EXPECT_EQ(test.y, 8);

	test /= 2;

	EXPECT_EQ(test.x, 3);
	EXPECT_EQ(test.y, 4);

	EXPECT_EQ(constructorTestVec2D.x, 10);
	EXPECT_EQ(constructorTestVec2D.y, 11);
}

TEST(cpu, Vec_3D_Test)
{
	cpu::Vec3D<int> test{ 1,2,3 };
	static const cpu::Vec3D<int> temp(4,5,6);

	static const cpu::Vec<int, 3> constructorTestVec{ 10,11,12 };
	static const cpu::Vec3D<int> constructorTestVec2D(constructorTestVec);

	EXPECT_EQ(test.x, 1);
	EXPECT_EQ(test.y, 2);
	EXPECT_EQ(test.z, 3);

	test = temp;

	EXPECT_EQ(test.x, 4);
	EXPECT_EQ(test.y, 5);
	EXPECT_EQ(test.z, 6);

	test += temp;

	EXPECT_EQ(test.x, 8);
	EXPECT_EQ(test.y, 10);
	EXPECT_EQ(test.z, 12);

	test -= temp;

	EXPECT_EQ(test.x, 4);
	EXPECT_EQ(test.y, 5);
	EXPECT_EQ(test.z, 6);

	test *= 2;

	EXPECT_EQ(test.x, 8);
	EXPECT_EQ(test.y, 10);
	EXPECT_EQ(test.z, 12);

	test /= 2;

	EXPECT_EQ(test.x, 4);
	EXPECT_EQ(test.y, 5);
	EXPECT_EQ(test.z, 6);

	EXPECT_EQ(constructorTestVec2D.x, 10);
	EXPECT_EQ(constructorTestVec2D.y, 11);
	EXPECT_EQ(constructorTestVec2D.z, 12);
}

TEST(cpu, Vec_4D_Test)
{
	cpu::Vec4D<int> test{ 1,2,3,4 };
	static const cpu::Vec4D<int> temp(5, 6, 7, 8);

	static const cpu::Vec<int, 4> constructorTestVec { 10,11,12,13 };
	static const cpu::Vec4D<int> constructorTestVec2D(constructorTestVec);

	EXPECT_EQ(test.x, 1);
	EXPECT_EQ(test.y, 2);
	EXPECT_EQ(test.z, 3);
	EXPECT_EQ(test.w, 4);

	test = temp;

	EXPECT_EQ(test.x, 5);
	EXPECT_EQ(test.y, 6);
	EXPECT_EQ(test.z, 7);
	EXPECT_EQ(test.w, 8);

	test += temp;

	EXPECT_EQ(test.x, 10);
	EXPECT_EQ(test.y, 12);
	EXPECT_EQ(test.z, 14);
	EXPECT_EQ(test.w, 16);

	test -= temp;

	EXPECT_EQ(test.x, 5);
	EXPECT_EQ(test.y, 6);
	EXPECT_EQ(test.z, 7);
	EXPECT_EQ(test.w, 8);

	test *= 2;

	EXPECT_EQ(test.x, 10);
	EXPECT_EQ(test.y, 12);
	EXPECT_EQ(test.z, 14);
	EXPECT_EQ(test.w, 16);

	test /= 2;

	EXPECT_EQ(test.x, 5);
	EXPECT_EQ(test.y, 6);
	EXPECT_EQ(test.z, 7);
	EXPECT_EQ(test.w, 8);

	EXPECT_EQ(constructorTestVec2D.x, 10);
	EXPECT_EQ(constructorTestVec2D.y, 11);
	EXPECT_EQ(constructorTestVec2D.z, 12);
	EXPECT_EQ(constructorTestVec2D.w, 13);
}
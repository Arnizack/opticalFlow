#include<gtest/gtest.h>
#include<FlowField.h>
#include<iostream>


TEST(FlowFieldTest, vectorTest)
{
	std::uint32_t width = rand() % 100 + 1;
	std::uint32_t height = rand() % 100 + 1;

	core::FlowField test_field(width, height);
	core::FlowVector temp_vector;

	ASSERT_EQ(test_field.GetWidth(), width) << "Widths differ";
	ASSERT_EQ(test_field.GetHeight(), height) << "Heights differ";

	for (std::uint32_t i = 0; i < height; i++)
	{
		for (std::uint32_t j = 0; j < width; j++)
		{
			temp_vector.vector_X = rand() % 100 + 1;
			temp_vector.vector_Y = rand() % 100 + 1;
			test_field.SetVector(j, i, temp_vector);
			EXPECT_EQ(test_field.GetVector(j, i).vector_X, temp_vector.vector_X) << "Vector X Values differ at " << j << " " << i;
			EXPECT_EQ(test_field.GetVector(j, i).vector_Y, temp_vector.vector_Y) << "Vector Y Values differ at " << j << " " << i;
		}
	}
	const core::FlowVector *copy_field = test_field.Data();

	for (std::uint32_t i = 0; i < height; i++)
	{
		for (std::uint32_t j = 0; j < width; j++)
		{
			EXPECT_EQ(test_field.GetVector(j, i).vector_X, copy_field[j + i * width].vector_X) << "Vector X Values differ at " << j << " " << i;
			EXPECT_EQ(test_field.GetVector(j, i).vector_Y, copy_field[j + i * width].vector_Y) << "Vector Y Values differ at " << j << " " << i;
		}
	}
}

TEST(FlowFieldTest, uscalingTest)
{
	std::uint32_t width = 2;
	std::uint32_t height = 2;
	std::uint32_t target_width = 4;
	std::uint32_t target_height = 4;

	core::FlowField test_field(width, height);

	core::FlowVector temp_vector;
	temp_vector.vector_X = 1;
	temp_vector.vector_Y = 1;

	for (std::int32_t i = 0; i < height; i++)
	{
		for (std::int32_t j = 0; j < width; j++)
		{
			temp_vector.vector_X = i+j+1;
			temp_vector.vector_Y = i+j+1;
			test_field.SetVector(j, i, temp_vector);
		}
	}

	core::FlowField upscaled_field = test_field.Upsize(target_height, target_height);

	ASSERT_EQ(target_width, upscaled_field.GetWidth()) << "Width differs";
	ASSERT_EQ(target_height, upscaled_field.GetHeight()) << "height differs";
	ASSERT_EQ(2, upscaled_field.GetVector(0, 0).vector_X);
}
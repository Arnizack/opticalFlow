#include<gtest/gtest.h>
#include"image_helper/ImageHelper.h"
#include"cpu_backend/Array.h"

#define _SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING

namespace imagehelper
{
	TEST(image_helper, test1)
	{
		std::array<float, 16> expected_img =
		{ 0,
			1.0 / 16.0,2.0 / 16.0,3.0 / 16.0,4.0 / 16.0,
			5.0 / 16.0,6.0 / 16.0,7.0 / 16.0,8.0 / 16.0,
			9.0 / 16.0,10.0 / 16.0,11.0 / 16.0,12.0 / 16.0,
			13.0 / 16.0,14.0 / 16.0,15.0 / 16.0 };

		std::string path = "test.png";
		std::array<const size_t, 2> shape = { 4,4 };
		auto img = std::make_shared< cpu_backend::Array<float, 2>>
			(shape, expected_img.data());

		imagehelper::SaveImage(path, img);
		Image opened_img = imagehelper::OpenImage(path);
		float* actual_data = opened_img.data->data();
		float* expected_data = expected_img.data();

		for (int i = 0; i < 16; i++)
		{
			EXPECT_NEAR(actual_data[i], expected_data[i],0.05);
		}

	}
}
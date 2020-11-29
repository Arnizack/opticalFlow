#include"cpu_backend/image/warper/GrayWarper.h"
#include"cpu_backend/Array.h"
#include"gtest/gtest.h"
#include"image_helper/ImageHelper.h"
#include"test_utilities/Timer.h"

namespace cpu_backend
{
	namespace testing
	{

		template<typename T>
		std::vector<T> arange(T start, T stop, T step = 1) {
			std::vector<T> values;
			for (T value = start; value < stop; value += step)
				values.push_back(value);
			return values;
		}

		TEST(GrayWarperTest, test1)
		{
			
			std::array<float, 4> img_data = { 1,2,3,4 };
			std::array<double, 8> flow_data = { 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 };

			Array<float, 2> test({ 1,1 });

			auto ptr_img = std::make_shared<Array<float, 2>>(
				std::array<const size_t,2>({ 2,2 }),img_data.data());
			auto ptr_flow = std::make_shared<Array<double, 3>>(
				std::array<const size_t, 3>({2, 2,2 }),flow_data.data());



			GrayWarper warper;

			warper.SetImage(ptr_img);
			auto result = warper.Warp(ptr_flow);

			std::array<float, 4> actual_warped;

			result->CopyDataTo(actual_warped.data());
			
			//BorderTyp Zeros
			std::array<float, 4> expected_warped = { 2.0906096 , 1.1082518 , 3.3014038 , 0.44431114 };
			//BorderTyp Replicate
			//std::array<float, 4> expected_warped = { 0.8274412, 1.4844971, 3.140686 , 3.0 };


			for (int i = 0; i < 4; i++)
			{
				EXPECT_NEAR(expected_warped[i], actual_warped[i],0.06);
			}
			
		}
		
		TEST(GrayWarperTest, test2)
		{

			const size_t width = 32;
			const size_t height = 32;

			auto img_data = arange<float>(0,1, 1 / ((float) width*height));
			auto flow_data = arange<double>(0,1,1/((float) width*height*2));

	

			auto ptr_img = std::make_shared<Array<float, 2>>(
				std::array<const size_t, 2>({ height,width }), img_data.data());

			imagehelper::SaveImage("before_warp.png", ptr_img);

			auto ptr_flow = std::make_shared<Array<double, 3>>(
				std::array<const size_t, 3>({ 2, height,width }), flow_data.data());


			GrayWarper warper;

			warper.SetImage(ptr_img);
			auto result = warper.Warp(ptr_flow);

			imagehelper::SaveImage("after_warp.png", result);

		}
		/*
		TEST(GrayWarperTest, speed)
		{
			
			const size_t width = 1920;
			const size_t height = 1080;

			auto img_data = arange<float>(0, 1, 1 / ((float)width * height));
			auto flow_data = arange<double>(0, 1, 1 / ((float)width * height * 2));



			auto ptr_img = std::make_shared<Array<float, 2>>(
				std::array<const size_t, 2>({ height,width }), img_data.data());

			

			auto ptr_flow = std::make_shared<Array<double, 3>>(
				std::array<const size_t, 3>({ 2, height,width }), flow_data.data());


			GrayWarper warper;

			warper.SetImage(ptr_img);

			test_utilities::Timer timer("GrayWarper");

			auto result = warper.Warp(ptr_flow);

			timer.Stop();

		}*/


	}
}
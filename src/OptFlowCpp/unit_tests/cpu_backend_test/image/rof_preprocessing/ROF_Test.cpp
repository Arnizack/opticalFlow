#include"gtest/gtest.h"

#include "cpu_backend/image/ROFPreProcessing/ROFPreProcessing.h"

#include "utilities/image_helper/ImageHelper.h"

namespace cpu_backend
{
	namespace testing
	{
		TEST(ROFPreProcessing, ROF_Image_Test)
		{
			auto settings = std::make_shared<ROFPreProcessingSettings>();

			auto arr_factory2D = std::make_shared<ArrayFactory<float, 2>>(ArrayFactory<float, 2>());
			auto arr_factory3D = std::make_shared<ArrayFactory<float, 3>>(ArrayFactory<float, 3>());

			auto statistics = std::make_shared<Statistics<float>>(Statistics<float>());

			auto arith_vector = std::make_shared<ArithmeticVector<float, 2>>(ArithmeticVector<float, 2>(arr_factory2D));

			auto rof_preprocessing = ROFPreProcessing(settings, arr_factory2D, arr_factory3D, statistics, arith_vector);

			auto image_read = imagehelper::OpenImage("E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10-gray.png");

			auto image = arr_factory3D->CreateFromSource(image_read.data->data(), { image_read.color_count, image_read.height, image_read.width });

			auto out_img = std::static_pointer_cast<Array<float, 3>>(rof_preprocessing.Process(image));

			auto arith_chained = std::make_shared<ArithmeticChained<float, 3>>(ArithmeticChained<float, 3>(arr_factory3D));

			auto tex_img = arith_chained->Sub(image, out_img);

			imagehelper::SaveImage("E:\\dev\\denoised_img.png", out_img);

			imagehelper::SaveImage("E:\\dev\\textured_img.png", tex_img);
		}

		TEST(ROFPreProcessing, ROF_Test)
		{
			auto settings = std::make_shared<ROFPreProcessingSettings>();

			auto arr_factory2D = std::make_shared<ArrayFactory<float, 2>>(ArrayFactory<float, 2>());
			auto arr_factory3D = std::make_shared<ArrayFactory<float, 3>>(ArrayFactory<float, 3>());

			auto statistics = std::make_shared<Statistics<float>>(Statistics<float>());

			auto arith_vector = std::make_shared<ArithmeticVector<float, 2>>(ArithmeticVector<float, 2>(arr_factory2D));

			auto rof_preprocessing = ROFPreProcessing(settings, arr_factory2D, arr_factory3D, statistics, arith_vector);

			auto in = std::static_pointer_cast<Array<float,3>>(arr_factory3D->Full(1, { 3, 10, 10 }));

			for (int i = 0; i < 300; i++)
			{
				(*in)[i] = i + 1;
			}

			auto out = rof_preprocessing.Process(in);
		}
    }
}
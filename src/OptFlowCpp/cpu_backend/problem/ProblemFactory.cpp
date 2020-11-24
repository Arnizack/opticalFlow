
#include "ProblemFactory.h"
#include"../Array.h"
#include"../image/inner/ConvertToGrayscale.h"

cpu_backend::ProblemFactory::ProblemFactory(std::shared_ptr<ArrayFactory<float, 2>> grayscale_factory)
	:_grayscale_factory(grayscale_factory)
{
}

std::shared_ptr<core::IGrayCrossFilterProblem> cpu_backend::ProblemFactory::CreateGrayCrossFilterProblem(std::shared_ptr<core::IArray<float, 3>> first_image, std::shared_ptr<core::IArray<float, 3>> seconde_image)
{
	auto cpu_first_img = std::static_pointer_cast<Array<float, 3>>(first_image);
	auto cpu_second_img = std::static_pointer_cast<Array<float, 3>>(seconde_image);
	auto problem = std::make_shared< core::IGrayCrossFilterProblem>();
	
	size_t height = first_image->Shape[1];
	size_t width = first_image->Shape[2];

	std::array<const size_t, 2> gray_shape = { height,width };
	
	problem->CrossFilterImage = first_image;
	auto first_gray_img = std::static_pointer_cast<Array<float, 2>>(_grayscale_factory->Zeros(gray_shape));
	auto second_gray_img = std::static_pointer_cast<Array<float, 2>>(_grayscale_factory->Zeros(gray_shape));
	
	ConvertColorToGrayscale(cpu_first_img->Data(), width, height, first_gray_img->Data());
	ConvertColorToGrayscale(cpu_second_img->Data(), width, height, second_gray_img->Data());

	problem->FirstFrame = first_gray_img;
	problem->SecondFrame = second_gray_img;

	return problem;

}



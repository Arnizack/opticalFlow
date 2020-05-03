#pragma once
#include"ImageRGB.h"
#include"FlowField.h"

enum class ImplemetationMode
{
	CPU,
	CUDA
};

class OpticalFlow
{
public:
	OpticalFlow(float sigma_distance, float sigma_color, float convergence_threshold, ImplemetationMode mode = ImplemetationMode::CPU);
	FlowField ComputeFlow(ImageRGB TemplateImage, ImageRGB NextImage);
};


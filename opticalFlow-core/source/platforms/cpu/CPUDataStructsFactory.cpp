#include "CPUDataStructsFactory.hpp"
#include"datastructurs/Host_Mat.h"
#include"datastructurs/Host_Array.h"

namespace cpu
{
	std::shared_ptr<datastructurs::IDeviceArray<float>> CPUDataStructsFactory::createArray(float* data, int itemCount)
	{
		return std::make_shared<Host_Array<float>>(data, itemCount);
	}
	std::shared_ptr<datastructurs::IDeviceArray<int>> CPUDataStructsFactory::createArray(int* data, int itemCount)
	{
		return std::make_shared<Host_Array<int>>(data, itemCount);
	}
	std::shared_ptr<datastructurs::IDevice2DMatrix<float, 2>> CPUDataStructsFactory::createMatrix2D(float* data, int width, int heigth)
	{
		return std::make_shared<Host_Mat<Vec2D<float>,float, 2>>(data,width,heigth);
		
	}
	std::shared_ptr<datastructurs::IDevice2DMatrix<float, 4>> CPUDataStructsFactory::createMatrix4D(float* data, int width, int heigth)
	{
		return std::make_shared<Host_Mat<Vec4D<float>, float, 4>>(data, width, heigth);
	}

	std::shared_ptr<datastructurs::IDevice2DMatrix<float, 1>> CPUDataStructsFactory::createMatrix1D(float* data, int width, int heigth)
	{
		return std::make_shared<Host_Mat<float,float, 1>>(data, width, heigth);
	}

	std::shared_ptr<datastructurs::IDeviceTextureRGBA> CPUDataStructsFactory::createTextureRGBA(float* data, int width, int heigth)
	{
		//return std::make_unique<Host_Mat2D<Vec4D<float>, float, 4>>(data, width, heigth);
		return nullptr;
	}




}

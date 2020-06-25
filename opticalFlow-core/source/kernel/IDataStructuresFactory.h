#pragma once
#include"datastructurs/DeviceData.h"

#include<memory>

namespace kernel
{
	class IDataStructuresFactory
	{
	public:
		
		virtual std::unique_ptr<datastructurs::IDeviceArray<float>> createArray(float* data, int itemCount) = 0;

		virtual std::unique_ptr<datastructurs::IDeviceArray<int>> createArray(int* data, int itemCount) = 0;

		virtual std::unique_ptr<datastructurs::IDevice2DMatrix<float, 1>> createMatrix1D(float* data, int width, int heigth) = 0;
		virtual std::unique_ptr<datastructurs::IDevice2DMatrix<float,2>> createMatrix2D(float* data, int width, int heigth) = 0;
		virtual std::unique_ptr<datastructurs::IDevice2DMatrix<float,4>> createMatrix4D(float* data, int width, int heigth) = 0;

		
		virtual std::unique_ptr<datastructurs::IDeviceTextureRGBA> createTextureRGBA(float* data, int width, int heigth) = 0;


	};

}
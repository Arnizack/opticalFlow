#pragma once
#include"datastructurs/DeviceData.h"

#include<memory>

namespace kernelbackend
{
	class IDataStrcuturesFactory
	{
	public:
		
		virtual std::unique_ptr<datastructurs::IDeviceArray<float>> createArray(float* data, int itemCount) = 0;

		virtual std::unique_ptr<datastructurs::IDeviceArray<int>> createArray(int* data, int itemCount) = 0;

		
		virtual std::unique_ptr<datastructurs::IDevice2DMatrix<float,2>> createMatrix2D(float* data, int width, int heigth) = 0;
		virtual std::unique_ptr<datastructurs::IDevice2DMatrix<float,4>> createMatrix4D(float* data, int width, int heigth) = 0;
		virtual std::unique_ptr<datastructurs::IDevice2DMatrix<float, 6>> createMatrix6D(float* data, int width, int heigth) = 0;

		
		virtual std::unique_ptr<datastructurs::IDeviceTexture> createTexture(float* data, int width, int heigth) = 0;


	};

}
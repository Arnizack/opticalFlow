#pragma once
#include"kernel/IDataStructuresFactory.h"

namespace cuda
{
	class CUDADataStructsFactory : public kernel::IDataStructuresFactory
	{
		// Inherited via IDataStructuresFactory
		virtual std::shared_ptr<datastructures::IDeviceArray<float>> createArray(float* data, int itemCount) override;
		virtual std::shared_ptr<datastructures::IDeviceArray<int>> createArray(int* data, int itemCount) override;
		virtual std::shared_ptr<datastructures::IDevice2DMatrix<float, 1>> createMatrix1D(float* data, int width, int heigth) override;
		virtual std::shared_ptr<datastructures::IDevice2DMatrix<float, 2>> createMatrix2D(float* data, int width, int heigth) override;
		virtual std::shared_ptr<datastructures::IDevice2DMatrix<float, 4>> createMatrix4D(float* data, int width, int heigth) override;
		virtual std::shared_ptr<datastructures::IDeviceTextureRGBA> createTextureRGBA(float* data, int width, int heigth) override;
	};
}
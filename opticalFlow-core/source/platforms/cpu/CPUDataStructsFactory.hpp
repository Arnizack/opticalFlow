#pragma once
#include"kernel/IDataStructuresFactory.h"

namespace cpu
{
	class CPUDataStructsFactory : public kernel::IDataStrcuturesFactory
	{
		// Inherited via IDataStrcuturesFactory
		virtual std::unique_ptr<datastructurs::IDeviceArray<float>> createArray(float* data, int itemCount) override;
		virtual std::unique_ptr<datastructurs::IDeviceArray<int>> createArray(int* data, int itemCount) override;
		virtual std::unique_ptr<datastructurs::IDevice2DMatrix<float, 2>> createMatrix2D(float* data, int width, int heigth) override;
		virtual std::unique_ptr<datastructurs::IDevice2DMatrix<float, 4>> createMatrix4D(float* data, int width, int heigth) override;

		

		// Inherited via IDataStrcuturesFactory
		virtual std::unique_ptr<datastructurs::IDevice2DMatrix<float, 1>> createMatrix1D(float* data, int width, int heigth) override;

		virtual std::unique_ptr<datastructurs::IDeviceTextureRGBA> createTextureRGBA(float* data, int width, int heigth) override;

	};

}
                
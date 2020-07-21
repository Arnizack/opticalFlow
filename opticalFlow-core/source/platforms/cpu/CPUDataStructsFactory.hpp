#pragma once
#include"kernel/IDataStructuresFactory.h"

namespace cpu
{
	
	class CPUDataStructsFactory : public kernel::IDataStructuresFactory
	{
		// Inherited via IDataStrcuturesFactory
		virtual std::shared_ptr<datastructures::IDeviceArray<float>> createArray(float* data, int itemCount) override;
		virtual std::shared_ptr<datastructures::IDeviceArray<int>> createArray(int* data, int itemCount) override;
		virtual std::shared_ptr<datastructures::IDevice2DMatrix<float, 2>> createMatrix2D(float* data, int width, int heigth) override;
		virtual std::shared_ptr<datastructures::IDevice2DMatrix<float, 4>> createMatrix4D(float* data, int width, int heigth) override;

		

		// Inherited via IDataStrcuturesFactory
		virtual std::shared_ptr<datastructures::IDevice2DMatrix<float, 1>> createMatrix1D(float* data, int width, int heigth) override;

		virtual std::shared_ptr<datastructures::IDeviceTextureRGBA> createTextureRGBA(float* data, int width, int heigth) override;

	};

}
                
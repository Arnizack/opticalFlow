#include "CUDADataStructsFactory.hpp"
#include "platforms/cuda/datastructures/HostDevice2DMatrix.h"
#include "platforms/cuda/datastructures/HostDeviceArray.h"


std::shared_ptr<datastructures::IDeviceArray<float>> cuda::CUDADataStructsFactory::createArray(float* data, int itemCount)
{
    return std::make_shared<datastructures::HostDeviceArray<float>>(data, itemCount);
}

std::shared_ptr<datastructures::IDeviceArray<int>> cuda::CUDADataStructsFactory::createArray(int* data, int itemCount)
{
    return std::make_shared<datastructures::HostDeviceArray<int>>(data, itemCount);
}

std::shared_ptr<datastructures::IDevice2DMatrix<float, 1>> cuda::CUDADataStructsFactory::createMatrix1D(float* data, int width, int heigth)
{
    return std::make_shared<datastructures::HostDevice2DMatrix<float,1>>(data,width,heigth);
}

std::shared_ptr<datastructures::IDevice2DMatrix<float, 2>> cuda::CUDADataStructsFactory::createMatrix2D(float* data, int width, int heigth)
{
    return std::make_shared<datastructures::HostDevice2DMatrix<float, 2>>(data, width, heigth);
}

std::shared_ptr<datastructures::IDevice2DMatrix<float, 4>> cuda::CUDADataStructsFactory::createMatrix4D(float* data, int width, int heigth)
{
    return std::make_shared<datastructures::HostDevice2DMatrix<float, 4>>(data, width, heigth);
}

std::shared_ptr<datastructures::IDeviceTextureRGBA> cuda::CUDADataStructsFactory::createTextureRGBA(float* data, int width, int heigth)
{
    return nullptr;
}

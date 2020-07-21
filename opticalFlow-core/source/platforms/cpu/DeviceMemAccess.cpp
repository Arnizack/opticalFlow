#include "DeviceMemAccess.hpp"
#include"datastructures/Host_Array.h"
#include"datastructures/Host_Mat.h"

namespace cpu
{
	template<class _Mat,typename T>
	void loadArray(const _Mat& mat, T* src)
	{
		mat.copyTo(src);
	}

	void DeviceMemAccess::load(float* dst, const datastructures::IDeviceArray<float>& data)
	{
		const Host_Array<float>& arr = static_cast<const Host_Array<float>&>(data);
		loadArray(arr, dst);
	}

	void DeviceMemAccess::load(int* dst, const datastructures::IDeviceArray<int>& data)
	{
		const Host_Array<int>& arr = static_cast<const Host_Array<int>&>(data);
		loadArray(arr, dst);
	}

	void DeviceMemAccess::load(float* dst, const datastructures::IDevice2DMatrix<float, 1>& data)
	{
		/*
		const Host_Mat<float,float,1>& arr = static_cast<const Host_Mat<float,float,1>&>(data);
		loadArray(arr, dst);*/
	}

	void DeviceMemAccess::load(float* dst, const datastructures::IDevice2DMatrix<float, 2>& data)
	{
		/*
		const Host_Mat<Vec2D<float>, float, 2>& arr = static_cast<const Host_Mat<Vec2D<float>, float, 2>&>(data);
		loadArray(arr, dst);*/
	}

	void DeviceMemAccess::load(float* dst, const datastructures::IDevice2DMatrix<float, 4>& data)
	{
		/*
		const Host_Mat<Vec4D<float>, float, 4>& arr = static_cast<const Host_Mat<Vec4D<float>, float, 4>&>(data);
		loadArray(arr, dst);*/
	}

}

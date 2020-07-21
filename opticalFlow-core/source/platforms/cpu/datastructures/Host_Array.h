#pragma once
#include"datastructures/DeviceData.h"
#include<vector>
#include<algorithm>
namespace cpu
{
	template<class T>
	class Host_Array : public datastructures::IDeviceArray<T>
	{
	private:
		
		std::vector<T> data;
	public:
		Host_Array(T* src, int size)
			: data(size), datastructures::IDeviceArray<T>(size)
		{

			std::copy_n(src, size, data.data());
		}

		void copyTo(T* dst) const
		{
			std::copy_n(data.data(), ItemCount, dst);
		}

		std::vector<T>* getKernelData()
		{
			return &data;
		}
	};
}
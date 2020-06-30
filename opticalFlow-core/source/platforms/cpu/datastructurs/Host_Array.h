#pragma once
#include"datastructurs/DeviceData.h"
#include<vector>
#include<algorithm>
namespace cpu
{
	template<class T>
	class Host_Array : public datastructurs::IDeviceArray<T>
	{
	private:
		
		std::vector<T> data;
	public:
		Host_Array(T* src, int size)
			: data(size), datastructurs::IDeviceArray<T>(size)
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
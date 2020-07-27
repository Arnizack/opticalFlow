#pragma once
#include"datastructures/DeviceData.h"
#include <vector>
#include <algorithm>
#include <initializer_list>

namespace cpu
{
	template<class T>
	class HostArray : public datastructures::IDeviceArray<T>
	{
	private:
		
		std::vector<T> data;
	public:
		
		HostArray(const T* const src, const int& size)
			: data(size), datastructures::IDeviceArray<T>(size)
		{
			std::copy_n(src, size, data.data());
		}
		

		HostArray(std::initializer_list<T> list, const size_t size)
			: data(list), datastructures::IDeviceArray<T>(size)
		{}

		void copyTo(T* dst) const
		{
			std::copy_n(data.data(), ItemCount, dst);
		}

		T* getKernelData()
		{
			return data.data();
		}

		std::vector<T> getVector()
		{
			return data;
		}
	};
}
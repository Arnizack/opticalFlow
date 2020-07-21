#pragma once
namespace datastructures
{
	class IDeviceObj
	{
	public:
		IDeviceObj() = default;
		IDeviceObj(const IDeviceObj& obj) = delete;
		virtual ~IDeviceObj() //= 0;
		{}
	};

	template<typename T>
	class IDeviceArray : public IDeviceObj
	{
	public:
		IDeviceArray(T* const& arr, const size_t& _ItemCount)
			: host_array(arr), ItemCount(_ItemCount)
		{}

		virtual ~IDeviceArray() override
		{}

		T* host_array;
		const size_t ItemCount;
	};

	template<typename T,size_t VectorDimensions>
	class IDevice2DMatrix : public IDeviceArray<T>
	{
	public:
		IDevice2DMatrix(const T*& matrix, const size_t& width, const size_t& height)
			: host_array(matrix), Width(width), Heigth(height)
		{}

		virtual ~IDevice2DMatrix() override
		{}

		T* host_array;
		const size_t Width;
		const size_t Heigth;
	};

	class IDeviceTextureRGBA : public IDeviceObj
	{
	};

	class IDeviceTextureGrayScale : public IDeviceObj
	{
		
	};
}
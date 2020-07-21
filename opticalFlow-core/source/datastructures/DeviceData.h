#pragma once
namespace datastructures
{
	class IDeviceObj
	{
	public:
		IDeviceObj() = default;
		IDeviceObj(const IDeviceObj& obj) = delete;
		virtual ~IDeviceObj() = 0;
	};

	template<typename T>
	class IDeviceArray : public IDeviceObj
	{
	public:
		IDeviceArray(const size_t& _ItemCount)
			: ItemCount(_ItemCount)
		{}

		const size_t ItemCount;
	};

	template<typename T,size_t VectorDimensions>
	class IDevice2DMatrix : public IDeviceArray<T>
	{
	public:
		IDevice2DMatrix(const size_t& width, const size_t& height)
			: Width(width), Heigth(height), IDeviceArray(width*height)
		{}
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
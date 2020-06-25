#pragma once
namespace datastructurs
{
	class IDeviceObj
	{
	public:
		IDeviceObj(const IDeviceObj& obj) = delete;
		virtual ~IDeviceObj() = 0;
	};

	template<typename T>
	class IDeviceArray : public IDeviceObj
	{
	public:
		const int ItemCount;
	};

	template<typename T,int VectorDimensions>
	class IDevice2DMatrix : public IDeviceArray<T>
	{
	public:
		const int Width;
		const int Heigth;
	};




	class IDeviceTextureRGBA : public IDeviceObj
	{
	};

	class IDeviceTextureGrayScale : public IDeviceObj
	{
		
	};
}
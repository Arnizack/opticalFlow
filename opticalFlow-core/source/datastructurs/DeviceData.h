#pragma once
namespace datastructurs
{
	class IDeviceObj
	{
	public:
		IDeviceObj() = default;
	
		IDeviceObj(const IDeviceObj& obj) = delete;
		//virtual ~IDeviceObj() = 0;
	};

	template<typename T>
	class IDeviceArray : public IDeviceObj
	{
	public:
		IDeviceArray(int itemCount);
		const int ItemCount;
	};

	template<typename T, int VectorDimensions>
	class IDevice2DMatrix : public IDeviceObj
	{
	public:
		IDevice2DMatrix(int width, int heigth);

		const int Width;
		const int Heigth;
	};




	class IDeviceTextureRGBA : public IDeviceObj
	{
	};

	class IDeviceTextureGrayScale : public IDeviceObj
	{

	};

	template<typename T>
	IDeviceArray<T>::IDeviceArray(int itemCount)
		: ItemCount(itemCount)
	{
	}

	template<typename T, int VectorDimensions>
	IDevice2DMatrix<T, VectorDimensions>::IDevice2DMatrix(int width, int heigth)
		: Width(width), Heigth(heigth)
	{		
	}
}


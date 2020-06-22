#pragma once
namespace datastructurs
{
	class IDeviceObj
	{
	public:
		IDeviceObj(const IDeviceObj& obj) = delete;
		virtual ~IDeviceObj()=0;
	};

	template<typename T>
	class IDeviceArray : public IDeviceObj
	{
	public:
		const int ItemCount;
	};

	template<typename T,size_t VectorDimensions>
	class IDevice2DMatrix : public IDeviceArray<T>
	{
	public:
		const int Width;
		const int Heigth;
	};




	class IDeviceTexture : public IDevice2DMatrix<float,4>
	{
	};
}
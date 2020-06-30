#pragma once
#include"kernel/IDeviceMemAccess.h"
namespace cpu
{
	class DeviceMemAccess : public kernel::IDeviceMemAccess
	{
	public:


		

		// Inherited via IDeviceMemAccess
		virtual void load(float* dst, const datastructurs::IDeviceArray<float>& data) override;

		virtual void load(int* dst, const datastructurs::IDeviceArray<int>& data) override;

		virtual void load(float* dst, const datastructurs::IDevice2DMatrix<float, 1>& data) override;

		virtual void load(float* dst, const datastructurs::IDevice2DMatrix<float, 2>& data) override;

		virtual void load(float* dst, const datastructurs::IDevice2DMatrix<float, 4>& data) override;

	};

}
                
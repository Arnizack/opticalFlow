#pragma once
#include"ArrayHelper.h"

namespace cpu_backend
{
	template<class T>
	inline T BilinearInterpolateAt(const float& x, const float& y, T* img, size_t width, size_t height)
	{
		float x_lower = floor(x);
		float x_upper = ceil(x);
		float y_lower = floor(y);
		float y_upper = ceil(y);

		T val_lower_left = _inner::GetValueAt<T, Padding::ZEROS>(x_lower, y_lower, width, height, img);
		T val_lower_right = _inner::GetValueAt<T, Padding::ZEROS>(x_upper, y_lower, width, height, img);
		T val_upper_left = _inner::GetValueAt<T, Padding::ZEROS>(x_lower, y_upper, width, height, img);
		T val_uppper_right = _inner::GetValueAt<T, Padding::ZEROS>(x_upper, y_upper, width, height, img);

		return
			val_lower_left * (x - x_lower) * (y - y_lower) +
			val_lower_right * (x_upper - x) * (y - y_lower) +
			val_upper_left * (x - x_lower) * (y_upper - y) +
			val_uppper_right * (x_upper - x) * (y_upper - y);
	
	}
}
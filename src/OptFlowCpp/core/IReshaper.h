#pragma once
#include"IArray.h"


namespace core
{
	template<class InnerTyp>
	class IReshaper
	{
	public:
		virtual IArray<InnerTyp, 1> Reshape1D(IContainer<InnerTyp> container) = 0;
		virtual IArray<InnerTyp, 2> Reshape2D(
			IContainer<InnerTyp> container, std::array<size_t,2> shape) = 0;
		virtual IArray<InnerTyp, 3> Reshape3D(
			IContainer<InnerTyp> container, std::array<size_t, 3> shape) = 0;
	};
}
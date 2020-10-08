#pragma once
#include"IArray.h"
#include<memory>
#include<array>

namespace core
{
	template<class InnerTyp>
	class IReshaper
	{
	public:
		virtual std::shared_ptr < IArray<InnerTyp, 1>> 
			Reshape1D(std::shared_ptr<IContainer<InnerTyp>> container) = 0;
		
		virtual std::shared_ptr < IArray<InnerTyp, 2>> 
			Reshape2D(std::shared_ptr<IContainer<InnerTyp>> container, std::array<size_t,2> shape) = 0;
		
		virtual std::shared_ptr < IArray<InnerTyp, 3>> 
			Reshape3D(std::shared_ptr<IContainer<InnerTyp>>  container, 
				std::array<size_t, 3> shape) = 0;
	};
}
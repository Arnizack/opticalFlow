#pragma once
#include"../IContainer.h"
#include"../IArray.h"
#include"../pch.h"
namespace core
{
	namespace linalg
	{

		

		template<class InnerTyp, size_t InputDimCount,size_t OuputDimCount>
		class ILinearOperator
		{
			using InVector = std::shared_ptr<IArray<InnerTyp, InputDimCount>>;
			using OutVector = std::shared_ptr<IArray<InnerTyp, OuputDimCount>>;

		public:
			virtual OutVector operator()(const InVector vec) = 0;
			
		};

		
	}
}
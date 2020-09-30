#pragma once
#include"IContainer.h"
#include<memory>

namespace core
{
	template<class InnerTyp>
	class IStatistics
	{
		using ConstPtrArray = const std::shared_ptr<IContainer<InnerTyp>>;
	public:
		virtual InnerTyp NormL2(ConstPtrArray x)=0;
		virtual InnerTyp StandardDeviation(ConstPtrArray x) = 0;
		virtual InnerTyp Sum(ConstPtrArray x) = 0;
		virtual InnerTyp Mean(ConstPtrArray x) = 0;
	};

};


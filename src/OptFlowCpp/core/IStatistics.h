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
		//virtual InnerTyp NormL2(ConstPtrArray x)=0;
		virtual double Variance(ConstPtrArray x) = 0;
		virtual /*InnerTyp*/ double StandardDeviation(ConstPtrArray x) = 0;
		virtual /*InnerTyp*/ double Sum(ConstPtrArray x) = 0;
		virtual /*InnerTyp*/ double Mean(ConstPtrArray x) = 0;
	};
}


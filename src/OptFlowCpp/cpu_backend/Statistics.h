#pragma once

#include "core/IStatistics.h"

namespace cpu_backend
{
	template<class InnerTyp>
	class Statistics : public core::IStatistics<InnerTyp>
	{
		using ConstPtrArray = const std::shared_ptr<IContainer<InnerTyp>>;
	public:
		virtual InnerTyp NormL2(ConstPtrArray x) override
		{

		}

		virtual InnerTyp StandardDeviation(ConstPtrArray x) override
		{

		}

		virtual InnerTyp Sum(ConstPtrArray x) override
		{

		}

		virtual InnerTyp Mean(ConstPtrArray x) override
		{

		}
	};
}
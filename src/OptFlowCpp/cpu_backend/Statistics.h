#pragma once
#include "core/IStatistics.h"
#include "Container.h"

#include <memory>
#include <math.h>
#include <omp.h>

namespace cpu_backend
{
    template<class InnerTyp>
    class Statistics final : public core::IStatistics<InnerTyp>
    {
		using ConstPtrArray = const std::shared_ptr<core::IContainer<InnerTyp>>;

	public:
		//virtual InnerTyp NormL2(ConstPtrArray x)=0;

		virtual double Variance(ConstPtrArray x) override
		{
			InnerTyp* data = std::static_pointer_cast<Container<InnerTyp>>(x)->Data();

			const size_t size = x->Size();
			double mean = Mean(x);
			double sum = 0;

			#pragma omp parallel for reduction(+: sum)
			for (int i = 0; i < size; i++)
			{
				sum += (data[i] - mean) * (data[i] - mean);
			}

			return sum / size;
		}

		virtual double StandardDeviation(ConstPtrArray x) override
		{
			//double mean = Mean(x);
			//double sum = 0;

			//const size_t size = x->Size();
			//auto data = std::static_pointer_cast<Container<InnerTyp>>(x)->Data();

			//#pragma omp parallel for reduction(+: sum)
			//for (int i = 0; i < size; i++)
			//{
			//	sum += (data[i] - mean) * (data[i] - mean);
			//}

			//double variance = sum / size;

			return sqrt(Variance(x));
		}

		virtual double Sum(ConstPtrArray x) override
		{
			double sum = 0;

			const size_t size = x->Size();
			auto x_cpu = std::static_pointer_cast<Container<InnerTyp>>(x);
			auto data = x_cpu->Data();

			#pragma omp parallel for reduction(+: sum)
			for (int i = 0; i < size; i++)
			{
				sum += data[i];
			}

			return sum;
		}

		virtual double Mean(ConstPtrArray x) override
		{
			double sum = Sum(x);
			const size_t size = x->Size();

			return sum / size;
		}
    };
}
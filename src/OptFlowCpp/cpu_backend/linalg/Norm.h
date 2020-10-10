#pragma once
#include "core/linalg/IOperator.h"
#include "../Array.h"

#include <memory>
#include <math.h>
#include <cblas.h>


namespace cpu_backend
{
	template<typename InnerTyp, size_t DimCount>
	class Norm : public core::IOperator<std::shared_ptr< core::IArray<InnerTyp, DimCount>>, std::shared_ptr<InnerTyp>>
	{
		using InputTyp = std::shared_ptr<core::IArray<InnerTyp, DimCount>>;
		using OutputTyp = std::shared_ptr<InnerTyp>;

	public:
		virtual OutputTyp Apply(const InputTyp vec) override
		{
			//calculates the norm
			return CalculateNorm(vec);
		}

		virtual void ApplyTo(OutputTyp dst, const InputTyp vec) override
		{
			//calculates the norm
			dst = CalculateNorm(vec);
			return;
		}

	private:
		OutputTyp CalculateNorm(const InputTyp& vec)
		{
			std::shared_ptr<Array<InnerTyp, DimCount>> in = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(vec);

			InnerTyp norm = 0;
			const size_t size = (*in).Size();

			for (size_t i = 0; i < size; i++)
			{
				norm += (*in)[i] * (*in)[i];
			}

			norm = (InnerTyp)sqrt((double)norm);

			return std::make_shared<InnerTyp>(norm);
		}
	};

	template<size_t DimCount>
	class Norm<double, DimCount> : public core::IOperator<std::shared_ptr< core::IArray<double, DimCount>>, std::shared_ptr<double>>
	{
		using InputTyp = std::shared_ptr<core::IArray<double, DimCount>>;
		using OutputTyp = std::shared_ptr<double>;

	public:
		virtual OutputTyp Apply(const InputTyp vec) override
		{
			//calculates the norm
			return CalculateNorm(vec);
		}

		virtual void ApplyTo(OutputTyp dst, const InputTyp vec) override
		{
			//calculates the norm
			dst = CalculateNorm(vec);
			return;
		}

	private:
		OutputTyp CalculateNorm(const InputTyp& vec)
		{
			std::shared_ptr<Array<double, DimCount>> in = std::dynamic_pointer_cast<Array<double, DimCount>>(vec);

			double norm = cblas_dnrm2((*in).Size(), &(*in)[0], 1);

			return std::make_shared<double>(norm);
		}
	};

	template<size_t DimCount>
	class Norm<float, DimCount> : public core::IOperator<std::shared_ptr< Array<float, DimCount>>, std::shared_ptr<float>>
	{
		using InputTyp = std::shared_ptr<core::IArray<float, DimCount>>;
		using OutputTyp = std::shared_ptr<float>;

	public:
		virtual OutputTyp Apply(const InputTyp vec) override
		{
			//calculates the norm
			return CalculateNorm(vec);
		}

		virtual void ApplyTo(OutputTyp dst, const InputTyp vec) override
		{
			//calculates the norm
			dst = CalculateNorm(vec);
			return;
		}

	private:
		OutputTyp CalculateNorm(const InputTyp& vec)
		{
			std::shared_ptr<Array<float, DimCount>> in = std::dynamic_pointer_cast<Array<float, DimCount>>(vec);

			float norm = cblas_snrm2((*in).Size(), &(*in)[0], 1);

			return std::make_shared<float>(norm);
		}
	};
}
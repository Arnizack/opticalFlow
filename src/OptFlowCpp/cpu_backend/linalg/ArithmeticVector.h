#pragma once
#include "core/linalg/IArithmeticVector.h"
#include "../Array.h"

#include <memory>
#include <cblas.h>

namespace cpu_backend
{
	/*
	* ALL
	*/
	template<class InnerTyp, size_t DimCount>
	class ArithmeticVector : public core::IArithmeticVector<InnerTyp, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<InnerTyp, 1>>;

	public:
		// ||vec|| / norm(vec)
		virtual double NormEuclidean(const PtrVector vec) override
		{
			std::shared_ptr<Array<InnerTyp, DimCount>> in = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(vec);

			double norm = 0;
			const size_t size = (*in).Size();

			for (size_t i = 0; i < size; i++)
			{
				norm += (*in)[i] * (*in)[i];
			}

			return sqrt(norm);
		}

		// <a, b>
		virtual double ScalarProduct(const PtrVector a, const PtrVector b) override
		{
			double out = 0;

			std::shared_ptr<Array<InnerTyp, 1>> in_a = std::dynamic_pointer_cast<Array<InnerTyp, 1>>(a);
			std::shared_ptr<Array<InnerTyp, 1>> in_b = std::dynamic_pointer_cast<Array<InnerTyp, 1>>(b);

			for (size_t i = 0; i < _data.Size(); i++)
			{
				out += (*in_a)[i] * (*in_b)[i];
			}

			return out;
		}
	};

	/*
	* FLOAT
	*/
	template<size_t DimCount>
	class ArithmeticVector : public core::IArithmeticVector<float, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<float, 1>>;

	public:
		// ||vec|| / norm(vec)
		virtual double NormEuclidean(const PtrVector vec) override
		{
			std::shared_ptr<Array<float, DimCount>> in = std::dynamic_pointer_cast<Array<float, DimCount>>(vec);

			return (double) cblas_snrm2((*vec).Size(), &(*vec)[0], 1)
		}

		// <a, b>
		virtual double ScalarProduct(const PtrVector a, const PtrVector b) override
		{
			std::shared_ptr<Array<float, 1>> in_a = std::dynamic_pointer_cast<Array<float, 1>>(a);
			std::shared_ptr<Array<float, 1>> in_b = std::dynamic_pointer_cast<Array<float, 1>>(b);

			return (double) cblas_sdot((*in_a).Size(), &(*in_a)[0], 1, &(*in_b)[0], 1);
		}
	};


}
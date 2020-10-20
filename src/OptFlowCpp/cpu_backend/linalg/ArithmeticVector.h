#pragma once
#include "core/linalg/IArithmeticVector.h"
#include "ArithmeticBasic.h"
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
		using PtrMatrix = std::shared_ptr<core::IArray<InnerTyp, DimCount>>;

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

			for (size_t i = 0; i < (*in_a).Size(); i++)
			{
				out += (*in_a)[i] * (*in_b)[i];
			}

			return out;
		}

		// x = fac * A
		virtual PtrMatrix Scale(const double& fac, const PtrMatrix a) override
		{
			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::make_shared<Array<InnerTyp, DimCount>>(*a);

			for (size_t i = 0; i < (*out).Size(); i++)
			{
				(*out)[i] = fac * (*out)[i];
			}

			return out;
		}

		// A = fac * A
		virtual void ScaleTo(const double& fac, const PtrMatrix a) override
		{
			std::shared_ptr<Array<InnerTyp, DimCount>> in_a = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);

			for (size_t i = 0; i < (*in_a).Size(); i++)
			{
				(*in_a)[i] = fac * (*in_a)[i];
			}

			return;
		}
	};

	/*
	* DOUBLE
	*/
	template<size_t DimCount>
	class ArithmeticVector<double, DimCount> : public core::IArithmeticVector<double, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<double, 1>>;
		using PtrMatrix = std::shared_ptr<core::IArray<double, DimCount>>;

	public:
		// ||vec|| / norm(vec)
		virtual double NormEuclidean(const PtrVector vec) override
		{
			std::shared_ptr<Array<double, DimCount>> in = std::dynamic_pointer_cast<Array<double, DimCount>>(vec);

			return cblas_dnrm2((*in).Size(), &(*in)[0], 1);
		}

		// <a, b>
		virtual double ScalarProduct(const PtrVector a, const PtrVector b) override
		{
			std::shared_ptr<Array<double, 1>> in_a = std::dynamic_pointer_cast<Array<double, 1>>(a);
			std::shared_ptr<Array<double, 1>> in_b = std::dynamic_pointer_cast<Array<double, 1>>(b);

			return cblas_ddot((*in_a).Size(), &(*in_a)[0], 1, &(*in_b)[0], 1);
		}

		// x = fac * A
		virtual PtrMatrix Scale(const double& fac, const PtrMatrix a) override
		{
			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(*a);

			cblas_dscal((*out).Size(), fac, &(*out)[0], 1);

			return out;
		}

		// A = fac * A
		virtual void ScaleTo(const double& fac, const PtrMatrix a) override
		{
			std::shared_ptr<Array<double, DimCount>> in_a = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

			cblas_dscal((*in_a).Size(), fac, &(*in_a)[0], 1);

			return;
		}
	};

	/*
	* FLOAT
	*/
	template<size_t DimCount>
	class ArithmeticVector<float, DimCount> : public core::IArithmeticVector<float, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<float, 1>>;
		using PtrMatrix = std::shared_ptr<core::IArray<float, DimCount>>;

	public:
		// ||vec|| / norm(vec)
		virtual double NormEuclidean(const PtrVector vec) override
		{
			std::shared_ptr<Array<float, DimCount>> in = std::dynamic_pointer_cast<Array<float, DimCount>>(vec);

			return (double)cblas_snrm2((*in).Size(), &(*in)[0], 1);
		}

		// <a, b>
		virtual double ScalarProduct(const PtrVector a, const PtrVector b) override
		{
			std::shared_ptr<Array<float, 1>> in_a = std::dynamic_pointer_cast<Array<float, 1>>(a);
			std::shared_ptr<Array<float, 1>> in_b = std::dynamic_pointer_cast<Array<float, 1>>(b);

			return (double)cblas_sdot((*in_a).Size(), &(*in_a)[0], 1, &(*in_b)[0], 1);
		}

		// x = fac * A
		virtual PtrMatrix Scale(const double& fac, const PtrMatrix a) override
		{
			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(*a);

			cblas_sscal((*out).Size(), fac, &(*out)[0], 1);

			return out;
		}

		// A = fac * A
		virtual void ScaleTo(const double& fac, const PtrMatrix a) override
		{
			std::shared_ptr<Array<float, DimCount>> in_a = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

			cblas_sscal((*in_a).Size(), fac, &(*in_a)[0], 1);

			return;
		}
	};


}
#pragma once
#include "core/linalg/ILinearOperator.h"
#include "core/IArray.h"

#include "../Array.h"

namespace cpu_backend
{
	template<class Innertyp>
	class LinearSystemMatrix : public core::ILinearOperator<std::shared_ptr<core::IArray<Innertyp, 1>>, std::shared_ptr<core::IArray<Innertyp, 1>>>
	{
		using InputTyp = std::shared_ptr<core::IArray<Innertyp, 1>>; //Vector
		using OutputTyp = std::shared_ptr<core::IArray<Innertyp, 1>>; //Vector

		using Ptr2DMatrix = std::shared_ptr<core::IArray<Innertyp, 2>>; //2DMatrix

	public:

		/*
		* Adds Matrix-Vector Multiplication to a 2D Array
		*/

		LinearSystemMatrix(const std::shared_ptr<core::IArray<Innertyp, 2>> data)
			: _data(std::dynamic_pointer_cast<Array<Innertyp, 2>>(data))
		{}

		virtual OutputTyp Apply(const InputTyp vec) override
		{
			std::shared_ptr<Array<Innertyp, 1>> dst_cpu = std::make_shared<Array<Innertyp, 1>>(Array<Innertyp, 1>({ _data->Shape[1] }));
			OutputTyp dst = std::dynamic_pointer_cast<core::IArray<Innertyp, 1>>(dst_cpu);

			ApplyTo(dst, vec);

			return dst;
		}

		virtual void ApplyTo(OutputTyp dst, const InputTyp vec) override
		{
			std::shared_ptr<Array<Innertyp, 1>> dst_cpu = std::dynamic_pointer_cast<Array<Innertyp, 1>>(dst);
			std::shared_ptr<Array<Innertyp, 1>> vec_cpu = std::dynamic_pointer_cast<Array<Innertyp, 1>>(vec);

			const size_t height = _data->Shape[0];
			const size_t width = _data->Shape[1];

			for (size_t i = 0; i < height; i++)
			{
				(*dst_cpu)[i] = (*_data)[i * width] * (*vec_cpu)[0];

				for (size_t j = 0; j < width; j++)
				{
					(*dst_cpu)[i] += (*_data)[i * width + j] * (*vec_cpu)[j];
				}
			}

			return;
		}

		virtual std::shared_ptr<core::ILinearOperator<OutputTyp, InputTyp>> Transpose() override
		{
			const size_t height = _data->Shape[0];
			const size_t width = _data->Shape[1];

			std::shared_ptr<Array<Innertyp, 2>> transposed = std::make_shared<Array<Innertyp, 2>>(Array<Innertyp, 2>( { width, height }) );

			for (size_t i = 0; i < height; i++)
			{
				for (size_t j = 0; j < width; j++)
				{
					(*transposed)[j * height + i] = (*_data)[i * width + j];
				}
			}

			std::shared_ptr<LinearSystemMatrix<Innertyp>> out = std::make_shared<LinearSystemMatrix<Innertyp>>(LinearSystemMatrix<Innertyp>(transposed) );

			return std::static_pointer_cast<core::ILinearOperator<OutputTyp, InputTyp>>(out);
		}

		virtual bool IsSymetric() override
		{
			const size_t height = _data->Shape[0];
			const size_t width = _data->Shape[1];

			if (height != width)
			{
				return false;
			}

			for (size_t i = 0; i < height; i++)
			{
				for (size_t j = 0; j < width; j++)
				{
					if ((*_data)[j * height + i] != (*_data)[i * width + j])
					{
						return false;
					}
				}
			}

			return true;
		}

	private:
		std::shared_ptr<Array<Innertyp, 2>> _data;
	};
}
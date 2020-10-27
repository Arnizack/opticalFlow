#pragma once
#include "core/linalg/ILinearOperator.h"
#include "core/IArray.h"

#include "../Array.h"
#include "../ArrayFactory.h"

namespace cpu_backend
{
	template<class InnerTyp>
	class LinearSystemMatrix : public core::ILinearOperator<std::shared_ptr<core::IArray<InnerTyp, 1>>, std::shared_ptr<core::IArray<InnerTyp, 1>>>
	{
		using InputTyp = std::shared_ptr<core::IArray<InnerTyp, 1>>; //Vector
		using OutputTyp = std::shared_ptr<core::IArray<InnerTyp, 1>>; //Vector

		using Ptr2DMatrix = std::shared_ptr<core::IArray<InnerTyp, 2>>; //2DMatrix

		using PtrArrayFactory = std::shared_ptr<core::IArrayFactory<InnerTyp, 1>>;

	public:

		/*
		* Adds Matrix-Vector Multiplication to a 2D Array
		*/

		LinearSystemMatrix(const PtrArrayFactory factory, std::shared_ptr<core::IArray<InnerTyp, 2>> data)
			: _factory(std::dynamic_pointer_cast<ArrayFactory<InnerTyp, 1>>(factory)), _data(std::dynamic_pointer_cast<Array<InnerTyp, 2>>(data))
		{}

		virtual OutputTyp Apply(const InputTyp vec) override
		{
			std::shared_ptr<Array<InnerTyp, 1>> dst = std::dynamic_pointer_cast<Array<InnerTyp, 1>>( _factory->Zeros(vec->Shape));

			ApplyTo(dst, vec);

			return dst;
		}

		virtual void ApplyTo(OutputTyp dst, const InputTyp vec) override
		{
			std::shared_ptr<Array<InnerTyp, 1>> dst_cpu = std::dynamic_pointer_cast<Array<InnerTyp, 1>>(dst);
			std::shared_ptr<Array<InnerTyp, 1>> vec_cpu = std::dynamic_pointer_cast<Array<InnerTyp, 1>>(vec);

			const size_t height = _data->Shape[0];
			const size_t width = _data->Shape[1];

			for (size_t i = 0; i < height; i++)
			{
				(*dst_cpu)[i] = (*_data)[i * width] * (*vec_cpu)[0];

				for (size_t j = 1; j < width; j++)
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

			std::shared_ptr<Array<InnerTyp, 2>> transposed = std::make_shared<Array<InnerTyp, 2>>(Array<InnerTyp, 2>( { width, height }) );

			for (size_t i = 0; i < height; i++)
			{
				for (size_t j = 0; j < width; j++)
				{
					(*transposed)[j * height + i] = (*_data)[i * width + j];
				}
			}

			std::shared_ptr<LinearSystemMatrix<InnerTyp>> out = std::make_shared<LinearSystemMatrix<InnerTyp>>(LinearSystemMatrix<InnerTyp>(_factory, transposed) );

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
		std::shared_ptr<Array<InnerTyp, 2>> _data;
		std::shared_ptr<ArrayFactory<InnerTyp, 1>> _factory;
	};
}
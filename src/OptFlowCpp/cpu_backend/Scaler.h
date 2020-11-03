#pragma once
#include <omp.h>

#include "core/IScaler.h"
#include "core/IArrayFactory.h"
#include "image/inner/BicubicInterpolate.h"
#include "Array.h"

namespace cpu_backend
{
	template<class InnerTyp, size_t DimCount>
	class Scaler : public core::IScaler<InnerTyp, DimCount>
	{
		using PtrArray = std::shared_ptr<core::IArray<InnerTyp, DimCount>>;
		using PtrArrayFactroy = std::shared_ptr<core::IArrayFactory<InnerTyp, DimCount>>;

	public:
		Scaler (PtrArrayFactroy factory)
			: _factroy(factory)
		{}

		virtual PtrArray Scale(const PtrArray input, const size_t& dst_width,
			const size_t& dst_height) override
		{
			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(_factroy->Zeros({ dst_width, dst_height}));

			auto image = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(input);

			const size_t input_width = input->Shape[0];
			const size_t input_height = input->Shape[1];

			const float width_realtion = dst_width / input_width;
			const float height_relation = dst_height / input_height;

			float x_proj;
			float y_proj;

			#pragma omp parallel for
			for (size_t y = 0; y < dst_height; y++) // column y
			{

				y_proj = (y - 0.5) / height_relation;

				for (size_t x = 0; x < dst_width; x++) // row x
				{
					x_proj = (x - 0.5) / width_realtion;

					(*out)[y * dst_width + x] = _inner::BicubicInerpolateAt<InnerTyp>(x_proj, y_proj, image->Data(), input_width, input_height);
				}
			}

			return out;
		}

	private:
		PtrArrayFactroy _factroy;
	};

	/*
	* 2D
	*/
	template<class InnerTyp>
	class Scaler<InnerTyp, 2> : public core::IScaler<InnerTyp, 2>
	{
		using PtrArray = std::shared_ptr<core::IArray<InnerTyp, 2>>;
		using PtrArrayFactroy = std::shared_ptr<core::IArrayFactory<InnerTyp, 2>>;

	public:
		Scaler(PtrArrayFactroy factory)
			: _factroy(factory)
		{}

		virtual PtrArray Scale(const PtrArray input, const size_t& dst_width,
			const size_t& dst_height) override
		{
			std::shared_ptr<Array<InnerTyp, 2>> out = std::dynamic_pointer_cast<Array<InnerTyp, 2>>(_factroy->Zeros({ dst_width, dst_height }));

			auto image = std::dynamic_pointer_cast<Array<InnerTyp, 2>>(input);

			const size_t input_width = input->Shape[0];
			const size_t input_height = input->Shape[1];

			const float width_realtion = dst_width / input_width;
			const float height_relation = dst_height / input_height;

			float x_proj;
			float y_proj;

			#pragma omp parallel for
			for (size_t y = 0; y < dst_height; y++) // column y
			{

				y_proj = (y - 0.5) / height_relation;

				for (size_t x = 0; x < dst_width; x++) // row x
				{
					x_proj = (x - 0.5) / width_realtion;

					(*out)[y * dst_width + x] = _inner::BicubicInerpolateAt<InnerTyp>(x_proj, y_proj, image->Data(), input_width, input_height);
				}
			}

			return out;
		}

	private:
		PtrArrayFactroy _factroy;
	};

	/*
	* 3D
	*/
	template<class InnerTyp>
	class Scaler<InnerTyp, 3> : public core::IScaler<InnerTyp, 3>
	{
		using PtrArray = std::shared_ptr<core::IArray<InnerTyp, 3>>;
		using PtrArrayFactroy = std::shared_ptr<core::IArrayFactory<InnerTyp, 3>>;

	public:
		Scaler(PtrArrayFactroy factory)
			: _factroy(factory)
		{}

		virtual PtrArray Scale(const PtrArray input, const size_t& dst_width,
			const size_t& dst_height) override
		{
			std::shared_ptr<Array<InnerTyp, 3>> out = std::dynamic_pointer_cast<Array<InnerTyp, 3>>(_factroy->Zeros({ dst_width, dst_height, 3 }));

			auto image = std::dynamic_pointer_cast<Array<InnerTyp, 3>>(input);

			size_t input_width = input->Shape[0];
			size_t input_height = input->Shape[1];
			const size_t input_depth = input->Shape[2];

			const float width_realtion = dst_width / input_width;
			const float height_relation = dst_height / input_height;

			float x_proj;
			float y_proj;
			size_t y_offset;
			size_t dst_offset;
			size_t input_offset;

			#pragma omp parallel for
			for (int z = 0; z < input_depth; z++)
			{
				dst_offset = dst_height * dst_width * z;
				input_offset = input_width * input_height * z;

				for (size_t y = 0; y < dst_height; y++) // column y
				{
					y_proj = (y - 0.5) / height_relation;
					y_offset = y * dst_width;

					for (size_t x = 0; x < dst_width; x++) // row x
					{
						x_proj = (x - 0.5) / width_realtion;

						(*out)[dst_offset + y_offset + x] = _inner::BicubicInerpolateAt<InnerTyp>(x_proj, y_proj, image->Data(), input_width, input_height, input_offset);
					}
				}
			}

			return out;
		}

	private:
		PtrArrayFactroy _factroy;
	};
}
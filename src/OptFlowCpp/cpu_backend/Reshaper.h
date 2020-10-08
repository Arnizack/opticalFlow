#pragma once
#include "core/IReshaper.h"
#include "Array.h"

namespace cpu_backend
{
	template<class InnerTyp>
	class Reshaper : public core::IReshaper<InnerTyp>
	{
	public:
		virtual std::shared_ptr < core::IArray<InnerTyp, 1>>
			Reshape1D(std::shared_ptr<core::IContainer<InnerTyp>> container) override
		{
			std::array<const size_t, 1> shape = { container.get()->Size() };
			return return_data<1>(shape, container);
		}

		virtual std::shared_ptr < core::IArray<InnerTyp, 2>>
			Reshape2D(std::shared_ptr<core::IContainer<InnerTyp>> container, std::array<const size_t, 2> shape) override
		{
			return return_data<2>(shape , container);
		}

		virtual std::shared_ptr < core::IArray<InnerTyp, 3>>
			Reshape3D(std::shared_ptr<core::IContainer<InnerTyp>> container,
				std::array<const size_t, 3> shape) override
		{
			return return_data<3>(shape, container);
		}

	private:
		template<size_t DimCount>
		std::shared_ptr < core::IArray<InnerTyp, DimCount>> return_data(std::array<const size_t, DimCount>& shape, std::shared_ptr<core::IContainer<InnerTyp>>& container)
		{
			const size_t size = container.get()->Size();

			std::unique_ptr<InnerTyp[]> ptr(new InnerTyp[size]);
			container.get()->CopyDataTo(ptr.get());

			Array<InnerTyp, DimCount> out(shape, size, ptr.get());
			return std::make_shared<Array<InnerTyp, DimCount>>(out);
		}
	};
}
#pragma once
#include "..\core\IArray.h"
#include "CpuContainer.h"

namespace cpu
{
	template<class InnerTyp, size_t DimCount>
	class Array : public core::IArray<InnerTyp, DimCount>
	{
	public:
		Array(size_t(&shape)[DimCount], const InnerTyp *const src) : core::IArray<InnerTyp, DimCount>(shape)
		{
			int i = 0;
			const InnerTyp* current = args.begin();
			while (current != args.end())
			{
				_data[i] = *current;
				++current;
				++i;
			}
		}

		virtual size_t Size() override
		{
			size = 1;
			for (int i = 0; i < DimCount; i++)
			{
				size *= Shape[i];
			}
			return size;
		}

		virtual bool CopyDataTo(InnerTyp*& destination) override
		{
			destination = &_data.get()[0];
			return true;
		}
	protected:
	private:
		std::shared_ptr<Eigen::Array<InnerTyp, Eigen::Dynamic, Eigen::Dynamic>[]> _data = std::make_shared<Eigen::Array<InnerTyp, Eigen::Dynamic, Eigen::Dynamic>[]>(DimCount - 2);
	};
}
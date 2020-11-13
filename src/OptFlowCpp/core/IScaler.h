#pragma once
#include <memory>

namespace core
{

	template<class InnerTyp> /*, size_t DimCount>*/
	class IScaler
	{
	public:
		//using PtrArray = std::shared_ptr<IArray<InnerTyp, DimCount>>;
		using PtrInnerTyp = std::shared_ptr<InnerTyp>;

		virtual PtrInnerTyp Scale(const PtrInnerTyp input, const size_t& dst_width,
			const size_t& dst_height) = 0;
	};
	
}
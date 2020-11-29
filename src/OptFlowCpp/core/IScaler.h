#pragma once
#include <memory>

namespace core
{

	template<class InnerTyp> 
	class IScaler
	{
	public:
		virtual std::shared_ptr<InnerTyp> Scale(const std::shared_ptr<InnerTyp> input, const size_t& dst_width,
			const size_t& dst_height) = 0;
	};
	
}
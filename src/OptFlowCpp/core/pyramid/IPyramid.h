#pragma once
#include<memory>
namespace core {

	template <class InnerTyp>
	class IPyramid {
	public:
		virtual std::shared_ptr<InnerTyp> NextLevel() = 0;
		virtual bool IsEndLevel() = 0;
	};


}

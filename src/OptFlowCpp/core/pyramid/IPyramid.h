#pragma once
namespace core {

	template <class T>
	class IPyramid {
	public:
		virtual T NextLevel() = 0;
		virtual bool IsEndLevel() = 0;
	};


}

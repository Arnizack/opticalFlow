#pragma once
namespace core {
	namespace pyramid {

		template <class T>
		class IPyramid {
		public:
			virtual T NextLevel() = 0;
			virtual bool IsEndLevel() = 0;
		};

	}
}

#pragma once
#include<vector>
#include<array>
#include"IPyramid.h"
#include<memory>

namespace core
{
	template<class InnerTyp>
	class IPyramidBuilder
	{
	public:

		virtual void SetScaleFactors(std::vector<double> factors) = 0;
		virtual void SetScaleFactor(double factor, std::array<size_t, 2> min_resolution) = 0;
		virtual void SetResolutions(std::vector < std::array<size_t, 2>> resolutions) = 0;
		virtual std::shared_ptr<IPyramid<InnerTyp>> Create(std::shared_ptr<InnerTyp> last_level) = 0;
	};
}
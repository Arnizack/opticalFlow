#pragma once
#include"IPenalty.h"

namespace core
{

	template<class T>
	class IBlendablePenalty : public IPenalty<T>
	{
	public:
		virtual void SetBlendFactor(double blend_factor) = 0;
	};

}
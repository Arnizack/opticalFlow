#pragma once
#include "core/IContainer.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace cpu_backend
{
    template<class InnerTyp>
    class Container : public core::IContainer<InnerTyp>
    {
    public:
		/*
		* Interface for CpuContainer specific Methods
		*/

		virtual std::shared_ptr<std::vector<InnerTyp>> GetRef() = 0;

		virtual InnerTyp* Data() = 0;
    };
}
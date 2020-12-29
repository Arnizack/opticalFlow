#pragma once
#include "../IArray.h"
#include <memory>

namespace core
{
    template<class InnerTyp, size_t DimCount>
    class IPreProcessor
    {
        using PtrArray = std::shared_ptr<IArray<InnerTyp, DimCount>>;

    public:
        virtual PtrArray Process(PtrArray img) = 0;
    };
}
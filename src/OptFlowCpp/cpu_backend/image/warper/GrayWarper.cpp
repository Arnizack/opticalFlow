#pragma once
#include"pch.h"
#include"GrayWarper.h"

namespace cpu_backend
{
    using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
    using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;

    void GrayWarper::SetImage(PtrGrayImg Image)
    {
    }
    PtrGrayImg GrayWarper::Warp(PtrFlowField Flow)
    {
        return PtrGrayImg();
    }
}
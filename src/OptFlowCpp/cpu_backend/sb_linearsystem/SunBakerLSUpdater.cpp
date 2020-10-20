#pragma once
#include"pch.h"
#include"SunBakerLSUpdater.h"

namespace cpu_backend
{
    using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
    using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
    void SunBakerLSUpdater::SetFramePair(PtrGrayImg first_image, PtrGrayImg second_image)
    {
    }
    void SunBakerLSUpdater::UpdateParameter(PtrFlowField linearization_points, double relaxation)
    {
    }
    std::shared_ptr<core::ILinearProblem<double>> SunBakerLSUpdater::Update()
    {
        return std::shared_ptr<core::ILinearProblem<double>>();
    }
}
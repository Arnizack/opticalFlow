#include "IMinimizer.hpp"

#include <windows.h>
#include <ppl.h>
#include<algorithm>

cpu::IMinimizer::IMinimizer(float sigma_d, float sigma_c, uint8_t sampleCount)
{
    Sigma_c = sigma_c;
    Sigma_d = sigma_d;
    SampleCount = sampleCount;
}

core::FlowField cpu::IMinimizer::minimize(std::unique_ptr<core::FlowField>& initialFlow, const std::shared_ptr<core::ImageRGB>& templateFrame,
    const std::shared_ptr<core::ImageRGB>& nextFrame)
{
    uint32_t width = templateFrame->GetWidth();
    uint32_t heigth = templateFrame->GetHeight();
    core::FlowField resultFow(width,heigth);

    kdtree::KDTree tree(templateFrame);
    kdtree::KDTreeData treeData = tree.Build(Sigma_d, Sigma_c, SampleCount);

    concurrency::parallel_for(0, static_cast<int>(width), [&](int x)
        //for (uint32_t x = 0; x < width; x++)
    {
        for (uint32_t y = 0; y < heigth; y++)
        {
            resultFow.SetVector(x, y, minimizeAtPixel(treeData, *templateFrame, *nextFrame, x, y, initialFlow->GetVector(x, y)));
        }
    }
    );
    return resultFow;
}



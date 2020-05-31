#include "ReliabilityFactory.hpp"
#include"OpticalFlowMath.h"

namespace cpu::bilateralfilter
{
    ReliabilityFactory::ReliabilityFactory(uint8_t filterSize) : FilterSize(filterSize)
{
}
float ReliabilityFactory::ReliabilityAt(uint32_t x, uint32_t y, const core::FlowField & flow, const core::ImageRGB & templateFrame, const core::ImageRGB & nextFrame)
{
    uint8_t filterHalf = FilterSize / 2;
    uint32_t width = templateFrame.GetWidth();
    uint32_t heigth = templateFrame.GetHeight();
    
    uint32_t xMin = std::max((int)x - filterHalf, (int)0);
    uint32_t xMax = std::min(x + filterHalf, width);

    uint32_t yMin = std::max((int)y - filterHalf, (int)0);
    uint32_t yMax = std::min(y + filterHalf, heigth);
    //Calc Min and Mean

    float minError = 1000;
    float meanError = 0;

    for (uint32_t xChild = xMin; xChild < xMax; xChild++)
    {

        for (uint32_t yChild = yMin; xChild < yMax; xChild++)
        {
            core::FlowVector vec = flow.GetVector(xChild, yChild);
            float error = Error(x, y, vec, templateFrame, nextFrame);
            minError = std::min(minError, error);
            meanError += error;
        }
    }
    meanError /= (xMax - xMin) * (yMax - yMin);

    return meanError - minError;
}
float ReliabilityFactory::Error(const uint32_t x, const uint32_t y, const core::FlowVector& vec, const core::ImageRGB& templateFrame, const core::ImageRGB& nextFrame)
{
    auto vecTemplate = templateFrame.GetPixel(x,y);
    int x_new = x + vec.vector_X;
    int y_new = y + vec.vector_Y;

    if (x_new < 0 || x_new >= templateFrame.GetWidth())
        return -1;

    if (y_new < 0 || y_new >= templateFrame.GetHeight())
        return -1;


    auto vecNext = nextFrame.GetPixel(x_new, y_new);
    return core::ColorSqueredDistance(vecTemplate, vecNext);   

}
}

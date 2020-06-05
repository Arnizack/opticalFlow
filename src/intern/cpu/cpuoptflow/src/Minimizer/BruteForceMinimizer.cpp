#include "BruteForceMinimizer.hpp"
#include"ErrorFunctions.hpp"
#include<cmath>
#include"OpticalFlowMath.h"


namespace cpu
{
    BruteForceMinimizer::BruteForceMinimizer(float sigma_d, float sigma_c, uint8_t sampleCount, uint8_t searchRegionSize)
        :IMinimizer(sigma_d,sigma_c, sampleCount)
    {
        
        SearchRegionSize = searchRegionSize;

    }

    core::FlowVector BruteForceMinimizer::minimizeAtPixel(kdtree::KDTreeData& treeData, 
        const core::ImageRGB& templateFrame, const core::ImageRGB& nextFrame, 
        uint32_t x, uint32_t y, core::FlowVector& initialFlowVector)
    {
        uint32_t width = templateFrame.GetWidth();
        uint32_t heigth = templateFrame.GetHeight();

        auto color_0 = templateFrame.GetPixel(x, y);
        auto samplePoints = kdtree::queryKDTree(treeData, x, y, color_0);
        uint8_t regionHalf = SearchRegionSize / 2;
        //flow Vector coordinates
        float minError = -1;

        int32_t u_result = initialFlowVector.vector_X;
        int32_t v_result = initialFlowVector.vector_Y;

        int32_t u_min = initialFlowVector.vector_X - regionHalf;
        int32_t v_min = initialFlowVector.vector_Y - regionHalf;

        int32_t u_max = initialFlowVector.vector_X + regionHalf;
        int32_t v_max = initialFlowVector.vector_Y + regionHalf;

        if (u_min + x < 0)
            u_min = -x;

        if (v_min + y < 0)
            v_min = -y;

        if (u_max + x >= width)
            u_max = width - x-1;

        if (v_max + y >= heigth)
            v_max = heigth - y - 1;


        for (int32_t u = u_min; u < u_max; u++)
        {
            for (int32_t v = v_min; v < v_max; v++)
            {
                float divider = 0;
                float fullError = 0;
                
                int filterSize = 5;
                /*
                for (int32_t x2 = std::max((uint32_t)0, x - filterSize); x2 < std::min(x + filterSize, width); x2++)
                {
                    for (int32_t y2 = std::max((uint32_t)0, y - filterSize); y2 < std::min(y + filterSize, heigth); y2++)
                    {
                        int32_t x_shifted = x2 + u;
                        int32_t y_shifted = y2 + v;
                        core::Color nextColor(0, 0, 0);

                        auto color = templateFrame.GetPixel(x2, y2);
                        float distance_d = core::Distance(x2, y2, x, y);
                        float distanceSqr_c = core::ColorSqueredDistance(color, color_0);
                        float distanceSqr_d = distance_d * distance_d;
                        float exponent_d = -distanceSqr_d / (2 * Sigma_d);
                        float exponent_c = -distanceSqr_c / (2 * Sigma_c);
                        float weigth = exp(exponent_c + exponent_d);

                        if (0 <= x_shifted && x_shifted < width && 0 <= y_shifted && y_shifted < heigth)
                        {
                            nextColor = nextFrame.GetPixel(x_shifted, y_shifted);
                        }
                        float distanceSquared =
                            core::ColorSqueredDistance(core::Color(color.Red, color.Green, color.Blue), nextColor);
                        float error = CharbonnierError(distanceSquared);
                        fullError += weigth * error;
                        divider += weigth;
                    }
                }
                */
                
                for (const auto& sample : samplePoints)
                {

                    int32_t x_shifted = sample.X + u;
                    int32_t y_shifted = sample.Y + v;
                    core::Color nextColor(0,0,0);

                    if(0<=x_shifted && x_shifted< width && 0<=y_shifted && y_shifted < heigth)
                    {

                        nextColor = nextFrame.GetPixel(x_shifted, y_shifted);
                        
                    }
                    float distanceSquared =
                        core::ColorSqueredDistance(core::Color(sample.R, sample.G, sample.B), nextColor);
                    float error = CharbonnierError(distanceSquared);
                    fullError += sample.Weight * error;
                    divider += sample.Weight;
                }
                
                if (divider == 0)
                    fullError = NAN;
                else
                    fullError /= divider;

                if (fullError < minError || minError<0)
                {
                    u_result = u;
                    v_result = v;
                    minError = fullError;
                }
            }
        }
        return core::FlowVector(u_result, v_result);

    }



}

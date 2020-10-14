#pragma once

#include"ArrayHelper.h"
#include"stddef.h"
#include"math.h"
#include<iostream>

namespace cpu_backend
{
    namespace _inner
    {
        
        inline void GetCoefficients(float* coeffs, const float& x)
        {
            const float A = -0.75f;

            coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
            coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
            coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
            coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
        }

        template<class T>
        inline T BicubicInerpolateAt(const float& x,const float& y,const T* grid, 
        const size_t& width, const size_t& height)
        {
            //Reference
            //https://theailearner.com/tag/bicubic-interpolation/
            float coeffs_x[4];
            float coeffs_y[4];

            const float x_remap = x-floor(x);
            const float y_remap = y-floor(y);

            GetCoefficients(coeffs_x,x_remap);
            GetCoefficients(coeffs_y,y_remap);


            T result = 0;

            for(int coeffs_y_idx = 0; coeffs_y_idx<4; coeffs_y_idx++)
            {
                int y_grid = floor(y) - 1 + coeffs_y_idx;
                T weight_x = 0;
                for(int coeffs_x_idx = 0; coeffs_x_idx < 4; coeffs_x_idx ++)
                {
                    int x_grid = floor(x) - 1 + coeffs_x_idx;
                    //opencv default: Padding::Zeros
                    T grid_val = GetValueAt<T,Padding::ZEROS>(x_grid,y_grid,width,height,grid);
                    T& coeff_x = coeffs_x[coeffs_x_idx];
                    weight_x+=coeff_x*grid_val;
                   
                }

                T& coeff_y = coeffs_y[coeffs_y_idx];
                result+=coeff_y*weight_x;
            }

            return result;
        }

        
    }
}
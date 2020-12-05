#pragma once 
#include"../convolution1D.h"
#include<algorithm>
namespace cpu_backend
{
    namespace _inner
    {
        template<class T,Direction _Direction, Padding _Padding>
        void SobelFilter(T* image, int width, int height, T* destination)
        {
            T* temp =(T*) malloc(width*height*sizeof(T));
            T first_kernel[3] = {1,0,-1};
            T second_kernel[3] = {1,2,1};
            if(_Direction == Direction::Y)
            {
                std::swap(first_kernel,second_kernel);
            }
            Convolute1D<T,_Padding,Direction::X>(image,width,height,first_kernel,3,temp);
            Convolute1D<T,_Padding,Direction::Y>(temp,width,height,second_kernel,3,destination);
            free(temp);
        }
    }
}

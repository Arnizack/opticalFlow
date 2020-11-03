#pragma once
#include<algorithm>
#include<cassert>
#include<iostream>

namespace cpu_backend
{
    enum class Padding
	{
		ZEROS, REPLICATE, SYMMETRIC
	};

    namespace _inner
    {
        template<class T>
        constexpr const T& clamp( const T& v, const T& lo, const T& hi )
        {
            assert( !(hi < lo) );
            return (v < lo) ? lo : (hi < v) ? hi : v;
        }

       

        template<class T, Padding padding>
		inline T GetValueAt(const int& x, const  int& y, const  int width, const  int height, const T* img, const int offset = 0)
		{
            int x_remap = 0;
            int y_remap = 0;

            switch (padding)
            {
            case Padding::REPLICATE :
                x_remap = clamp(x,0,width-1);
                y_remap = clamp(y,0,height-1);
                return img[width*y_remap+x_remap];

            case Padding::SYMMETRIC:
                x_remap = x;
                y_remap = y;
                if (x >= width)
                    x_remap = 2 * width - x - 1;
                if (x < 0)
                    x_remap = abs(x) - 1;

                if (y >= height)
                    y_remap = 2 * height - y - 1;
                if (y < 0)
                    y_remap = abs(y) - 1;
                return img[width * y_remap + x_remap];

            default:
                //default padding Zeros
                if (x >= width || y >= height || x < 0 || y < 0)
                    return 0;
                return img[offset + width * y + x];
            }
		}
        template<class T>
        void PrintArray(const T* array, int width)
        {
            for(int i = 0; i<width-1; i++)
                std::cout<<array[i]<<", ";
            if(width>0)
                std::cout<<array[width-1];
        }

    }
}
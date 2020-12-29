#pragma once
#include"filters/SobelFilter.h"
#include"IncreaseMaskThickness.h"
namespace cpu_backend
{
    
    namespace _inner
    {
       
        template<class T, Padding _Padding>
        void EdgeDetection(T* image, int width, int height,int color_count,T threshold, int edge_thickness, 
            bool* is_edge_image)
        {
            
            T* edges = (T*)calloc(width * height , sizeof(T));
            T* edges_x = (T*) malloc(width*height*sizeof(T));
            T* edges_y = (T*) malloc(width*height*sizeof(T));
            for(int color_channel_idx = 0; color_channel_idx< color_count;color_channel_idx++)
            {
                T* color_channel = image + width * height * color_channel_idx;
                SobelFilter<T,Direction::X,_Padding>(color_channel,width,height,edges_x);
                SobelFilter<T,Direction::Y,_Padding>(color_channel,width,height,edges_y);
                for (int i = 0; i < width * height; i++)
                    edges[i] += abs(edges_x[i]) + abs(edges_y[i]);
            }
            bool* is_edge_temp = (bool*) malloc(width*height*sizeof(bool));
            
            for(int i = 0; i<width*height;i++)
            {
                is_edge_temp[i] = (edges[i]/(float)color_count > threshold);
            }

            IncreaseMaskThickness(is_edge_temp, width, height, edge_thickness, is_edge_image);
            free(edges_x);
            free(edges_y);
            free(is_edge_temp);
            free(edges);

        }
    }
}
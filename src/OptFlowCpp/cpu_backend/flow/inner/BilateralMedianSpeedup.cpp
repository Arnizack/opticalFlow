#include"BilateralMedianSpeedup.h"
#include"BilateralMedian.h"
#include"../../image/inner/ArrayHelper.h"
#include<algorithm>

namespace cpu_backend
{
    void BilateralMedianEdgeSpeedup(double* flow,
            double* auxiliary_flow,
            double* log_occlusion,
            float* image,
            double filter_influence,
            double auxiliary_influence,
            double sigma_distance,
            double sigma_color,
            bool* is_edge_map,
            int bilateral_filter_length,
            int median_filter_length,
            size_t width,
            size_t height,
            size_t color_channel_count,
    
            double* destination) 
    {
        size_t weights_max_length = bilateral_filter_length * bilateral_filter_length ;
        size_t median_list_max_length = std::max<size_t>(weights_max_length * 2 + 1, median_filter_length*median_filter_length);

        #pragma omp parallel for
        for (int y = 0; y < height; y++)
        {
            double* weights = (double*) malloc(weights_max_length * sizeof(double));
            double* median_list_x = (double*)malloc(median_list_max_length * sizeof(double));

            double* median_list_y = (double*)malloc(median_list_max_length * sizeof(double));
            for (int x = 0; x < width; x++)
            {
                if(is_edge_map[y*width+x])
                {
                    BilateralMedianAt(x, y, flow, auxiliary_flow, log_occlusion, image, weights,
                        median_list_x,median_list_y, filter_influence, auxiliary_influence, sigma_distance, sigma_color, bilateral_filter_length,
                        width, height, color_channel_count,destination);
                }else
                {
                    MedianFilterFlowAt(x,y,flow,median_filter_length,median_list_x,median_list_y,width,height,destination);
                }

            }
            free(weights);
            free(median_list_x);
            free(median_list_y);
        }
    }
    
    void MedianFilterFlowAt(int x, int y,
            double* flow, int filter_length,
            double* median_list_x,
            double* median_list_y,
            size_t width,
            size_t height,double* destination) 
    {
        int filter_length_half = filter_length/2;

        double* median_iter_x = median_list_x;
        double* median_iter_y = median_list_y;
        
        double* flow_y = flow;
        double* flow_x = flow+width*height;

        for(int src_y = y - filter_length_half; src_y<= y+filter_length_half;src_y++)
        {
            for(int src_x = x - filter_length_half;src_x <= x+filter_length_half;src_x++)
            {
                *(median_iter_x) = _inner::GetValueAt<double,Padding::SYMMETRIC>(src_x,src_y,width,height,flow_x);
                *(median_iter_y) = _inner::GetValueAt<double,Padding::SYMMETRIC>(src_x,src_y,width,height,flow_y);
                median_iter_y++;
                median_iter_x++;
            }
        }
        int coord = y*width+x;
        destination[coord] = _inner::_median(median_list_y,filter_length*filter_length);
        destination[coord+width*height] = _inner::_median(median_list_x,filter_length*filter_length);
    }

    


}
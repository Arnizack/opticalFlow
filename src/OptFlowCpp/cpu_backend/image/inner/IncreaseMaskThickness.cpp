#include "IncreaseMaskThickness.h"
#include"stdlib.h"
#include<algorithm>


namespace cpu_backend
{
    void IncreaseMaskThickness(bool* mask, int width, int height, int thickness, bool* destination) 
    {
        int thickness_half = thickness / 2;

            bool* mask_temp = (bool*)malloc(width * height * sizeof(bool));

            for (int dst_y = 0; dst_y < height; dst_y++)
            {
                for (int dst_x = 0; dst_x < width; dst_x++)
                {
                    int src_x_min = std::max(dst_x - thickness_half, 0);
                    int src_x_max = std::min(dst_x + thickness_half, width - 1);
                    bool is_masked = false;
                    for (int src_x = src_x_min; src_x <= src_x_max && !is_masked; src_x++)
                    {
                        is_masked = mask[dst_y * width + src_x];
                    }
                    mask_temp[dst_y * width + dst_x] = is_masked;
                }
            }

            for (int dst_y = 0; dst_y < height; dst_y++)
            {
                for (int dst_x = 0; dst_x < width; dst_x++)
                {
                    int src_y_min = std::max(dst_y - thickness_half, 0);
                    int src_y_max = std::min(dst_y + thickness_half, height - 1);
                    bool is_masked = false;
                    for (int src_y = src_y_min; src_y <= src_y_max && !is_masked; src_y++)
                    {
                        is_masked = mask_temp[src_y * width + dst_x];
                    }
                    destination[dst_y * width + dst_x] = is_masked;
                }
            }
            free(mask_temp);
    }
}
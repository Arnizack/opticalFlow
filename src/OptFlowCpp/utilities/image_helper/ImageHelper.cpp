#pragma once
#include"ImageHelper.h"
#include"libs/imageLib/imageLib/imageLib.h"

namespace imagehelper
{
    void SaveImage(std::string filepath, float* img, size_t width, size_t height, size_t color_count)
    {
        CByteImage lib_img(width,height,color_count);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < color_count; c++)
                {
                    size_t coord = c * width * height + y * width + x;
                    lib_img.Pixel(x, y, c) = img[coord] * 255.0;

                }
            }
        }

        WriteImage(lib_img, filepath.data());

        
    }
    void SaveImage(std::string filepath, Image img)
    {
        SaveImage(filepath, img.data->data(), img.width, img.height, img.color_count);
    }
    void SaveImage(std::string filepath, std::shared_ptr < core::IArray<float, 2>> img)
    {
        Image data_img;
        data_img.width = img->Shape[1];
        data_img.height = img->Shape[0];
        data_img.color_count = 1;
        data_img.data = std::make_shared < std::vector<float>>(data_img.width * data_img.height);
        img->CopyDataTo(data_img.data->data());
        SaveImage(filepath, data_img);
    }

    void SaveImage(std::string filepath, std::shared_ptr < core::IArray<float, 3>> img)
    {
        Image data_img;
        data_img.width = img->Shape[2];
        data_img.height = img->Shape[1];
        data_img.color_count = img->Shape[0];
        data_img.data = std::make_shared < std::vector<float>>(
            data_img.width * data_img.height * data_img.color_count);
        img->CopyDataTo(data_img.data->data());
        SaveImage(filepath, data_img);
    }

    Image OpenImage(std::string filepath)
    {
        CByteImage lib_img;
        size_t width, height, color_count;

        ReadImage(lib_img, filepath.data());
        
        auto shape = lib_img.Shape();
        height = shape.height;
        width = shape.width;
        color_count = shape.nBands;

        Image img;
        img.width = width;
        img.height = height;
        img.color_count = color_count;
        img.data = std::make_shared<std::vector<float>>(width*height*color_count);

        float* img_data = img.data->data();

        for(int c = 0; c < color_count; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float val = ((float)lib_img.Pixel(x, y, c)) / 255.0;
                    img_data[c * width * height + y * width + x] = val;

                }
            }
        }
        return img;

    }

    float& Image::Pixel(size_t x, size_t y, size_t channel)
    {
        return data->operator[](channel* width* height + y * width + x);
    }

}
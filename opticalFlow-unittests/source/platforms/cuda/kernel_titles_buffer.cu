
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<gtest/gtest.h>


#include <stdio.h>
#include<memory>
#include<vector>
#include"platforms/cuda/schedulers/tilesScheduler.cuh"

#include"platforms/cuda/helper/errorCheckingMacro.cuh"
#include<cuda_occupancy.h>
#include<math.h>
#include"platforms/cuda/datastructures/tilesBuffer.cuh"

#include"platforms/cuda/datastructures/kernelInfo.cuh"
#include"platforms/cuda/kernelLauncher.hpp"
#include <assert.h>
#include <cooperative_groups.h>

#define DD 2


#define testx 1
#define testy 2

namespace cuda
{

    struct addKernelTilesSharedDev
    {
        __device__ static void kernel(KernelInfo& kinfo, int* src, int* dst, int dimX, int dimY,int x_padding, int y_padding)
        {
            tilesBufferRF<int> buffer = allocTilesBufferRF<int>(kinfo, { dimX,dimY }, { 1,1 }, { x_padding,y_padding });
            //printf("alloc1\n");
            tilesBufferRF<int> buffer2 = allocTilesBufferRF<int>(kinfo, { dimX,dimY }, { 1,1 }, { x_padding,y_padding });
            //printf("alloc2\n");

            int2 dim = { dimX,dimY };
            cooperative_groups::thread_block block = cooperative_groups::this_thread_block();


            tilesScheduler2D(0, { dimX,dimY }, { 1,1 }, { x_padding,y_padding },
                [](const int2& idx, int* src, int* dst, const int2& dim, tilesBufferRF<int>& buffer)
            {

            
                int& item = buffer[idx];
                int res = src[idx.x + idx.y * dim.x] * 2;
                item = res;
            
            
            }, src, dst, dim, buffer);
        
            tilesScheduler2D(0, { dimX,dimY }, { 1,1 }, { x_padding,y_padding },
                [](const int2& idx, int* src, int* dst, const int2& dim, tilesBufferRF<int>& buffer)
            {


                int& item = buffer[idx];
                int res = src[idx.x + idx.y * dim.x] *2;
                item = res;


            }, src, dst, dim, buffer2);
        
            block.sync();
            tilesScheduler2D(0, { dimX,dimY }, { 1,1 }, { 0,0 },
                [](const int2& idx, int* src, int* dst, const int2& dim, tilesBufferRF<int>& buffer)
            {

          
                dst[idx.x + idx.y * dim.x] = buffer[idx];
           


            }, src, dst, dim, buffer);
            tilesScheduler2D(0, { dimX,dimY }, { 1,1 }, { 0,0 },
                [](const int2& idx, int* src, int* dst, const int2& dim, tilesBufferRF<int>& buffer)
            {


                dst[idx.x + idx.y * dim.x] = buffer[idx];



            }, src, dst, dim, buffer2);
        
        
       
        }
    };
    struct KernelKeeper
    {
        __device__ static void kernel(KernelInfo& kinfo, int test,int test2)
        {
            printf("%d %d\n", test,test2);
        }
    };

    TEST(cuda, kernel_titles_buffer)
    {
    
        int blockSize = 1;   // The launch configurator returned block size 
        int minGridSize = 1; // The minimum grid size needed to achieve the 
                         // maximum occupancy for a full device launch 
        int gridSize;    // The actual grid size needed, based on input size 
        /*
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
            addKernelTiles, 0, 0);
        */
        printf("blockSize: %d\n", blockSize);
        printf("gridSize: %d\n", minGridSize);

    
        int dimX = 14;
        int dimY = 13;

        //
        int x_padding = 5;
        int y_padding = 5;

        unsigned long N = (dimX+ 2 * x_padding)*(dimY+2*y_padding);

        std::vector<int>h_src(N);
        std::vector<int>h_dst(N);

    
        int a = 0;
        for (int& val : h_src)
            val = a++;


        int* d_src, *d_dst;
        cudaMalloc(&d_src, N * sizeof(int));
        cudaMalloc(&d_dst, N * sizeof(int));

        checkCUDAError(cudaMemcpy(d_src, h_src.data(), N * sizeof(int), cudaMemcpyHostToDevice));

        checkCUDAError(cudaMemcpy(d_dst, h_dst.data(), N * sizeof(int), cudaMemcpyHostToDevice));

        d_src += y_padding * dimX +x_padding;

        KernelLauncher lauchner(1200,1);

        lauchner.considerTilesBuffer(dimX, dimY, 1, 1, x_padding, y_padding, sizeof(int));
        lauchner.considerTilesBuffer(dimX, dimY, 1, 1, x_padding, y_padding, sizeof(int));
        lauchner.launch<addKernelTilesSharedDev,int*,int*,int,int,int,int>(d_src, d_dst, dimX, dimY, x_padding, y_padding);


    

        checkCUDAError(cudaDeviceSynchronize());
    
        checkCUDAError(cudaMemcpy( h_dst.data(), d_dst, N * sizeof(int), cudaMemcpyDeviceToHost));


        cudaFree(d_dst);
        cudaFree(d_src);
    
        int count = 0;
        for (int i = 0; i < dimY*dimX; i++)
        {
        
            EXPECT_EQ(h_dst[i], h_src[i + y_padding * dimX + x_padding] * 2);
            /*
            if (h_dst[i] != h_src[i+y_padding*dimX+x_padding] *2)
            {
                count++;
                printf("Fehler %d,%d i:%d\n", i%dimX,i/dimX,i);

                //break;
            }*/
        }

    
    }
}

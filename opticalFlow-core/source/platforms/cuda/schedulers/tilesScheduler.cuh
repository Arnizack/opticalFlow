
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#define __CUDA_INTERNAL_COMPILATION__
#include<math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__

#include <nvfunctional>
#include <stdio.h>


/*
PSEUDO Code:
def GridStripLoop2D(x,y,xDim,yDim,step):
	minYStep = int(step/xDim)
	switch = step%xDim
	while(x<xDim and y<yDim):
		print(x,y)
		if(x<(xDim-switch)):
			y+=minYStep
			x+=switch
		else:
			y+=minYStep+1
			x+=switch-xDim
*/
namespace cuda
{

	template<typename... ARGS, typename _inst>
	inline __device__ void _rectangleBlockSchedular2D(const int2& start,const int2& end,
		_inst Instruction, ARGS&&... args)
	{
	
		int dimX = end.x - start.x;
		const int stepSize = blockDim.x* blockDim.y* blockDim.z;
		int minYStep = stepSize / dimX;
		int rest = stepSize % dimX;


		int2 index = {threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y,start.y};
		/*
		if (blockIdx.x == 13 && blockIdx.y == 1)
		{
			printf("_rectangleBlockSchedular2D 1 idx: %d,%d\n", index.x, index.y);
			printf("dimX: %d\n", dimX);
		}*/

		index.y += index.x / dimX;
		index.x = start.x + (index.x % dimX);
		/*
		if (blockIdx.x == 13 && blockIdx.y == 1)
		{
			printf("_rectangleBlockSchedular2D 2 idx: %d,%d\n", index.x, index.y);
		}*/


		//#pragma unroll (4) //plus 1 !!!
		for(int i = index.x+index.y*dimX;i<= end.x-1+(end.y-1)*dimX ;i+=stepSize)

		{
			/*
			if (blockIdx.x == 13 && blockIdx.y == 1)
			{
				printf("_rectangleBlockSchedular2D 3 idx: %d,%d\n", index.x, index.y);
			}*/
			Instruction(index, std::forward<ARGS>(args)...);
			int d = 0;
			if (index.x >= dimX - rest + start.x)
				d = 1;
			index.y += minYStep + d;
			index.x += rest - dimX * d;
		
		}
	}

	template<typename... ARGS, typename _inst>
	inline __device__ void _rectangleBlockSchedular2DTrans(const int2& start, const int2& end,
		_inst Instruction, ARGS&&... args)
	{

	

		int dimY = end.y - start.y;
		const int stepSize = blockDim.x * blockDim.y * blockDim.z;
		int minXStep = stepSize / dimY;
		int rest = stepSize % dimY;
		//printf("start: %d,%d end: %d,%d stepSize: %d\n", start.x, start.y, end.x, end.y,stepSize);

		int2 index = {start.x, threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y};

		index.x += index.y / dimY;
		index.y = start.y + (index.y % dimY);

		//printf("rest: %d minXStep: %d\n", rest, minXStep);

		//#pragma unroll (4)
		for (int i = index.x * dimY + index.y;  i <= (end.x-1) * dimY + (end.y-1)  ; i += stepSize)

		{
			//printf("test [%d,%d] i:%d\n", index.x, index.y,i);
			Instruction(index, std::forward<ARGS>(args)...);
			int d = 0;
			if (index.y >= dimY - rest + start.y)
				d = 1;
			index.x += minXStep + d;
			index.y += rest - dimY * d;

		}
	}



	inline __device__ void _calcTilesBlockSize(const int2& dimenions, const int2& tilesSize, const int2& padding, int2& min_out, int2& max_out)
	{

	

		int xBlockCoords = (int)ceilf((float)dimenions.x / (float)(tilesSize.x * gridDim.x)) * tilesSize.x;
		int yBlockCoords = (int)ceilf((float)dimenions.y / (float)(tilesSize.y * gridDim.y)) * tilesSize.y;
		/*
		if(blockIdx.x==0 && blockIdx.y == 0)
		{
		
		}*/

	



		min_out.x = blockIdx.x * xBlockCoords ;
		min_out.y = blockIdx.y * yBlockCoords ;
		min_out.x = min(min_out.x, dimenions.x);
		min_out.y = min(min_out.y, dimenions.y);

		min_out.x -= padding.x;
		min_out.y -= padding.y;


		max_out.x = (blockIdx.x + 1) * xBlockCoords;
		max_out.y = (blockIdx.y + 1) * yBlockCoords;

		max_out.x = min(max_out.x, dimenions.x);
		max_out.y = min(max_out.y, dimenions.y);

		max_out.x += padding.x;
		max_out.y += padding.y;
		/*
		if (blockIdx.x == 13 && blockIdx.y == 1)
		{
		
			printf("\n x: %d = ceilf(%d /(%d * %d)) * %d\n y: %d = ceilf(%d /(%d * %d)) * %d\n block: %d,%d \n min: %d,%d max: %d,%d\n",
				xBlockCoords, dimenions.x, tilesSize.x, gridDim.x, tilesSize.x,
				yBlockCoords, dimenions.y, tilesSize.y, gridDim.y, tilesSize.y,
				blockIdx.x, blockIdx.y, min_out.x,min_out.y, max_out.x,max_out.y
			);
		}*/
	}

	template<typename... ARGS,typename _inst>
	inline __device__ void tilesScheduler2D(int Info, const int2& dimenions, const int2& tilesSize, const int2& padding,
		_inst Instruction, ARGS&&... args)
	{
		/*//little bit faster
		int xBlockCoords = (int)ceilf((float)dimenions.x / (float)(tilesSize.x * gridDim.x)) * tilesSize.x;
		int yBlockCoords = (int)ceilf((float)dimenions.y / (float)(tilesSize.y * gridDim.y)) * tilesSize.y;

		int xMin = blockIdx.x * xBlockCoords - padding.x;
		int yMin = blockIdx.y * yBlockCoords - padding.y;
	
		int xMax = (blockIdx.x + 1) * xBlockCoords;
		int yMax = (blockIdx.y + 1) * yBlockCoords;

		xMax = min(xMax, dimenions.x);
		yMax = min(yMax, dimenions.y-1);

		xMax += padding.x;
		yMax += padding.y;
		*/
		int2 min;
		int2 max;

		_calcTilesBlockSize(std::forward<const int2& >(dimenions),std::forward<const int2&>(tilesSize),std::forward<const int2&>(padding), min, max);
		if(min.x!=max.x && min.y != max.y)
			_rectangleBlockSchedular2D(min, max, Instruction, std::forward<ARGS>(args)...);
	
	}



	template<typename... ARGS, typename _inst>
	inline __device__ void tilesSchedular2DTrans(int Info, const int2& dimenions, const int2& tilesSize, const int2& padding,
		_inst Instruction, ARGS&&... args)
	{

		int2 min;
		int2 max;

		_calcTilesBlockSize(std::forward<const int2& >(dimenions), std::forward<const int2&>(tilesSize), std::forward<const int2&>(padding), min, max);
		if (min.x != max.x && min.y != max.y)
			_rectangleBlockSchedular2DTrans(min, max, Instruction, std::forward<ARGS>(args)...);

	}
}
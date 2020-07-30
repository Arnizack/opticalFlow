#pragma once

#include"tempMacros.h"
//#include"platforms/cpu/CPUBackend.h"
//#include"platforms/cuda/CUDABackend.cuh"

namespace kernels
{

template<class SBE>
class multKernel
{
	public:


		typedef typename SBE::ds::template Array<float>			Array;
		typedef typename SBE::dt::kernelInfo			kernInfo;
		typedef typename SBE::sh						schedulars;
	
		static DEVICE void instruction(int idx, Array src,
			float scalar,
			Array dst)
		{

			dst[idx] = src[idx] * scalar;
		}


		static DEVICE void kernel
		(kernInfo kInfo, int size,Array src,
			float scalar,
			Array dst )
		{
			
			schedulars::gridStripSchedular(kInfo, size, instruction, src,scalar,dst);

		}

	};

}
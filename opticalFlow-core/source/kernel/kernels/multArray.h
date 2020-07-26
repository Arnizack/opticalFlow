#pragma once

#include"tempMacros.h"
//#include"platforms/cpu/CPUBackend.h"

namespace kernels
{

template<class SBE>
class multKernel
{
	public:
	

		typedef  typename SBE::ds::template ArrayMetaData<float>	ArrayMD;

		typedef typename SBE::ds::template Array<float>			Array;
		typedef typename SBE::dt::kernelInfo			kernInfo;
		typedef typename SBE::sh						schedulars;
	
		static DEVICE void instruction(int idx, ArrayMD srcMD,
			float scalar,
			ArrayMD dstMD)
		{

			SBE::ds::Array<float>& src = *srcMD;
			SBE::ds::Array<float>& dst = *dstMD;
			dst[idx] = src[idx] * scalar;
		}


		static DEVICE void kernel
		(kernInfo kInfo, int size, ArrayMD src,
			float scalar,
			ArrayMD dst )
		{
			
			schedulars::gridStripSchedular(kInfo, size, instruction, src,scalar,dst);

		}

	};

}
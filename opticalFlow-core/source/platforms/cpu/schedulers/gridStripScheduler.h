#pragma once
#include"platforms/cpu/launchers/kernelInfo.h"
#include<functional>
#include"platforms/cpu/CPUMacros.h"
#include<memory>

namespace cpu
{

	template<typename _inst,typename... ARGS>
	void gridStripSchedular(kernelInfo info, int itemCount, _inst Instruction, ARGS... args)
	{
		
		int min = (itemCount * info.index) / info.threadCount;
		int max = (itemCount * (info.index+1)) / info.threadCount;

		for (int index = min; index < max; index++)
		{
			Instruction(index, std::forward<ARGS>(args)...);
			
		}
	}
}
#pragma once
#include<functional>
#include"kernelInfo.h"

namespace cpu
{
	template<typename _kern,typename ...ARGS>
	bool launchSerial(_kern kernel, ARGS... args)
	{
		kernelInfo kInfo;
		kInfo.threadCount = 4;
		for (int i = 0; i < 4; i++)
		{
			kInfo.index = i;
			kernel(kInfo, args...);
		}
		return true;
	}
}
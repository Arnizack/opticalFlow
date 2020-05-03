#pragma once
template<typename T>
class ICUDATransferable
{
public:
	virtual T GetDataForGPU();
	virtual ~ICUDATransferable();
};


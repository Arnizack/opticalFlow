#pragma once
#include"ICUDATransferable.h"
#include"FlowField.h"

struct CUDAFlowFildDATA
{

};

class CudaFlowField : public ICUDATransferable<CUDAFlowFildDATA>
{
public:
	CudaFlowField(const FlowField& flowFlied);
	CudaFlowField(uint32_t wight, uint32_t height);
	~CudaFlowField();

	CUDAFlowFildDATA GetDataForGPU();
	CudaFlowField GetFlowFieldToCpu();

	CudaFlowField Upsize(uint32_t target_wight, uint32_t target_height);
};


#pragma once
#include"../linalg/IOperator.h"
#include"../IArray.h"
#include<memory>

namespace core
{
	using PtrFlowField = std::shared_ptr<IArray<double, 3>>;
	using PtrColorImage = std::shared_ptr<IArray<float, 3>>;
	class ICrossFlowFilter : public IOperator<PtrFlowField,PtrFlowField>
	{
	public:
		virtual void SetAuxiliaryInfluence(double influence) = 0;
		virtual void SetCrossFilterImage(PtrColorImage image) = 0;

	};
}
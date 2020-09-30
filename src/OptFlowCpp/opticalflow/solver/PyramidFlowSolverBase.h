#pragma once
#include"core/solver/IFlowFieldSolver.h"
#include"PyramidSettings.h"
#include"FlowFieldSolverBase.h"
#include<memory>

namespace optflow
{
	namespace solver
	{
		
		template<class InnerTyp, size_t DimCount, class LayerSettingsTyp>
		
		class PyramidFlowSolverBase 
			: public FlowFieldSolverBase<InnerTyp,DimCount,PyramidSettings<LayerSettingsTyp>>
		{

		public:
			
			using PtrFlowField = std::shared_ptr<IArray<double, 3>>;
			using PtrImage = std::shared_ptr<IArray<InnerTyp, DimCount>>;

			PtrFlowField Solve(const PtrImage first_frame, 
				const PtrImage second_frame, const PtrFlowField InitialGuess) override final;

			virtual PtrFlowField SolveLayer()
			

		};
		template<class InnerTyp, size_t DimCount, class LayerSettingsTyp>
		inline std::shared_ptr<core::IArray<double, 3>> 
			PyramidFlowSolverBase<InnerTyp, DimCount, LayerSettingsTyp>::Solve(
				const std::shared_ptr<core::IArray<InnerTyp, DimCount>> first_frame, 
				const std::shared_ptr<core::IArray<InnerTyp, DimCount>> second_frame, 
				const std::shared_ptr<core::IArray<double, 3>> InitialGuess)
		{
			return nullptr;
		}
	}
}
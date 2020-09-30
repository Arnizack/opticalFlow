#pragma once
#include<memory>
#include"core/solver/IFlowFieldSolver.h"
#include"core/IArrayFactory.h"

namespace optflow
{
	template<class InnerTyp, size_t DimCount, class SettingsTyp>
	class FlowFieldSolverBase : public core::solver::IFlowFieldSolver<InnerTyp,DimCount,SettingsTyp>
	{
	protected:
		std::shared_ptr<core::IArrayFactory<double, 3>> _FlowFactory;

	public:
		FlowFieldSolverBase(std::shared_ptr<core::IArrayFactory<double, 3>> flow_factory)
			: _FlowFactory(flow_factory)
		{}

		using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
		using PtrImage = std::shared_ptr<core::IArray<InnerTyp, DimCount>>;

		PtrFlowField Solve(const PtrImage first_frame, 
			const PtrImage second_frame,
			SettingsTyp settings) override final;
		
		PtrFlowField Solve(const PtrImage first_frame,
			const PtrImage second_frame,
			const PtrFlowField InitialGuess,
			SettingsTyp settings) = 0;


	};






	template<class InnerTyp, size_t DimCount, class SettingsTyp>
	inline std::shared_ptr<core::IArray<double, 3>>
		FlowFieldSolverBase<
		InnerTyp, DimCount, SettingsTyp>::Solve(
			const std::shared_ptr<core::IArray<InnerTyp, DimCount>> first_frame, 
			const std::shared_ptr<core::IArray<InnerTyp, DimCount>> second_frame,
			SettingsTyp settings)
	{
		size_t width = first_frame->Shape[2];
		size_t height = first_frame->Shape[1];

		auto init_flow = _FlowFactory->Zeros({ 2,height,width });
		return Solve(first_frame, second_frame, init_flow, settings);
	}

	

}
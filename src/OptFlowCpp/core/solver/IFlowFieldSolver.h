#pragma once
#include"../IArray.h"
#include<memory>

namespace core
{
	namespace solver
	{
		template<class InnerTyp,size_t DimCount,class SettingsTyp>
		class IFlowFieldSolver
		{

		public:
			using PtrFlowField = std::shared_ptr<IArray<double, 3>>;
			using PtrImage = std::shared_ptr<IArray<InnerTyp, DimCount>>;

			virtual PtrFlowField Solve(const PtrImage first_frame, 
				const PtrImage second_frame,
				SettingsTyp settings) = 0;
			virtual PtrFlowField Solve(const PtrImage first_frame, 
				const PtrImage second_frame, 
				const PtrFlowField InitialGuess,
				SettingsTyp settings) = 0;

		};
	}
}
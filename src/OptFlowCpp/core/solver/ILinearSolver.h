#pragma once
#include "..\IArray.h"
#include "..\linalg\ILinearOperator.h"
#include <memory>
namespace core
{
	namespace solver
	{
		template<class InnerTyp, size_t DimCount, class SettingsTyp>
		class ILinearSolver
		{
		public:
			using PtrVector = std::shared_ptr<IArray<InnerTyp, 1>>;
			using PtrMatrix = std::shared_ptr<linalg::ILinearOperator<PtrVector, PtrVector>>;

			virtual PtrVector Solve(const PtrMatrix input_matrix, const PtrVector input_vector) = 0; Solve(const PtrMatrix input_matrix, const PtrVector input_vector) = 0;
			virtual PtrVector Solve(const PtrMatrix input_matrix, const PtrVector input_vector, const PtrMatrix preconditioner_matrix, const PtrVector initial_guess) = 0;
		};
	}
}
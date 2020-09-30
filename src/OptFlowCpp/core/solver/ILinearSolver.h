#pragma once
#include "..\IContainer.h"
#include <memory>
namespace core
{
	namespace solver
	{
		template<class InnerTyp, class SettingsTyp>
		class ILinearSolver
		{
		public:
			using PtrMatrix = std::shared_ptr<IContainer<InnerTyp>>;
			using PtrVector = std::shared_ptr<IContainer<InnerTyp>>;

			virtual Solve(const PtrMatrix input_matrix, const PtrVector input_vector) = 0;
			virtual Solve(const PtrMatrix input_matrix, const PtrVector input_vector, const PtrMatrix preconditioner_matrix, const PtrVector initial_guess) = 0;
		};
	}
}
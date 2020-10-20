#pragma once
#include "../framework.h"

#include "core/solver/ILinearSolver.h"
#include "core/IArrayFactory.h"
#include "core/linalg/IArithmeticChained.h"
#include "core/linalg/IArithmeticVector.h"

namespace optflow_solvers
{
	template<class InnerTyp>
	class ConjugateGradientSolver : public core::ILinearSolver<InnerTyp>
	{
		using PtrVector = std::shared_ptr< core::IArray<InnerTyp, 1>>;
		using PtrLinearOperator = std::shared_ptr<ILinearOperator<PtrVector, PtrVector>>;

	public:
		ConjugateGradientSolver( 
		std::shared_ptr< core::IArrayFactory<InnerTyp, 1>> arr_factory,
		std::shared_ptr< core::IArithmeticVector<InnerTyp, 1>> arith_vector,
		std::shared_ptr< core::IArithmeticChained<InnerTyp, 1>> arith_chained,
		const double tol,
		const size_t iter) 
			: _arr_factory(arr_factory), _arith_vector(arith_vector), _arith_chained(arith_chained), _tol(tol), _iter(iter)
		{}

		virtual PtrVector Solve(std::shared_ptr < core::ILinearProblem<InnerTyp>> problem) override
		{
			PtrVector x = (*_arr_factory).Zeros( (*b).Shape);

			return Solve(problem, x);
		}

		virtual PtrVector Solve(std::shared_ptr < core::ILinearProblem<InnerTyp>> problem, const PtrVector initial_guess) override
		{
			PtrLinearOperator A = (*problem).LinearOperator;
			PtrVector b = (*problem).Vector;

			PtrVector x = std::make_shared<core::IArray<InnerTyp, 1>> (*initial_guess);

			PtrVector r_current = (*_arith_vector).Sub(b, (*A_dot_product).Apply(x));
			PtrVector d = (*_arith_vector).Sub(b, (*A_dot_product).Apply(x));

			PtrVector z;
			PtrVector r_next;
			double alpha;
			double beta;

			for (size_t i = 0; i < _iter; i++)
			{
				(*A).ApplyTo(z, d);

				alpha = (*_arith_vector).ScalarProduct(r_current, r_current) / (*_arith_vector).ScalarProduct(d, z);

				(*_arith_chained).AddTo(x, x, (*_arith_vector).Scale(alpha, d));

				(*_arith_chained).SubTo(r_next, r_current, (*_arith_vector).Scale(alpha, z));

				if ((*_arith_vector).NormEuclidean(r_next) < _tol)
				{
					return x;
				}

				beta = (*_arith_vector).ScalarProduct(r_next, r_next) / (*_arith_vector).ScalarProduct(r_current, r_current);

				(*_arith_chained).AddTo(d, r_next, (*_arith_vector).Scale(beta, d));

				r_current = r_next;
			}

			return x;
		}

	private:
		const double _tol;
		const size_t _iter;

		std::shared_ptr< core::IArrayFactory<InnerTyp, 1>> _arr_factory;
		std::shared_ptr< core::IArithmeticVector<InnerTyp, 1>> _arith_vector;
		std::shared_ptr< core::IArithmeticChained<InnerTyp, 1>> _arith_chained;
	};
}
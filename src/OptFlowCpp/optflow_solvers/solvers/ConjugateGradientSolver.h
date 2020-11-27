#pragma once
#include "../framework.h"
#include"settings/CGSolverSettings.h"
#include "core/solver/ILinearSolver.h"
#include "core/IArrayFactory.h"
#include "core/linalg/IArithmeticChained.h"
#include "core/linalg/IArithmeticVector.h"
#include "core/Logger.h"
#include<iostream>

namespace optflow_solvers
{
	
	template<class InnerTyp>
	class ConjugateGradientSolver : public core::ILinearSolver<InnerTyp>
	{
		using PtrVector = std::shared_ptr< core::IArray<InnerTyp, 1>>;
		using PtrLinearOperator = std::shared_ptr<core::ILinearOperator<PtrVector, PtrVector>>;

	public:
		ConjugateGradientSolver( 
		std::shared_ptr< core::IArrayFactory<InnerTyp, 1>> arr_factory,
		std::shared_ptr< core::IArithmeticVector<InnerTyp, 1>> arith_vector,
		std::shared_ptr< core::IArithmeticChained<InnerTyp, 1>> arith_chained,
		std::shared_ptr < CGSolverSettings> settings
		) 
			: _arr_factory(arr_factory), _arith_vector(arith_vector), _arith_chained(arith_chained), 
			_tol(settings->Tolerance), _iter(settings->Iterations)
		{}

		virtual PtrVector Solve(std::shared_ptr < core::ILinearProblem<InnerTyp>> problem) override
		{
			PtrVector initial_guess = _arr_factory->Zeros({ problem->Vector->Shape[0] });

			return Solve(problem, initial_guess);
		}

		virtual PtrVector Solve(std::shared_ptr < core::ILinearProblem<InnerTyp>> problem, const PtrVector initial_guess) override
		{
			OPF_LOG_TRACE("Solve CG");
			PtrLinearOperator A = problem->LinearOperator;
			PtrVector b = problem->Vector;

			PtrVector x = initial_guess;

			PtrVector r_current = _arith_chained->Sub(b, A->Apply(x)); // r_current = b - A.dot(x)
			PtrVector d = _arith_chained->Sub(b, A->Apply(x)); // d = r

			PtrVector z = _arr_factory->Zeros({ b->Shape });
			PtrVector r_next = _arr_factory->Zeros({ b->Shape });
			double alpha;
			double beta;

			for (size_t i = 0; i < _iter; i++)
			{
				A->ApplyTo(z, d); //z = A.dot(d)

				alpha = _arith_vector->ScalarDivScalar(r_current, r_current, d, z); // alpha = <r_current, r_current> / <d, z>

				_arith_chained->ScaleAddTo(x, alpha, d, x); // x = alpha * d + x

				_arith_chained->ScaleAddTo(r_next, -alpha, z, r_current); // r_next  = r_current - alpha * z

				if (_arith_vector->NormEuclidean(r_next) < _tol)
				{
					return x;
				}

				beta = _arith_vector->ScalarDivScalar(r_next, r_next, r_current, r_current); // beta = <r_next, r_next> / <r_current, r_current>

				_arith_chained->ScaleAddTo(d, beta, d, r_next); // d = beta*d + r_next

				std::swap(r_current, r_next);
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
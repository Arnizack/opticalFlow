#include "optflow_solvers/solvers/ConjugateGradientSolver.h"

#include "unit_tests/core_mock_adaptor/MockIArrayFactory1D.h"
#include "unit_tests/core_mock_adaptor/linalg/MockIArithmeticChained.h"
#include "unit_tests/core_mock_adaptor/linalg/MockIArithmeticVector.h"
#include "unit_tests/core_mock_adaptor/solver/problem/FakeLinearProblem.h"

#include "cpu_backend/linalg/LinearSystemMatrix.h"
#include "cpu_backend/ArrayFactory.h"
#include "cpu_backend/Array.h";
#include "cpu_backend/linalg/ArithmeticChained.h";
#include "cpu_backend/linalg/ArithmeticVector.h";

namespace optflow_solvers
{
	TEST(ConjugateGradientSolver, MockTest)
	{
		double tol = 0.001;
		size_t iter = 1;

		//ArrayFactory
		core::testing::MockIArrayFactory1D* arr_factory = new core::testing::MockIArrayFactory1D;
		std::shared_ptr<core::IArrayFactory<double, 1>> ptr_arr_factory(arr_factory);

		//ArithmeticBasic
		core::testing::MockIArithmeticChained1D* arith_chained = new core::testing::MockIArithmeticChained1D;
		std::shared_ptr<core::IArithmeticChained<double, 1>> ptr_arith_chained(arith_chained);

		//ArithmeticVector
		core::testing::MockIArithmeticVector1D* arith_vector = new core::testing::MockIArithmeticVector1D;
		std::shared_ptr<core::IArithmeticVector<double, 1>> ptr_arith_vector(arith_vector);

		//Settings
		auto settings = std::make_shared<CGSolverSettings>();
		settings->Iterations = iter;
		settings->Tolerance = tol;

		//Solver Obj
		ConjugateGradientSolver<double> cg_solver(ptr_arr_factory, ptr_arith_vector, ptr_arith_chained, settings);

		//Linear Problem
		core::testing::FakeLinearProblem* problem = new core::testing::FakeLinearProblem;
		std::shared_ptr<core::ILinearProblem<double>> ptr_problem(problem);

		//Expectations
		using ::testing::_;

		EXPECT_CALL(*arith_vector, NormEuclidean(_)).WillRepeatedly([](const std::shared_ptr<core::IArray<double, 1>> vec) -> double {return 1;}); //macht nicht was es soll

		EXPECT_CALL(*arr_factory, Zeros(_)).Times(3);


		EXPECT_CALL(*arith_chained, Sub(_, _)).Times(2);

		EXPECT_CALL(*arith_chained, ScaleAddTo(_, _, _, _)).Times(2*iter); // 3


		EXPECT_CALL(*arith_vector, NormEuclidean(_)).Times(1 * iter);

		EXPECT_CALL(*arith_vector, ScalarDivScalar(_, _, _, _)).Times(1 * iter); //2

		//call
		cg_solver.Solve(ptr_problem);
	}

	TEST(ConjugateGradientSolver, Test)
	{
		double tol = 0.00001;
		size_t iter = 100;

		//ArrayFactory
		cpu_backend::ArrayFactory<double, 1> arr_factory;
		std::shared_ptr<cpu_backend::ArrayFactory<double, 1>> ptr_arr_factory = std::make_shared<cpu_backend::ArrayFactory<double, 1>>(arr_factory);

		//ArithmeticBasic
		cpu_backend::ArithmeticChained<double, 1> arith_chained(ptr_arr_factory);
		std::shared_ptr<cpu_backend::ArithmeticChained<double, 1>> ptr_arith_chained = std::make_shared<cpu_backend::ArithmeticChained<double, 1>>(arith_chained);

		//ArithmeticVector
		cpu_backend::ArithmeticVector<double, 1> arith_vector(ptr_arr_factory);
		std::shared_ptr<cpu_backend::ArithmeticVector<double, 1>> ptr_arith_vector = std::make_shared<cpu_backend::ArithmeticVector<double, 1>>(arith_vector);

		//Settings
		auto settings = std::make_shared<CGSolverSettings>();
		settings->Iterations = iter;
		settings->Tolerance = tol;

		//Solver Obj
		ConjugateGradientSolver<double> cg_solver(ptr_arr_factory, ptr_arith_vector, ptr_arith_chained, settings);

		//Problem
		double mat_arr[16] = { 4,8,2,5, 8,8,6,1, 2,6,8,4, 5,1,4,7 };
		cpu_backend::Array<double, 2> mat({ 4,4 }, mat_arr);
		cpu_backend::LinearSystemMatrix<double> linear_sys_mat(ptr_arr_factory, std::make_shared<cpu_backend::Array<double, 2>>(mat));

		double vector_arr[4] = { 8,4,6,8 };
		cpu_backend::Array<double, 1> vector({ 4 }, vector_arr);

		core::ILinearProblem<double> problem;
		problem.Vector = std::make_shared<cpu_backend::Array<double, 1>>(vector); //Problem Vector
		problem.LinearOperator = std::make_shared<cpu_backend::LinearSystemMatrix<double>>(linear_sys_mat); //Problem Linear Operator

		//solve
		std::shared_ptr<core::IArray<double, 1>> out = cg_solver.Solve(std::make_shared<core::ILinearProblem<double>>(problem));
		std::shared_ptr<cpu_backend::Array<double, 1>> out_cpu = std::dynamic_pointer_cast<cpu_backend::Array<double, 1>>(out);

		double compare[4] = { 0.08333333,  0.30555556, -0.02777778,  1.05555556 };
		double temp;
		for (size_t i = 0; i < 4; i++)
		{
			temp = (*out_cpu)[i] - compare[i];
			//EXPECT_DOUBLE_EQ(compare[i], (*out_cpu)[i]);

			if (temp < 0)
			{
				temp *= -1;
			}

			if (temp > tol)
			{
				EXPECT_EQ(0, 1);
			}
		}
	}
}
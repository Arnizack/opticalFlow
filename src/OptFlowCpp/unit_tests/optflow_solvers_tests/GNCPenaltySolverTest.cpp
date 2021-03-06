
#include"optflow_solvers/solvers/GNCPenaltySolver.h"
#include"core_mock_adaptor/solver/MockIFlowFieldSolver.h"
#include"core_mock_adaptor/penalty/MockIBlenablePenalty.h"
#include"core_mock_adaptor/MockIArrayFactoryD3.h"
#include"core_mock_adaptor/solver/problem/MockIProblemFactory.h"
#include"core_mock_adaptor/MockIArray.h"
#include"core_mock_adaptor/solver/problem/FakeGrayCrossProblem.h"
#include"core_mock_adaptor/solver/problem/FakeGrayCrossPenaltyProblem.h"
#include"core_mock_adaptor/solver/MockIFlowSolverIterator.h"

namespace optflow_solvers
{

	TEST(GNCPenaltySolver, test1) {


		int gnc_steps = 3;



		core::testing::MockIFlowFieldSolver* first_solver =
			new core::testing::MockIFlowFieldSolver;
		

		std::shared_ptr< core::testing::MockIFlowFieldSolver>ptr_first_solver (first_solver);

		using IStandardSolver = core::IFlowFieldSolver<
			std::shared_ptr<core::IGrayPenaltyCrossProblem>>;

		using PtrStandardFlowSolver = std::shared_ptr<IStandardSolver>;


		std::shared_ptr<core::IFlowSolverIterator<core::IGrayPenaltyCrossProblem>> solver_iterator =
		std::make_shared<core::testing::MockIFlowSolverIterator>(2, ptr_first_solver);
		//solvers.push_back(std::shared_ptr<IStandardSolver>(&third_solver));

		using PtrGrayScale = std::shared_ptr <core::IArray<float, 2>>;
		using ::testing::Return;

		core::testing::MockIBlendablePenalty<double>* penalty_mock =
			new core::testing::MockIBlendablePenalty<double>;

		std::shared_ptr<core::IBlendablePenalty<double>> ptr_penalty(penalty_mock);

		core::testing::MockIArrayFactoryD3* flow_factory = new core::testing::MockIArrayFactoryD3;

		std::shared_ptr<core::IArrayFactory<double, 3>> ptr_flow_factory(flow_factory);

		core::testing::MockIProblemFactory* problem_factory
			= new core::testing::MockIProblemFactory;

		core::testing::FakeGrayCrossPenaltyProblem* penalty_problem
			= new core::testing::FakeGrayCrossPenaltyProblem;
		std::shared_ptr<core::IGrayPenaltyCrossProblem> ptr_penalty_problem(penalty_problem);

		using PtrPenaltyProblem = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

		EXPECT_CALL(*problem_factory, CreateGrayPenaltyCrossProblem)
			.WillOnce([&]()->PtrPenaltyProblem {
			int test = 0;
			return ptr_penalty_problem;
		});



		//ptr_penalty_problem->CrossFilterImage = std::shared_ptr<core::IArray<float, 3>>(nullptr);

		std::shared_ptr<core::IProblemFactory> ptr_problem_factory(problem_factory);

		auto settings = std::make_shared<GNCPenaltySolverSettings>();
		settings->GNCSteps = gnc_steps;
		GNCPenaltySolver gnc_solver(settings, solver_iterator, ptr_penalty,
			ptr_flow_factory, ptr_problem_factory);

		using ::testing::_;


		EXPECT_CALL(*penalty_mock, SetBlendFactor((double)0)).Times(1);


		EXPECT_CALL(*penalty_mock, SetBlendFactor((double)0.5)).Times(1);
		EXPECT_CALL(*penalty_mock, SetBlendFactor((double)1)).Times(1);


		EXPECT_CALL(*first_solver, Solve(_, _)).Times(3);

		EXPECT_CALL(*flow_factory, Zeros(std::array<const size_t, 3>({ 2,0,0 }))).Times(1);

		core::testing::FakeGrayCrossProblem* problem = 
			new core::testing::FakeGrayCrossProblem;


		std::shared_ptr<core::IGrayCrossFilterProblem> ptr_problem(problem);

		gnc_solver.Solve(ptr_problem);



	}

}
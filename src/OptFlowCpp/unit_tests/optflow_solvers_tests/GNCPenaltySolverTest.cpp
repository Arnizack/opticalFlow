
#include"optflow_solvers/solvers/GNCPenaltySolver.h"
#include"unit_tests/core_mock_adaptor/solver/MockIFlowFieldSolver.h"
#include"unit_tests/core_mock_adaptor/penalty/MockIBlenablePenalty.h"
#include"unit_tests/core_mock_adaptor/MockIArrayFactoryD3.h"
#include"unit_tests/core_mock_adaptor/solver/problem/MockIProblemFactory.h"
#include"unit_tests/core_mock_adaptor/MockIArray.h"
#include"unit_tests/core_mock_adaptor/solver/problem/FakeGrayCrossProblem.h"
#include"unit_tests/core_mock_adaptor/solver/problem/FakeGrayCrossPenaltyProblem.h"

namespace optflow_solvers
{

	TEST(GNCPenaltySolver, test1) {


		int gnc_steps = 3;



		core::testing::MockIFlowFieldSolver* first_solver =
			new core::testing::MockIFlowFieldSolver;
		core::testing::MockIFlowFieldSolver* second_solver =
			new core::testing::MockIFlowFieldSolver;
		core::testing::MockIFlowFieldSolver* third_solver =
			new core::testing::MockIFlowFieldSolver;

		using IStandardSolver = core::IFlowFieldSolver<
			std::shared_ptr<core::IGrayPenaltyCrossProblem>>;

		using PtrStandardFlowSolver = std::shared_ptr<IStandardSolver>;


		std::vector< PtrStandardFlowSolver> solvers;
		solvers.push_back(std::shared_ptr<IStandardSolver>(first_solver));
		solvers.push_back(std::shared_ptr<IStandardSolver>(second_solver));
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


		GNCPenaltySolver gnc_solver(gnc_steps, solvers, ptr_penalty, 
			ptr_flow_factory, ptr_problem_factory);

		using ::testing::_;


		EXPECT_CALL(*penalty_mock, SetBlendFactor((double)0)).Times(1);


		EXPECT_CALL(*penalty_mock, SetBlendFactor((double)0.5)).Times(1);
		EXPECT_CALL(*penalty_mock, SetBlendFactor((double)1)).Times(1);


		EXPECT_CALL(*first_solver, Solve(_, _)).Times(1);
		EXPECT_CALL(*second_solver, Solve(_, _)).Times(2);

		EXPECT_CALL(*flow_factory, Zeros(std::array<const size_t, 3>({ 2,0,0 }))).Times(1);

		core::testing::FakeGrayCrossProblem* problem = 
			new core::testing::FakeGrayCrossProblem;


		std::shared_ptr<core::IGrayCrossFilterProblem> ptr_problem(problem);

		gnc_solver.Solve(ptr_problem);



	}

}
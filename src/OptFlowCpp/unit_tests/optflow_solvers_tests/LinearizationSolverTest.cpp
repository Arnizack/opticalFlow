#include"gtest/gtest.h"
#include"optflow_solvers/solvers/LinearizationSolver.h"
#include"unit_tests/core_mock_adaptor/flow/MockICrossFlowFilter.h"
#include"unit_tests/optflow_solvers_mock_adaptor/linearsystems/FakeISunBakerLSUpdater.h"
#include"unit_tests/core_mock_adaptor/solver/MockILinearSolverD.h"
#include"unit_tests/core_mock_adaptor/MockIReshaperD.h"
#include"unit_tests/core_mock_adaptor/image/MockIGrayWarper.h"
#include"unit_tests/core_mock_adaptor/MockIArrayFactoryD3.h"
#include"unit_tests/core_mock_adaptor/linalg/MockIArithmeticBasic.h"
#include"unit_tests/core_mock_adaptor/solver/problem/FakeGrayCrossPenaltyProblem.h"


namespace optflow_solvers
{
	namespace testing
	{
		TEST(LinearizaionSolverTest, test1)
		{
			using namespace core::testing;
			double start_relaxation = 0.1;
			double end_relaxation = 10;
			double relaxation_steps = 3;

			auto mock_cross_filter = std::make_shared<MockICrossFlowFilter>();
			std::shared_ptr<core::ICrossFlowFilter> cross_filter = mock_cross_filter;

			auto mock_linear_system_builder = std::make_shared<FakeISunBakerLSUpdater>();
			std::shared_ptr<ISunBakerLSUpdater> linear_system_builder = mock_linear_system_builder;

			auto mock_linear_solver = std::make_shared < MockILinearSolverD>();
			std::shared_ptr<core::ILinearSolver<double>> linear_solver = mock_linear_solver;

			auto mock_flow_reshaper = std::make_shared<MockIReshaperD>();
			std::shared_ptr<core::IReshaper<double>> flow_reshaper = mock_flow_reshaper;

			auto mock_warper = std::make_shared<MockIGrayWarper>();
			std::shared_ptr<core::IGrayWarper> warper = mock_warper;

			auto mock_flow_factory = std::make_shared<MockIArrayFactoryD3>();
			std::shared_ptr<core::IArrayFactory<double, 3>> flow_factory = mock_flow_factory;

			auto mock_flow_arithmetic = std::make_shared<MockIArithmeticBasic3D>();
			std::shared_ptr<core::IArithmeticBasic<double, 3>> flow_arithmetic 
				= mock_flow_arithmetic;

			auto settings = std::make_shared<LinearizationSolverSettings>();
			settings->StartRelaxation = start_relaxation;
			settings->EndRelaxation = end_relaxation;
			settings->RelaxationSteps = relaxation_steps;

			LinearizationSolver solver(
				settings,
				cross_filter,
				linear_system_builder,
				linear_solver,
				flow_reshaper,
				warper,
				flow_factory,
				flow_arithmetic
			);

			auto fake_flow_problem = std::make_shared<FakeGrayCrossPenaltyProblem>();
			std::shared_ptr<core::IGrayPenaltyCrossProblem> flow_problem = fake_flow_problem;

			using ::testing::_;

			EXPECT_CALL(*mock_flow_factory, Zeros(std::array<const size_t, 3>({ 2,0,0 }))).Times(4);
			EXPECT_CALL(*mock_flow_arithmetic, AddTo(_, _, _)).Times(3);
			EXPECT_CALL(*mock_flow_arithmetic, SubTo(_, _, _)).Times(3);
			EXPECT_CALL(*mock_cross_filter, SetCrossFilterImage(_)).Times(1);
			EXPECT_CALL(*mock_cross_filter, ApplyTo(_,_)).Times(3);
			
			double first_relax = 0.10000000000000007;
			double second_relax = 9.306902992863403;
			double third_relax = 10;

			EXPECT_CALL(*mock_cross_filter, SetAuxiliaryInfluence(first_relax));
			EXPECT_CALL(*mock_cross_filter, SetAuxiliaryInfluence(second_relax));
			EXPECT_CALL(*mock_cross_filter, SetAuxiliaryInfluence(third_relax));
			EXPECT_CALL(*mock_linear_system_builder, SetFramePair(_,_));
			EXPECT_CALL(*mock_linear_system_builder, UpdateParameter(_, first_relax));
			EXPECT_CALL(*mock_linear_system_builder, UpdateParameter(_, second_relax));
			EXPECT_CALL(*mock_linear_system_builder, UpdateParameter(_, third_relax));

			EXPECT_CALL(*mock_flow_reshaper,Reshape1D(_) ).Times(3);
			EXPECT_CALL(*mock_flow_reshaper, Reshape3D(_,_)).Times(3);

			EXPECT_CALL(*mock_linear_solver, Solve(_, _)).Times(3);
			EXPECT_CALL(*mock_warper, Warp(_)).Times(1);
			EXPECT_CALL(*mock_warper, SetImage(_)).Times(1);

			solver.Solve(fake_flow_problem);
		}
	}
}
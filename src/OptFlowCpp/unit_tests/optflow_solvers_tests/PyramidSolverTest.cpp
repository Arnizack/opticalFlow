#include"gmock/gmock.h"
#include"optflow_solvers/solvers/PyramidSolver.h"
#include"core_mock_adaptor/MockIArrayFactoryD3.h"
#include"core_mock_adaptor/pyramid/FakeIPyramidBuilder.h"
#include"core_mock_adaptor/flow/MockIFlowScaler.h"
#include"core_mock_adaptor/solver/MockIFlowFieldSolver.h"
#include"core_mock_adaptor/solver/problem/FakeGrayCrossPenaltyProblem.h"
#include"core_mock_adaptor/pyramid/FakeIPyramid.h"
#include"core/pyramid/IPyramid.h"


namespace optflow_solvers
{
	TEST(PyramidSolverTest, test1)
	{
		using namespace core::testing;
		int pyramid_levels = 3;
		
		auto mock_flow_factory = std::make_shared<MockIArrayFactoryD3>();
		std::shared_ptr<core::IArrayFactory<double, 3>> flow_factory
			= std::static_pointer_cast<core::IArrayFactory<double, 3>, MockIArrayFactoryD3>
			(mock_flow_factory);

		auto mock_problem = std::make_shared<FakeGrayCrossPenaltyProblem>();
		auto problem = std::static_pointer_cast<
			core::IGrayPenaltyCrossProblem, FakeGrayCrossPenaltyProblem>(mock_problem);

		using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;
		auto fake_pyramid = std::make_shared<FakeIPyramid>(pyramid_levels, problem);
		auto pyramid = std::static_pointer_cast<core::IPyramid<core::IGrayPenaltyCrossProblem>, FakeIPyramid>(fake_pyramid);


		auto mock_pyramid_builder =
			std::make_shared<FakeIPyramidBuilder>(pyramid);
		std::shared_ptr<core::IPyramidBuilder< core::IGrayPenaltyCrossProblem>> pyramid_builder
			= std::static_pointer_cast<
			core::IPyramidBuilder< core::IGrayPenaltyCrossProblem>,
			FakeIPyramidBuilder >(mock_pyramid_builder);

		auto mock_flow_scaler = std::make_shared<MockIFlowScaler>();
		std::shared_ptr<core::IScaler<core::IArray<double,3>>> flow_scaler
			= std::static_pointer_cast<core::IScaler<core::IArray<double, 3>>,MockIFlowScaler>(mock_flow_scaler);

		auto mock_inner_solver = std::make_shared<MockIFlowFieldSolver>();
		std::shared_ptr<core::IFlowFieldSolver<PtrProblemTyp>> inner_solver
			= std::static_pointer_cast<core::IFlowFieldSolver<PtrProblemTyp>,
			MockIFlowFieldSolver >(mock_inner_solver);

		PyramidSolver solver(flow_factory,pyramid_builder,flow_scaler,inner_solver);

		

		using ::testing::_;
		EXPECT_CALL(*mock_inner_solver, Solve(_, _)).Times(3);
		
		EXPECT_CALL(*mock_flow_factory, Zeros(std::array<const size_t, 3>({ 2,0,0 }))).Times(1);

		
		EXPECT_CALL(*mock_flow_scaler, Scale(_, 0, 0)).Times(3);
		
		solver.Solve(problem);

	}
}
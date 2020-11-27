set(AUTOSRC
"main.cpp"
"test.cpp"
"MockIArray.h"
"MockIArrayFactory1D.h"
"MockIArrayFactoryD3.h"
"MockIReshaperD.h"
"flow/MockICrossFlowFilter.h"
"flow/MockIFlowScaler.h"
"image/MockIGrayWarper.h"
"linalg/MockIArithmeticBasic.h"
"linalg/MockIArithmeticChained.h"
"linalg/MockIArithmeticVector.h"
"linalg/MockILinearOperator.h"
"penalty/MockIBlenablePenalty.h"
"penalty/MockIPenalty.h"
"pyramid/FakeIPyramid.cpp"
"pyramid/FakeIPyramidBuilder.cpp"
"pyramid/FakeIPyramid.h"
"pyramid/FakeIPyramidBuilder.h"
"solver/MockIFlowSolverIterator.cpp"
"solver/MockIFlowFieldSolver.h"
"solver/MockIFlowSolverIterator.h"
"solver/MockILinearSolverD.h"
"solver/problem/FakeGrayCrossPenaltyProblem.cpp"
"solver/problem/FakeGrayCrossProblem.cpp"
"solver/problem/FakeLinearProblem.cpp"
"solver/problem/FakeGrayCrossPenaltyProblem.h"
"solver/problem/FakeGrayCrossProblem.h"
"solver/problem/FakeLinearProblem.h"
"solver/problem/MockILinearProblem.h"
"solver/problem/MockIProblemFactory.h"
)

set(INC
    "../"
)

set(LIBS
    core
)

optflow_add_gmock_lib(core_mock_adaptor "${AUTOSRC}" "${INC}" "${LIBS}" )
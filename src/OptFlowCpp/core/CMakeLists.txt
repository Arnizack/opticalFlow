set(AUTOSRC
"Logger.cpp"
"pch.cpp"
"Base.h"
"IArray.h"
"IArrayFactory.h"
"IContainer.h"
"IReshaper.h"
"IScaler.h"
"IStatistics.h"
"Logger.h"
"pch.h"
"flow/ICrossFlowFilter.h"
"image/IColorConvolution.h"
"image/IColorFilter.h"
"image/IColorScaler.h"
"image/IGrayscaleConvolution.h"
"image/IGrayscaleFilter.h"
"image/IGrayscaleScaler.h"
"image/IGrayWarper.h"
"linalg/IArithmeticBasic.h"
"linalg/IArithmeticChained.h"
"linalg/IArithmeticVector.h"
"linalg/ILinearOperator.h"
"linalg/IOperator.h"
"penalty/IBlendablePenalty.h"
"penalty/IPenalty.h"
"pyramid/IPyramid.h"
"pyramid/IPyramidBuilder.h"
"solver/IFlowFieldSolver.h"
"solver/IFlowSolverIterator.h"
"solver/ILinearSolver.h"
"solver/problem/IColor2FrameProblem.h"
"solver/problem/IColorCrossFilterProblem.h"
"solver/problem/IGray2FrameProblem.h"
"solver/problem/IGrayCrossFilterProblem.h"
"solver/problem/IGrayPenaltyCrossProblem.h"
"solver/problem/IGrayPenaltyProblem.h"
"solver/problem/ILinearProblem.h"
"solver/problem/IPenaltyProblem.h"
"solver/problem/IProblemFactory.h"
)

set(INC 
    "../"
)

set(LIBS
    spdlog
)

message("Toolchain: ${CMAKE_TOOLCHAIN_FILE}")

optflow_add_library(core "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)


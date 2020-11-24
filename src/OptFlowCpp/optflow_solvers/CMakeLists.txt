set(AUTOSRC
"OpticalFlowApplication.cpp"
"pch.cpp"
"framework.h"
"OpticalFlowApplication.h"
"pch.h"
"linearsystems/ISunBakerLSUpdater.h"
"pyramid/GrayPenaltyCrossPyramidBuilder.cpp"
"pyramid/GrayPenaltyCrossPyramidBuilder.h"
"pyramid/Pyramid.h"
"pyramid/PyramidBuilder.h"
"solvers/GNCPenaltySolver.cpp"
"solvers/IncrementalSolver.cpp"
"solvers/LinearizationSolver.cpp"
"solvers/PyramidSolver.cpp"
"solvers/PyramidSolverIterator.cpp"
"solvers/ConjugateGradientSolver.h"
"solvers/GNCPenaltySolver.h"
"solvers/IncrementalSolver.h"
"solvers/LinearizationSolver.h"
"solvers/PyramidSolver.h"
"solvers/PyramidSolverIterator.h"
"solvers/TwoSolverIterator.h"
"solvers/settings/CGSolverSettings.h"
"solvers/settings/GNCPenaltySolverSettings.h"
"solvers/settings/IncrementalSolverSettings.h"
"solvers/settings/LinearizationSolverSettings.h"
)

set(INC 
    "../"
)

set(LIBS
    core
    debug

)

optflow_add_library(optflow_solvers "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)
target_precompile_headers(optflow_solvers PRIVATE "pch.h")

if(MSVC)
    #Precompiled Header will be included
    target_compile_options(optflow_solvers PRIVATE "/FI")
endif()
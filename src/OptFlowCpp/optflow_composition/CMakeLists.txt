set(AUTOSRC
"ContainerInstaller.cpp"
"CPUBackendInstaller.cpp"
"FlowSolverInstaller.cpp"
"Backends.h"
"ContainerInstaller.h"
"ContainerOptions.h"
"CPUBackendInstaller.h"
"CPUBackendOptions.h"
"FlowSolverInstaller.h"
"FlowSolverOptions.h"
"CPUBackendInstallers/CPUBackendContainer.cpp"
"CPUBackendInstallers/RegisterCPUBackend.cpp"
"CPUBackendInstallers/RegisterCPULinalg.cpp"
"CPUBackendInstallers/CPUBackendContainer.h"
"CPUBackendInstallers/RegisterCPUBackend.h"
"CPUBackendInstallers/RegisterCPULinalg.h"
"FlowSolverInstallers/GNCSolverContainer.cpp"
"FlowSolverInstallers/LevelSolverContainer.cpp"
"FlowSolverInstallers/PyramidContainer.cpp"
"FlowSolverInstallers/PyramidIteratorContainer.cpp"
"FlowSolverInstallers/GNCSolverContainer.h"
"FlowSolverInstallers/LevelSolverContainer.h"
"FlowSolverInstallers/PyramidContainer.h"
"FlowSolverInstallers/PyramidIteratorContainer.h"
)

set(INC 
    "../"
)

set(LIBS
    core
    cpu_backend
    optflow_solvers
    hypodermic
)

optflow_add_library(optflow_composition "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)


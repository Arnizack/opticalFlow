set(AUTOSRC
"JSONHandler.cpp"
"JSONHelper.cpp"
"JSONHandler.h"
"JSONHelper.h"
"Parser/FlowContainerParser.cpp"
"Parser/LevelContainerParser.cpp"
"Parser/PyramidContainerParser.cpp"
"Parser/FlowContainerParser.h"
"Parser/LevelContainerParser.h"
"Parser/PyramidContainerParser.h"
"Parser/CPUBackendParser/CPUBackendParser.cpp"
"Parser/CPUBackendParser/CPUBackendParser.h"
"Parser/FlowSolverParser/CGSolverParser.cpp"
"Parser/FlowSolverParser/GNCPenaltySolverParser.cpp"
"Parser/FlowSolverParser/IncrementalSolverParser.cpp"
"Parser/FlowSolverParser/LinearizationSolverParser.cpp"
"Parser/FlowSolverParser/CGSolverParser.h"
"Parser/FlowSolverParser/GNCPenaltySolverParser.h"
"Parser/FlowSolverParser/IncrementalSolverParser.h"
"Parser/FlowSolverParser/LinearizationSolverParser.h"
)

set(INC 
"../"
)

set(LIBS
optflow_composition
nlohmann_json
boost

)

optflow_add_library(json_settings "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)
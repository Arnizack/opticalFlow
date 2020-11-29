set(AUTOSRC
"opticalflow.cpp"
"Base.h"
"opticalflow.h"
)
set(INC 
    "../"
)

set(LIBS
    optflow_composition
    json_settings
    optflow_solvers
    debug_helper
)

optflow_add_library(opticalflow "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)
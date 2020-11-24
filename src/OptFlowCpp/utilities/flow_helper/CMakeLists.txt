set(AUTOSRC
"FlowHelper.cpp"
"FlowHelper.h"
)

set(INC 
    "../../"
)

set(LIBS
    flowcode
    core
)

message("Toolchain: ${CMAKE_TOOLCHAIN_FILE}")

optflow_add_library(flow_helper "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)
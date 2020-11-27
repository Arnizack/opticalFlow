set(AUTOSRC
"ImageLogger.cpp"
"Base.h"
"ImageLogger.h"
)

set(INC 
    "../"
)

set(LIBS
    core
    flow_helper
    image_helper
    
)

message("Toolchain: ${CMAKE_TOOLCHAIN_FILE}")

optflow_add_library(debug_helper "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)
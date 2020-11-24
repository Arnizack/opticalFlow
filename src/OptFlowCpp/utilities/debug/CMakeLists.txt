set(AUTOSRC
"ImageLogger.cpp"
"ImageLogger.h"
)

set(INC 
    "../../"
)

set(LIBS
    core
    flow_helper
    image_helper
    
)

message("Toolchain: ${CMAKE_TOOLCHAIN_FILE}")

optflow_add_library(debug "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)
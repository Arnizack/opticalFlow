set(AUTOSRC
"ImageHelper.cpp"
"ImageHelper.h"
)

set(INC 
    "../../"
)

set(LIBS
    imagelib
    core
)

message("Toolchain: ${CMAKE_TOOLCHAIN_FILE}")

optflow_add_library(image_helper "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)
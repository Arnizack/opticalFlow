set(AUTOSRC
"Timer.cpp"
"Timer.h"
)

set(INC
    "../../"
)

set(LIBS
)

optflow_add_library(test_utilities "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)
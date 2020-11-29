set(SRC
"colorcode.cpp"
"flowIO.cpp"
"colorcode.h"
"flowIO.h"
)

set(INC 
    "../"
)

set(LIBS
    imagelib
)

optflow_add_library(flowcode "${SRC}" "${INC}" "${LIBS}" STATIC)


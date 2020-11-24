set(SRC
"imageLib/Convert.cpp"
"imageLib/Convolve.cpp"
"imageLib/Image.cpp"
"imageLib/ImageIO.cpp"
"imageLib/ImageIOpng.cpp"
"imageLib/RefCntMem.cpp"
"imageLib/Convert.h"
"imageLib/Convolve.h"
"imageLib/Copyright.h"
"imageLib/Error.h"
"imageLib/Image.h"
"imageLib/ImageIO.h"
"imageLib/imageLib.h"
"imageLib/RefCntMem.h"
)

set(INC 
    "imageLib"
)

set(LIBS
    libpng
)

optflow_add_library(imagelib "${SRC}" "${INC}" "${LIBS}" STATIC)


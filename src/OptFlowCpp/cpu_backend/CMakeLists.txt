set(AUTOSRC
"pch.cpp"
"Array.h"
"ArrayFactory.h"
"ArrayScaler.h"
"Base.h"
"Container.h"
"pch.h"
"Reshaper.h"
"Statistics.h"
"flow/CrossBilateralMedianFilter.cpp"
"flow/FlowFieldScaler.cpp"
"flow/CrossBilateralMedianFilter.h"
"flow/FlowFieldScaler.h"
"flow/inner/BilateralMedian.cpp"
"flow/inner/BilateralMedianSpeedup.cpp"
"flow/inner/LogOcclusion.cpp"
"flow/inner/BilateralMedian.h"
"flow/inner/BilateralMedianSpeedup.h"
"flow/inner/LogOcclusion.h"
"image/inner/ConvertToGrayscale.cpp"
"image/inner/IncreaseMaskThickness.cpp"
"image/inner/WarpLinearColorImage.cpp"
"image/inner/ArrayHelper.h"
"image/inner/BicubicInterpolate.h"
"image/inner/BicubicScale.h"
"image/inner/BilinearInterpolate.h"
"image/inner/ConvertToGrayscale.h"
"image/inner/convolution1D.h"
"image/inner/convolution2D.h"
"image/inner/DerivativeCalculator.h"
"image/inner/DownScaleGaussianGrayScale.h"
"image/inner/EdgeDetection.h"
"image/inner/GaussianScale.h"
"image/inner/IncreaseMaskThickness.h"
"image/inner/IteratorHelper.h"
"image/inner/WarpLinearColorImage.h"
"image/inner/filters/Gaussian1DFilter.h"
"image/inner/filters/SobelFilter.h"
"image/warper/GrayWarper.cpp"
"image/warper/GrayWarper.h"
"linalg/ArithmeticBasic.h"
"linalg/ArithmeticChained.h"
"linalg/ArithmeticVector.h"
"linalg/LinearSystemMatrix.h"
"penalty/CharbonnierPenalty.cpp"
"penalty/QuadraticPenalty.cpp"
"penalty/CharbonnierPenalty.h"
"penalty/QuadraticPenalty.h"
"problem/ProblemFactory.cpp"
"problem/GrayPenaltyCrossProblemScaler.h"
"problem/ProblemFactory.h"
"sb_linearsystem/SunBakerLinearOp.cpp"
"sb_linearsystem/SunBakerLSUpdater.cpp"
"sb_linearsystem/SunBakerLinearOp.h"
"sb_linearsystem/SunBakerLSUpdater.h"
)

set(INC 
    "../"
)

set(LIBS
    core
    openblas
    debug_helper

)

optflow_add_library(cpu_backend "${AUTOSRC}" "${INC}" "${LIBS}" STATIC)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_backend OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(cpu_backend PRIVATE "/fp:fast")
endif()
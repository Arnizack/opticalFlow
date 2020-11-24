set(AUTOSRC
"array_test.cpp"
"GrayPenaltyCrossProblemScaler_Test.cpp"
"main.cpp"
"Reshape_Test.cpp"
"Scaler_Test.cpp"
"flow/BilateralMedianFilterTest.cpp"
"flow/CrossBilateralMedianFilterTest.cpp"
"flow/FlowFieldScaler_Test.cpp"
"flow/OcclusionTest.cpp"
"image/GrayWarperTest.cpp"
"image/inner/BicubicUpscale_Test.cpp"
"image/inner/convolution1DTest.cpp"
"image/inner/convolution2DTest.cpp"
"image/inner/DerivativeTest.cpp"
"image/inner/DownScaleTest.cpp"
"image/inner/GaussianFilterTest.cpp"
"image/inner/WarpLinearColorTest.cpp"
"linalg/ArithmeticBasic_test.cpp"
"linalg/ArithmeticChained_Test.cpp"
"linalg/ArithmeticVector_Test.cpp"
"penalty/CharbonnierPenaltyTest.cpp"
"Pyramid/PyramidBuilder_Test.cpp"
"sb_linearsystem/SunBakerLinSystemTest.cpp"
)

set(INC
    "../../"
)

set(LIBS
    core
    cpu_backend
    test_utilities
    image_helper
)

optflow_add_gtest(cpu_backend_test "${AUTOSRC}" "${INC}" "${LIBS}")
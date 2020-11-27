set(AUTOSRC
"ConjugateGradientSolverTest.cpp"
"GNCPenaltySolverTest.cpp"
"LinearizationSolverTest.cpp"
"main.cpp"
"PyramidSolverTest.cpp"
)

set(INC
    "../"
)

set(LIBS
    core
    cpu_backend
    optflow_solvers
    optflow_solvers_mock_adaptor
    test_utilities
    image_helper
    debug
)

optflow_add_gtest(optflow_solvers_test "${AUTOSRC}" "${INC}" "${LIBS}")
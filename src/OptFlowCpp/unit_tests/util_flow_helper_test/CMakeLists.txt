set(AUTOSRC
"FlowHelperTest.cpp"
"main.cpp"
)

set(INC
    "../../"
)

set(LIBS
    core
    flow_helper
    image_helper
)

optflow_add_gtest(util_flow_helper_test "${AUTOSRC}" "${INC}" "${LIBS}")
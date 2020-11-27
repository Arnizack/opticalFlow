set(AUTOSRC
"ImageTest.cpp"
"main.cpp"
)

set(INC
    "../"
)

set(LIBS
    core
    image_helper
)

optflow_add_gtest(util_image_helper_test "${AUTOSRC}" "${INC}" "${LIBS}")
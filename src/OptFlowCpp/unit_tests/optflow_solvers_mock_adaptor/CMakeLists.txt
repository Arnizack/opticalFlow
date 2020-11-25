set(AUTOSRC
"linearsystems/FakeISunBakerLSUpdater.cpp"
"linearsystems/FakeISunBakerLSUpdater.h"
)

set(INC
    "../../"
)

set(LIBS
    core
    core_mock_adaptor
    optflow_solvers
)

optflow_add_gmock_lib(optflow_solvers_mock_adaptor "${AUTOSRC}" "${INC}" "${LIBS}" )
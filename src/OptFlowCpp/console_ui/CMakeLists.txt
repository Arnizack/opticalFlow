set(AUTOSRC
"main.cpp"
"Base.h"
"commandline/IClEngine.h"
"commandline/commands/ICLArgument.h"
"commandline/commands/ICLCommand.h"
"commandline/commands/IClOption.h"
"commandline_boost/ProgramOptions.cpp"
"commandline_boost/SetupOptions.cpp"
"commandline_boost/ProgramOptions.h"
"commandline_boost/SetupOptions.h"
)

set(INC 
"../"
)

set(LIBS
optflow_composition
json_settings
optflow_solvers
debug_helper
boost

)

optflow_add_executable(console_ui "${AUTOSRC}" "${INC}" "${LIBS}")
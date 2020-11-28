set(AUTOSRC
"main.cpp"
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
opticalflow
boost

)

optflow_add_executable(console_ui "${AUTOSRC}" "${INC}" "${LIBS}")
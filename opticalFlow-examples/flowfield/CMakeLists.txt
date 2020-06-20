add_executable(example-flow-field "main.cpp")

target_link_libraries(example-flow-field opticalFlow-examples-utilities)

set_target_properties(example-flow-field PROPERTIES FOLDER "Examples")
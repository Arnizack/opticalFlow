cmake_minimum_required(VERSION 3.16)

project(optical_flow LANGUAGES CXX)

enable_testing()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/macros.cmake)

add_subdirectory(libs)
add_subdirectory(core)
add_subdirectory(utilities)
add_subdirectory(optflow_solvers)
add_subdirectory(cpu_backend)
add_subdirectory(optflow_composition)
add_subdirectory(json_settings)
add_subdirectory(console_ui)

add_subdirectory(unit_tests)
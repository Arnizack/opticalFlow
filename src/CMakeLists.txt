cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(HalloWorld VERSION 0.1)


# Must use GNUInstallDirs to install libraries into correct
# locations on all platforms.
include(GNUInstallDirs)

enable_testing()
set(OPTION INCLUDE_TESTS OFF)

add_subdirectory(libs)

add_subdirectory(printerLib)

add_subdirectory(Programm)

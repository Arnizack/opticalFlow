cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(OpticalFlow VERSION 0.1)


# Must use GNUInstallDirs to install libraries into correct
# locations on all platforms.
include(GNUInstallDirs)

enable_testing()
set(OPTION INCLUDE_TESTS OFF)

add_subdirectory(libs)

#Add the Subdirectories here with add_subdirectory()

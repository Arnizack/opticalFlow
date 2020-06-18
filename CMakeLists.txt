cmake_minimum_required(VERSION 3.15.7 FATAL_ERROR)
project(OpticalFlow LANGUAGES CXX CUDA VERSION 0.1)

enable_language(CUDA)

# Must use GNUInstallDirs to install libraries into correct
# locations on all platforms.
include(GNUInstallDirs)


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(opticalFlow-core)
add_subdirectory(opticalFlow-core-naive)
add_subdirectory(opticalFlow-unittests)
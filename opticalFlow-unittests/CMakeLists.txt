enable_testing()

add_subdirectory(extern)

find_package(CUDAToolkit)

set(OPTFLOWTESTSSOURCE "")

add_subdirectory(source)

message(STATUS "TEST UNIT SOURCE")
message(STATUS ${OPTFLOWTESTSSOURCE}) 

add_executable(opticalFlow-unittests
                ${OPTFLOWTESTSSOURCE}
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

target_link_libraries(opticalFlow-unittests PUBLIC
    opticalFlow-core opticalFlow-core-naive gtest_main gmock_main "-static" 
) 

target_link_libraries(opticalFlow-unittests PRIVATE CUDA::toolkit)

#set_property(TARGET opticalFlow-unittests PROPERTY
#  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

target_include_directories( opticalFlow-unittests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../opticalFlow-core/source)
target_include_directories(opticalFlow-unittests PRIVATE source)
add_subdirectory(extern)

set(SOURCE)

add_subdirectory(source)

add_executable(opticalFlow-unittests
                ${SOURCE}
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

target_link_libraries(opticalFlow-unittests PUBLIC
    opticalFlow-core opticalFlow-core-naive gtest_main "-static"
) 

#set_property(TARGET opticalFlow-unittests PROPERTY
#  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

target_include_directories( opticalFlow-unittests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_include_directories(opticalFlow-unittests PRIVATE source)
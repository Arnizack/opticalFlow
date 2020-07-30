set(SOURCE)

add_subdirectory(source)


find_package(CUDAToolkit)

add_library(opticalFlow-core 
    ${SOURCE})

#$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/opticalFlow-core/opticalFlow>
target_include_directories(opticalFlow-core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source>
    $<INSTALL_INTERFACE:opticalFlow>
    PRIVATE source
)

target_link_libraries(opticalFlow-core PRIVATE CUDA::toolkit)

#exporting the lib
install(TARGETS opticalFlow-core  EXPORT opticalFlow-core-config
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT opticalFlow-core-config DESTINATION share/opticalFlow-core/cmake)

export(TARGETS opticalFlow-core FILE opticalFlow-core-config.cmake)
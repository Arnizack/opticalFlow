set(SOURCE)

add_subdirectory(source)

message(${SOURCE})

add_library(opticalFlow-core-naive
    ${SOURCE})
    
target_include_directories(opticalFlow-core-naive PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/>
    $<INSTALL_INTERFACE:opticalFlow>
    PRIVATE source
)



#exporting the lib
install(TARGETS opticalFlow-core-naive  EXPORT opticalFlow-core-naive-config
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT opticalFlow-core-naive-config DESTINATION share/opticalFlow-core-naive/cmake)

export(TARGETS opticalFlow-core-naive FILE opticalFlow-core-naive-config.cmake)
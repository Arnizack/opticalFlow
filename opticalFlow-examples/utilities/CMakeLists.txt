
set(SOURCE)

add_subdirectory(optFlowExampleUtilities)

add_library(opticalFlow-examples-utilities ${SOURCE})

set_target_properties(opticalFlow-examples-utilities PROPERTIES FOLDER "Examples")

#find_package(Modules CONFIG REQUIRED)
find_package(OpenImageIO CONFIG REQUIRED)

if(TARGET OpenImageIO::OpenImageIO)
	message("OpenImageIO was found")

else()
	message("OpenImageIO was not found")
endif()
message(${CMAKE_PREFIX_PATH})

find_package(Boost REQUIRED COMPONENTS system filesystem thread)

find_package(JPEG REQUIRED)



target_include_directories(opticalFlow-examples-utilities PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/optFlowExampleUtilities>
    $<INSTALL_INTERFACE:optFlowExampleUtilities>
    PRIVATE optFlowExampleUtilities
)


target_link_libraries(opticalFlow-examples-utilities PUBLIC 
opticalFlow-core 
PRIVATE JPEG::JPEG Boost::system Boost::filesystem Boost::thread OpenImageIO::OpenImageIO OpenImageIO::OpenImageIO_Util ) 


#exporting the lib
install(TARGETS opticalFlow-examples-utilities  EXPORT opticalFlow-examples-utilities-config
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT opticalFlow-examples-utilities-config DESTINATION share/opticalFlow-examples-utilities/cmake)

export(TARGETS opticalFlow-examples-utilities FILE opticalFlow-examples-utilities-config.cmake)
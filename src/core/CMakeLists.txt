#------------------------------------------------------------------------------------------
#******************************************************************************************
#------------------------------------------------------------------------------------------
#Settings:
set(LIBNAME core)

#cpp Files for the Library
set(LIB_SOURCE 
	src/FlowField.cpp
	src/ImageRGB.cpp
	)

#In this DIRECTORY are the Headers, which can be accessed from outside Code
set(LIB_PUBLIC_INCLUDE_DIRECTORY include)

#In this DIRECTORY are the Headers, which can only be accessed from Code inside the Library
set(LIB_PRIVATE_INCLUDE_DIRECTORY src)

#ccp Files for the Unit TESTS

set(TEST_SOURCE
	tests/ImageRGBTest.cpp)

set(LIB_DEPENDENCIES logger_Lib loggerHelper_Lib)

#Ende Settings
#------------------------------------------------------------------------------------------
#******************************************************************************************
#------------------------------------------------------------------------------------------

project(${LIBNAME} LANGUAGES CXX)

set(LIB_TARGET ${LIBNAME}_Lib)

set(LIBCONFIG ${LIBNAME}Config)

set(TEST_TARGET ${LIBNAME}TESTS)

#Creating the Library
add_library(${LIB_TARGET} 
	${LIB_SOURCE})

#setting the include dirs
target_include_directories(${LIB_TARGET} PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${LIB_PUBLIC_INCLUDE_DIRECTORY}>
$<INSTALL_INTERFACE:${LIB_PUBLIC_INCLUDE_DIRECTORY}>
PRIVATE ${LIB_PRIVATE_INCLUDE_DIRECTORY}
)

#find_package(Modules CONFIG REQUIRED)
find_package(OpenImageIO CONFIG REQUIRED)

if(TARGET OpenImageIO::OpenImageIO)
	message("OpenImageIO was found")

else()
	message("OpenImageIO was not found")
endif()
message(${CMAKE_PREFIX_PATH})

find_package(Boost REQUIRED COMPONENTS system filesystem thread)

target_link_libraries(${LIB_TARGET} PUBLIC 
${LIB_DEPENDENCIES} PRIVATE Boost::system Boost::filesystem Boost::thread OpenImageIO::OpenImageIO OpenImageIO::OpenImageIO_Util ) 



#exporting the lib
install(TARGETS ${LIB_TARGET} EXPORT ${LIBCONFIG}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ${LIBCONFIG} DESTINATION share/${LIBNAME}/cmake)

export(TARGETS ${LIB_TARGET} FILE ${LIBCONFIG}.cmake)




#Unit-Tests




add_executable(${TEST_TARGET} 
				${TEST_SOURCE})

target_compile_definitions(${TEST_TARGET} PRIVATE __TESTDATADIR__="${CMAKE_CURRENT_SOURCE_DIR}/tests/data" )
target_link_libraries(${TEST_TARGET} PUBLIC
${TESTFRAMEWORK} ${LIB_TARGET} ) 

target_include_directories( ${TEST_TARGET} PRIVATE $<TARGET_PROPERTY:${LIB_TARGET},INCLUDE_DIRECTORIES>)
target_include_directories(${TEST_TARGET} PRIVATE tests)

add_test(
	NAME ${TEST_TARGET} 
	COMMAND ${TEST_TARGET}
)

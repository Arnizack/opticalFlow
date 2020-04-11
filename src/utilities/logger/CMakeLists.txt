#------------------------------------------------------------------------------------------
#******************************************************************************************
#------------------------------------------------------------------------------------------
#Settings:
set(LIBNAME logger)

#cpp Files for the Library
set(LIB_SOURCE 
	src/logger.cpp
	src/ConsoleLogger.cpp
	src/FileLogger.cpp
	)

#In this DIRECTORY are the Headers, which can be accessed from outside Code
set(LIB_PUBLIC_INCLUDE_DIRECTORY include)

#In this DIRECTORY are the Headers, which can only be accessed from Code inside the Library
set(LIB_PRIVATE_INCLUDE_DIRECTORY src)

#ccp Files for the Unit TESTS

set(TEST_SOURCE
	tests/loggerConsoleLogger.cpp
	tests/loggerFileLogger.cpp)

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

target_compile_features(${LIB_TARGET} PUBLIC cxx_variadic_templates )

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


target_link_libraries(${TEST_TARGET} PUBLIC
${TESTFRAMEWORK} ${LIB_TARGET} ) 

target_include_directories( ${TEST_TARGET} PRIVATE $<TARGET_PROPERTY:${LIB_TARGET},INCLUDE_DIRECTORIES>)

add_test(
	NAME ${TEST_TARGET} 
	COMMAND ${TEST_TARGET}
)

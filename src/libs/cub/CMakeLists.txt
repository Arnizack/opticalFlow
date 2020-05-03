
#------------------------------------------------------------------------------------------
#******************************************************************************************
#------------------------------------------------------------------------------------------
#Settings:
set(LIBNAME cub)


#In this DIRECTORY are the Headers, which can be accessed from outside Code
set(LIB_PUBLIC_INCLUDE_DIRECTORY cub/)

#ccp Files for the Unit TESTS

#Ende Settings
#------------------------------------------------------------------------------------------
#******************************************************************************************
#------------------------------------------------------------------------------------------



project(${LIBNAME} LANGUAGES CUDA CXX )

set(LIB_TARGET ${LIBNAME}_Lib)

set(LIBCONFIG ${LIBNAME}Config)

set(TEST_TARGET ${LIBNAME}TESTS)

#Creating the Library
add_library(${LIB_TARGET} 
	INTERFACE)

target_compile_features(${LIB_TARGET} INTERFACE cxx_std_17)
#set_target_properties( ${LIB_TARGET}
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


#setting the include dirs
target_include_directories(${LIB_TARGET}  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/${LIB_PUBLIC_INCLUDE_DIRECTORY}>
    $<INSTALL_INTERFACE:${LIB_PUBLIC_INCLUDE_DIRECTORY}>
)
#exporting the lib
install(TARGETS ${LIB_TARGET} EXPORT ${LIBCONFIG}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ${LIBCONFIG} DESTINATION share/${LIBNAME}/cmake)

export(TARGETS ${LIB_TARGET} FILE ${LIBCONFIG}.cmake)
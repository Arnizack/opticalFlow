if(MSVC)
    message("Use x64 binaries for openblas")
    add_library(openblas STATIC IMPORTED GLOBAL)
    message("openblas imported_location: ${CMAKE_CURRENT_SOURCE_DIR}/bin/openblas.lib")
    set_target_properties(openblas PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/bin/openblas.lib")
    #set_target_properties(openblas PROPERTIES IMPORTED_IMPLIB "bin/openblas.lib"
    #set_target_properties(openblas PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include")
    target_include_directories(openblas INTERFACE "include")
else()
    find_package(OpenBLAS CONFIG REQUIRED)
    add_library(openblas INTERFACE)
    target_link_libraries(openblas INTERFACE OpenBLAS::OpenBLAS)
endif()


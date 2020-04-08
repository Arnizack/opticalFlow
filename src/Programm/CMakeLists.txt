add_executable(app 
printHalloWorld.cpp)


if(TARGET printerLib_Lib)
	message("printerLib_Lib wurde gefunden")
else()
	message("printerLib_Lib wurde nicht gefunden")
endif()

target_link_libraries(app printerLib_Lib)
add_subdirectory(googletest)

set(TESTFRAMEWORK gtest_main PARENT_SCOPE)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

if(TARGET gtest_main)
	message("gtest_main cloud be located")
else()
	message("gtest_main cloud be located")
endif()

add_subdirectory(cub)

if(TARGET cub_Lib)
	message("cub_Lib cloud be located")
else()
	message("cub_Lib cloud be located")
endif()

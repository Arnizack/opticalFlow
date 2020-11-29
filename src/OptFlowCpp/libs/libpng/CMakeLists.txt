find_package(PNG REQUIRED)

add_library(libpng INTERFACE)
target_link_libraries(libpng INTERFACE PNG::PNG)

if (TARGET libpng)
    message("libpng found")
else()
    message("libpng not found")
endif()
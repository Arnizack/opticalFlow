
file(
     GLOB_RECURSE source
     "${CMAKE_CURRENT_SOURCE_DIR}"
     *.cpp *.hpp *.h *.c
)

add_executable(ImagePyramide ${source})

set_target_properties(ImagePyramide PROPERTIES FOLDER "Examples")

target_link_libraries(ImagePyramide PUBLIC 
${LIB_DEPENDENCIES} PRIVATE JPEG::JPEG Boost::system Boost::filesystem Boost::thread OpenImageIO::OpenImageIO OpenImageIO::OpenImageIO_Util ) 



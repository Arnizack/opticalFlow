file(
     GLOB_RECURSE source_list
     "${CMAKE_CURRENT_SOURCE_DIR}"
     *.cpp *.hpp *.h *.c *.cu *.cuh
)

set(OPTFLOWTESTSSOURCE ${source_list} PARENT_SCOPE)




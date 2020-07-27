find_package(CUDAToolkit)

add_executable(example-cuda-data-structure "main.cu")

target_link_libraries(opticalFlow-core PRIVATE CUDA::toolkit)

target_link_libraries(example-cuda-data-structure opticalFlow-examples-utilities)

set_target_properties(example-cuda-data-structure PROPERTIES FOLDER "Examples/Cuda")
// errorChecking.cuh
#ifndef CHECK_CUDA_ERROR_H
#define CHECK_CUDA_ERROR_H

// This could be set with a compile time flag ex. DEBUG or _DEBUG
// But then would need to use #if / #ifdef not if / else if in code
#define FORCE_SYNC_GPU 0
#define PRINT_ON_SUCCESS 1

cudaError_t checkAndPrint(const char* name, int sync = 0);
cudaError_t checkCUDAError(const char* name, int sync = 0);

#endif // CHECK_CUDA_ERROR_H
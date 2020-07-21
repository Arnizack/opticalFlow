// errorChecking.cu
#include "errorChecking.cuh"
#include <stdio.h>

cudaError_t checkAndPrint(const char* name, int sync) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        const char* errorMessage = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error check \"%s\" returned ERROR code: %d (%s) %s \n", name, err, errorMessage, (sync) ? "after sync" : "");
    }
    else if (PRINT_ON_SUCCESS) {
        printf("CUDA error check \"%s\" executed successfully %s\n", name, (sync) ? "after sync" : "");
    }
    return err;
}

cudaError_t checkCUDAError(const char* name, int sync) {
    cudaError_t err = cudaSuccess;
    if (sync || FORCE_SYNC_GPU) {
        err = checkAndPrint(name, 0);
        cudaDeviceSynchronize();
        err = checkAndPrint(name, 1);
    }
    else {
        err = checkAndPrint(name, 0);
    }
    return err;
}
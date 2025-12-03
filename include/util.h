/**
 * Author: Prachit Amin
 * ECE 4122
 * 12/1/2025
 * Defines helper functions/macros ensuring safe CUDA setup.
 */

#ifndef UTIL_H
#define UTIL_H

#define IMAGE_DIM 224

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(ans)                      \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

/**
 * @brief Helper function for CHECK_ERROR macro to handle CUDA errors
 * @param code CUDA error code to check
 * @param file Source file where the error occurred
 * @param line Line number where the error occurred
 * @param abort Whether to abort execution on error (default: true)
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/**
 * @brief Helper function to compute output dimensions of a convolutional layer
 * @param inputDim Input dimension (height or width)
 * @param stride Convolution stride
 * @param pad Padding size
 * @param kernelDim Kernel dimension (height or width)
 * @return Output dimension after convolution
 */
inline int computeDim(int inputDim, int stride, int pad, int kernelDim)
{
    return (inputDim + 2 * pad - kernelDim) / stride + 1;
}
#endif
#include "ResNetDev.h"
#include "util.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define EPSILON 1e-5

__global__ void conv2d_kernel(const float *input,    // [in_channels, H, W]
                              const float *weight,   // [out_channels, in_channels, kH, kW]
                              const float *bnWeight, //
                              const float *bnBias,   //
                              const float *bnMean,   //
                              const float *bnVar,    //
                              float *output,         // [out_channels, H_out, W_out]
                              int in_channels, int out_channels, // channels
                              int H, int W, int outH, int outW,  // dims
                              int kernel_size, int stride, int padding);

void launchConvKernel(float *image, float *output, const ConvLayerDev &conv, const BatchNormDev &bn,
                      int inputDim, int stride, int pad);
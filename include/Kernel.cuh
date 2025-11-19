#ifndef KERNEL_CUH
#define KERNEL_CUH

// #include "CopyModel.h"
#include "ResNetDev.h"
#include "util.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

__global__ void downsample_kernel(const float *input,     // [in_ch, H, W]
                                  const float *weight,    // [out_ch, in_ch, 1, 1]
                                  const float *bn_weight, //
                                  const float *bn_bias,   //
                                  const float *bn_mean,   //
                                  const float *bn_var,    //
                                  float *output,          // [out_ch, H/2, W/2]
                                  int in_ch, int out_ch, int H, int W, float epsilon);

__global__ void maxpool_kernel(const float *input, // [C, H, W]
                               float *output,      // [C, H_out, W_out]
                               int C, int H, int W, int kernel_size, int stride, int padding);

__global__ void adaptive_avgpool_kernel(const float *input, // [C, H, W]
                                        float *output,      // [C, 1, 1]
                                        int C, int H, int W);

__global__ void fc_kernel(const float *input,  // [in_features]
                          const float *weight, // [out_features, in_features]
                          const float *bias,   // [out_features]
                          float *output,       // [out_features]
                          int in_features, int out_features);

void launchConvKernel(float *image, float *output, const ConvLayerDev &conv, const BatchNormDev &bn,
                      int inputDim, int stride, int pad);
void launchDownsampleKernel(float *input, float *output, const DownsampleDev &ds, int H, int W);

void runBasicBlock(const BasicBlockDev &bb, float *input, float *output, int inputChannels,
                   int inputH, int inputW, int stride1);

void launchMaxPoolKernel(float *input, float *output, int H, int W, int C, int kernel_size,
                         int stride, int padding);

void launchAdaptiveAvgPoolKernel(float *input, float *output, int H, int W, int C);

void launchFCKernel(float *input, float *output, const FullyConnectedDev &fc, int in_features,
                    int out_features);

void launchReLUKernel(float *input, float *output, int size);

#endif
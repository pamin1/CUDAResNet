/**
 * Author: Prachit Amin
 * ECE 4122
 * 12/1/2025
 * Declares the CUDA kernels and launch functions used by the ResNet model.
 */

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "ResNetDev.h"
#include "util.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EPSILON 1e-5

/**
 * @brief CUDA Convolution implementation
 * @param input Input tensor
 * @param weight Convolution weights
 * @param bnWeight Batch normalization weight
 * @param bnBias Batch normalization bias
 * @param bnMean Batch normalization mean
 * @param bnVar Batch normalization variance
 * @param output Output tensor
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param H Input height
 * @param W Input width
 * @param outH Output height
 * @param outW Output width
 * @param kernel_size Convolution kernel size
 * @param stride Convolution stride
 * @param padding Convolution padding
 */
__global__ void conv2d_kernel(const float *input, const float *weight, const float *bnWeight,
                              const float *bnBias, const float *bnMean, const float *bnVar,
                              float *output, int in_channels, int out_channels, int H, int W,
                              int outH, int outW, int kernel_size, int stride, int padding);

/**
 * @brief CUDA Downsample kernel for reducing spatial dimensions
 * @param input Input tensor
 * @param weight Convolution weights
 * @param bn_weight Batch normalization weight
 * @param bn_bias Batch normalization bias
 * @param bn_mean Batch normalization mean
 * @param bn_var Batch normalization variance
 * @param output Output tensor
 * @param in_ch Number of input channels
 * @param out_ch Number of output channels
 * @param H Input height
 * @param W Input width
 * @param epsilon Epsilon value for batch normalization
 */
__global__ void downsample_kernel(const float *input, const float *weight, const float *bn_weight,
                                  const float *bn_bias, const float *bn_mean, const float *bn_var,
                                  float *output, int in_ch, int out_ch, int H, int W,
                                  float epsilon);

/**
 * @brief CUDA ReLU activation kernel
 * @param input Input tensor
 * @param output Output tensor
 * @param size Size of the tensor
 */
__global__ void relu_kernel(const float *input, float *output, int size);

/**
 * @brief CUDA element wise addition kernel
 * @param a First input tensor
 * @param b Second input tensor
 * @param output Output tensor
 * @param size Size of the tensors
 */
__global__ void add_kernel(const float *a, const float *b, float *output, int size);

/**
 * @brief CUDA Max Pooling kernel
 * @param input Input tensor
 * @param output Output tensor
 * @param C Number of channels
 * @param H Input height
 * @param W Input width
 * @param kernel_size Pooling kernel size
 * @param stride Pooling stride
 * @param padding Pooling padding
 */
__global__ void maxpool_kernel(const float *input, float *output, int C, int H, int W,
                               int kernel_size, int stride, int padding);

/**
 * @brief CUDA Adaptive Average Pooling kernel
 * @param input Input tensor
 * @param output Output tensor
 * @param C Number of channels
 * @param H Input height
 * @param W Input width
 */
__global__ void adaptive_avgpool_kernel(const float *input, float *output, int C, int H, int W);

/**
 * @brief CUDA Fully Connected layer kernel
 * @param input Input tensor
 * @param weight Weight matrix
 * @param bias Bias vector
 * @param output Output tensor
 * @param in_features Number of input features
 * @param out_features Number of output features
 */
__global__ void fc_kernel(const float *input, const float *weight, const float *bias,
                          float *output, int in_features, int out_features);

/**
 * @brief Launch convolution kernel
 * @param image Input image data
 * @param output Output data
 * @param conv Convolution layer parameters
 * @param bn Batch normalization parameters
 * @param inputDim Input dimension
 * @param stride Convolution stride
 * @param pad Convolution padding
 */
void launchConvKernel(float *image, float *output, const ConvLayer &conv, const BatchNorm &bn,
                      int inputDim, int stride, int pad);

/**
 * @brief Launch downsample kernel
 * @param input Input data
 * @param output Output data
 * @param ds Downsample layer parameters
 * @param H Input height
 * @param W Input width
 */
void launchDownsampleKernel(float *input, float *output, const Downsample &ds, int H, int W);

/**
 * @brief Run a complete basic block of ResNet
 * @param bb Basic block parameters
 * @param input Input data
 * @param output Output data
 * @param inputChannels Number of input channels
 * @param inputH Input height
 * @param inputW Input width
 * @param stride1 Stride for first convolution
 */
void runBasicBlock(const BasicBlock &bb, float *input, float *output, int inputChannels,
                   int inputH, int inputW, int stride1);

/**
 * @brief Launch max pooling kernel
 * @param input Input data
 * @param output Output data
 * @param H Input height
 * @param W Input width
 * @param C Number of channels
 * @param kernel_size Pooling kernel size
 * @param stride Pooling stride
 * @param padding Pooling padding
 */
void launchMaxPoolKernel(float *input, float *output, int H, int W, int C, int kernel_size,
                         int stride, int padding);

/**
 * @brief Launch adaptive average pooling kernel
 * @param input Input data
 * @param output Output data
 * @param H Input height
 * @param W Input width
 * @param C Number of channels
 */
void launchAdaptiveAvgPoolKernel(float *input, float *output, int H, int W, int C);

/**
 * @brief Launch fully connected layer kernel
 * @param input Input data
 * @param output Output data
 * @param fc Fully connected layer parameters
 * @param in_features Number of input features
 * @param out_features Number of output features
 */
void launchFCKernel(float *input, float *output, const FullyConnected &fc, int in_features,
                    int out_features);

/**
 * @brief Launch element-wise addition kernel
 * @param a First input data
 * @param b Second input data
 * @param output Output data
 * @param size Size of the data arrays
 */
void launchAddKernel(float *a, float *b, float *output, int size);

/**
 * @brief Launch ReLU activation kernel
 * @param input Input data
 * @param output Output data
 * @param size Size of the data array
 */
void launchReLUKernel(float *input, float *output, int size);

#endif
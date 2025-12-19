# CUDA ResNet Implementation

[CUDAResNet](https://github.com/pamin1/CUDAResNet)

## Overview

This past semester, I set out to implement a ResNet-18 Architecture using custom CUDA Kernels. This handled the full implementation stack, including image preprocessing, model parsing, device memory allocation, and the model implementation using CUDA kernels and launch functions.

## Image Preprocessing

Image preprocessing is a critical step because the ResNet-18 architecture expects normalized input of a specific size and format. The preprocessing pipeline involves several transformations:

First, I converted the color space from BGR (OpenCV's default) to RGB, resized the image, and applied center cropping to produce a 224×224 image.

Next came data type conversion. Image data is typically stored as `uint8` (unsigned char) values ranging from [0, 255]. I converted this to FP32 (32-bit floating point) and scaled the values to [0, 1] to match the model's expected input format.

Finally, I normalized each RGB channel using ImageNet statistics and stored the result in a host array with CHW (Channel-Height-Width) layout:
```cpp
// ImageNet normalization statistics
const float mean[3] = {0.485f, 0.456f, 0.406f};
const float stdv[3] = {0.229f, 0.224f, 0.225f};

for (int h = 0; h < 224; ++h)
{
    auto row = cropped.ptr<cv::Vec3f>(h);
    for (int w = 0; w < 224; ++w)
    {
        size_t base = h * 224 + w;
        host[0 * 224 * 224 + base] = (row[w][0] - mean[0]) / stdv[0]; // R
        host[1 * 224 * 224 + base] = (row[w][1] - mean[1]) / stdv[1]; // G
        host[2 * 224 * 224 + base] = (row[w][2] - mean[2]) / stdv[2]; // B
    }
}
```

The CHW layout (all red pixels, then all green, then all blue) enables coalesced memory accesses in the GPU convolution kernels later on, where threads processing the same channel can read contiguous memory locations.

## Model Parsing and Device Allocation

Using PyTorch, I exported the model as two files: a JSON manifest defining the layer-by-layer architecture of ResNet-18, and an NPZ file containing the trained weights. To parse these into a cohesive layer structure, I used nlohmann/json and cnpy libraries to read their respective formats, allocate host arrays with the appropriate sizes, populate them with weights, and copy everything to the GPU in a single initialization function.

This straightforward approach works well because ResNet-18 is relatively compact (~11M parameters, ~44MB of weights), fitting comfortably into GPU global memory. This means I can load all weights once at startup and keep them resident throughout inference, avoiding the need to stream layers on and off the device.

## Model Implementation

The model implementation consolidates into a single header file that exposes one function to execute the entire ResNet-18 inference pipeline. I created wrapper functions for each kernel operation (convolution, batch normalization, ReLU, pooling), allowing the main execution function to compose these operations with architecture-specific parameters like stride and padding values.

ResNet-18's structure consists of five stages, where each stage after the first contains basic residual blocks. Each basic block performs two 3×3 convolutions with batch normalization, adds the residual connection, and applies ReLU activation. To reduce kernel launch overhead, I fused these operations into a single `basicBlock` kernel that handles the complete residual computation in one pass.

## Results

I created an image classifier using the ResNet-18 architecture with pre-trained weights, validating correctness by benchmarking against PyTorch's reference model.

I also benchmarked inference times against both PyTorch's GPU (CUDA) and CPU implementations. My initial implementation was 8–9× slower than PyTorch CUDA, but through iterative optimizations, I improved performance to within 3.38× of their highly optimized implementation.

| Implementation                 | Latency (ms/img) | Throughput (img/s) | vs PyTorch CUDA |  vs Direct Conv  |
| :----------------------------- | :--------------: | :----------------: | :-------------: | :--------------: |
| **PyTorch CUDA (cuDNN)**       |      4.281       |       233.58       |    Baseline     |   5.46x faster   |
| **Custom CUDA - Tensor Cores** |      14.497      |       68.97        |  3.38x slower   | **2.39x faster** |
| PyTorch CPU                    |      18.550      |       53.92        |  4.33x slower   |   1.87x faster   |
| Custom CUDA - Shared Memory    |      34.744      |       28.78        |  8.11x slower   |     Baseline     |
| Custom CUDA - Naive            |      39.208      |       25.51        |  9.16x slower   |   1.13x slower   |

*Benchmarked with 1000 samples after 50 warmup iterations*

The journey from 9× to 3.38× slower involved several major iterations:

- **Shared Memory Tiling**: Cached input data to reduce global memory accesses
- **Tensor Core Acceleration**: Switched from direct convolution to implicit GEMM, unlocking specialized matrix multiply hardware
- **Kernel Fusion**: Combined convolution, batch normalization, and ReLU operations to eliminate intermediate memory writes

The most significant breakthrough came from adopting the implicit GEMM algorithm, which restructures 2D convolution as matrix multiplication—enabling tensor core utilization for a 2.39× speedup over my best direct convolution approach.
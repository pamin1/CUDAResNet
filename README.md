# CUDA ResNet
A from-scratch implementation of ResNet-18 inference using custom CUDA kernels, exploring GPU optimization techniques from naive direct convolution to tensor core-accelerated implicit GEMM.

## Project Overview

This project implements ResNet-18 image classification entirely in CUDA C++, progressing through multiple optimization stages to understand GPU performance characteristics. The implementation handles the complete inference pipeline from image preprocessing to final classification.

**Key Achievement**: Achieved 30% of PyTorch's highly-optimized CUDA performance using custom kernels, with a 2.39x speedup through tensor core utilization over direct convolution approaches.

## Usage
Clone:
```
git clone https://github.com/pamin1/CUDAResNet.git
cd CUDAResNet
git submodule update --init --recursive
```

Build:
```
mkdir build && cd build
cmake ..
make
```

Run:
```
./resnet
```

## Performance Benchmarks
### Accuracy
Using the same weights across all test groups so the inferences are deterministic, thus accuracy testing will be redundant.

### Performance
| Implementation                 | Latency (ms/img) | Throughput (img/s) | vs PyTorch CUDA |  vs Direct Conv  |
| :----------------------------- | :--------------: | :----------------: | :-------------: | :--------------: |
| **PyTorch CUDA (cuDNN)**       |      4.281       |       233.58       |    Baseline     |   5.46x faster   |
| **Custom CUDA - Tensor Cores** |      14.497      |       68.97        |  3.38x slower   | **2.39x faster** |
| PyTorch CPU                    |      18.550      |       53.92        |  4.33x slower   |   1.87x faster   |
| Custom CUDA - Shared Memory    |      34.744      |       28.78        |  8.11x slower   |     Baseline     |
| Custom CUDA - Naive            |      39.208      |       25.51        |  9.16x slower   |   1.13x slower   |

*Benchmarked with 1000 samples after 50 warmup iterations*

## Implementation Highlights

- **Tensor Core Acceleration**: Implicit GEMM algorithm using WMMA API for FP16 matrix operations
- **Full Inference Stack**: Custom image preprocessing, model parsing (JSON + NPZ), and classification
- **Profiler-Driven Development**: Extensive use of NSight Compute for bottleneck analysis

## Documentation

Detailed documentation is available in the docs and also Medium articles:

- **[Implementation Details](docs/implementation-details.md) ([Medium Post](https://medium.com/@prachitamin12/cuda-resnet-18-implementation-2f32240d61ee))**: Architecture, data flow, and kernel design
- More documentation and articles to come

## Key Learnings

1. **Tensor cores provide massive speedup** - Over 2x improvement by restructuring convolution as matrix multiplication
2. **Kernel profiling is revealing** - Profiling can explain abstract issues and bottlenecks in real kernels, as well as help compare various implementations
3. **Algorithmic choice matters** - Switching from direct convolution to implicit GEMM unlocked hardware capabilities

## Future Improvements

- Experimenting with tile sizes (32×32, 64×64) for better occupancy and compute/memory usage
- Improved memory coalescing and prefetching strategies
- Profile PyTorch implementation to see runtime differences

## Technologies

- CUDA C++ with WMMA API for tensor cores
- OpenCV for image preprocessing
- nlohmann/json for model architecture parsing
- cnpy for NumPy .npz weight loading
- NSight Compute for performance profiling


## Acknowledgments

Inspired by PyTorch's cuDNN implementation and the original ResNet paper (He et al., 2015).
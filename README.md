# CUDA ResNet

## Objective
Explore and implement the ResNet architecture using CUDA acceleration. 

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

| Implementation             | Latency (ms/img) | Throughput (img/s) | vs PyTorch CUDA |
| :------------------------- | :--------------: | :----------------: | :-------------: |
| **Baseline: PyTorch CUDA** |      4.281       |       233.58       |    Baseline     |
| Custom CUDA - v1.3         |      16.061      |       60.26        |  3.75x slower   |
| **Baseline: PyTorch CPU**  |      18.550      |       53.92        |  4.33x slower   |
| Custom CUDA - v1.1         |      34.744      |       28.78        |  8.11x slower   |
| Custom CUDA - v1.2         |      37.628      |       26.56        |  8.79x slower   |
| Custom CUDA - v1.0         |      39.208      |       25.51        |  9.16x slower   |

**v1.3 (Shared Memory and Tensor Cores)**: I took a look at the approach cuDNN made and realized I was handicapped by my utilization of the GPU hardware. CUDA libraries heavily optimize for the hardware, primarily using the tensor cores which perform matrix multiplications and additions with significantly higher throughput than CUDA core implementations. This requires changing the shared memory structure and much of the original implementation to fit the Implicit GEMM approach, but its very rewarding with a 116% speed up over my best direct convolution implementation. Additionally, it beat out the PyTorch CPU optimized implementation by 2.5ms and closed the gap to within 3.75x of PyTorch's optimized CUDA implementation.

**v1.2 (Shared Memory Image and Weights)**: Shared Memory Image and Weights is actually slower, so my hypothesis from v1.1 was wrong. I used Nsight Compute (ncu) to try and debug this further because the memory bandwidth speed up should have easily given more than a 5ms speed up. It turns out that I was facing a combination of Memory and Compute bound issues. I also learned that I was memory latency bound, instead of bandwidth bound, which has an important distinction. Read more on the profiling process using Nsight Compute [here](#using-nsight-compute)

**v1.1 (Shared Memory Image)**: Shared Memory Image stores only the image in shared memory. There is mild reuse of the image data so it makes sense there is not as extreme of a speed up. The real speed likely remains to be seen through storing the kernel weights.
Better optimizations will be seen by improving the data reuse pattern.

**v1.0 (Naive)**: Naive implementation consistently ~9x slower than CUDA accelerated PyTorch, only 2x slower than PyTorch CPU. Removed variable sample size; implemented warmup of 50 inferences before beginning timing and commited to 1000 samples for testing.

### Using Nsight Compute
After implementing shared memory optimization for input image and weight tiling, performance didn't improve as expected. Surprisingly, smaller (8x8) block sizes achieved better runtime performance than larger (16x16, 32x32, 64x64) blocks, contradicting typical GPU optimization wisdom where larger blocks usually perform better. At this point I suspected profiling the kernel with Nsight Compute would give more intuition to what was slowing the kernel down.

#### Understanding the Bottleneck: Memory-Latency Bound
Nsight Compute revealed my kernel was **Memory-Latency Bound**, which differs from the more commonly discussed compute-bound and memory-bandwidth-bound cases:

**Compute Bound**: The GPU's arithmetic units are saturated, so there's sufficient data being supplied, but the computational throughput limits performance. 
- Optimization focus: algorithmic improvements, reducing instruction count, or using more efficient math operations.

**Memory-Bandwidth Bound**: Memory throughput is the bottleneck. The compute units sit idle waiting for data because the volume of data transfer exceeds the memory bandwidth capacity. 
- Optimization focus: reducing total bytes transferred, better caching, or data reuse.

**Memory-Latency Bound**: While memory bandwidth utilization may be reasonable, performance is limited by the *time delay* to access memory. This occurs when there aren't enough active warps to hide memory access latencies through instruction interleaving. 
- Optimization focus: increasing occupancy, improving memory access patterns to reduce latency, or restructuring to allow more instruction-level parallelism.

#### Key Insights
This profiling exercise revealed that **memory access pattern correctness trumps occupancy optimization**. A kernel with perfect coalescing and 60% occupancy *might* outperform one with uncoalesced accesses and 99% occupancy, because:
1. Memory traffic increases with uncoalesced memory accesses, saturating memory bandwidth.
3. More requests increase average latency
4. Even high occupancy can't hide long memory stalls

### Next Steps
At this point, I believe I have exhausted my resources with the direct convolution approach and am ready to explore more sophisticated algorithmic implementations to achieve real speedups. 

After researching cuDNN's implementation for forward convolutions, I discovered the fundamental algorithmic difference: while I implemented a basic Direct Convolution algorithm with a sliding window accumulation approach, cuDNN uses the Implicit GEMM approach. This restructuring is critical because it contrstucts the convolution operation to match computations that modern GPU hardware is designed to accelerate. The implicit GEMM approach reorganizes convolution into matrix multiply patterns, which unlocks tensor core utility for convolution.

## Why ResNet?
### Vanishing Gradient Problem
As the number of layers in a model increase, the performance of the models decreases:
* During training, the model identifies errors and adjusts the weights using back-propagation.
* As the model move backwards through the model, the model compute the gradients of the lost w.r.t to each weight and update them accordingly (Gradient Descent Algorithm)
* As more layers are attached to the model, the result of gradient descent through back-propagation becomes negligible. 
* Since the early layers are foundational to overall feature detection, if those layers are poorly trained, the model will have worse performance due to early errors.

### How does ResNet address this? 
By adding Residual/Skip connections (layer bypassing). Generally, the $(n-1)^{th}$ layer bypasses the $n^{th}$ layer, and adds to the output of the $(n+1)^{th}$ . 

*Why is this important?* There are two cases: the $n^{th}$ layer provides information or it doesn't. In either case, the model can use the input to the $n^{th}$, because it should provide useful information. This way whether the $n^{th}$ layer is useful or not, the model should get some more information from the previous layer, allowing in consistent/improved performance, as the number of layers increases. 

TL,DR: ResNet will maintain or improve model performance by allowing more layers to provide information deeper in the network, improving upon typical CNN performance.

## Implementation Organization
The implementation handles the entire inference stack, from image preprocessing to top-k results.

### Image Preprocessing
The first layer of the ResNet18 architecture uses a 224x224x3 convolution. What this means in context is that the input data is a 224x224 array with 3 channels. In the context of an image this would mean a 224x224 input image, and split into R, G, and B channels. 

I used OpenCV to parse the input image. It converts an arbitrarily sized input image into the required 224x224 size and scales the uchar[0-255] to floats[0-1], matching the required input format for the model weights.

### Model Parsing
The model is split between two files. The JSON manifest contains the model architecture describing the shapes and sizes of each layer. The npz file contains the actual weights used for kernel operations at each layer. Each layer is copied into a host structure that contains the layers sizes as well as a float array of the weights.

5 total layers:

0. Layer 0: convolution, batch norm
1. Layer 1:
   1. 2x (convolution, batch norm)
   2. 2x (convolution, batch norm)
2. Layer 2:
   1. 2x (convolution, batch norm) + DS
   2. 2x (convolution, batch norm)
3. Layer 3:
   1. 2x (convolution, batch norm) + DS
   2. 2x (convolution, batch norm)
4. Layer 4:
   1. 2x (convolution, batch norm) + DS
   2. 2x (convolution, batch norm)
5. Fully Connected

Using nlohmann json, I am able to parse the architecture/layers sizes into the model.
I parse the NPZ file similarily to get the layer weights and allocate them to the GPU.

On the first version, I had two separate structs for holding the model data. One struct held the data on the host and the other on the device. This added a ton of bloat to the codebase and implementation.

After refactoring, the memory allocation step has been integrated into one setup function such that the actual implementation does not require copying and freeing later on. This all fits on the comfortably on the GPU since the ResNet18 architecture is relatively small (~11M params, ~44MB). Similar to initialization, freeing is also all done in one function.

### Model Implementation
The implementation runs the layers in the logical order defined by the architecture. I use hardcoded values for striding and padding.

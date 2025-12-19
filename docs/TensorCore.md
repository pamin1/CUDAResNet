# Implicit GEMM and the Tensor Core Approach
The underlying algorithm/technique that the cuDNN library uses is the **Implicit GEMM** algorithm. This essentially patterns the Direct Convolution approach as a matrix multiplication, such that the Tensor Cores can be utilized rather than the CUDA Cores, leverging the hardware built for this purpose!

## How do Tensor Cores work?
Tensor Cores do a tiled matrix multiplication and accumulate, given 2 input matrices, and a single accumulation matrix. Having dedicated hardware for Matrix Multiplication is incredibly important, because of the fact that a complex computation can now be done in a single clock cycle; regular CUDA Cores can't achieve this kind of performance on their own!

The input matrices typically use half precision (FP16) and the output accumulation is stored as FP32/FP64; a benefit of accumulation precision. This allows tighter use of memory without the lack of precision in the output. 

Tensor cores differ from CUDA cores in that the programmer is not in control of each individual thread and its output, but rather a warp, or group of 32 threads together. These 32 threads work in synchronization under the warp and tensor core to perform the MatMul computation much faster than having individual threads compute their own outputs. The programmer is left to maintain the storage, access, and use of data by the warps.

## Where does the speed up come from?
Notes from the NCU analysis of Implicit GEMM vs Direct Convolution kernels:
* 70% decrease in Shared Memory usage
* 250% increase in L1 cache hit rate
* 8% decrease in L2 cache hit rate and 66% decrease in L2 cache throughput
* BUT 171% increase in device memory throughput
* WMMA barely utilizes device peak memory throughput, everything remained well below 20% peak usage. The direct convolution approach used way more shared memory usage, near 50%, but still remained significantly slower.
* WMMA had significant decreases in compute (25%), memory (60%), and cache (66%) throughput, but had a 170% increase in DRAM throughput

At this point the kernel is seems to be so computationally efficient, that it is seems more so memory latency bound. It is likely still partially compute bound as well, but that might improve with better tuned block and grid sizing due to the fact that the kernel has a significantly reduced occupancy. Improving the kernel occupancy and the memory access patterns are probably the last optimizations to bring my kernel within ~10% of the PyTorch implementation.
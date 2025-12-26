/**
 * Author: Prachit Amin
 * ECE 4122
 * 12/1/2025
 * Implements all CUDA Kernels and launching functions.
 */

#include "Kernel.cuh"
#include <mma.h>
using namespace nvcuda;

// tile dimensions
const int TILE_M = 32; // output spatial dimension tile
const int TILE_N = 32; // output channel tile
const int TILE_K = 16; // input channel * kernel_size^2 tile

__global__ void conv2d_kernel(const float *input, const float *weight, const float *bnWeight, const float *bnBias, const float *bnMean, const float *bnVar, float *output, int in_channels, int out_channels, int H, int W, int outH, int outW, int kernel_size, int stride, int padding, bool ReLU, const float *residual)
{
    __shared__ half smem_input[TILE_M * TILE_K];
    __shared__ half smem_weight[TILE_N * TILE_K];

    // WMMA tile matrices
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    int block_row = blockIdx.y; // output spatial tile
    int block_col = blockIdx.z; // output channel tile

    // warp and lane indices
    int warpM = (threadIdx.x / 32) / (TILE_N / 16); // warp row in block
    int warpN = (threadIdx.x / 32) % (TILE_N / 16); // warp col in block
    int laneId = threadIdx.x % 32;

    // ouput position
    int out_row_base = block_row * TILE_M + warpM * 16;
    int out_col_base = block_col * TILE_N + warpN * 16;

    // convert linear output position to (out_y, out_x, out_c)
    int spatial_base = block_row * TILE_M;
    int oc_base = block_col * TILE_N;

    wmma::fill_fragment(acc_frag, 0.0f);

    int K2 = kernel_size * kernel_size;
    int total_K = in_channels * K2;
    int num_k_tiles = (total_K + TILE_K - 1) / TILE_K;

    // each thread loads multiple elements
    int num_input_elements = TILE_M * TILE_K;
    int elements_per_thread = (num_input_elements + blockDim.x - 1) / blockDim.x;

    // tile over K dimension
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++)
    {
        int k_base = k_tile * TILE_K;

        // input
        for (int i = 0; i < elements_per_thread; i++)
        {
            int idx = threadIdx.x * elements_per_thread + i;
            if (idx < num_input_elements)
            {
                int spatial_idx = idx / TILE_K;
                int k_idx = k_base + (idx % TILE_K);

                // convert spatial_idx to (out_y, out_x)
                int linear_pos = spatial_base + spatial_idx;
                int out_y = linear_pos / outW;
                int out_x = linear_pos % outW;

                if (out_y < outH && out_x < outW && k_idx < total_K)
                {
                    int ic = k_idx / K2;
                    int k_offset = k_idx % K2;
                    int ky = k_offset / kernel_size;
                    int kx = k_offset % kernel_size;

                    // compute input position
                    int in_y = out_y * stride - padding + ky;
                    int in_x = out_x * stride - padding + kx;

                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                    {
                        int in_idx = ic * H * W + in_y * W + in_x;
                        smem_input[idx] = __float2half(input[in_idx]);
                    }
                    else
                    {
                        smem_input[idx] = __float2half(0.0f);
                    }
                }
                else
                {
                    smem_input[idx] = __float2half(0.0f);
                }
            }
        }

        // weight
        int num_weight_elements = TILE_N * TILE_K;
        elements_per_thread = (num_weight_elements + blockDim.x - 1) / blockDim.x;

        for (int i = 0; i < elements_per_thread; i++)
        {
            int idx = threadIdx.x * elements_per_thread + i;
            if (idx < num_weight_elements)
            {
                int oc = oc_base + (idx / TILE_K);
                int k_idx = k_base + (idx % TILE_K);

                if (oc < out_channels && k_idx < total_K)
                {
                    int ic = k_idx / K2;
                    int k_offset = k_idx % K2;
                    int ky = k_offset / kernel_size;
                    int kx = k_offset % kernel_size;

                    int w_idx = oc * in_channels * K2 + ic * K2 + ky * kernel_size + kx;
                    smem_weight[idx] = __float2half(weight[w_idx]);
                }
                else
                {
                    smem_weight[idx] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // Accumulate across TILE_K
        for (int k = 0; k < TILE_K; k += 16)
        {
            // load from shared memory into fragments
            int a_offset = warpM * 16 * TILE_K + k;
            int b_offset = warpN * 16 * TILE_K + k;

            wmma::load_matrix_sync(a_frag, &smem_input[a_offset], TILE_K);
            wmma::load_matrix_sync(b_frag, &smem_weight[b_offset], TILE_K);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    __shared__ float smem_output[TILE_M * TILE_N];

    int output_offset = warpM * 16 * TILE_N + warpN * 16;
    wmma::store_matrix_sync(&smem_output[output_offset], acc_frag, TILE_N, wmma::mem_row_major);

    __syncthreads();

    int num_output_elements = TILE_M * TILE_N;
    elements_per_thread = (num_output_elements + blockDim.x - 1) / blockDim.x;

    // fused bn + add + relu
    for (int i = 0; i < elements_per_thread; i++)
    {
        int idx = threadIdx.x * elements_per_thread + i;
        if (idx < num_output_elements)
        {
            int spatial_idx = idx / TILE_N;
            int oc = oc_base + (idx % TILE_N);

            int linear_pos = spatial_base + spatial_idx;
            int out_y = linear_pos / outW;
            int out_x = linear_pos % outW;

            if (out_y < outH && out_x < outW && oc < out_channels)
            {
                float val = smem_output[idx];

                // BN
                float eps = 1e-5f;
                float std = sqrtf(bnVar[oc] + eps);
                val = (val - bnMean[oc]) / std;
                val = val * bnWeight[oc] + bnBias[oc];

                // residual
                if (residual != nullptr)
                {
                    int out_idx = oc * outH * outW + out_y * outW + out_x;
                    val += residual[out_idx];
                }

                // relu
                if (ReLU && val < 0.0f)
                {
                    val = 0.0f;
                }

                int out_idx = oc * outH * outW + out_y * outW + out_x;
                output[out_idx] = val;
            }
        }
    }
}

__global__ void downsample_kernel(const float *input, const float *weight, const float *bn_weight, const float *bn_bias, const float *bn_mean, const float *bn_var, float *output, int in_ch, int out_ch, int H, int W, float epsilon)
{
    int oc = blockIdx.z;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;

    int outH = (H + 1) / 2;
    int outW = (W + 1) / 2;

    if (out_h >= outH || out_w >= outW)
        return;

    int in_h = out_h * 2;
    int in_w = out_w * 2;

    float sum = 0.0f;
    for (int ic = 0; ic < in_ch; ic++)
    {
        int input_idx = ic * H * W + in_h * W + in_w;
        int weight_idx = oc * in_ch + ic;
        sum += input[input_idx] * weight[weight_idx];
    }

    float scale = bn_weight[oc] / sqrtf(bn_var[oc] + epsilon);
    float bias = bn_bias[oc] - scale * bn_mean[oc];
    float normalized = scale * sum + bias;

    int output_idx = oc * outH * outW + out_h * outW + out_w;
    output[output_idx] = normalized;
}

__global__ void add_kernel(const float *a, const float *b, float *output, int size, bool ReLU)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float result = a[idx] + b[idx];
        if (ReLU)
        {
            result = fmaxf(0.0f, result);
        }
        output[idx] = result;
    }
}

__global__ void maxpool_kernel(const float *input, float *output, int C, int H, int W, int kernel_size, int stride, int padding)
{
    int c = blockIdx.z;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    if (out_h >= outH || out_w >= outW)
        return;

    float max_val = -INFINITY;

    for (int kh = 0; kh < kernel_size; kh++)
    {
        for (int kw = 0; kw < kernel_size; kw++)
        {
            int in_h = out_h * stride + kh - padding;
            int in_w = out_w * stride + kw - padding;

            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W)
            {
                int input_idx = c * H * W + in_h * W + in_w;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }

    int output_idx = c * outH * outW + out_h * outW + out_w;
    output[output_idx] = max_val;
}

__global__ void adaptive_avgpool_kernel(const float *input, float *output, int C, int H, int W)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= C)
        return;

    float sum = 0.0f;
    int size = H * W;

    for (int i = 0; i < size; i++)
    {
        sum += input[c * size + i];
    }

    output[c] = sum / size;
}

__global__ void fc_kernel(const float *input, const float *weight, const float *bias, float *output, int in_features, int out_features)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= out_features)
        return;

    float sum = 0.0f;
    for (int i = 0; i < in_features; i++)
    {
        int weight_idx = out_idx * in_features + i;
        sum += input[i] * weight[weight_idx];
    }

    output[out_idx] = sum + bias[out_idx];
}

void launchMaxPoolKernel(float *input, float *output, int H, int W, int C, int kernel_size, int stride, int padding)
{
    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 block(32, 32, 1);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, C);

    maxpool_kernel<<<grid, block>>>(input, output, C, H, W, kernel_size, stride, padding);
}

void launchConvKernel(float *image, float *output, const ConvLayer &conv, const BatchNorm &bn, int inputDim, int stride, int pad, bool ReLU, const float *residual)
{
    int outputH = computeDim(inputDim, stride, pad, conv.kernelSize);
    int outputW = computeDim(inputDim, stride, pad, conv.kernelSize);

    // num warps per dimension
    int warps_per_M = TILE_M / 16;
    int warps_per_N = TILE_N / 16;
    int warps_per_block = warps_per_M * warps_per_N;

    // 32 threads per warp * number of warps
    dim3 block(warps_per_block * 32);

    // grid dims
    int total_spatial = outputH * outputW;
    int num_spatial_tiles = (total_spatial + TILE_M - 1) / TILE_M;
    int num_channel_tiles = (conv.outputSize + TILE_N - 1) / TILE_N;

    dim3 grid(1, num_spatial_tiles, num_channel_tiles);

    conv2d_kernel<<<grid, block>>>(
        image, conv.d_weight, bn.d_weight, bn.d_bias,
        bn.d_runningMean, bn.d_runningVar, output,
        conv.inputSize, conv.outputSize,
        inputDim, inputDim, outputH, outputW,
        conv.kernelSize, stride, pad, ReLU, residual);

    CHECK_ERROR(cudaGetLastError());
}

void launchDownsampleKernel(float *input, float *output, const Downsample &ds, int H, int W)
{
    int outH = (H + 1) / 2;
    int outW = (W + 1) / 2;

    dim3 block(32, 32, 1);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, ds.weight.outputSize);

    downsample_kernel<<<grid, block>>>(input, ds.weight.d_weight, ds.bn.d_weight, ds.bn.d_bias, ds.bn.d_runningMean, ds.bn.d_runningVar, output, ds.weight.inputSize, ds.weight.outputSize, H, W, 1e-5f);
    CHECK_ERROR(cudaGetLastError());
}

void launchAddKernel(float *a, float *b, float *output, int size, bool ReLU)
{
    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;
    add_kernel<<<gridSize, blockSize>>>(a, b, output, size, ReLU);
    CHECK_ERROR(cudaGetLastError());
}

void runBasicBlock(const BasicBlock &bb, float *input, float *output, int inputChannels, int inputH, int inputW, int stride1)
{
    int midH = (inputH + 2 * 1 - 3) / stride1 + 1;
    int midW = (inputW + 2 * 1 - 3) / stride1 + 1;
    int outH = midH;
    int outW = midW;

    float *temp1, *identity;
    size_t temp1_size = bb.conv1.outputSize * midH * midW * sizeof(float);
    size_t identity_size = bb.conv2.outputSize * outH * outW * sizeof(float);

    CHECK_ERROR(cudaMalloc(&temp1, temp1_size));
    CHECK_ERROR(cudaMalloc(&identity, identity_size));

    // Conv1 + BN1 + ReLU (no residual)
    launchConvKernel(input, temp1, bb.conv1, bb.bn1, inputH, stride1, 1, true, nullptr);

    // Prepare residual
    if (bb.hasDownsample)
    {
        launchDownsampleKernel(input, identity, bb.ds, inputH, inputW);
    }
    else
    {
        size_t copy_size = inputChannels * inputH * inputW * sizeof(float);
        CHECK_ERROR(cudaMemcpy(identity, input, copy_size, cudaMemcpyDeviceToDevice));
    }

    // Conv2 + BN2 + Add + ReLU (all fused!)
    launchConvKernel(temp1, output, bb.conv2, bb.bn2, midH, 1, 1, true, identity);

    cudaFree(temp1);
    cudaFree(identity);
}

void launchAdaptiveAvgPoolKernel(float *input, float *output, int H, int W, int C)
{
    int blockSize = 256;
    int gridSize = (C + blockSize - 1) / blockSize;

    adaptive_avgpool_kernel<<<gridSize, blockSize>>>(input, output, C, H, W);
    CHECK_ERROR(cudaGetLastError());
}

void launchFCKernel(float *input, float *output, const FullyConnected &fc, int in_features, int out_features)
{
    int blockSize = 256;
    int gridSize = (out_features + blockSize - 1) / blockSize;

    fc_kernel<<<gridSize, blockSize>>>(input, fc.d_weight, fc.d_bias, output, in_features, out_features);
    CHECK_ERROR(cudaGetLastError());
}
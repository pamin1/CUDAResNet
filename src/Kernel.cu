/**
 * Author: Prachit Amin
 * ECE 4122
 * 12/1/2025
 * Implements all CUDA Kernels and launching functions.
 */

#include "Kernel.cuh"

__global__ void conv2d_kernel(const float *input, const float *weight, const float *bnWeight, const float *bnBias, const float *bnMean, const float *bnVar, float *output, int in_channels, int out_channels, int H, int W, int outH, int outW, int kernel_size, int stride, int padding)
{
    // shared memory - initialize the weights by channel
    // tile each channel
    extern __shared__ float smem[];

    int hOutStart = blockIdx.y * blockDim.y;
    int wOutStart = blockIdx.x * blockDim.x;

    int hInStart = hOutStart * stride - padding;
    int wInStart = wOutStart * stride - padding;

    int hTile = blockDim.y * stride + (kernel_size - 1);
    int wTile = blockDim.x * stride + (kernel_size - 1);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    int total_elements = TILE_CHANNELS * hTile * wTile;

    int oc = blockIdx.z;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ic += TILE_CHANNELS)
    {
        for (int i = tid; i < total_elements; i += total_threads)
        {
            int c = i / (hTile * wTile);
            int h = (i / wTile) % hTile;
            int w = i % wTile;

            int cGlobal = ic + c;
            int hGlobal = hInStart + h;
            int wGlobal = wInStart + w;

            if ((hGlobal >= 0 && hGlobal < H) && (wGlobal >= 0 && wGlobal < W) && (cGlobal >= 0 && cGlobal < in_channels))
            {
                int globalIdx = cGlobal * H * W + hGlobal * W + wGlobal;
                smem[i] = input[globalIdx];
            }
            else
            {
                smem[i] = 0.0f;
            }
        }
        __syncthreads();

        if (out_h < outH && out_w < outW)
        {

            for (int c = 0; c < TILE_CHANNELS && (ic + c) < in_channels; c++)
            {
                for (int kh = 0; kh < kernel_size; kh++)
                {
                    for (int kw = 0; kw < kernel_size; kw++)
                    {
                        int hLocal = threadIdx.y * stride + kh;
                        int wLocal = threadIdx.x * stride + kw;

                        int smem_idx = c * hTile * wTile + hLocal * wTile + wLocal;
                        int weight_idx = oc * in_channels * kernel_size * kernel_size + (ic + c) * kernel_size * kernel_size + kh * kernel_size + kw;

                        if (smem_idx < total_elements)
                        {
                            sum += smem[smem_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (out_h < outH && out_w < outW)
    {
        int output_idx = oc * outH * outW + out_h * outW + out_w;

        // Correct BatchNorm fusion formula
        float scale = bnWeight[oc] / sqrtf(bnVar[oc] + EPSILON);
        float bias = bnBias[oc] - scale * bnMean[oc];

        output[output_idx] = scale * sum + bias;
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

__global__ void relu_kernel(const float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void add_kernel(const float *a, const float *b, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = a[idx] + b[idx];
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

    dim3 block(16, 16, 1);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, C);

    maxpool_kernel<<<grid, block>>>(input, output, C, H, W, kernel_size, stride, padding);
}

void launchConvKernel(float *image, float *output, const ConvLayer &conv, const BatchNorm &bn, int inputDim, int stride, int pad)
{
    int outputH = computeDim(inputDim, stride, pad, conv.kernelSize);
    int outputW = computeDim(inputDim, stride, pad, conv.kernelSize);

    dim3 block(16, 16, 1);
    dim3 grid((outputW + block.x - 1) / block.x, (outputH + block.y - 1) / block.y, conv.outputSize);

    conv2d_kernel<<<grid, block>>>(image, conv.d_weight, bn.d_weight, bn.d_bias, bn.d_runningMean, bn.d_runningVar, output, conv.inputSize, conv.outputSize, inputDim, inputDim, outputH, outputW, conv.kernelSize, stride, pad);
}

void launchDownsampleKernel(float *input, float *output, const Downsample &ds, int H, int W)
{
    int outH = (H + 1) / 2;
    int outW = (W + 1) / 2;

    dim3 block(16, 16, 1);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, ds.weight.outputSize);

    downsample_kernel<<<grid, block>>>(input, ds.weight.d_weight, ds.bn.d_weight, ds.bn.d_bias, ds.bn.d_runningMean, ds.bn.d_runningVar, output, ds.weight.inputSize, ds.weight.outputSize, H, W, 1e-5f);
}

void launchAddKernel(float *a, float *b, float *output, int size)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    add_kernel<<<gridSize, blockSize>>>(a, b, output, size);
}

void launchReLUKernel(float *input, float *output, int size)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(input, output, size);
}

void runBasicBlock(const BasicBlock &bb, float *input, float *output, int inputChannels, int inputH, int inputW, int stride1)
{
    bool isDownsample = (stride1 == 2);

    int midH = (inputH + 2 * 1 - 3) / stride1 + 1;
    int midW = (inputW + 2 * 1 - 3) / stride1 + 1;
    int outH = midH;
    int outW = midW;

    float *temp1, *temp2, *identity;
    size_t temp1_size = bb.conv1.outputSize * midH * midW * sizeof(float);
    size_t temp2_size = bb.conv2.outputSize * outH * outW * sizeof(float);

    cudaMalloc(&temp1, temp1_size);
    cudaMalloc(&temp2, temp2_size);
    cudaMalloc(&identity, temp2_size);

    launchConvKernel(input, temp1, bb.conv1, bb.bn1, inputH, stride1, 1);
    launchReLUKernel(temp1, temp1, bb.conv1.outputSize * midH * midW);

    launchConvKernel(temp1, temp2, bb.conv2, bb.bn2, midH, 1, 1);

    if (bb.hasDownsample)
    {
        launchDownsampleKernel(input, identity, bb.ds, inputH, inputW);
    }
    else
    {
        size_t copy_size = inputChannels * inputH * inputW * sizeof(float);
        cudaMemcpy(identity, input, copy_size, cudaMemcpyDeviceToDevice);
    }

    launchAddKernel(temp2, identity, output, bb.conv2.outputSize * outH * outW);

    launchReLUKernel(output, output, bb.conv2.outputSize * outH * outW);

    cudaFree(temp1);
    cudaFree(temp2);
    cudaFree(identity);
}

void launchAdaptiveAvgPoolKernel(float *input, float *output, int H, int W, int C)
{
    int blockSize = 256;
    int gridSize = (C + blockSize - 1) / blockSize;

    adaptive_avgpool_kernel<<<gridSize, blockSize>>>(input, output, C, H, W);
}

void launchFCKernel(float *input, float *output, const FullyConnected &fc, int in_features, int out_features)
{
    int blockSize = 256;
    int gridSize = (out_features + blockSize - 1) / blockSize;

    fc_kernel<<<gridSize, blockSize>>>(input, fc.d_weight, fc.d_bias, output, in_features, out_features);
}
/**
 * Author: Prachit Amin
 * ECE 4122
 * 12/1/2025
 * Implements all CUDA Kernels and launching functions.
 */

#include "Kernel.cuh"

__global__ void conv2d_kernel(const float *input, const float *weight, const float *bnWeight, const float *bnBias, const float *bnMean, const float *bnVar, float *output, int in_channels, int out_channels, int H, int W, int outH, int outW, int kernel_size, int stride, int padding)
{
    extern __shared__ float smem[];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int oc = blockIdx.z;

    // smem dimensions
    int smem_width = blockDim.x * stride + kernel_size - 1;
    int smem_height = blockDim.y * stride + kernel_size - 1;
    int totalElem = smem_width * smem_height;
    int blockStride = blockDim.x * blockDim.y;

    // partition shared memory
    int smem_img_size = smem_width * smem_height;
    float *smem_img = smem;
    float *smem_weight = smem + smem_img_size; // weights after image tile

    // block start position in output
    int block_out_x = blockIdx.x * blockDim.x;
    int block_out_y = blockIdx.y * blockDim.y;

    // thread position in output
    int out_x = block_out_x + threadIdx.x;
    int out_y = block_out_y + threadIdx.y;

    // block start in input space
    int block_input_start_x = block_out_x * stride - padding;
    int block_input_start_y = block_out_y * stride - padding;

    float sum = 0.0f;

    // loop over input channels
    for (int ic = 0; ic < in_channels; ic++)
    {
        // cooperative tiling
        for (int i = tid; i < totalElem; i += blockStride)
        {
            int smem_x = i % smem_width;
            int smem_y = i / smem_width;

            // map to global input position
            int in_x = block_input_start_x + smem_x;
            int in_y = block_input_start_y + smem_y;

            if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H)
            {
                int input_idx = ic * (H * W) + in_y * W + in_x;
                smem_img[i] = input[input_idx];
            }
            else
            {
                smem_img[i] = 0.0f;
            }
        }

        int weightsPerChannel = kernel_size * kernel_size;
        for (int i = tid; i < weightsPerChannel; i += blockStride)
        {
            int kh = i / kernel_size;
            int kw = i % kernel_size;
            int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
            smem_weight[i] = weight[weight_idx];
        }
        __syncthreads();

        // compute convolution if this thread produces a valid output
        if (out_x < outW && out_y < outH)
        {
            // thread output position in input
            int thread_input_start_x = out_x * stride - padding;
            int thread_input_start_y = out_y * stride - padding;

            // block input start offset
            int smem_base_x = thread_input_start_x - block_input_start_x;
            int smem_base_y = thread_input_start_y - block_input_start_y;

            // convolve
            for (int kh = 0; kh < kernel_size; kh++)
            {
                for (int kw = 0; kw < kernel_size; kw++)
                {
                    int smem_x = smem_base_x + kw;
                    int smem_y = smem_base_y + kh;
                    int smem_idx = smem_y * smem_width + smem_x;

                    int smem_w_idx = kh * kernel_size + kw;
                    sum += smem_img[smem_idx] * smem_weight[smem_w_idx];
                }
            }
        }
        __syncthreads();
    }

    // apply batch norm and write output
    if (out_x < outW && out_y < outH)
    {
        float scale = bnWeight[oc] / sqrtf(bnVar[oc] + EPSILON);
        float bn_output = scale * (sum - bnMean[oc]) + bnBias[oc];

        int output_idx = oc * (outH * outW) + out_y * outW + out_x;
        output[output_idx] = bn_output;
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

    dim3 block(8, 8, 1);
    dim3 grid((outputW + block.x - 1) / block.x, (outputH + block.y - 1) / block.y, conv.outputSize);

    // smem sizing
    int radius = conv.kernelSize / 2;
    int smemImageSize = (block.x * stride + conv.kernelSize - 1) * (block.y * stride + conv.kernelSize - 1) * sizeof(float);
    int smemWeightSize = conv.kernelSize * conv.kernelSize;
    int totalSmem = (smemImageSize + smemWeightSize) * sizeof(float);

    conv2d_kernel<<<grid, block, totalSmem>>>(image, conv.d_weight, bn.d_weight, bn.d_bias, bn.d_runningMean, bn.d_runningVar, output, conv.inputSize, conv.outputSize, inputDim, inputDim, outputH, outputW, conv.kernelSize, stride, pad);
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